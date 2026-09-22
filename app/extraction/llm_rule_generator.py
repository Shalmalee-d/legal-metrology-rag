"""Local Ollama-backed proposals for regulatory compliance rules."""

import json
import logging
import re
import unicodedata
from typing import Any

import httpx
from pydantic import ValidationError

from app.config import (
    get_ollama_base_url,
    get_ollama_connect_timeout_seconds,
    get_ollama_max_output_tokens,
    get_ollama_model,
    get_ollama_timeout_seconds,
)
from app.extraction.rule_schema import ComplianceRule
from app.validation.product_checkability import validate_product_rule
from app.validation.rule_validator import validate_rule


logger = logging.getLogger(__name__)


# Compact structured-output contract for Ollama /api/generate.
# "format": "json" only guarantees valid JSON, not schema-conformant output.
# A JSON Schema in `format` constrains the decoder to this shape. Real
# regulatory chunks average ~1700 chars (max 2400); asking for up to two rules
# with full evidence regularly exceeds the 256-token output budget and ends
# truncated mid-string. One concise rule per call fits reliably.
OLLAMA_RESPONSE_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "rules": {
            "type": "array",
            "maxItems": 1,
            "items": {
                "type": "object",
                "properties": {
                    "rule_id": {"type": "string"},
                    "parameter": {"type": "string"},
                    "condition": {
                        "type": "string",
                        "enum": [
                            "must_exist",
                            "must_not_exist",
                            "must_equal",
                            "must_be_greater_than",
                            "must_be_less_than",
                            "must_match",
                        ],
                    },
                    "requirement": {"type": "string"},
                    "evidence_text": {"type": "string"},
                    "applies_to": {"type": ["object", "null"]},
                    "expected_value": {"type": ["string", "number", "boolean", "null"]},
                    "expected_unit": {"type": ["string", "null"]},
                    "category": {"type": "string"},
                    "title": {"type": "string"},
                    "applies_when": {"type": "string"},
                    "check_type": {"type": "string"},
                    "check_parameters": {"type": "object"},
                    "evidence_required": {"type": "array", "items": {"type": "string"}},
                    "effective_from": {"type": ["string", "null"]},
                },
                "required": ["rule_id", "parameter", "condition", "requirement", "evidence_text"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["rules"],
    "additionalProperties": False,
}


def _build_request(model: str, prompt: str) -> dict:
    """Compact deterministic request: schema-constrained, no reasoning stream."""
    return {
        "model": model,
        "prompt": prompt,
        "format": OLLAMA_RESPONSE_SCHEMA,
        "stream": False,
        # Qwen3 otherwise spends its response budget in reasoning and can
        # leave ``response`` empty. The prompt requires final JSON only.
        "think": False,
        # Bounded for one concise single-line rule object. Do not raise this
        # to fix truncation; keep the output small instead.
        "options": {"num_predict": get_ollama_max_output_tokens(), "temperature": 0},
    }


class OllamaUnavailableError(RuntimeError):
    """The local model could not be contacted or returned an unusable response."""

    MODEL_TIMEOUT = "MODEL_TIMEOUT"
    MODEL_TRANSPORT_ERROR = "MODEL_TRANSPORT_ERROR"
    MODEL_JSON_ERROR = "MODEL_JSON_ERROR"
    MODEL_SCHEMA_ERROR = "MODEL_SCHEMA_ERROR"

    def __init__(self, message: str, *, error_type: str | None = None,
                 timeout_type: str | None = None, kind: str | None = None) -> None:
        super().__init__(message)
        self.error_type = error_type
        self.timeout_type = timeout_type
        self.kind = kind


class OllamaModelError(OllamaUnavailableError):
    """Ollama was reachable but a model response was structurally unusable.

    Subclasses :class:`OllamaUnavailableError` so existing ``except`` clauses
    keep working, while :attr:`kind` distinguishes malformed JSON
    (``MODEL_JSON_ERROR``) from schema-invalid output (``MODEL_SCHEMA_ERROR``).
    Only this error class is retried, at most once.
    """


class OllamaRuleGenerator:
    """Generate source-validated ``ComplianceRule`` proposals through local Ollama.

    By default, connection settings come from ``OLLAMA_BASE_URL``,
    ``OLLAMA_MODEL``, and ``OLLAMA_TIMEOUT_SECONDS``. Optional constructor
    values are useful for isolated tests and do not introduce cloud services.
    """

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        timeout_seconds: float | None = None,
    ) -> None:
        self.base_url = (base_url or get_ollama_base_url()).rstrip("/")
        self.model = model or get_ollama_model()
        self.timeout_seconds = timeout_seconds if timeout_seconds is not None else get_ollama_timeout_seconds()

    def generate_rules(self, chunk: dict, source_document: str,
                         diagnostics: dict | None = None) -> list[ComplianceRule]:
        """Request JSON proposals and reject any rule lacking chunk evidence.

        Malformed or schema-invalid model output is retried once with the same
        deterministic prompt/schema (two attempts total). Transport/timeout
        errors and deterministic validator rejections are never retried.
        """
        # Chunks without normative signals cannot yield a validated rule: the
        # validator requires normative evidence from the same vocabulary.
        # Skip the model call entirely (Devanagari text still goes to the
        # model since Hindi normative wording uses a separate vocabulary).
        chunk_text = chunk.get("text", "") if isinstance(chunk, dict) else ""
        if not chunk_needs_model_call(chunk_text):
            if diagnostics is not None:
                diagnostics.update(skipped_no_candidates=True, proposed=0)
            return []
        request = _build_request(self.model, _prompt(chunk))
        timeout = httpx.Timeout(
            connect=get_ollama_connect_timeout_seconds(),
            read=self.timeout_seconds,
            write=30.0,
            pool=get_ollama_connect_timeout_seconds(),
        )
        for attempt in (1, 2):
            try:
                if diagnostics is not None:
                    diagnostics["attempts"] = attempt
                response = httpx.post(
                    f"{self.base_url}/api/generate",
                    json=request,
                    timeout=timeout,
                )
                response.raise_for_status()
                try:
                    payload = response.json()
                except (ValueError, json.JSONDecodeError) as error:
                    raise OllamaModelError(
                        "Local Ollama rule generation returned invalid data.",
                        error_type=type(error).__name__,
                        kind=OllamaUnavailableError.MODEL_JSON_ERROR,
                    ) from error
                return _parse_rules(payload, chunk, source_document,
                                    diagnostics=diagnostics)
            except httpx.TimeoutException as error:
                timeout_type = "read" if isinstance(error, httpx.ReadTimeout) else "connect"
                logger.error("Local Ollama rule generation %s timeout: %s", timeout_type, error)
                raise OllamaUnavailableError(
                    f"Local Ollama rule generation failed ({timeout_type} timeout).",
                    error_type=type(error).__name__, timeout_type=timeout_type,
                    kind=OllamaUnavailableError.MODEL_TIMEOUT,
                ) from error
            except httpx.HTTPError as error:
                logger.error("Local Ollama rule generation failed: %s", error)
                raise OllamaUnavailableError(
                    "Local Ollama rule generation failed (HTTP error).",
                    error_type=type(error).__name__,
                    kind=OllamaUnavailableError.MODEL_TRANSPORT_ERROR,
                ) from error
            except OllamaModelError as error:
                if attempt == 1:
                    logger.warning(
                        "Retrying chunk %s after %s (%s).",
                        chunk.get("chunk_id", "?") if isinstance(chunk, dict) else "?",
                        error.kind,
                        error.error_type,
                    )
                    continue
                logger.error(
                    "Local Ollama rule generation failed after 2 attempts (%s): %s",
                    error.kind,
                    error,
                )
                raise


# Deterministic normative-candidate signals. Case-insensitive textual match
# only; this identifies potentially normative spans, never legal meaning.
# English signals mirror the validator's normative vocabulary plus the
# applicability wording ("applicable", "only") used by real advisories for
# operative coverage/permission sentences; Hindi signals match the
# validator's Hindi normative set so Hindi-normative chunks get candidates.
_NORMATIVE_CANDIDATE_SIGNALS = (
    "shall not apply",
    "shall",
    "must",
    "required",
    "allowed",
    "permitted",
    "covered",
    "prohibited",
    "applicable",
    "only",
    "होगा",
    "होंगे",
    "चाहिए",
    "लागू नहीं",
    "अपेक्षित",
    "निषिद्ध",
)
_NORMATIVE_CANDIDATE_RE = re.compile(
    "|".join(r"\b" + re.escape(sig) + r"\b" for sig in sorted(_NORMATIVE_CANDIDATE_SIGNALS, key=len, reverse=True)),
    re.IGNORECASE,
)
# Conservative clause splitter: blank-line paragraphs first, then sentence
# boundaries. Single newlines (PDF line wraps) are kept inside the unit so a
# wrapped sentence is never torn apart. Abbreviation-style fragments such as
# "G." or "No." rarely end a unit because a split requires following text to
# start with an alphanumeric, quote, or Devanagari character.
_CLAUSE_SPLIT_RE = re.compile(r"(?<=[.!?;])\s+(?=[A-Za-z0-9\(\"“‘\u0900-\u097f])")
_PARAGRAPH_SPLIT_RE = re.compile(r"\r?\n\s*\r?\n")
_MIN_CANDIDATE_CHARS = 20


def extract_normative_candidates(chunk_text: Any) -> list[dict]:
    """List verbatim normative candidate spans in deterministic order.

    Each item has ``index`` (1-based), ``text`` (exact contiguous substring of
    ``chunk_text``), ``start``, and ``end`` offsets. Matching is a
    case-insensitive textual signal check only. Page attribution stays at the
    existing chunk level: chunks carry ``source_pages`` and 11 of 216 chunks
    span two pages, so exact offset-to-page mapping is not inferred here.
    Chunks without signals safely yield ``[]``.
    """
    if not isinstance(chunk_text, str) or not chunk_text.strip():
        return []
    candidates: list[dict] = []
    cursor = 0
    for paragraph in _PARAGRAPH_SPLIT_RE.split(chunk_text):
        if not paragraph.strip():
            continue
        for unit in _CLAUSE_SPLIT_RE.split(paragraph):
            text = unit.strip()
            if len(text) < _MIN_CANDIDATE_CHARS:
                continue
            if not _NORMATIVE_CANDIDATE_RE.search(text):
                continue
            start = chunk_text.find(text, cursor)
            if start < 0:
                start = chunk_text.find(text)
            if start < 0:
                continue
            end = start + len(text)
            cursor = end
            candidates.append({"index": len(candidates) + 1, "text": text, "start": start, "end": end})
    return candidates


def chunk_needs_model_call(chunk_text: Any) -> bool:
    """Whether a chunk justifies a model call.

    Chunks without normative candidate signals cannot yield a validated rule:
    the validator requires normative evidence from the same English/Hindi
    vocabulary the candidate signals mirror. Such chunks skip the model, so
    zero-candidate chunks produce zero Ollama HTTP requests.
    """
    if not isinstance(chunk_text, str) or not chunk_text.strip():
        return False
    return bool(extract_normative_candidates(chunk_text))


def _candidate_block(chunk_text: str) -> str:
    candidates = extract_normative_candidates(chunk_text)
    if not candidates:
        return (
            "RELEVANT CANDIDATE SENTENCES: none found.\n"
            "If no candidate supports a valid actionable rule, return {\"rules\": []}."
        )
    lines = [
        "RELEVANT CANDIDATE SENTENCES (verbatim excerpts from the regulatory text above; "
        "the full text remains the complete context):"
    ]
    for candidate in candidates:
        lines.append(f"CANDIDATE {candidate['index']}:")
        lines.append(f"\"{candidate['text']}\"")
    return "\n".join(lines)


def _prompt(chunk: dict) -> str:
    return f"""You extract product-facing Legal Metrology Packaged Commodities compliance rules for a downstream Rule Engine.
Use ONLY the supplied regulatory text. Do not use outside knowledge and do not invent requirements. Do not infer a requirement that is not supported by the source text.

Extract ONLY an explicit, observable requirement, prohibition, threshold, declaration, measurement, dimensional, pricing, quantity, marking, or labelling requirement, or a conditional applicability/exemption that the Rule Engine can check against a packaged commodity, product, package, or package label.
Every rule must answer: "What can the downstream Rule Engine actually check on a product or package?"
Focus on: packaged commodities, packages, labels, declarations, net quantity, MRP/retail sale price inclusive of taxes, manufacturer/packer/importer name and address, dimensions and letter/numeral height, mandatory markings and information, and other concrete package/product conditions.

Ignore document titles, short titles, Gazette or publication identifiers, REGD numbers, rule or amendment names and numbers, section/rule citations, headings, dates of publication, commencement or effective-date statements, and administrative information UNLESS that passage directly establishes a product/package requirement. Do not create rules merely saying "Rule X was amended", "corrigendum issued", or "document effective on date Y". An amendment is eligible only when the text states its regulatory effect as a new, modified, or removed product/package requirement.

Each rule needs a concrete parameter, an actionable requirement stating what must be present or absent on the package, a valid condition, and an exact contiguous evidence_text quote. When the rule states an exemption, exception, or non-applicability for a package or category, you MUST include applies_to identifying the affected package or category from the source (for example {{"package_type": "export_package"}}); do not leave it null. Otherwise include applies_to only when the text states an applicability or exemption condition. Numeric threshold fields are governed by the unconditional comparison rule below, not by optionality. A definition alone is not a rule; it qualifies only when it creates a structured applicability or exemption condition the Rule Engine can use.

Do NOT extract document metadata or administrative text as product rules. Return no rule for document titles, short titles, Gazette or publication identifiers, rule or amendment names, section/rule numbers, citations, headings, commencement/effective dates by themselves, "these rules may be called", "these rules shall come into force", bare definitions, or the mere existence of a mentioned concept. In particular, never turn text such as a document identifier, a rule name, "package", "regulatory document", or "package must exist/comply" into a vague must_exist rule.

Use only: must_exist, must_not_exist, must_equal, must_be_greater_than, must_be_less_than, must_match. If you choose a comparison condition (must_be_less_than, must_be_greater_than, must_equal, or another numeric comparison), you MUST provide both expected_value and expected_unit. Never omit them and never put the numeric threshold only inside requirement; a bare value such as "50 kg" as the requirement is forbidden. If the source does not provide a clear numeric value AND unit supporting a comparison, DO NOT choose a comparison condition. Return {{"rules": []}} or use an appropriate non-comparison condition only when the source genuinely supports it. Write requirement as one complete actionable sentence, e.g. "Packages of agricultural farm produce up to 50 kg are covered under the Rules." Quote as evidence_text the contiguous operative sentence or clause containing shall, must, required, allowed, permitted, covered under the Rules, prohibited, or not permitted; never use a bare noun phrase such as "packages of agriculture farm produce upto 50 kg". Requirement and evidence MUST refer to the same threshold and the same regulatory effect: do not combine a 50 kg requirement with 25 kg evidence. When a chunk contains both (a) an exclusion or exception threshold and (b) an applicability or permission statement for the target product, do NOT automatically use the first numeric clause and do NOT select an exclusion clause merely because it contains "shall not apply" and a number; choose the clause describing the actual regulatory effect for the target product. Extract a rule only when the source states such an actionable effect; otherwise return {{"rules": []}}. Never invent or paraphrase evidence.

Return JSON only as a single-line compact object with no markdown, no ```json fences, no explanations, no reasoning, no comments, and no trailing text. Return at most ONE concise high-confidence rule; prefer fewer precise rules over many weak rules, and never split one requirement into several rules. Each item has: {{"rule_id": str, "parameter": str, "condition": str, "requirement": str, "evidence_text": str}} plus applies_to only when supported, plus expected_value and expected_unit which are REQUIRED for comparison conditions. Additionally always include the product-catalog overlay: {{"category": str, "title": str, "applies_when": str, "check_type": str}} with optional {{"check_parameters": object, "evidence_required": [str], "effective_from": str|null}}. Category is "common" for general rules or the exact product category filename stem (for example "food", "electronics", "medical_devices"). Title is a short human label; applies_when is one clause stating when the rule applies; check_type is a short snake_case effect label such as must_exist, must_not_exist, must_equal, must_be_greater_than, must_be_less_than, must_match, conditional_exemption, or permitted_practice. If the chunk contains no actionable compliance requirement, return {{"rules": []}} exactly (empty rules, return [] ).
Evidence_text must be the complete contiguous source sentence or clause that establishes the same regulatory effect as the generated rule; do not paraphrase it. Before choosing evidence, compare parameter, condition, expected_value, and expected_unit against the proposed evidence span. The evidence must support the exact threshold and effect: a 50 kg rule must never use 25 kg exclusion evidence. When several numeric clauses exist, do not select one merely because it contains a number, "shall", "shall not apply", or a similar parameter; it must establish the actual rule being generated. For an applicability or permission rule, prefer the operative sentence stating that applicability or permission for the target product over an earlier exception or exclusion clause. Never build a rule from one sentence and evidence from a different regulatory effect. If no contiguous span directly supports the exact generated rule, return {{"rules": []}}. Evidence must not be a noun phrase or isolated fragment; it must contain enough of the operative sentence or clause to establish the rule. Keep evidence to one sentence or clause (about 250 characters or less); longer passages waste the output budget and risk truncation. Empty, invented, or unrelated evidence is invalid. Keep requirement concise (one sentence).
Evidence_text MUST be copied verbatim from exactly one presented candidate below: the complete candidate text, not a fragment assembled from several candidates and never from an unlisted passage. A 50 kg rule must never cite the 25 kg candidate. If no candidate supports a valid actionable rule, return {{"rules": []}}. The full regulatory text above remains available for surrounding context such as applicability, definitions, and exceptions; candidates only highlight potentially normative spans and are not legal conclusions.

Regulatory text:\n{chunk['text']}\n\n{_candidate_block(chunk['text'])}"""


def _normalise_with_source_positions(text: str) -> tuple[str, list[int]]:
    """Return comparison text and an index back to the original source text.

    PDF text often inserts line breaks, soft hyphens, curly quotes, or spacing
    around punctuation.  These are formatting differences, not different legal
    words.  We normalize only those differences, retaining every letter and
    number in its original order so a paraphrase still cannot match.
    """
    output: list[str] = []
    positions: list[int] = []
    pending_separator = False
    for source_index, character in enumerate(text):
        normalized = unicodedata.normalize("NFKC", character).casefold()
        for normalized_character in normalized:
            if normalized_character.isalnum():
                if pending_separator and output:
                    output.append(" ")
                    positions.append(source_index)
                output.append(normalized_character)
                positions.append(source_index)
                pending_separator = False
            else:
                pending_separator = bool(output)
    return "".join(output), positions


def _requirement_source_words(source_text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+|[\u0900-\u097f]+", source_text.lower()))


def _split_concatenated_token(token: str, source_words: set[str]) -> str:
    """Split one concatenated token using exact source vocabulary only."""
    lowered = token.lower()
    if lowered in source_words or len(lowered) < 4:
        return token
    # Preserve leading/trailing punctuation while repairing the core word.
    prefix_match = re.match(r"^([^A-Za-z0-9]*)([A-Za-z0-9]+)([^A-Za-z0-9]*)$", token)
    if not prefix_match:
        return token
    leading, core, trailing = prefix_match.groups()
    core_lower = core.lower()
    if core_lower in source_words or len(core_lower) < 4:
        return token
    for pos in range(2, len(core_lower) - 1):
        left, right = core_lower[:pos], core_lower[pos:]
        if len(left) < 2 or len(right) < 2:
            continue
        if left in source_words and right in source_words:
            if core.isupper():
                fixed = left.upper() + " " + right.upper()
            elif core[0].isupper():
                fixed = core[:pos] + " " + core[pos:].lower()
            else:
                fixed = core[:pos] + " " + core[pos:]
            return f"{leading}{fixed}{trailing}"
    return token


# Tiny fallback for unambiguous function-word concatenations. These strings are
# never valid English words in regulatory text; splitting is whitespace repair,
# not spellchecking. Evidence-anchored splitting above handles the general case.
_SAFE_SPACING_FIXES = (
    (re.compile(r"\bonthe\b", re.IGNORECASE), "on the"),
    (re.compile(r"\bofthe\b", re.IGNORECASE), "of the"),
    (re.compile(r"\binthe\b", re.IGNORECASE), "in the"),
    (re.compile(r"\btothe\b", re.IGNORECASE), "to the"),
    (re.compile(r"\bforthe\b", re.IGNORECASE), "for the"),
)


def _normalize_requirement_text(requirement: Any, source_text: Any = None) -> Any:
    """Deterministic whitespace repair for model-generated requirements only.

    Evidence is never touched. This collapses whitespace and repairs missing
    spaces using exact source vocabulary (e.g. "onthe" -> "on the" when both
    "on" and "the" appear in the chunk). No spellchecking or rewording.
    """
    if not isinstance(requirement, str):
        return requirement
    text = re.sub(r"\s+", " ", requirement).strip()
    if not text:
        return text
    # Restore a space after a comma before a letter ("prepackage,the").
    # Digits are excluded so thousands separators ("1,000") are preserved.
    text = re.sub(r",(?=[A-Za-z\u0900-\u097f])", ", ", text)
    if isinstance(source_text, str) and source_text.strip():
        source_words = _requirement_source_words(source_text)
        parts = text.split(" ")
        repaired = [_split_concatenated_token(part, source_words) for part in parts]
        text = " ".join(repaired)
        text = re.sub(r"\s+", " ", text).strip()
    for pattern, replacement in _SAFE_SPACING_FIXES:
        def _preserve_case(match: re.Match) -> str:
            hit = match.group(0)
            if hit.isupper():
                return replacement.upper()
            if hit[0].isupper():
                return replacement.capitalize()
            return replacement
        text = pattern.sub(_preserve_case, text)
    return text


def _grounded_evidence(evidence: str, source_text: str) -> str | None:
    """Return the exact source span for formatting-equivalent evidence only."""
    normalized_evidence, _ = _normalise_with_source_positions(evidence)
    normalized_source, source_positions = _normalise_with_source_positions(source_text)
    # A bare word is not enough context to substantiate a legal requirement.
    if len(normalized_evidence) < 12:
        return None
    start = normalized_source.find(normalized_evidence)
    if start < 0:
        return None
    end = start + len(normalized_evidence) - 1
    return source_text[source_positions[start]:source_positions[end] + 1].strip()


_MAX_REPORTED_REJECTIONS = 3
_MAX_REJECTION_CHARS = 200


_PRODUCT_OVERLAY_KEYS = (
    "category", "title", "applies_when", "check_type",
    "check_parameters", "evidence_required", "effective_from",
)


def _parse_rules(payload: Any, chunk: dict, source_document: str,
                 diagnostics: dict | None = None) -> list[ComplianceRule]:
    def _fail(kind: str, message: str) -> OllamaModelError:
        return OllamaModelError(message, error_type="ValueError", kind=kind)

    response_text = payload.get("response") if isinstance(payload, dict) else None
    if not isinstance(response_text, str):
        raise _fail(OllamaUnavailableError.MODEL_SCHEMA_ERROR,
                    "Ollama response did not contain a JSON response string.")
    stripped = response_text.strip()
    if not stripped:
        raise _fail(OllamaUnavailableError.MODEL_SCHEMA_ERROR,
                    "Ollama response was empty.")
    # Tolerate only harmless markdown fencing. Never scan prose for a
    # JSON-looking substring: truncated or trailing-text output must fail.
    if stripped.startswith("```") and stripped.endswith("```"):
        inner = stripped[3:-3].strip()
        if inner.lower().startswith("json"):
            inner = inner[4:].strip()
        stripped = inner
    # Qwen may expose reasoning separately, or wrap it in a legacy think
    # section. Reasoning is never treated as a rule proposal.
    if "</think>" in stripped:
        stripped = stripped.split("</think>", 1)[1].strip()
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError as error:
        raise OllamaModelError(
            "Local Ollama rule generation returned invalid data.",
            error_type=type(error).__name__,
            kind=OllamaUnavailableError.MODEL_JSON_ERROR,
        ) from error
    candidates = parsed.get("rules", []) if isinstance(parsed, dict) else parsed
    if not isinstance(candidates, list):
        raise _fail(OllamaUnavailableError.MODEL_SCHEMA_ERROR,
                    "Ollama rule response must contain a rules list.")
    if diagnostics is not None:
        diagnostics.update(proposed=len(candidates), empty_response=not candidates,
                           validation_rejected=0, rejection_reasons=[])

    rules = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            if diagnostics is not None:
                diagnostics["validation_rejected"] += 1
                if len(diagnostics["rejection_reasons"]) < _MAX_REPORTED_REJECTIONS:
                    diagnostics["rejection_reasons"].append("candidate is not a JSON object.")
            continue
        # Product-catalog overlay rides alongside the extraction fields. It is
        # validated at persistence time, never here, and never invented: only
        # keys the model actually supplied are carried forward.
        overlay = {key: candidate.pop(key) for key in _PRODUCT_OVERLAY_KEYS if key in candidate}
        # Source traceability belongs to the pipeline, not the LLM.
        candidate["source_document"] = source_document
        candidate["source_pages"] = chunk["source_pages"]
        candidate["status"] = "active"
        evidence = candidate.get("evidence_text", "")
        source_evidence = (
            _grounded_evidence(evidence, chunk["text"])
            if isinstance(evidence, str)
            else None
        )
        if source_evidence is None:
            logger.warning("Rejected LLM rule with evidence absent from source chunk.")
            if diagnostics is not None:
                diagnostics["validation_rejected"] += 1
                if len(diagnostics["rejection_reasons"]) < _MAX_REPORTED_REJECTIONS:
                    diagnostics["rejection_reasons"].append(
                        f"{candidate.get('rule_id', '?')}: evidence absent from source chunk."[:_MAX_REJECTION_CHARS])
            continue
        # Persist the exact extracted source passage for auditability, never a
        # model-normalized rendition of it.
        candidate["evidence_text"] = source_evidence
        if isinstance(candidate.get("requirement"), str):
            # Model decoding can drop a space ("onthe"). Repair requirement
            # spacing deterministically using the chunk vocabulary. Evidence
            # is never modified to hide such issues.
            candidate["requirement"] = _normalize_requirement_text(
                candidate["requirement"], chunk["text"]
            )
        try:
            rule = ComplianceRule(**candidate)
        except ValidationError as error:
            logger.warning("Rejected LLM rule that failed schema validation: %s", error)
            if diagnostics is not None:
                diagnostics["validation_rejected"] += 1
                if len(diagnostics["rejection_reasons"]) < _MAX_REPORTED_REJECTIONS:
                    diagnostics["rejection_reasons"].append(
                        f"{candidate.get('rule_id', '?')}: schema: {str(error).splitlines()[0] if str(error).splitlines() else error}"[:_MAX_REJECTION_CHARS])
            continue
        errors = validate_rule(rule)
        if errors:
            logger.warning("Rejected LLM rule that failed deterministic validation: %s", rule.rule_id)
            if diagnostics is not None:
                diagnostics["validation_rejected"] += 1
                if len(diagnostics["rejection_reasons"]) < _MAX_REPORTED_REJECTIONS:
                    diagnostics["rejection_reasons"].append(
                        f"{rule.rule_id}: {errors[0]}"[:_MAX_REJECTION_CHARS])
            continue
        product_errors = validate_product_rule(rule)
        if product_errors:
            logger.warning("Rejected LLM rule that is not product-checkable: %s", rule.rule_id)
            if diagnostics is not None:
                diagnostics["validation_rejected"] += 1
                if len(diagnostics["rejection_reasons"]) < _MAX_REPORTED_REJECTIONS:
                    diagnostics["rejection_reasons"].append(
                        f"{rule.rule_id}: {product_errors[0]}"[:_MAX_REJECTION_CHARS])
            continue
        rule._product_overlay = overlay
        rules.append(rule)
    return rules


def generate_rules_from_chunk(chunk: dict, source_document: str,
                              diagnostics: dict | None = None) -> list[ComplianceRule]:
    """Backward-compatible function wrapper around :class:`OllamaRuleGenerator`."""
    return OllamaRuleGenerator().generate_rules(chunk, source_document, diagnostics=diagnostics)
