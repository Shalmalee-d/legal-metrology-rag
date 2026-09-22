import json
import logging
import os
import tempfile
import time
from datetime import date
from pathlib import Path

from app.extraction.product_rule import ProductRule
from app.extraction.rule_schema import ComplianceRule
from app.validation.rule_validator import validate_rule


logger = logging.getLogger(__name__)


RULES_DIR = Path("data/rules")
RULES_FILE = RULES_DIR / "compliance_rules.json"
_DEFAULT_RULES_DIR = Path("data/rules")
_DEFAULT_RULES_FILE = _DEFAULT_RULES_DIR / "compliance_rules.json"

# Curated product-rule store (Rule Engine contract). The RAG pipeline never
# writes these files; it owns only RAG_GENERATED_FILENAME below.
PRODUCT_COMMON_FILENAME = "compliance_rules.json"
PRODUCT_README_FILENAME = "README.json"
# RAG pipeline working snapshot. Formerly the default rules file; separated so
# fingerprint-gated RAG updates can never overwrite the curated catalog.
RAG_GENERATED_FILENAME = "rag_generated.json"


class CategoryNotFoundError(ValueError):
    """A requested product category has no rule file."""


def _resolve_rules_dir() -> Path:
    if RULES_DIR != _DEFAULT_RULES_DIR:
        return RULES_DIR
    from app.config import get_rules_dir

    return get_rules_dir()


def _resolve_rag_store() -> Path:
    """RAG pipeline snapshot location.

    An explicitly configured rules file (test redirect or deployment
    override) always wins; otherwise the pipeline uses its dedicated store
    so curated catalog files are never overwritten.
    """
    if RULES_FILE != _DEFAULT_RULES_FILE:
        return RULES_FILE
    from app.config import get_rules_file

    configured = get_rules_file()
    if configured != _DEFAULT_RULES_FILE:
        return configured
    return _resolve_rules_dir() / RAG_GENERATED_FILENAME


def load_rules() -> list[ComplianceRule]:
    rules_file = _resolve_rag_store()
    if not rules_file.exists():
        return []

    try:
        with open(rules_file, "r", encoding="utf-8") as file:
            data = json.load(file)
    except json.JSONDecodeError as error:
        raise ValueError(f"Rule repository is corrupted: {rules_file}") from error

    if not isinstance(data, list):
        raise ValueError("Rule repository must contain a JSON list.")

    return [ComplianceRule(**item) for item in data]


def save_rules(rules: list[ComplianceRule]) -> None:
    invalid = {
        rule.rule_id: validate_rule(rule)
        for rule in rules
        if validate_rule(rule)
    }
    if invalid:
        details = "; ".join(f"{rule_id}: {', '.join(errors)}" for rule_id, errors in invalid.items())
        raise ValueError(f"Refusing to persist invalid compliance rules: {details}")
    rules_dir = _resolve_rules_dir()
    rules_file = _resolve_rag_store()
    rules_dir.mkdir(parents=True, exist_ok=True)

    # Replace only after the complete JSON snapshot has been written. This lets
    # readers continue using the previous valid snapshot if a write fails.
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=rules_dir,
        prefix=f".{rules_file.name}.",
        suffix=".tmp",
        delete=False,
    ) as file:
        temporary_file = Path(file.name)
        json.dump(
            [rule.model_dump() for rule in rules],
            file,
            indent=2,
            ensure_ascii=False,
        )
        file.flush()
        os.fsync(file.fileno())
    try:
        _replace_file_safely(temporary_file, _resolve_rag_store())
    finally:
        temporary_file.unlink(missing_ok=True)


def _replace_file_safely(temporary_file: Path, destination: Path) -> None:
    """Retry an atomic replace; never truncate a previous valid snapshot."""
    last_error: PermissionError | None = None
    for _ in range(5):
        try:
            os.replace(temporary_file, destination)
            return
        except PermissionError as error:
            last_error = error
            time.sleep(0.25)
    assert last_error is not None
    raise last_error


def add_rule(rule: ComplianceRule) -> None:
    """Add a version once; merge repeat evidence for the same logical version."""
    rules = load_rules()
    for index, existing in enumerate(rules):
        if existing.rule_id == rule.rule_id and existing.version == rule.version:
            pages = sorted(set(existing.source_pages + rule.source_pages))
            evidence = existing.evidence_text
            if rule.evidence_text not in evidence:
                evidence = f"{evidence}\n\n{rule.evidence_text}"
            rules[index] = existing.model_copy(update={"source_pages": pages, "evidence_text": evidence})
            save_rules(rules)
            return
    rules.append(rule)
    save_rules(rules)


def update_rule(rule: ComplianceRule) -> None:
    """Replace an exact version or supersede older versions of that rule."""
    rules = load_rules()
    updated_rules = []

    for existing in rules:
        if existing.rule_id == rule.rule_id:
            if existing.version == rule.version:
                updated_rules.append(rule)
            else:
                updated_rules.append(
                    existing.model_copy(update={"status": "superseded"})
                )
        else:
            updated_rules.append(existing)

    if not any(
        existing.rule_id == rule.rule_id
        and existing.version == rule.version
        for existing in rules
    ):
        updated_rules.append(rule)

    save_rules(updated_rules)


def update_rules(rules_to_update: list[ComplianceRule]) -> dict[str, int]:
    """Activate a completed document update with one atomic repository write."""
    # Fail fast: no structurally/semantically invalid candidate may enter
    # activation, even if save_rules() would also refuse the final snapshot.
    incoming_invalid = {
        rule.rule_id: validate_rule(rule)
        for rule in rules_to_update
        if validate_rule(rule)
    }
    if incoming_invalid:
        details = "; ".join(f"{rule_id}: {', '.join(errors)}" for rule_id, errors in incoming_invalid.items())
        raise ValueError(f"Refusing to activate invalid compliance rules: {details}")
    # Old snapshots may predate mandatory validation.  They are never exposed
    # as active rules, and a subsequent valid activation does not re-persist
    # them as though they were accepted regulatory knowledge.
    current_rules = [rule for rule in load_rules() if not validate_rule(rule)]
    affected_documents = {
        rule.source_document_id for rule in rules_to_update if rule.source_document_id
    }
    incoming_by_document: dict[str, set[str]] = {}
    for rule in rules_to_update:
        if rule.source_document_id:
            incoming_by_document.setdefault(rule.source_document_id, set()).add(
                rule.source_identity or rule.rule_id
            )
    rules_superseded = 0
    # Retire any previously active rule from an updated source that is absent
    # from its newly validated source snapshot.  Other documents are retained.
    retired = []
    for existing in current_rules:
        document_id = existing.source_document_id
        identity = existing.source_identity or existing.rule_id
        if (
            document_id in affected_documents
            and identity not in incoming_by_document.get(document_id, set())
            and existing.status == "active"
        ):
            retired.append(existing.model_copy(update={"status": "superseded"}))
            rules_superseded += 1
        else:
            retired.append(existing)
    current_rules = retired
    rules_added = 0
    for rule in rules_to_update:
        identity = rule.source_identity or rule.rule_id
        replaced = False
        next_rules = []
        for existing in current_rules:
            existing_identity = existing.source_identity or existing.rule_id
            if existing_identity != identity:
                next_rules.append(existing)
            elif existing.version == rule.version:
                next_rules.append(rule)
                replaced = True
            else:
                next_rules.append(existing.model_copy(update={"status": "superseded"}))
        if not replaced:
            next_rules.append(rule)
            rules_added += 1
        current_rules = next_rules
    save_rules(current_rules)
    return {"rules_added": rules_added, "rules_updated": len(rules_to_update), "rules_superseded": rules_superseded}


def get_active_rules() -> list[ComplianceRule]:
    rules = load_rules()
    today = date.today()
    active: list[ComplianceRule] = []
    for rule in rules:
        if rule.status != "active" or validate_rule(rule):
            continue
        if rule.effective_from and date.fromisoformat(rule.effective_from) > today:
            continue
        if rule.effective_until and date.fromisoformat(rule.effective_until) < today:
            continue
        active.append(rule)
    return active


# ---------------------------------------------------------------------------
# Curated product-rule catalog (Rule Engine contract).
#
# data/rules/compliance_rules.json holds the canonical common rules and
# data/rules/<category>.json holds category-specific rules. These files are
# curated source of truth: read-only for every runtime path. The RAG pipeline
# above owns only its dedicated generated store and can neither read nor
# write the catalog through these functions.
# ---------------------------------------------------------------------------


def list_categories() -> list[str]:
    """Category identifiers from rule files present on disk.

    The identifier is the filename stem (``food`` for ``food.json``).
    Reserved names (common file, README, RAG store, backups, dotfiles) are
    never treated as categories and are not hardcoded per-category.
    """
    rules_dir = _resolve_rules_dir()
    if not rules_dir.exists():
        return []
    excluded = {PRODUCT_COMMON_FILENAME, PRODUCT_README_FILENAME, RAG_GENERATED_FILENAME}
    categories = []
    for path in rules_dir.glob("*.json"):
        if path.name in excluded or path.name.startswith(".") or ".bak." in path.name:
            continue
        categories.append(path.stem)
    return sorted(categories)


def _load_product_envelope(path: Path) -> list[ProductRule]:
    """Parse one catalog file; failures name the file and stop that file only."""
    try:
        with open(path, "r", encoding="utf-8") as file:
            data = json.load(file)
    except FileNotFoundError as error:
        raise ValueError(f"Product rule file not found: {path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"Product rule file is malformed: {path}") from error
    if isinstance(data, dict):
        records = data.get("rules")
    elif isinstance(data, list):
        records = data
    else:
        records = None
    if not isinstance(records, list):
        raise ValueError(f"Product rule file must contain a rules list: {path}")
    rules = []
    for item in records:
        if not isinstance(item, dict):
            raise ValueError(f"Product rule file contains a non-object rule: {path}")
        try:
            rules.append(ProductRule(**item))
        except Exception as error:
            raise ValueError(
                f"Invalid product rule {item.get('rule_id', '?')!r} in {path}: {error}"
            ) from error
    return rules


def load_common_rules() -> list[ProductRule]:
    """All structurally valid common rules from compliance_rules.json."""
    return _load_product_envelope(_resolve_rules_dir() / PRODUCT_COMMON_FILENAME)


def load_category_rules(category: str) -> list[ProductRule]:
    """All structurally valid rules for one category (possibly empty).

    Raises :class:`CategoryNotFoundError` for an unknown category.
    """
    if category not in list_categories():
        raise CategoryNotFoundError(f"Unknown product category: {category!r}")
    return _load_product_envelope(_resolve_rules_dir() / f"{category}.json")


def is_applicable(rule: ProductRule, today: date | None = None) -> bool:
    """Whether a product rule currently applies (active, in effect)."""
    if rule.status != "active":
        return False
    if rule.effective_from and date.fromisoformat(rule.effective_from) > (today or date.today()):
        return False
    return True


def load_applicable_rules(category: str | None, today: date | None = None) -> list[ProductRule]:
    """Active common rules plus the active rules of one category.

    ``category=None`` returns the common set only. Category-specific files
    are never duplicated into the common set: the Rule Engine combines them
    here instead.
    """
    applicable = [rule for rule in load_common_rules() if is_applicable(rule, today)]
    if category is not None:
        applicable.extend(rule for rule in load_category_rules(category)
                          if is_applicable(rule, today))
    return applicable


def load_all_active_rules(today: date | None = None) -> list[ProductRule]:
    """Active common rules plus every category's active rules.

    One malformed category file never hides the rest of the catalog here;
    the offending file still fails loudly when addressed directly.
    """
    rules = [rule for rule in load_common_rules() if is_applicable(rule, today)]
    for category in list_categories():
        try:
            rules.extend(rule for rule in load_category_rules(category)
                         if is_applicable(rule, today))
        except ValueError:
            logger.warning("Skipping unreadable category file: %s", category)
    return rules


class ProductRuleConversionError(ValueError):
    """A validated extraction cannot become a catalog rule without invention."""


def _normalize_rule_text(text: str) -> str:
    import re
    import unicodedata

    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip().casefold()


_MEANINGFUL_COMPARE_FIELDS = (
    "title", "requirement", "applies_when", "check_type",
    "check_parameters", "evidence_required", "effective_from", "status",
)


def to_product_rule(rule: ComplianceRule, *, source_title: str,
                    source_pages: list[int]) -> ProductRule:
    """Deterministically convert a validated extraction into a catalog rule.

    Only fields the model actually supplied (via its product overlay) or the
    pipeline traceability provides are used. Anything required but missing
    fails instead of being fabricated. ``rule_id`` is resolved against the
    target file: an existing rule with the same normalized requirement keeps
    its ID (amendment path), otherwise a deterministic per-file ID is minted.
    """
    overlay = getattr(rule, "_product_overlay", None) or {}
    missing = [key for key in ("category", "title", "applies_when", "check_type")
               if not isinstance(overlay.get(key), str) or not overlay[key].strip()]
    if missing:
        raise ProductRuleConversionError(
            f"Cannot persist rule without model-supplied {', '.join(missing)}; refusing to invent them."
        )
    category = overlay["category"].strip()
    if category != "common" and category not in list_categories():
        raise CategoryNotFoundError(f"Unknown product category: {category!r}")
    requirement = (rule.requirement or "").strip()
    if not requirement:
        raise ProductRuleConversionError("Cannot persist a rule with an empty requirement.")
    check_parameters = overlay.get("check_parameters")
    if check_parameters is None:
        if rule.expected_value is not None:
            check_parameters = {"value": rule.expected_value}
            if rule.expected_unit is not None:
                check_parameters["unit"] = rule.expected_unit
        else:
            check_parameters = {}
    if not isinstance(check_parameters, dict):
        raise ProductRuleConversionError("check_parameters must be an object.")
    evidence_required = overlay.get("evidence_required")
    if evidence_required is None:
        evidence_required = []
    if not isinstance(evidence_required, list) or not all(isinstance(item, str) for item in evidence_required):
        raise ProductRuleConversionError("evidence_required must be a list of strings.")
    pages = [page for page in (source_pages or rule.source_pages) if isinstance(page, int) and page > 0]
    rule_id, _is_new = resolve_product_rule_id(requirement, category)
    return ProductRule(
        rule_id=rule_id,
        category=category,
        title=overlay["title"].strip(),
        requirement=requirement,
        applies_when=overlay["applies_when"].strip(),
        check_type=overlay["check_type"].strip(),
        check_parameters=check_parameters,
        evidence_required=evidence_required,
        source=[{"document": source_title, "page": pages[0] if pages else None}],
        source_text=rule.evidence_text,
        effective_from=overlay.get("effective_from"),
        status="active",
    )


def _catalog_path_for_category(category: str) -> Path:
    if category == "common":
        return _resolve_rules_dir() / PRODUCT_COMMON_FILENAME
    if category not in list_categories():
        raise CategoryNotFoundError(f"Unknown product category: {category!r}")
    return _resolve_rules_dir() / f"{category}.json"


def _read_envelope(path: Path) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as file:
            data = json.load(file)
    except FileNotFoundError as error:
        raise ValueError(f"Product rule file not found: {path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"Product rule file is malformed: {path}") from error
    if not isinstance(data, dict) or not isinstance(data.get("rules"), list):
        raise ValueError(f"Product rule file must contain a rules list: {path}")
    return data


def _write_envelope(path: Path, envelope: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     prefix=f".{path.name}.", suffix=".tmp",
                                     delete=False) as file:
        temporary = Path(file.name)
        json.dump(envelope, file, indent=2, ensure_ascii=False)
        file.flush()
        os.fsync(file.fileno())
    try:
        _replace_file_safely(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def resolve_product_rule_id(requirement: str, category: str) -> tuple[str, bool]:
    """Resolve a stable ID: existing on normalized-requirement match, else mint.

    Minting derives the ``LM_<PREFIX>`` prefix from the target file's own IDs
    and takes max sequence + 1, so IDs stay deterministic and human-consistent
    without any hardcoded category registry. Returns ``(rule_id, is_new)``.
    """
    import re

    path = _catalog_path_for_category(category)
    envelope = _read_envelope(path)
    records = envelope["rules"]
    ids = [item.get("rule_id") for item in records if isinstance(item, dict)]
    if len(set(ids)) != len(ids):
        raise ValueError(f"Duplicate rule_ids already present in {path}; refusing to modify.")
    wanted = _normalize_rule_text(requirement)
    for item in records:
        if isinstance(item, dict) and _normalize_rule_text(item.get("requirement", "")) == wanted:
            return item["rule_id"], False
    prefixes: dict[str, int] = {}
    top_sequence = 0
    for rule_id in ids:
        match = re.match(r"^(LM_[A-Z]+)_(\d+)$", str(rule_id))
        if not match:
            continue
        prefixes[match.group(1)] = prefixes.get(match.group(1), 0) + 1
        top_sequence = max(top_sequence, int(match.group(2)))
    if prefixes:
        prefix = sorted(prefixes.items(), key=lambda item: (-item[1], item[0]))[0][0]
        sequence = top_sequence + 1
    else:
        stem = re.sub(r"[^A-Za-z]", "", category).upper() or "GEN"
        prefix, sequence = f"LM_{stem}", 1
    return f"{prefix}_{sequence:03d}", True


def add_or_update_rule(rule: ProductRule) -> dict:
    """Append a new rule or replace one amended rule, nothing else.

    Returns ``{"action": "added"|"updated"|"noop", "rule_id": ..., "file": ...}``.
    Comparison uses meaningful content only, so source-text-only drift is a
    no-op. Unrelated rules, envelope metadata, and ``rule_count`` stay correct.
    """
    path = _catalog_path_for_category(rule.category)
    envelope = _read_envelope(path)
    records = envelope["rules"]
    ids = [item.get("rule_id") for item in records if isinstance(item, dict)]
    if len(set(ids)) != len(ids):
        raise ValueError(f"Duplicate rule_ids already present in {path}; refusing to modify.")
    dumped = rule.model_dump()
    action = "added"
    replaced = False
    next_records = []
    for item in records:
        if not isinstance(item, dict) or item.get("rule_id") != rule.rule_id:
            next_records.append(item)
            continue
        if all(item.get(field) == dumped.get(field) for field in _MEANINGFUL_COMPARE_FIELDS):
            return {"action": "noop", "rule_id": rule.rule_id, "file": str(path)}
        next_records.append(dumped)
        replaced = True
    if not replaced:
        next_records.append(dumped)
    else:
        action = "updated"
    envelope = {**envelope, "rules": next_records, "rule_count": len(next_records)}
    _write_envelope(path, envelope)
    return {"action": action, "rule_id": rule.rule_id, "file": str(path)}


def load_applicable_rules(category: str | None, today: date | None = None) -> list[ProductRule]:
    """Active common rules plus the active rules of one category.

    ``category=None`` returns the common set only. Category-specific files
    are never duplicated into the common set: the Rule Engine combines them
    here instead.
    """
    applicable = [rule for rule in load_common_rules() if is_applicable(rule, today)]
    if category is not None:
        applicable.extend(rule for rule in load_category_rules(category)
                          if is_applicable(rule, today))
    return applicable


def load_all_active_rules(today: date | None = None) -> list[ProductRule]:
    """Active common rules plus every category's active rules (RAG status use)."""
    rules = [rule for rule in load_common_rules() if is_applicable(rule, today)]
    for category in list_categories():
        try:
            rules.extend(rule for rule in load_category_rules(category)
                         if is_applicable(rule, today))
        except ValueError:
            logger.warning("Skipping unreadable category file: %s", category)
    return rules
