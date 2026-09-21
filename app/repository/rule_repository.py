import json
import os
import tempfile
import time
from datetime import date
from pathlib import Path

from app.extraction.rule_schema import ComplianceRule
from app.validation.rule_validator import validate_rule


RULES_DIR = Path("data/rules")
RULES_FILE = RULES_DIR / "compliance_rules.json"
_DEFAULT_RULES_DIR = Path("data/rules")
_DEFAULT_RULES_FILE = _DEFAULT_RULES_DIR / "compliance_rules.json"


def _resolve_rules_dir() -> Path:
    if RULES_DIR != _DEFAULT_RULES_DIR:
        return RULES_DIR
    from app.config import get_rules_dir

    return get_rules_dir()


def _resolve_rules_file() -> Path:
    if RULES_FILE != _DEFAULT_RULES_FILE:
        return RULES_FILE
    from app.config import get_rules_file

    return get_rules_file()


def load_rules() -> list[ComplianceRule]:
    rules_file = _resolve_rules_file()
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
    rules_file = _resolve_rules_file()
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
        _replace_file_safely(temporary_file, _resolve_rules_file())
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
