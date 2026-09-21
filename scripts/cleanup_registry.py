"""Developer-only registry recovery tool.

Does NOT run during tests. Does NOT filter production discovery.
Use explicitly to inspect or remove a known-bad record (e.g. a test URL
that previously polluted the real registry due to missing isolation).

Examples (PowerShell):
  .\\venv\\Scripts\\python.exe scripts\\cleanup_registry.py --list
  .\\venv\\Scripts\\python.exe scripts\\cleanup_registry.py --dry-run --remove-url https://example.gov.in/rules.pdf
  .\\venv\\Scripts\\python.exe scripts\\cleanup_registry.py --remove-url https://example.gov.in/rules.pdf

Safety:
- Always backs up data/registry/document_registry.json before writing.
- Requires explicit --remove-url; never bulk-deletes.
- Never touches data/rules/compliance_rules.json.
- Refuses to run when RAG_* test-isolation env vars point at tmp (to avoid
  accidentally cleaning a test registry).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path


PROD_REGISTRY = Path("data/registry/document_registry.json")


def _is_test_isolation_active() -> bool:
    # Autouse pytest fixture sets these; refuse to clean a tmp registry.
    return any(os.getenv(name) for name in ("RAG_REGISTRY_FILE", "RAG_REGISTRY_DIR", "RAG_DATA_ROOT"))


def _load() -> list[dict]:
    if not PROD_REGISTRY.exists():
        return []
    with open(PROD_REGISTRY, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Registry must contain a JSON list.")
    return data


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect/clean production document registry.")
    parser.add_argument("--list", action="store_true", help="List all registry records (url + status).")
    parser.add_argument("--remove-url", help="Exact normalized URL to remove (requires confirmation unless --yes).")
    parser.add_argument("--dry-run", action="store_true", help="Show what would change without writing.")
    parser.add_argument("--yes", action="store_true", help="Skip confirmation prompt.")
    args = parser.parse_args()

    if _is_test_isolation_active():
        print("Refusing: RAG_* isolation env vars are set (looks like a test run). Unset them to clean PROD registry.")
        return 2

    records = _load()
    print(f"Production registry: {PROD_REGISTRY} ({len(records)} records)")

    if args.list or (not args.remove_url):
        for r in records:
            print(f"- {r.get('url')} | status={r.get('status')} | fp={str(r.get('fingerprint'))[:12]} pp={str(r.get('processed_fingerprint'))[:12] if r.get('processed_fingerprint') else None}")
        if not args.remove_url:
            return 0

    target = args.remove_url
    assert target is not None
    matched = [r for r in records if r.get("url") == target]
    if not matched:
        # Try normalized comparison hint, but do not auto-delete similar URLs.
        print(f"No exact match for {target!r}. Use --list to see exact stored URLs.")
        return 1

    print(f"Matched {len(matched)} record(s) for removal:")
    for r in matched:
        print(json.dumps(r, indent=2, ensure_ascii=False))

    if args.dry_run:
        print("Dry run: no changes written.")
        return 0

    if not args.yes:
        confirm = input(f"Remove {len(matched)} record(s)? Type YES to confirm: ")
        if confirm.strip() != "YES":
            print("Aborted.")
            return 1

    backup = PROD_REGISTRY.with_suffix(f".json.bak.{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}")
    PROD_REGISTRY.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(PROD_REGISTRY, backup)
    print(f"Backup written to {backup}")

    remaining = [r for r in records if r.get("url") != target]
    tmp = PROD_REGISTRY.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(remaining, f, indent=2, ensure_ascii=False)
    os.replace(tmp, PROD_REGISTRY)
    print(f"Removed. {len(remaining)} records remain.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
