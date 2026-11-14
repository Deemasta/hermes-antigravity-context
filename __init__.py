"""antigravity-oauth-context: per-model context windows for `antigravity-oauth`.

The base plugin returns a flat 200000 for every model. This companion
re-registers the same provider name with a subclass whose
get_model_context_length() resolves, in order:

  1. per-family override   env ANTIGRAVITY_CONTEXT_OVERRIDES="gemini-4-hydrogen=524288,claude-opus-5-5=400000"
  2. bundled family table  context_table.json (or env ANTIGRAVITY_CONTEXT_TABLE=/path/custom.json,
                           merged over the bundled one; .yaml/.yml accepted if PyYAML is importable)
  3. cap                   env ANTIGRAVITY_CONTEXT_CAP=<int>, applied to 1-2 (default: no cap; useful
                           because agy reportedly compacts near 256K despite advertising 1M)
  4. None -> base plugin behaviour (ANTIGRAVITY_CONTEXT_LENGTH / 200000)

Matching is lowercased longest-family-prefix on a separator boundary (`-_:@/`), so
`gemini-4-hydrogen` matches `gemini-4-hydrogen-xhigh` but not `gemini-4-hydrogenx`. A future sibling
such as `gemini-4-hydrogen-lite` still inherits the parent value until it gets its own table entry. Invalid env values are ignored
with one warning each; nothing here raises.

LOAD ORDER: name the directory so it sorts AFTER `antigravity-oauth`. Provider plugins are imported in
sorted order and register_provider() is last-writer-wins. On Hermes 0.23.1 an earlier-sorting name also
happens to work (our get_provider_profile() miss forces a nested rescan that imports the base first), but
that is incidental; install() logs a warning if the base is not found.

The hot path (get_model_context_length) is table lookup (re-reads the small JSON each call; no subprocess). The `agy models`
subprocess is only run by report() / unmatched_live_models(), never at import
and never during resolution.
"""
from __future__ import annotations

import dataclasses
import json
import logging
import os
import shutil
import subprocess
from pathlib import Path

log = logging.getLogger(__name__)

BASE_NAME = "antigravity-oauth"
BUNDLED_TABLE = Path(__file__).resolve().parent / "context_table.json"
_warned: set[str] = set()


def _warn_once(msg: str) -> None:
    if msg not in _warned:
        _warned.add(msg)
        log.warning("antigravity-oauth-context: %s", msg)


def _to_pos_int(value) -> int | None:
    try:
        n = int(str(value).strip())
    except (TypeError, ValueError):
        return None
    return n if n > 0 else None


def _read_table_file(path) -> dict[str, int]:
    """Parse a table file into {family: context}; bad file/entries -> skipped."""
    out: dict[str, int] = {}
    try:
        p = Path(path)
        text = p.read_text(encoding="utf-8")
        if p.suffix.lower() in (".yaml", ".yml"):
            import yaml  # optional
            data = yaml.safe_load(text)
        else:
            data = json.loads(text)
        families = data["families"]
        items = families.items()
    except Exception as exc:  # noqa: BLE001 - never raise from a config read
        _warn_once(f"ignoring unreadable table {path}: {exc}")
        return out
    for fam, entry in items:
        n = _to_pos_int(entry.get("context") if isinstance(entry, dict) else entry)
        key = str(fam).strip().lower()
        if n is None or not key:  # an empty key would prefix-match every model
            _warn_once(f"ignoring invalid table entry {fam!r}")
        else:
            out[key] = n
    return out


def load_table() -> dict[str, int]:
    table = _read_table_file(BUNDLED_TABLE)
    custom = os.environ.get("ANTIGRAVITY_CONTEXT_TABLE", "").strip()
    if custom:
        table.update(_read_table_file(custom))
    return table


def parse_overrides(raw: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for part in raw.split(","):
        if not part.strip():
            continue
        fam, _, val = part.partition("=")
        n = _to_pos_int(val)
        if not fam.strip() or n is None:
            _warn_once(f"ignoring invalid ANTIGRAVITY_CONTEXT_OVERRIDES entry {part.strip()!r}")
        else:
            out[fam.strip().lower()] = n
    return out


def _family_lookup(table: dict[str, int], model: str) -> int | None:
    model = model.strip().lower()
    for fam in sorted(table, key=len, reverse=True):
        if model.startswith(fam) and (len(model) == len(fam) or model[len(fam)] in "-_:@/"):
            return table[fam]
    return None


def resolve(model) -> int | None:
    """Resolve a model id to a positive int, or None to defer to the base plugin."""
    if not isinstance(model, str) or not model.strip():
        return None
    value = _family_lookup(parse_overrides(os.environ.get("ANTIGRAVITY_CONTEXT_OVERRIDES", "")), model)
    if value is None:
        value = _family_lookup(load_table(), model)
    if value is None:
        return None
    raw_cap = os.environ.get("ANTIGRAVITY_CONTEXT_CAP", "").strip()
    if raw_cap:
        cap = _to_pos_int(raw_cap)
        if cap is None:
            _warn_once(f"ignoring invalid ANTIGRAVITY_CONTEXT_CAP {raw_cap!r}")
        else:
            value = min(value, cap)
    return value


# --- live model discovery (diagnostics only; never on the resolution path) ---

def live_model_ids(timeout: float = 5.0) -> list[str] | None:
    """Model ids from `agy models` (tab-separated `id<TAB>label`). None if unavailable."""
    agy = shutil.which("agy")
    if not agy:
        return None
    try:
        res = subprocess.run([agy, "models"], capture_output=True, text=True, timeout=timeout,
                             stdin=subprocess.DEVNULL)
        return [ln.split()[0] for ln in res.stdout.splitlines()
                if ln.strip() and any(k in ln.split()[0].lower() for k in ("gemini", "claude", "gpt"))]
    except Exception as exc:  # noqa: BLE001
        log.debug("antigravity-oauth-context: agy models failed: %s", exc)
        return None


def unmatched_live_models(ids: list[str] | None = None) -> list[str]:
    """Live ids matching no family in the table; warns once per id."""
    ids = live_model_ids() if ids is None else ids
    table = load_table()
    missing = [m for m in (ids or []) if _family_lookup(table, m) is None]
    for m in missing:
        _warn_once(f"live model {m!r} matches no family in the context table (falls back to base value)")
    return missing


def report(ids: list[str] | None = None) -> dict[str, int | None]:
    """{live model id: resolved context or None}. Run: python <plugin-dir>/__init__.py"""
    ids = live_model_ids() if ids is None else ids
    unmatched_live_models(ids)
    return {m: resolve(m) for m in ids or []}


# --- registration ---

def install() -> bool:
    """Re-register the base profile with per-model resolution. No-op if base is absent."""
    try:
        from providers import get_provider_profile, register_provider
        base = get_provider_profile(BASE_NAME)
        if base is None:
            log.warning("antigravity-oauth-context: %s profile not found; context windows unchanged", BASE_NAME)
            return False

        class ContextAwareProfile(type(base)):
            def get_model_context_length(self, model):
                try:
                    value = resolve(model)
                except Exception:  # noqa: BLE001
                    value = None
                return value if value is not None else super().get_model_context_length(model)

        fields = {f.name: getattr(base, f.name) for f in dataclasses.fields(base) if f.init}
        register_provider(ContextAwareProfile(**fields))
        return True
    except Exception as exc:  # noqa: BLE001 - a companion must never break provider loading
        log.debug("antigravity-oauth-context: install skipped: %s", exc)
        return False


install()


if __name__ == "__main__":  # diagnostics: the hyphenated dir name is not importable
    print(json.dumps(report(), indent=2))
