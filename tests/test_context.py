import dataclasses
import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


@dataclasses.dataclass
class FakeProfile:
    name: str = "antigravity-oauth"
    base_url: str = "agy://local"

    def get_model_context_length(self, model):
        return 200_000


def load(monkeypatch, with_base=True):
    registry = {"antigravity-oauth": FakeProfile(base_url="x://y")} if with_base else {}
    prov = types.ModuleType("providers")
    prov.get_provider_profile = registry.get
    prov.register_provider = lambda p: registry.__setitem__(p.name, p)
    monkeypatch.setitem(sys.modules, "providers", prov)
    spec = importlib.util.spec_from_file_location("ctxplug", ROOT / "__init__.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod._warned.clear()
    return mod, registry


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for k in ("ANTIGRAVITY_CONTEXT_TABLE", "ANTIGRAVITY_CONTEXT_OVERRIDES", "ANTIGRAVITY_CONTEXT_CAP"):
        monkeypatch.delenv(k, raising=False)


@pytest.mark.parametrize("model,want", [
    ("gemini-4-hydrogen", 1048576), ("gemini-4-hydrogen-xhigh", 1048576),
    ("Gemini-4-Argon-Medium", 1048576), ("gemini-3.8-flash-high", 1048576),
    ("claude-opus-5-5-high", 1000000), ("claude-sonnet-5-5-medium", 1000000),
    ("gpt-oss-120b-medium", 131072),
])
def test_family_prefix(monkeypatch, model, want):
    mod, reg = load(monkeypatch)
    p = reg["antigravity-oauth"]
    assert p.get_model_context_length(model) == want
    assert type(p.get_model_context_length(model)) is int


def test_registration_copies_fields(monkeypatch):
    _, reg = load(monkeypatch)
    p = reg["antigravity-oauth"]
    assert p.base_url == "x://y" and type(p) is not FakeProfile


def test_unknown_falls_back_to_base(monkeypatch):
    _, reg = load(monkeypatch)
    p = reg["antigravity-oauth"]
    assert p.get_model_context_length("some-unknown-model") == 200_000
    assert p.get_model_context_length(None) == 200_000
    assert p.get_model_context_length("") == 200_000


def test_longest_prefix_wins(monkeypatch, tmp_path):
    t = tmp_path / "t.json"
    t.write_text(json.dumps({"families": {"gemini-4": 111, "gemini-4-hydrogen-lite": {"context": 222}}}))
    monkeypatch.setenv("ANTIGRAVITY_CONTEXT_TABLE", str(t))
    mod, _ = load(monkeypatch)
    assert mod.resolve("gemini-4-hydrogen-lite-high") == 222
    assert mod.resolve("gemini-4-other") == 111
    assert mod.resolve("gemini-4-hydrogen-high") == 1048576  # bundled still merged


def test_overrides(monkeypatch):
    monkeypatch.setenv("ANTIGRAVITY_CONTEXT_OVERRIDES", "gemini-4-hydrogen=524288, Claude-Opus-5-5=400000")
    mod, reg = load(monkeypatch)
    p = reg["antigravity-oauth"]
    assert p.get_model_context_length("gemini-4-hydrogen-high") == 524288
    assert p.get_model_context_length("claude-opus-5-5-low") == 400000
    assert p.get_model_context_length("gemini-4-argon") == 1048576


def test_cap(monkeypatch):
    monkeypatch.setenv("ANTIGRAVITY_CONTEXT_CAP", "262144")
    mod, _ = load(monkeypatch)
    assert mod.resolve("gemini-4-hydrogen") == 262144
    assert mod.resolve("gpt-oss-120b-medium") == 131072  # below cap untouched
    assert mod.resolve("unknown") is None  # cap does not invent values


def test_garbage_env_ignored_with_single_warning(monkeypatch, caplog):
    monkeypatch.setenv("ANTIGRAVITY_CONTEXT_OVERRIDES", "foo,gemini-4-argon=abc,=5,claude-opus-5-5=-3,gemini-4-hydrogen=500")
    monkeypatch.setenv("ANTIGRAVITY_CONTEXT_CAP", "lots")
    monkeypatch.setenv("ANTIGRAVITY_CONTEXT_TABLE", "/nonexistent/table.json")
    mod, _ = load(monkeypatch)
    with caplog.at_level("WARNING"):
        for _ in range(3):
            assert mod.resolve("gemini-4-argon") == 1048576
            assert mod.resolve("claude-opus-5-5") == 1000000
    assert mod.resolve("gemini-4-hydrogen") == 500
    msgs = [r.getMessage() for r in caplog.records]
    assert len(msgs) == len(set(msgs)) == 6


@pytest.mark.parametrize("cap", ["0", "-5", "abc", "1.5", " "])
def test_bad_cap_ignored(monkeypatch, cap):
    monkeypatch.setenv("ANTIGRAVITY_CONTEXT_CAP", cap)
    mod, _ = load(monkeypatch)
    assert mod.resolve("gemini-4-hydrogen") == 1048576


@pytest.mark.parametrize("model,want", [
    ("gemini-4-hydrogenx", None), ("gpt-oss-1200b", None), ("gemini-4-hydrogen", 1048576),
    ("  GEMINI-4-HYDROGEN-HIGH ", 1048576), ("gemini-4-hydrogen:high", 1048576),
])
def test_prefix_boundary(monkeypatch, model, want):
    mod, _ = load(monkeypatch)
    assert mod.resolve(model) == want


def test_bad_table_entries_skipped(monkeypatch, tmp_path):
    t = tmp_path / "t.json"
    t.write_text('{"families": {"": 5, "  ": 6, "g": true, "h": 1.5, "i": -1, "j": "x", " Gemini-9 ": 7}}')
    monkeypatch.setenv("ANTIGRAVITY_CONTEXT_TABLE", str(t))
    mod, _ = load(monkeypatch)
    assert mod.resolve("unknown-model") is None  # empty key must not match everything
    assert mod.resolve("gemini-9-x") == 7
    assert "g" not in mod.load_table() and "h" not in mod.load_table()


@pytest.mark.parametrize("body", ["", '{"families": ', "[]", '{"families": []}', '{"families": 3}', "null", "\xff\xfe"])
def test_corrupt_custom_table_fails_open(monkeypatch, tmp_path, body):
    t = tmp_path / "t.json"
    t.write_bytes(body.encode("latin-1"))
    monkeypatch.setenv("ANTIGRAVITY_CONTEXT_TABLE", str(t))
    mod, reg = load(monkeypatch)
    assert reg["antigravity-oauth"].get_model_context_length("gemini-4-hydrogen") == 1048576  # bundled intact


def test_corrupt_bundled_table_falls_back_to_base(monkeypatch, tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text('{"families": {"gemini-4-hyd')
    mod, reg = load(monkeypatch)
    monkeypatch.setattr(mod, "BUNDLED_TABLE", bad)
    assert reg["antigravity-oauth"].get_model_context_length("gemini-4-hydrogen") == 200_000
    monkeypatch.setattr(mod, "BUNDLED_TABLE", tmp_path / "missing.json")
    assert reg["antigravity-oauth"].get_model_context_length("gemini-4-hydrogen") == 200_000


def test_yaml_table(monkeypatch, tmp_path):
    t = tmp_path / "t.yaml"
    t.write_text("families:\n  gemini-4-hydrogen: 300000\n")
    monkeypatch.setenv("ANTIGRAVITY_CONTEXT_TABLE", str(t))
    mod, _ = load(monkeypatch)
    try:
        import yaml  # noqa: F401
    except ImportError:
        assert mod.resolve("gemini-4-hydrogen") == 1048576  # no PyYAML: file ignored, bundled intact
    else:
        assert mod.resolve("gemini-4-hydrogen") == 300000


def test_resolver_exception_falls_back_to_base(monkeypatch):
    mod, reg = load(monkeypatch)
    monkeypatch.setattr(mod, "resolve", lambda m: 1 / 0)
    assert reg["antigravity-oauth"].get_model_context_length("gemini-4-hydrogen") == 200_000


def test_install_twice_still_works(monkeypatch):
    mod, reg = load(monkeypatch)
    assert mod.install() is True
    assert reg["antigravity-oauth"].get_model_context_length("claude-opus-5-5") == 1000000
    assert reg["antigravity-oauth"].get_model_context_length("zzz") == 200_000


def test_non_dataclass_base_is_noop(monkeypatch):
    prov = types.ModuleType("providers")
    prov.get_provider_profile = lambda n: object()
    prov.register_provider = lambda p: pytest.fail("must not register")
    monkeypatch.setitem(sys.modules, "providers", prov)
    spec = importlib.util.spec_from_file_location("ctxplug3", ROOT / "__init__.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # must not raise


def test_missing_base_is_noop(monkeypatch):
    mod, reg = load(monkeypatch, with_base=False)
    assert reg == {}
    assert mod.install() is False


def test_missing_providers_module_is_noop(monkeypatch):
    monkeypatch.setitem(sys.modules, "providers", None)  # import raises ImportError
    spec = importlib.util.spec_from_file_location("ctxplug2", ROOT / "__init__.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # must not raise


def test_bundled_table_file(monkeypatch):
    data = json.loads((ROOT / "context_table.json").read_text())["families"]
    assert set(data) == {"gemini-4-hydrogen", "gemini-4-argon", "gemini-3.8-flash",
                         "claude-opus-5-5", "claude-sonnet-5-5", "gpt-oss-120b"}
    for fam, e in data.items():
        assert e["context"] > 0 and e["source"] and isinstance(e["confirmed"], bool)
    assert [f for f, e in data.items() if not e["confirmed"]] == ["claude-opus-5-5", "claude-sonnet-5-5", "gpt-oss-120b"]
    mod, _ = load(monkeypatch)
    assert mod.load_table()["gemini-4-hydrogen"] == 1048576


def test_discovery_stubbed(monkeypatch, caplog):
    mod, _ = load(monkeypatch)
    ids = ["gemini-4-hydrogen-high", "brand-new-gemini-9"]
    with caplog.at_level("WARNING"):
        rep = mod.report(ids)
    assert rep == {"gemini-4-hydrogen-high": 1048576, "brand-new-gemini-9": None}
    assert "brand-new-gemini-9" in caplog.text


def test_live_ids_parse_and_fail_open(monkeypatch):
    mod, _ = load(monkeypatch)
    seen = {}

    def fake_run(cmd, **kw):
        seen.update(cmd=cmd, **kw)
        return types.SimpleNamespace(stdout="Fetching available models...\ngemini-4-argon-high\tGemini 4\nclaude-opus-5-5-low\nnoise\n")
    monkeypatch.setattr(mod.shutil, "which", lambda _: "/fake/agy")
    monkeypatch.setattr(mod.subprocess, "run", fake_run)
    assert mod.live_model_ids() == ["gemini-4-argon-high", "claude-opus-5-5-low"]
    assert seen["cmd"] == ["/fake/agy", "models"] and seen["timeout"] and "shell" not in seen
    monkeypatch.setattr(mod.subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(mod.subprocess.TimeoutExpired("agy", 5)))
    assert mod.live_model_ids() is None
    monkeypatch.setattr(mod.shutil, "which", lambda _: None)
    assert mod.live_model_ids() is None and mod.report() == {}


def test_resolve_never_calls_agy(monkeypatch):
    mod, reg = load(monkeypatch)
    monkeypatch.setattr(mod.subprocess, "run", lambda *a, **k: pytest.fail("agy called on hot path"))
    assert reg["antigravity-oauth"].get_model_context_length("gemini-4-hydrogen") == 1048576
