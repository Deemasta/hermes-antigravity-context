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
    t.write_text(json.dumps({"families": {"gemini-4": 111, "gemini-4-hydrogen-x": {"context": 222}}}))
    monkeypatch.setenv("ANTIGRAVITY_CONTEXT_TABLE", str(t))
    mod, _ = load(monkeypatch)
    assert mod.resolve("gemini-4-hydrogen-xhigh") == 222
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


def test_cap_zero_or_negative_ignored(monkeypatch):
    monkeypatch.setenv("ANTIGRAVITY_CONTEXT_CAP", "0")
    mod, _ = load(monkeypatch)
    assert mod.resolve("gemini-4-hydrogen") == 1048576


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


def test_live_ids_cached_and_fail_open(monkeypatch):
    mod, _ = load(monkeypatch)
    calls = []

    def fake_run(cmd, **kw):
        calls.append(cmd)
        return types.SimpleNamespace(stdout="Fetching models...\ngemini-4-argon-high  x\nclaude-opus-5-5-low\nnoise\n")
    monkeypatch.setattr(mod.shutil, "which", lambda _: "/fake/agy")
    monkeypatch.setattr(mod.subprocess, "run", fake_run)
    assert mod.live_model_ids() == ["gemini-4-argon-high", "claude-opus-5-5-low"]
    mod.live_model_ids()
    assert len(calls) == 1 and calls[0][1:] == ["models"]
    monkeypatch.setattr(mod.subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(OSError("boom")))
    assert mod.live_model_ids(force=True) is None


def test_resolve_never_calls_agy(monkeypatch):
    mod, reg = load(monkeypatch)
    monkeypatch.setattr(mod.subprocess, "run", lambda *a, **k: pytest.fail("agy called on hot path"))
    assert reg["antigravity-oauth"].get_model_context_length("gemini-4-hydrogen") == 1048576
