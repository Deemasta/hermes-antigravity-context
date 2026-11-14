# antigravity-oauth-context

[![ci](https://github.com/Deemasta/hermes-antigravity-context/actions/workflows/ci.yml/badge.svg)](https://github.com/Deemasta/hermes-antigravity-context/actions/workflows/ci.yml)

Per-model context-window metadata for the Hermes `antigravity-oauth` provider.

The separately installed `antigravity-oauth` plugin (neerazz/hermes-antigravity-oauth) serves Google Antigravity models through the `agy` CLI, but its `get_model_context_length()` returns a hardcoded 200000 tokens for every model. A 1M-token Gemini model is therefore treated as a 200k one, and Hermes starts compressing conversation history far too early. This companion plugin registers a subclass of that provider profile and returns the right window per model from a bundled JSON family table, falling back to the base plugin (200000) for anything it does not know. It never raises: a bad table, bad env values, a missing base plugin or a missing `agy` binary all fail open.

The fix is metadata only. It changes what Hermes believes the window is, which moves the compression trigger. It does not change what the Antigravity service actually accepts.

## Requirements

- Hermes (tested on 0.23.1; `plugin.yaml` declares `requires_hermes: ">=0.21.0"`).
- The `antigravity-oauth` plugin installed, enabled and working.
- The `agy` CLI, only for the optional diagnostics.

## Install

**Critical invariant: the installed directory name must sort lexicographically after `antigravity-oauth`.** Provider plugins load in sorted order and registration is last-writer-wins, so this plugin has to load after the base one. `antigravity-oauth-context` satisfies this; do not rename it to something that sorts earlier. (On Hermes 0.23.1 an earlier-sorting name happened to still work, because a miss in `get_provider_profile()` forces a nested rescan that imports the base first. That is incidental; do not rely on it.)

### Manual

```sh
git clone https://github.com/Deemasta/hermes-antigravity-context $HERMES_HOME/plugins/antigravity-oauth-context
hermes plugins enable antigravity-oauth-context
```

`hermes plugins enable` adds the plugin to the `plugins.enabled` list in `config.yaml`, where `antigravity-oauth` is also listed. You can edit that list by hand instead. Then restart the gateway.

### Per-profile

Plugins live under `$HERMES_HOME/plugins/<name>/`, and each profile has its own `plugins.enabled` list. Install and enable the plugin in every profile that uses `antigravity-oauth`.

### Ansible / IaC (hermes-infra)

Add a `hermes_plugins` entry that pins `name`, `repo` and a commit sha. The `name` (and so the directory) MUST be `antigravity-oauth-context`.

```yaml
hermes_plugins:
  - name: antigravity-oauth-context
    repo: https://github.com/Deemasta/hermes-antigravity-context
    version: <commit sha>
```

The exact field names beyond `name` and `repo` depend on your role; check it.

## How a window is resolved

`get_model_context_length(model)` resolves in this order:

1. Per-family overrides from `ANTIGRAVITY_CONTEXT_OVERRIDES`.
2. The family table: the bundled `context_table.json`, with a custom table from `ANTIGRAVITY_CONTEXT_TABLE` merged over it.
3. The optional cap `ANTIGRAVITY_CONTEXT_CAP`, applied as `min(value, cap)` to the result of step 1 or 2.
4. If nothing matched, `None`, and the base plugin answers (`ANTIGRAVITY_CONTEXT_LENGTH` or 200000).

Matching is on the lowercased model id, using the longest family prefix that ends at a separator boundary (`-`, `_`, `:`, `@`, `/`) or at the end of the id. Model ids arrive with effort suffixes (`-high`, `-xhigh`, `-medium`, `-low`), so `gemini-4-hydrogen` matches `gemini-4-hydrogen-xhigh` but not `gemini-4-hydrogenx`. The cap applies only to matched models; unknown models are not capped.

A future sibling model (for example a hypothetical `gemini-4-hydrogen-lite`) inherits its parent family's window until the table gets its own, longer entry.

The table is re-read on every call (it is small); no subprocess runs during resolution.

## Configuration

All knobs are environment variables read at resolution time.

| Variable | Syntax | Default | Example |
| --- | --- | --- | --- |
| `ANTIGRAVITY_CONTEXT_TABLE` | Path to a `.json` file (or `.yaml`/`.yml` if PyYAML is importable) with the same shape as `context_table.json`. Merged over the bundled table. | unset (bundled table only) | `ANTIGRAVITY_CONTEXT_TABLE=/opt/data/my_table.json` |
| `ANTIGRAVITY_CONTEXT_OVERRIDES` | Comma-separated `family=tokens` pairs; matched with the same prefix rule and checked before the table. | unset | `ANTIGRAVITY_CONTEXT_OVERRIDES="gemini-4-hydrogen=524288,claude-opus-5-5=400000"` |
| `ANTIGRAVITY_CONTEXT_CAP` | Positive integer; upper bound on any matched window. | unset (no cap) | `ANTIGRAVITY_CONTEXT_CAP=262144` |

Invalid values (non-positive or non-integer numbers, malformed pairs, unreadable files, empty family keys) are ignored with one logged warning each; nothing raises.

A custom table file looks like this (a bare number is also accepted in place of the object):

```json
{"families": {"gemini-4-hydrogen": {"context": 524288}}}
```

Hermes's own `model.context_length` and `model_overrides.antigravity-oauth.<model>.context_window` still take precedence over this plugin.

## Bundled model table

From `context_table.json`:

| Family | Window (tokens) | Confirmed | Source |
| --- | --- | --- | --- |
| `gemini-4-hydrogen` | 1048576 | yes | https://ai.google.dev/gemini-api/docs/models/gemini-4-hydrogen |
| `gemini-4-argon` | 1048576 | yes | https://ai.google.dev/gemini-api/docs/models/gemini-4-argon |
| `gemini-3.8-flash` | 1048576 | yes | https://ai.google.dev/gemini-api/docs/tokens (1,048,576 input) |
| `claude-opus-5-5` | 1000000 | no | https://platform.claude.com/docs/en/models/opus-5-5/overview (upstream API value) |
| `claude-sonnet-5-5` | 1000000 | no | https://platform.claude.com/docs/en/models/sonnet-5-5/overview (upstream API value) |
| `gpt-oss-120b` | 131072 | no | https://huggingface.co/openai/gpt-oss-120b (upstream model card) |

The `claude-*` and `gpt-oss-120b` values are upstream vendor figures. The window Antigravity actually serves for them is unconfirmed.

To add or change an entry, edit the `families` object in `context_table.json`, or ship your own file and point `ANTIGRAVITY_CONTEXT_TABLE` at it (its entries win over the bundled ones). Each entry is `{"context": <int>, "confirmed": <bool>, "source": "<url>"}`. `confirmed` and `source` are documentation only; `confirmed: false` does not change behaviour.

## Diagnostics

Run the file directly (the hyphenated directory name is not importable as a module):

```sh
python $HERMES_HOME/plugins/antigravity-oauth-context/__init__.py
```

It prints a JSON object mapping each live model id (those containing `gemini`, `claude` or `gpt`) to its resolved window, or `null` where the base plugin's value would apply. It also logs a warning for every live id that matches no table family. The live ids come from `agy models` with a 5 second timeout and no stdin. If `agy` is missing, fails or times out, the report is an empty object `{}`. Importing the plugin as a provider never runs `agy`.

## Verify the fix

Use the interpreter of your Hermes install (`/opt/hermes/.venv/bin/python` on a docker-managed install):

```sh
HERMES_HOME=$HERMES_HOME /opt/hermes/.venv/bin/python -c 'import hermes_cli.plugins as P; P.discover_plugins(); from agent.model_metadata import get_model_context_length as g; print(g("gemini-4-hydrogen-high", provider="antigravity-oauth", base_url="agy://local"))'
```

Observed on a temporary `HERMES_HOME`, for `gemini-4-hydrogen-high`, `gemini-4-hydrogen-xhigh`, `gemini-4-argon-medium`, `gemini-3.8-flash-high`, `claude-opus-5-5-high`, `claude-sonnet-5-5-medium`, `gpt-oss-120b-medium`, `some-unknown-model`:

| Setup | Results |
| --- | --- |
| Without the plugin | `200000` x8 |
| With the plugin | `1048576 1048576 1048576 1048576 1000000 1000000 131072 200000` |
| With `ANTIGRAVITY_CONTEXT_CAP=262144` | `262144 262144 262144 262144 262144 262144 131072 200000` |

Consequence for compression: at `compression.threshold` 0.5 the trigger moves from 100k tokens (0.5 x the flat 200k window) to about 524k tokens (0.5 x 1048576) for a Gemini model.

## Caveats and residual risk

- The non-Gemini windows (`claude-*`, `gpt-oss-120b`) are upstream figures, not confirmed for Antigravity.
- A public report says `agy` compacts near 256K despite advertising 1M. If you see early compaction, set `ANTIGRAVITY_CONTEXT_CAP=262144`.
- Sibling models inherit their parent family's window until the table has an entry for them. Run the diagnostics to see which live ids match nothing.
- Correct load order depends on Hermes internals (sorted provider import, last-writer-wins registration).
- Not tested under `plugins.isolation: host`.
- A YAML custom table needs PyYAML, which the Hermes venv lacks. JSON always works.

## Development

The Hermes venv has no pytest, so use a separate venv:

```sh
python -m venv --system-site-packages .venv && .venv/bin/pip install pytest pyyaml && .venv/bin/python -m pytest tests -q
```

The suite currently has 42 tests (22 test functions plus parametrized cases). CI (`.github/workflows/ci.yml`) runs pytest on Python 3.12, 3.13 and 3.14 for pushes to `main` and for pull requests.

## License

MIT, see [LICENSE](LICENSE).

The proper fix belongs upstream in neerazz/hermes-antigravity-oauth. This plugin is a non-invasive stopgap that needs no fork of it.
