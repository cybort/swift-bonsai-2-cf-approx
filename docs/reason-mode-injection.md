# REASON mode injection — behavior & public catalog

Source of truth for mode **names / help-menu descriptions**: DavidAU Ultimate Heretic Turbo Brilliance model card help menu. Structured copy: [`configs/reason_modes.json`](../configs/reason_modes.json).

**Copyright boundary:** Full AI/AGI construct bodies (often 2k–8k tokens) live inside DavidAU’s copyrighted `chat_template.jinja`. This repo does **not** redistribute those bodies. We approximate training with short open primers + optional LM Studio distillation from your local loaded GGUF.

## How injection works

1. User message contains a tag, e.g. `{REASON:coder}` or `{REASON:icoder}` or `{REASON:agi-musk-curie}`.
2. Jinja chat template (embedded in the GGUF / LM Studio) finds the **first** tag in the **last** tagged user message, strips it from the visible prompt, and injects the matching system directives **before** thinking/generation.
3. Tag forms:
   - `{REASON:<mode>}` → reasoning (thinking on) + mode directive
   - `{REASON:i<mode>}` → instruct (thinking off) + same mode family
   - `{REASON:agi-a-b}` / `{REASON:iagi-a-b}` → dynamic merge of constructs `a`+`b` (prompt steers which facets dominate)
   - `{REASON:help}` or plain `help` → interactive help menu
   - `{REASON:off}` → no mode enhancement
4. Defaults when untagged: reasoning **`xhigh`**, instruct **`medium`** (Ultimate card). Some third-party template packs default untagged reasoning to `medium`.
5. Public Qwen effort steering snippets (safe to cite):
   - **xhigh:** “Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.”
   - **low:** “Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the conclusion without unnecessary elaboration.”
6. `x*` modes (e.g. `xcoder`, `xomni`) combine a specialized construct with the xhigh steering sentence.
7. Practical ops tips from the help menu: **new chat** when switching modes; **regen once** to clear app cache; higher quants → better construct fidelity.

## Mode groups (names only)

| Group | Modes |
|---|---|
| Frameworks | `omni`, `deeptree`, `hyper`, `socrates`, `logic`, `einstein`, `spoon`, `ultra`, `xultra` |
| Qwen native | `xhigh`, `high`, `medium`, `low` |
| X-combined | `xomni`, `xdeeptree`, `xhyper`, `xsocrates`, `xlogic`, `xeinstein`, `xspoon`, `xhelp`, `xbrainstorm`, `xcomedy`, `xbusiness`, `xstory`, `xteacher`, `x007`, `xcoder` |
| Experts | `holodeck`, `prompt`, `musk`, `curie`, `007`, `joe` |
| Characters | `pkd`, `neo`, `stark`, `matlock`, `house`, `vader`, `joker`, `quinn` |
| Supers | `brainstorm`, `business`, `coder`, `comedy`, `story`, `teacher` |
| Meta | `help`, `off` |
| AGI samples | `agi-musk-curie`, `agi-teacher-coder`, `agi-brainstorm-story`, … (see JSON) |

## Probe & distill with local LM Studio

This cloud VM cannot reach your machine’s `localhost:1234`. On the host where LM Studio has the DavidAU GGUF loaded with local server enabled:

```bash
# 1) Probe help + a few modes (writes artifacts/lmstudio/)
python3 scripts/lmstudio_probe_modes.py

# 2) Distill mode-tagged SFT from the live teacher
bash scripts/lmstudio_distill.sh

# 3) Rebuild prepared_train including LM Studio outputs
python3 scripts/prepare_proxy_datasets.py
python3 scripts/merge_lmstudio_distill.py
```

Expected LM Studio OpenAI base URL: `http://127.0.0.1:1234/v1` (override with `LMSTUDIO_BASE_URL`).

## What we put in training data

- Expanded **open mode primers** (`{REASON:…}` → short think + answer) covering frameworks, instruct (`i*`), x-combined, and a few AGI tags.
- Optional **live LM Studio distill** rows tagged `teacher=lmstudio-davidau`.
- TeichAI proxy SFT unchanged.
