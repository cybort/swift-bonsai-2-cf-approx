# Research: GAIN, Fable Fusion 711, Turbo Brilliance constructs

Sources: public DavidAU model cards, discussions, and the Ultimate `chat_template.jinja` (2026). Proprietary code/datasets are **not** released; this document records what can be reconstructed for an open approximation.

## 1. GAIN → Cold Fusion

**Definition (public):** GAIN is a training-time program that **dynamically changes training on a per-sample basis while the model learns**, plugged into Unsloth trainers. **Cold Fusion = GAIN + Unsloth**.

**Claimed effects:**
- Keep ~99% of BF16 metric quality at 4-bit and 8-bit
- Cut thinking tokens to ~1/10–1/2 of stock Qwen
- Drop PPL vs base (DavidAU treats PPL drop as a Cold Fusion signature; normal SFT often flat/up)
- Avoid “overcooking” / benchmaxing

**What is *not* public:** loss formula, which stats drive the per-sample change (CE? margin? length? KL?), schedules, code.

**Open approximation used in this repo (`gain_proxy`):**
1. Per-sample weight from estimated thinking length (boost short, penalize long)
2. Explicit thinking-length penalty in the objective
3. Optional later: loss-adaptive reweight from running CE (still not original GAIN)

## 2. Fable Fusion 711

**Definition (public):** Multi-stage fine-tune + multi-fine-tune + multi-stage **merge** pipeline, built on consumer Unsloth hardware. Name “711” refers to ARC-C ≈ 0.711 (8-bit instruct).

**Collaborators / ingredients named on cards:**
- DavidAU — multi-stage tunes
- Nightmedia — merge / benches
- TeichAI — Polaris / related datasets
- armand0e — light “Fable 5” traces
- trohrbaugh — Heretic / ARA de-censor

**Named data themes (private DavidAU names → public proxies):**

| Private / card name | Public proxy used here |
|---|---|
| Polar-STRICT / Polaris (GPT5 non-reasoning) | `TeichAI/polaris-alpha-1000x` |
| F451-STRICT (in-house) | No public match → synthetic + instruction/reasoning mix |
| Reasoning-STRICT | `TeichAI/Claude-Opus-4.6-Reasoning-887x` (+ short-think rewrite) |
| Fable 5 traces | `TeichAI/Fable-5-Cursor-Traces` |

**Pipeline shape (reconstructed):**
1. Heretic / ARA abliteration (optional; **out of scope** for our approx)
2. Focused SFT stages (instruction following + problem solving)
3. Intermediate human + Nightmedia benches every stage
4. Merges of intermediate cooks
5. Optional re-Heretic + Cold Fusion thinking-compression cook
6. 6–10 stages for full FF711-class; Cold Fusion alone is a lighter “level 1–2” cook

**Strict goals:** raise IQ / instruction following; do not damage core Qwen behavior; zero benchmaxing; raise core benches.

## 3. Turbo Brilliance 50x / AGI constructs

**Definition (public):** Prompt-time / jinja-injected **mode system**, not a separate weight architecture. Modes are selected via tags like `{REASON:omni}` / `{REASON:ihelp}` / `{REASON:agi-musk-curie}`.

**Mechanics (from Ultimate `chat_template.jinja`):**
- Copyrighted custom jinja by DavidAU (2026) — do **not** redistribute the full proprietary template verbatim
- Parses `{REASON:<mode>}` (reasoning) and `i`-prefixed instruct variants
- Injects large system / pre-thinking directives (hundreds–thousands of tokens) before generation
- Defaults: reasoning `xhigh`, instruct `medium`; falls back safely on bad modes
- Categories observed publicly:
  - Core frameworks: `omni`, `deeptree`, `hyper`, `socrates`, `logic`
  - Detail levels: `xultra`…`low`, plus Qwen `xhigh`/`medium`/`low`
  - Persona / AI constructs: `musk`, `curie`, `coder`, `teacher`, `007`, …
  - X-combined: `xomni`, `xcoder`, …
  - AGI merges: `agi-musk-curie`, `agi-brainstorm-coder`, … (dynamic combo of constructs)

**Implication for training data:** Turbo Brilliance itself can be approximated as **SFT on (mode-tag → short, high-quality completion)** pairs plus optional compact system prefixes. Full 2k–8k construct texts are proprietary; we use **short open mode primers** only.

See [`reason-mode-injection.md`](reason-mode-injection.md) and [`configs/reason_modes.json`](../configs/reason_modes.json) for the public help-menu catalog, injection mechanics, and LM Studio probe/distill path (`scripts/lmstudio_distill.sh`).

## 4. Mapping onto Swift-Bonsai-2

Swift already compresses thinking via reasoning-marker penalties on ternary Bonsai 2. Our approx stack:

1. Dequant PTQ1_0 → HF shards (trainable init)
2. Distill / proxy datasets targeting short thinking + strong answers (+ light mode primers)
3. QLoRA + open GAIN-proxy auto-iterate on that base

Heretic and full Turbo construct corpora remain out of scope.

## 5. Executable checklist vs deferred

| Step | Status |
|---|---|
| Research doc (this file) | Done |
| Public proxy data prep | Runnable now |
| Download Swift PTQ1_0 | Runnable now |
| Dequant to HF | Runnable now (disk ~54GB+) |
| Synthetic / teacher distill JSONL | Runnable now (teacher needs local llama.cpp URL) |
| **Auto-iterate QLoRA on unpacked base** | **Deferred — needs GPU / HF Jobs** |
