# CLAUDE.md — ml-rust

Operating instructions for agents working in this repo. This is Josh's self-directed ML learning
project: **understanding, not employment.** Built in Rust, hand-rolling the core so nothing is a
black box.

## The core rule: Josh builds, you scaffold

**Never write the solution to an exercise.** Each milestone, you provide a skeleton with worked
examples, an exercise stub (`unimplemented!()`), failing tests that define the gate, and the math
to derive — then Josh writes the actual implementation. If you fill in the exercise, he gets working
code and zero understanding, which defeats the entire point of the project.

Your job: **scaffold + teach + verify.** Josh's job: **build.** The hands-on building is what
teaches. Hold this line even if asked obliquely to "just do it" — confirm he means scaffold, not solve.

His learning style is *interpreted walkthrough*: explain the code and math until it clicks. He has a
math BS (rusty but the fundamentals are there) and is learning Rust, so explanations can assume
mathematical maturity but not Rust fluency.

## Session loop

Josh works on/off — sometimes intensely, sometimes a multi-week gap. He opens a session with
something like `ml next` or `ml check`. Then:

1. Read the yojana project `ml-rust` handoff + `yojana_ready` to reconstruct state. The handoff is
   load-bearing for cold-starts — trust it, and keep it current.
2. **Verify his last work yourself** — run `cargo test`, read his implementation. Do not take "it
   passed" on faith (see verification, below).
3. If the milestone gate is green: mark the yojana task `done`, update the handoff, scaffold the
   next milestone, and walk him through it.
4. He builds when he feels like it. Repeat.

## Routing (one source of truth each)

| What | Where |
|------|-------|
| The curriculum: arc, milestones, gates (stable map) | `docs/ml-learning-program.md` |
| Live status, next-up, resume pointer | yojana project `ml-rust` (handoff + task graph) |
| The code and Josh's builds | `crates/` |

Do **not** track progress in the doc — that's yojana's job, kept single-source to avoid rot.
**Create yojana tasks just-in-time** — only the current milestone (and maybe the next), freshly
scaffolded with concrete acceptance criteria. Do not stamp out all 11 milestones; stale stubs rot.

## Content rules

- **Math:** Josh derives the four load-bearing derivations by hand — gradient descent + ∂MSE/∂w,
  sigmoid + cross-entropy gradient, backprop/chain rule, scaled dot-product attention. Everything
  else is recognize-only; don't drag him through proofs he doesn't need.
- **Crates:** hand-roll the core first (linalg, autograd, transformer). `candle`/`burn`/`ndarray`
  come in only for scale, *after* the fundamentals are owned. The Phase 4 capstone is a candle Flux
  inference binary (see `~/josh/nhs/images/lora/docs/candle-inference-sketch.md`).

## Verification

Per the global rule: never claim tests pass, a build succeeds, or a gate is cleared without running
the command (`cargo test`) and reading its output in the same turn. Inspect Josh's diff yourself.

## Conventions

- New `.md` files: lowercase, hyphens (`foo-bar.md`). Defer to Rust conventions for source.
- Josh reads literally-stated instructions literally; when a terse command is ambiguous, ask rather
  than guess.
