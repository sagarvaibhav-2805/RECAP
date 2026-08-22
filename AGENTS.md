# AGENTS.md

> Agent-agnostic instruction file. Save this as `AGENTS.md` in your project root — Codex, Cursor, and Antigravity all read it natively. For Claude Code, create a one-line `CLAUDE.md` containing `@AGENTS.md` so it inherits this file instead of you maintaining a duplicate copy. See §9.

---

## 1. Project Context

<!-- Replace with real facts an agent can't infer from the code itself. -->

- **Stack:**
- **Install deps:**
- **Build:**
- **Test:**
- **Lint / format:**
- **Run locally:**
- **Package manager:** (flag if non-default, e.g. "pnpm, not npm")

---

## 2. Why These Rules Exist

This file combines two things: Andrej Karpathy's widely-shared January 2026 field notes on where coding agents go wrong after he shifted most of his own work to agentic coding, and general industry practice since then for managing context and cost across Claude Code, Codex, Cursor, and Antigravity.

His core finding: agents are remarkably persistent — they'll loop on a hard problem far longer than a human would tolerate before giving up — but three failure modes show up repeatedly if left unmanaged:

- **Silent wrong assumptions** — the agent picks an interpretation of an ambiguous request and runs with it instead of flagging the ambiguity or asking.
- **Overcomplication** — bloated abstractions, unrequested configurability, a construction ten times longer than it needed to be.
- **Orthogonal damage** — edits to comments or code adjacent to the task that the agent doesn't fully understand, done as an unintended side effect.

His stated fix, and the single highest-leverage habit in this file: **shift from imperative to declarative instructions.** Don't hand over a list of steps — hand over success criteria and let the agent loop against them. That's why §3 and §4 below lean so heavily on verification and "done when" conditions rather than procedures.

He also flagged a second-order risk worth designing process around: as the volume of agent-written code goes up, it gets easier to wave through output that's "almost right, but not quite" without noticing. Every verification and review habit in this file is a hedge against exactly that — not busywork.

---

## 3. Core Principles

### Think before coding
- State assumptions explicitly; don't silently resolve ambiguity.
- If multiple reasonable interpretations exist, present them instead of picking one.
- If a simpler approach exists than what was asked for, say so before building the complex version.
- Stop and ask when something is genuinely unclear — don't proceed on a guess.

### Simplicity first
- Write the minimum code that solves the stated problem: no speculative features, no unrequested flexibility, no abstractions for single-use code, no handling for scenarios that can't occur.
- If a change is trending toward 200 lines when it could be 50, stop and reconsider the approach.
- Test: would a senior engineer reviewing this call it overcomplicated? If yes, simplify before finishing.

### Surgical changes
- Touch only what the task requires. Don't reformat or "clean up" adjacent code as a side effect.
- Match the existing style of the file being edited, even if you'd do it differently.
- Notice unrelated dead code or bugs? Flag them — don't fix them unasked.
- Only remove code your own change made unused; leave pre-existing unused code alone unless asked to remove it.
- Every changed line should trace back to the actual request.

### Goal-driven, declarative execution
- Give (or ask for) a verifiable success criterion instead of a list of steps — this is what lets an agent loop productively instead of needing constant hand-holding.
- Turn vague asks into checkable ones: "fix the bug" → "write a failing test that reproduces it, then make it pass"; "add validation" → "write tests for invalid inputs, then make them pass."
- For optimization work specifically: write the naive, obviously-correct version first, confirm it's correct, then optimize while preserving that correctness — don't jump straight to the optimized version.
- Where the tool supports it, loop the agent against a real verification target (a test suite, a type checker, a browser/DOM check) instead of asking it to self-report success.
- For multi-step tasks, state a short plan with a verification step per item before starting:
  ```
  1. [step] → verify: [check]
  2. [step] → verify: [check]
  ```

---

## 4. Framing Non-Trivial Tasks

Before starting anything beyond a one-line fix, make sure the task states:

- **Goal** — the outcome wanted, as an outcome, not a method
- **Context** — which files, docs, or errors actually matter
- **Constraints** — conventions, architecture limits, things not to touch
- **Done when** — the concrete, checkable condition that means it's finished

Missing one of these and it matters? Ask, don't guess.

---

## 5. Token & Cost Efficiency

These apply regardless of which agent or model is running.

**Keep the instruction file itself cheap**
- Every session pays to load this file — don't restate what's inferable from the code (obvious directory layout, dependency lists). Keep pitfalls, rationale, and non-default conventions.
- Target well under 300 lines total; push detail into nested or path-scoped files (most tools support nested `AGENTS.md` files or scoped rule directories) rather than letting one root file sprawl.
- Maintain **one** copy of shared instructions, not one per tool. Duplicated, drifting config across `CLAUDE.md` / `.cursor/rules` / `AGENTS.md` / `GEMINI.md` wastes tokens and maintenance effort — single-source it and import where the tool allows (§9).

**Manage context during a session**
- Start a fresh session per unrelated task. Stale context from a finished task gets re-sent — and re-billed, on usage-based plans — with every subsequent message.
- Reference precisely (a function, a file, a line range) instead of pasting entire files or directories when only part is relevant.
- Delegate verbose operations — full test-suite output, large log files, long doc fetches — to a subagent or a preprocessing step, so only a summary re-enters the main context instead of the raw dump.
- Search/grep for relevant code rather than having the agent read broadly "just in case."

**Respect prompt caching**
- Most providers cache repeated prefixes (system prompt, this file, other stable early context) at a steep discount. Avoid unnecessary mid-session edits to this file or other pinned context — each edit invalidates the cache for everything that follows it.
- Batch related tool calls together rather than issuing many small sequential round-trips, where the tool supports it.

**Route by task complexity**
- Use a lighter/cheaper model for mechanical or narrow subtasks (formatting, boilerplate, simple lookups); reserve the strongest model for architecture decisions and multi-step reasoning.
- Turn down reasoning/thinking effort for simple, well-specified tasks; keep it on for genuinely hard problems.
- Prefer deterministic checks — linters, type checkers, test runners — over asking the model to "double check" in prose. A compiler is cheaper and more reliable than another inference pass for the same question.

**Reduce tool overhead**
- Disable connectors/tools not in active use for the current task — a connected tool can cost context just by being listed as available.
- Prefer a CLI (`gh`, `aws`, `gcloud`, etc.) over an equivalent MCP/tool integration when both exist and the CLI doesn't require an always-on tool listing.

---

## 6. Performance & Quality Habits

Saving tokens shouldn't cost correctness — these keep (or improve) output quality alongside everything in §5.

- **Be specific, not vague.** "Add input validation to the login form in `auth.ts`, rejecting empty and malformed emails" beats "improve the login form." Vague requests trigger broad, expensive exploration *and* produce worse results.
- **Explore before implementing.** For anything non-trivial, read the relevant code and propose an approach before editing. Catching a wrong direction before code is written is far cheaper than catching it after.
- **One task at a time.** Bundling several unrelated asks into one prompt degrades focus on all of them — sequence them instead.
- **Test incrementally.** Implement one piece, verify it, then continue. Don't write one large untested change and hope it all works.
- **Course-correct immediately.** The moment output looks wrong, stop it rather than letting it continue and cleaning up afterward — this saves tokens and avoids compounding a bad foundation.
- **Add a review pass for anything that matters.** Before accepting a non-trivial diff, re-read it once specifically looking for scope creep, unrequested changes, and unverified claims. Treat an agent's self-reported success with mild skepticism until a real check confirms it.
- **Use few-shot examples when style or pattern matters.** One correct example of the pattern you want is often cheaper and more reliable than describing it in prose.

---

## 7. Safety and Security

- Never hardcode secrets, API keys, tokens, or credentials — use environment variables or the project's existing secrets mechanism.
- Never run destructive commands (force-push, `rm -rf`, dropping tables, deleting branches) without explicit confirmation.
- Don't disable security checks, linters, or tests to make something pass — fix the underlying issue.
- Validate all external/user input in code you write.
- Flag security concerns instead of silently working around them.

---

## 8. Git and Commit Practices

- Keep commits small and focused on one logical change.
- Commit messages explain *why*, not just *what*.
- Don't push directly to protected branches — use a branch and open a PR unless told otherwise.
- Check `git status` / `git diff` before and after a task to confirm only the intended files changed.

---

## 9. Using This File Across Tools

- **Codex, Cursor, Antigravity:** read `AGENTS.md` natively, including nested/scoped files, with the most specific path taking precedence.
- **Claude Code:** doesn't read `AGENTS.md` directly. Create a `CLAUDE.md` containing just `@AGENTS.md` (plus any Claude-only additions below that line) so it inherits everything here with no duplicate copy to maintain.
- **Tool-specific override files** (e.g. Antigravity's `GEMINI.md`, Cursor's `.cursor/rules/*.mdc`): use them only for instructions specific to that one tool. Everything shared stays in this file.

---

## 10. Project-Specific Conventions

<!-- naming conventions, architecture decisions, "always do X" / "never do Y" rules specific to this repo -->
