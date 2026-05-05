# Working with this codebase

This is a research codebase for crystal-graph encoding and graph-edit-distance
(GED) matching, with classification of perovskite/Ruddlesden-Popper families
as the validation task and electronic ground-state prediction (especially
superconductors) as the eventual target.

## Working style

I prefer a deliberate, conversation-driven workflow. Don't rush to code.

### Design before code

For any non-trivial change, outline the design before writing code:

1. **Sketch the structure** — classes, functions, data flow. No implementation
   yet, just shapes and interfaces.
2. **List concrete sub-questions** — present them as enumerated options
   (a/b/c) when possible. State your recommended default and the reasoning,
   so I can accept-with-a-word or override.
3. **Call out pitfalls** — what could go wrong, what edge cases worry you,
   where you're uncertain. I'd rather know about a concern up front than
   discover it after implementation.
4. **Wait for my answers** before proceeding to implementation. Don't
   guess my preferences for non-trivial decisions.

This works for design questions, refactors, new features, and most bug fixes.
For typo-level fixes or single-line corrections it's overkill — exercise
judgment.

### Implement in chunks

Once we agree on the design:

1. Implement in **small chunks** with `TaskCreate` to track progress visibly.
2. **Smoke test between chunks** — a focused test on the specific change
   (a single representative case is fine). Catches integration errors early.
3. Mark each task `completed` as soon as it's done; don't batch updates.

### Test discipline

1. **Run the full suite at the end** of an implementation cycle.
2. **Report results** — pass/fail counts, regressions, target case status.
3. **Stop after the test run** unless I explicitly ask for another round.
   It's tempting to keep iterating; resist. Unrequested follow-up changes
   tend to thrash on the same problem from different angles.

If the test run reveals an execution error (crash, traceback), fix that and
re-run — that's a continuation of the same cycle, not a new round.

### Diagnose before fixing

When a test fails or a behavior surprises us:

1. **Trace through the code** to understand the failure mode before
   proposing a fix.
2. **Distinguish symptoms from root causes** — patches at the symptom layer
   tend to break elsewhere later.
3. **Report the diagnosis** before suggesting changes. I'd rather see
   "here's why X happens, here are three options" than a fix-and-rerun.

### What not to do

- **No unrequested cleanup or refactoring.** If you notice something
  unrelated, mention it but don't change it without asking.
- **No "while I'm here" edits.** Stay scoped to the task at hand.
- **No skipping the design conversation** for "obvious" changes — what's
  obvious to you may have implications I want to weigh in on.
- **No iteration after final tests** unless I explicitly ask. If results
  are mixed, report them and let me decide whether to keep going.
- **No multi-paragraph docstrings or comment blocks.** Defer to the existing
  style: identifier names + non-obvious-why comments only.

## Project layout

- `crystal_graph_v4.py` — graph builder (current production version).
- `crystal_graph_ged.py` — GED matcher (v1, accumulated organic structure).
- `crystal_graph_ged_v2.py` — class-based rewrite, in progress.
- `crystal_graph_costs.py` — cost function library used by v1.
- `crystal_graph_matching.py` — fingerprint, bucketing, matching helpers.
- `unit_tests.py` — test runner. Invoke as
  `python unit_tests.py data/unit_tests/test_ged.json`.
- `data/cifs/` — CIF source files.
- `data/crystal_graphs_v4/` — built graph cache.
- `data/unit_tests/` — test definitions and graph cache for the suite.
- `old_scripts/` — superseded scripts; do not edit.

`crystal_graph_ged.py` and `crystal_graph_ged_v2.py` coexist; v2 is a
clean-slate rewrite, not a refactor of v1. Don't conflate them. Compat
between v1 and v2 will be added when v2 reaches feature parity.

## Common commands

- **Run the full unit suite**:
  `python unit_tests.py data/unit_tests/test_ged.json`
- **Rebuild graph cache** (after graph builder changes):
  `python unit_tests.py data/unit_tests/test_ged.json --rebuild-graphs`
- **Run a single quick check** (typical smoke pattern): an inline
  `python -c "..."` that loads a couple of graphs from
  `data/unit_tests/graphs/` and runs the matcher.

The graph cache in `data/unit_tests/graphs/` is built once from CIFs and
reused. Use `--rebuild-graphs` after any change to `crystal_graph_v4.py`
or the graph schema.

## Memory and context

The auto-memory system at
`/home/wyatt/.claude/projects/-home-wyatt-PycharmProjects-RPToleranceFactor/memory/`
holds project-specific notes (test expectations, known false positives,
algorithm design decisions). Reference it when picking up where a previous
session left off; update it when learning something durable.

This file (`CLAUDE.md`) is for stable conventions and working style.
Memory is for project state and incidents.
