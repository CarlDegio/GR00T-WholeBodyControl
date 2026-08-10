# YOLOE Target-Numbered Output Design

## Goal

Remove the manual `--name` and `--exist-ok` output-directory controls from
both YOLOE reference-prompt CLIs. Every invocation creates a fresh directory
whose name is derived from the operator's target plus a monotonically
increasing integer, such as `table1`, then `table2`.

## Naming contract

- Manual, Codex, and Qwen runs share one numbering sequence inside the selected
  `--project` output root.
- The manual CLI uses `--target-name`; the automatic CLI uses `--target`.
- A filesystem-safe slug preserves Unicode letters and digits, converts each
  run of other characters to one underscore, strips leading/trailing
  underscores, and falls back to `target` if nothing remains. Thus
  `blue basket` becomes `blue_basket1` and `蓝色篮子` becomes `蓝色篮子1`.
- Existing directories matching the exact slug followed by a positive integer
  determine the next number. If `table1` and `table3` exist, the next directory
  is `table4`; gaps are not reused.
- Unrelated names such as `table-old`, `table2.txt`, or `other1` do not affect
  the sequence.

## Creation and safety

The shared pipeline exposes one directory allocator that creates the output
root if needed, scans matching directories, and attempts to create
`<slug><max+1>` with `exist_ok=False`. If another process wins the same name,
the allocator increments and retries. Existing output is never overwritten or
reused.

Both CLIs retain `--project` but no longer register `--name` or `--exist-ok`.
Their `main` functions validate and normalize the target before allocating the
directory, then pass the returned path through the existing inference pipeline.

## Compatibility and documentation

The change applies only to `visual_prompt.py` and `auto_refer_detect.py`; the
older text-prompt `infer.py` keeps its existing `--name` and `--exist-ok`
behavior. README examples for the two reference-image CLIs remove the deleted
arguments and explain automatic numbering.

## Verification

TDD covers first allocation, max-plus-one behavior, shared target slugging,
ignoring unrelated entries, and the removal of both CLI options. Then run all
focused tests, syntax compilation, both CLI help commands, and a real local
manual YOLOE smoke test that verifies two consecutive invocations create two
different numbered directories without overwriting either one.
