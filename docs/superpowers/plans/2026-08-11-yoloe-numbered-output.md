# YOLOE Target-Numbered Output Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make both reference-image YOLOE CLIs automatically create target-numbered output directories and remove `--name`/`--exist-ok` from those CLIs.

**Architecture:** Replace the reusable-name allocator with a shared atomic allocator that slugifies the operator target, scans exact numbered sibling directories, and creates `<slug><max+1>`. Both CLI adapters call this allocator using their existing target argument, while the older text-prompt CLI remains unchanged.

**Tech Stack:** Python 3.10, pathlib, regular expressions, argparse, pytest, existing Ultralytics YOLOE pipeline.

## Global Constraints

- Manual, Codex, and Qwen runs share one sequence inside `--project`.
- `table1` plus `table3` produces `table4`; gaps are not reused.
- Unicode letters/digits are preserved; other character runs become one underscore.
- Existing output is never overwritten or reused, including concurrent allocation.
- Remove `--name` and `--exist-ok` only from `visual_prompt.py` and `auto_refer_detect.py`.
- Keep `--project` and keep `infer.py` behavior unchanged.

---

### Task 1: Atomic target-numbered directory allocator

**Files:**
- Modify: `tools/yoloe26m/tests/test_reference_pipeline.py`
- Modify: `tools/yoloe26m/reference_pipeline.py`

**Interfaces:**
- Produces: `target_slug(target: str) -> str` and `prepare_numbered_run_dir(project: str | Path, target: str) -> Path`.
- Replaces: `prepare_run_dir(project, name, exist_ok)` for the two reference-image CLIs.

- [ ] **Step 1: Replace the old allocator tests with failing behavior tests.**

```python
def test_numbered_run_dir_uses_target_and_starts_at_one(tmp_path):
    assert prepare_numbered_run_dir(tmp_path, "table").name == "table1"

def test_numbered_run_dir_uses_max_existing_number_without_filling_gaps(tmp_path):
    (tmp_path / "table1").mkdir()
    (tmp_path / "table3").mkdir()
    assert prepare_numbered_run_dir(tmp_path, "table").name == "table4"

def test_numbered_run_dir_slugifies_spaces_and_preserves_unicode(tmp_path):
    assert prepare_numbered_run_dir(tmp_path, "blue basket").name == "blue_basket1"
    assert prepare_numbered_run_dir(tmp_path, "蓝色篮子").name == "蓝色篮子1"
```

Add a test proving `table2.txt`, `table-old`, and `other9` do not affect the sequence. Use real temporary directories and files rather than mocks.

- [ ] **Step 2: Run focused tests and verify RED.**

Run:

```bash
./.venv_inference/bin/python -m pytest \
  tools/yoloe26m/tests/test_reference_pipeline.py \
  -k 'numbered_run_dir' -v
```

Expected: collection fails because `prepare_numbered_run_dir` does not exist.

- [ ] **Step 3: Implement the minimal allocator.**

Use Unicode-aware `str.isalnum()` slugging and an exact escaped pattern:

```python
def target_slug(target: str) -> str:
    cleaned = "_".join(
        part for part in re.split(r"[^\w]+", target.strip(), flags=re.UNICODE) if part
    )
    return cleaned or "target"

def prepare_numbered_run_dir(project, target):
    root = Path(project).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    slug = target_slug(target)
    pattern = re.compile(rf"^{re.escape(slug)}([1-9]\d*)$")
    number = max(
        (int(match.group(1)) for item in root.iterdir()
         if item.is_dir() and (match := pattern.fullmatch(item.name))),
        default=0,
    ) + 1
    while True:
        candidate = root / f"{slug}{number}"
        try:
            candidate.mkdir()
            return candidate.resolve()
        except FileExistsError:
            number += 1
```

If regex `\w` behavior allows underscores as intended, keep it; explicitly reject an empty raw target before slugging so CLI validation remains meaningful.

- [ ] **Step 4: Run the focused allocator tests and full module tests.**

Expected: all allocator tests and the previous 32 tests pass after updating helper call sites in tests.

### Task 2: Remove manual output naming from both reference-image CLIs

**Files:**
- Modify: `tools/yoloe26m/tests/test_reference_pipeline.py`
- Modify: `tools/yoloe26m/visual_prompt.py`
- Modify: `tools/yoloe26m/auto_refer_detect.py`

**Interfaces:**
- Consumes: `prepare_numbered_run_dir(project, target)` from Task 1.
- Produces: parsers that reject `--name` and `--exist-ok`, and main functions that allocate from `target_name`/`target`.

- [ ] **Step 1: Write failing parser and main-boundary tests.**

```python
def test_manual_cli_no_longer_accepts_name_or_exist_ok():
    options = build_manual_parser()._option_string_actions
    assert "--name" not in options
    assert "--exist-ok" not in options

def test_auto_cli_no_longer_accepts_name_or_exist_ok():
    options = build_auto_parser()._option_string_actions
    assert "--name" not in options
    assert "--exist-ok" not in options
```

Also test the real allocator with the targets used by each parser so a wrong hard-coded default directory would fail.

- [ ] **Step 2: Run the new parser tests and verify RED.**

Expected: both tests fail because the parsers still register the two options.

- [ ] **Step 3: Modify both CLI adapters.**

Delete both argument registrations, import `prepare_numbered_run_dir`, and allocate with:

```python
run_dir = prepare_numbered_run_dir(args.project, args.target_name)
```

for the manual CLI and:

```python
run_dir = prepare_numbered_run_dir(args.project, target)
```

for the automatic CLI. Keep all model/provider/inference arguments unchanged.

- [ ] **Step 4: Run all focused tests and both `--help` commands.**

Expected: 0 failures, and neither help output contains `--name` or `--exist-ok`.

### Task 3: Documentation and end-to-end verification

**Files:**
- Modify: `tools/yoloe26m/README.md`
- Verify: all files from Tasks 1-2.

**Interfaces:**
- Produces: copy-paste commands without manual output naming and verified consecutive directories.

- [ ] **Step 1: Update only the reference-image README section.**

Remove `--name` from its three reference-image commands and replace the old overwrite paragraph with: outputs are automatically named `<target-slug><number>`, manual/Codex/Qwen share numbering, and existing results are never overwritten. Leave the older text-prompt `infer.py` examples unchanged.

- [ ] **Step 2: Run syntax and full focused tests.**

```bash
./.venv_inference/bin/python -m py_compile \
  tools/yoloe26m/reference_pipeline.py \
  tools/yoloe26m/visual_prompt.py \
  tools/yoloe26m/auto_refer_detect.py \
  tools/yoloe26m/tests/test_reference_pipeline.py
./.venv_inference/bin/python -m pytest \
  tools/yoloe26m/tests/test_reference_pipeline.py -v
```

Expected: compilation exit 0 and every test passes.

- [ ] **Step 3: Run two real manual YOLOE invocations in a temporary project root.**

Use the installed bus sample and model with `--target-name numbered-smoke`; invoke the same command twice without `--name` or `--exist-ok`. Expected directories are `numbered_smoke1` and `numbered_smoke2`, each containing nonempty `target_annotated.jpg` and `detections.json` with one bus detection and one mask.

- [ ] **Step 4: Audit scope and commit implementation.**

Run `git diff --check`, scan the two reference CLI help outputs, and inspect `git status --short`. Stage only the numbered-output implementation, tests, plan, and README; do not stage BasePose or unrelated files. Commit with:

```bash
git commit -m "feat: number YOLOE outputs by target"
```
