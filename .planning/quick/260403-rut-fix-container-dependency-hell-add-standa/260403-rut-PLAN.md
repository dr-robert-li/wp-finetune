---
phase: quick
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - scripts/merge_adapter.py
autonomous: true
requirements: []

must_haves:
  truths:
    - "merge_adapter.py has zero Unsloth imports"
    - "merge_adapter.py can be run as python3 scripts/merge_adapter.py without ModuleNotFoundError"
    - "Merge logic uses AutoModelForCausalLM + PeftModel only"
  artifacts:
    - path: "scripts/merge_adapter.py"
      provides: "LoRA merge without Unsloth dependency"
      contains: "AutoModelForCausalLM"
  key_links:
    - from: "scripts/merge_adapter.py"
      to: "transformers, peft"
      via: "AutoModelForCausalLM.from_pretrained + PeftModel.from_pretrained"
      pattern: "AutoModelForCausalLM\\.from_pretrained"
---

<objective>
Remove Unsloth dependency from merge_adapter.py and fix broken import path so the script runs cleanly in eval-toolbox or any container with peft+transformers.

Purpose: Unsloth's pip install destroys CUDA-enabled torch in NGC containers. The merge operation only needs AutoModelForCausalLM + PeftModel — Unsloth is unnecessary overhead with catastrophic side effects.
Output: A merge_adapter.py that uses only transformers+peft, can be invoked as `python3 scripts/merge_adapter.py`, and has no dependency on Unsloth or dgx_toolbox at import time.
</objective>

<execution_context>
@$HOME/.claude/get-shit-done/workflows/execute-plan.md
@$HOME/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@scripts/merge_adapter.py
@config/train_config.yaml
</context>

<tasks>

<task type="auto">
  <name>Task 1: Replace Unsloth with AutoModelForCausalLM and fix dgx_toolbox import</name>
  <files>scripts/merge_adapter.py</files>
  <action>
Three changes to scripts/merge_adapter.py:

1. **Remove Unsloth import and replace with AutoModelForCausalLM** (line 83):
   - Delete: `from unsloth import FastLanguageModel`
   - Replace the model loading block (lines 83-94) with:
     ```python
     from transformers import AutoModelForCausalLM  # noqa: PLC0415

     local_dir = str(resolve_path(config["model"]["local_dir"]))

     print(f"Loading base model from {local_dir} ...")
     model = AutoModelForCausalLM.from_pretrained(
         local_dir,
         torch_dtype=torch.bfloat16,
         device_map="auto",
     )
     ```
   - Note: FastLanguageModel.from_pretrained returned (model, tokenizer) — we only used model. AutoModelForCausalLM returns just the model. The tokenizer is loaded separately later from the extended tokenizer dir anyway.

2. **Remove dgx_toolbox import** (line 27):
   - Delete: `from scripts.dgx_toolbox import get_toolbox`
   - In `_verify_merged_model` (line 159), replace `dgx = get_toolbox()` with nothing — it was unused anyway. The fallback message on lines 160-161 already has all the info it needs (base_model path and adapter_dir). Just remove the `dgx = get_toolbox()` line.

3. **Update module docstring** (line 2):
   - Change "Load base model via Unsloth FastLanguageModel" to "Load base model via AutoModelForCausalLM"
   - Add note that script can be run as `python3 scripts/merge_adapter.py` (not just `-m`)

4. **Fix PYTHONPATH import pattern**: Add a sys.path fixup near the top (after existing imports, before PROJECT_ROOT) so the script works both as `python3 scripts/merge_adapter.py` and `python3 -m scripts.merge_adapter`:
   - This is NOT needed for the script itself (it only uses stdlib + transformers + peft + yaml now)
   - The `from scripts.dgx_toolbox` import was the ONLY thing requiring module-style invocation, and we are removing it
   - So no sys.path fixup needed — just removing the import fixes it

Do NOT change: the PeftModel usage (lines 97-100), the verification logic, the CLI argument parsing, the idempotency check, or the config loading.
  </action>
  <verify>
    <automated>cd /home/robert_li/Desktop/projects/wp-finetune && python3 -c "import ast; ast.parse(open('scripts/merge_adapter.py').read()); print('SYNTAX OK')" && python3 -c "
import ast, sys
tree = ast.parse(open('scripts/merge_adapter.py').read())
imports = [node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))]
for imp in imports:
    if isinstance(imp, ast.ImportFrom) and imp.module and 'unsloth' in imp.module:
        print('FAIL: unsloth import found'); sys.exit(1)
    if isinstance(imp, ast.ImportFrom) and imp.module and 'dgx_toolbox' in imp.module:
        print('FAIL: dgx_toolbox import found'); sys.exit(1)
print('NO UNSLOTH OR DGX_TOOLBOX IMPORTS — OK')
"</automated>
  </verify>
  <done>merge_adapter.py contains zero references to unsloth or dgx_toolbox, uses AutoModelForCausalLM.from_pretrained for base model loading, syntax-valid Python</done>
</task>

</tasks>

<verification>
- `grep -c "unsloth" scripts/merge_adapter.py` returns 0
- `grep -c "dgx_toolbox" scripts/merge_adapter.py` returns 0
- `grep -c "AutoModelForCausalLM" scripts/merge_adapter.py` returns at least 1
- `python3 -c "import ast; ast.parse(open('scripts/merge_adapter.py').read())"` succeeds
</verification>

<success_criteria>
- merge_adapter.py loads base model via transformers.AutoModelForCausalLM (not Unsloth)
- No import of unsloth anywhere in the file
- No import of scripts.dgx_toolbox anywhere in the file
- Script is syntactically valid Python
- Merge logic unchanged: load base -> load adapter via PeftModel -> merge_and_unload -> save -> verify tokens
</success_criteria>

<output>
After completion, create `.planning/quick/260403-rut-fix-container-dependency-hell-add-standa/260403-rut-SUMMARY.md`
</output>
