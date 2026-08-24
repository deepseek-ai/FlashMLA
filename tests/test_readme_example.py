# Guards the README usage example against API drift.
#
# The flash_mla_with_kvcache signature has softmax_scale between num_splits and
# causal, so a stale README example that passes causal/is_fp8_kvcache/indices
# positionally binds them to the wrong parameters and silently computes wrong
# attention output (e.g. softmax_scale=True). This test parses the README code
# blocks and checks every call against the real signature, without importing
# torch or the compiled extension, so it runs on any machine.

import ast
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
README = REPO_ROOT / "README.md"
INTERFACE = REPO_ROOT / "flash_mla" / "flash_mla_interface.py"


def get_param_names(func_name: str) -> list:
    tree = ast.parse(INTERFACE.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            return [a.arg for a in node.args.args]
    raise AssertionError(f"{func_name} not found in {INTERFACE}")


def readme_calls(func_name: str) -> list:
    calls = []
    for block in re.findall(r"```python\n(.*?)```", README.read_text(), re.DOTALL):
        # The example elides loop bodies with bare "...", which is valid Python,
        # but tolerate other snippet lines that are not.
        try:
            tree = ast.parse(block)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == func_name
            ):
                calls.append(node)
    return calls


def test_readme_decode_call_matches_signature():
    params = get_param_names("flash_mla_with_kvcache")
    boundary = params.index("softmax_scale")
    calls = readme_calls("flash_mla_with_kvcache")
    assert calls, "README no longer shows a flash_mla_with_kvcache example"
    for call in calls:
        n_pos = len(call.args)
        assert n_pos <= boundary, (
            f"README passes {n_pos} positional args to flash_mla_with_kvcache; "
            f"args from '{params[boundary]}' (index {boundary}) onward must be "
            f"keywords, or they bind to the wrong parameters"
        )
        for kw in call.keywords:
            assert kw.arg in params, (
                f"README passes unknown keyword '{kw.arg}' to flash_mla_with_kvcache"
            )


def test_readme_metadata_call_takes_no_args():
    calls = readme_calls("get_mla_metadata")
    assert calls, "README no longer shows a get_mla_metadata example"
    for call in calls:
        assert not call.args and not call.keywords, (
            "README passes arguments to get_mla_metadata, which ignores all "
            "arguments and returns an empty FlashMLASchedMeta"
        )


def test_readme_referenced_test_files_exist():
    for rel in re.findall(r"`(tests/[\w/]+\.py)`", README.read_text()):
        assert (REPO_ROOT / rel).is_file(), f"README references missing file {rel}"


if __name__ == "__main__":
    test_readme_decode_call_matches_signature()
    test_readme_metadata_call_takes_no_args()
    test_readme_referenced_test_files_exist()
    print("README example matches flash_mla_with_kvcache signature")
