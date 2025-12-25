from typing import List

import torch

def cdiv(x: int, y: int):
    return (x+y-1) // y

def check_is_allclose(name: str, ans: torch.Tensor, ref: torch.Tensor, abs_tol: float = 1e-5, rel_tol: float = 1e-2, cos_diff_tol: float = 1e-7):
    """
    Check if two tensors are close enough
    """
    def get_cos_diff(x: torch.Tensor, y: torch.Tensor) -> float:
        """
        Calculate the cosine diff between two tensors
        """
        x, y = x.double(), y.double()
        denominator = (x * x + y * y).sum().item()
        if denominator == 0:
            return 0
        sim = 2 * (x * y).sum().item() / denominator
        return 1 - sim
    assert ans.shape == ref.shape, f"`{name}` Shape mismatch: {ans.shape} vs {ref.shape}"
    
    ans = ans.clone().to(torch.float)
    ref = ref.clone().to(torch.float)

    # Deal with anomalies
    def deal_with_anomalies(val: float):
        ref_mask = (ref == val) if (val == val) else (ref != ref)
        ans_mask = (ans == val) if (val == val) else (ans != ans)
        ref[ref_mask] = 0.0
        ans[ans_mask] = 0.0
        if not torch.equal(ref_mask, ans_mask):
            print(f"`{name}` Anomaly number `{val}` mismatch: {ans_mask.sum().item()} in ans but {ref_mask.sum().item()} in ref")
            return False
        return True
    
    anomalies_check_passed = True
    anomalies_check_passed &= deal_with_anomalies(float("inf"))
    anomalies_check_passed &= deal_with_anomalies(float("-inf"))
    anomalies_check_passed &= deal_with_anomalies(float("nan"))

    if not anomalies_check_passed:
        return False

    cos_diff = get_cos_diff(ans, ref)
    raw_abs_err = torch.abs(ans-ref)
    raw_rel_err = raw_abs_err / (torch.abs(ref)+(1e-6))
    rel_err = raw_rel_err.masked_fill(raw_abs_err<abs_tol, 0)
    abs_err = raw_abs_err.masked_fill(raw_rel_err<rel_tol, 0)
    pass_mask = (abs_err < abs_tol) | (rel_err < rel_tol)

    if not pass_mask.all():
        print(f"`{name}` mismatch")
        max_abs_err_pos: int = torch.argmax(abs_err, keepdim=True).item()   # type: ignore
        max_rel_err_pos: int = torch.argmax(rel_err, keepdim=True).item()   # type: ignore
        def get_pos_in_tensor(t: torch.Tensor, pos: int) -> List[int]:
            result = []
            for size in t.shape[::-1]:
                result.append(pos % size)
                pos = pos // size
            assert pos == 0
            return result[::-1]
        print(f"max abs err: {torch.max(abs_err).item()}: pos {get_pos_in_tensor(ans, max_abs_err_pos)}, {ans.reshape(-1)[max_abs_err_pos].item()} vs {ref.reshape(-1)[max_abs_err_pos].item()}")
        print(f"max rel err: {torch.max(rel_err).item()}: pos {get_pos_in_tensor(ans, max_rel_err_pos)}, {ans.reshape(-1)[max_rel_err_pos].item()} vs {ref.reshape(-1)[max_rel_err_pos].item()}")
        print(f"{pass_mask.sum()} out of {pass_mask.numel()} passed ({pass_mask.sum()/pass_mask.numel()*100.0:.2f}%)")
        print(f"Cosine diff: {cos_diff} (threshold: {cos_diff_tol})")
        return False
    else:
        if abs(cos_diff) > cos_diff_tol:
            print(f"`{name}` mismatch: Cosine diff too large: {cos_diff} vs {cos_diff_tol}")
            return False
        return True