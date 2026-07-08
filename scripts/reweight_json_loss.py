#!/usr/bin/env python
"""Lever A — score-focused SFT loss for the judge (08.2 gap-closing).

Rationale: plain cross_entropy spreads gradient uniformly over ~1000 target tokens,
almost all of them prose CoT. The eval (Spearman rho) only reads the numbers in the
`<judge_output>{...}</judge_output>` JSON block. So we UP-WEIGHT that JSON span in the
per-token loss weights (renormalized to sum 1, keeping loss scale comparable to CE) and
train via forward_backward_custom. This concentrates capacity on the rank-carrying
tokens. NOTE: this is the feasible-on-Tinker approximation of a rank-aware loss — a true
listwise/pairwise ranking loss needs full-vocab logits Tinker's logprobs-only custom-loss
interface does not expose. See residual_audit.json: the gap is mid-band rank-compression,
which this targets only indirectly, so treat it as a hypothesis test, not a sure win.

Self-check: `python scripts/reweight_json_loss.py` builds one real batch, reweights it,
and asserts the boosted tokens actually decode to the judge_output JSON region.
"""
import numpy as np
import tinker

JSON_START = "<judge_output>"


def _tensordata(arr):
    """Rebuild a tinker.TensorData from a numpy array, matching the weights dtype."""
    return tinker.TensorData.from_numpy(np.asarray(arr, dtype=np.float32))


def reweight_datum(d, tok, alpha):
    """Multiply loss weight by `alpha` on the <judge_output> JSON span, renormalize to
    sum 1. Returns (n_boosted_tokens, total_target_tokens) for auditing. Mutates d."""
    w = np.array(d.loss_fn_inputs["weights"].data, dtype=np.float64)
    tt = np.array(d.loss_fn_inputs["target_tokens"].data, dtype=np.int64)
    idx = np.where(w > 0)[0]                       # target (assistant) token positions
    if len(idx) == 0:
        return 0, 0
    text = tok.decode(tt[idx].tolist())
    ji = text.find(JSON_START)
    if ji < 0:
        return 0, len(idx)                         # no JSON block; leave as-is
    n_before = len(tok.encode(text[:ji], add_special_tokens=False))
    json_pos = idx[n_before:]                       # JSON span -> end of target
    w[json_pos] *= alpha
    w /= w.sum()                                    # renormalize: keep loss scale ~CE
    d.loss_fn_inputs["weights"] = _tensordata(w)
    return len(json_pos), len(idx)


def reweight_batch(batch, tok, alpha):
    boosted = total = 0
    for d in batch:
        b, t = reweight_datum(d, tok, alpha)
        boosted += b
        total += t
    return boosted, total


def make_weighted_nll_loss():
    """Custom loss = -mean_i dot(logprobs_i, weights_i). With CE's sum-1 weights this
    reproduces cross_entropy exactly; with reweighted weights it is score-focused NLL."""
    import torch

    def loss_fn(data, logprobs_list):
        terms = []
        for d, lp in zip(data, logprobs_list):
            w = torch.tensor(np.array(d.loss_fn_inputs["weights"].data), dtype=torch.float32)
            lp = lp.float()
            L = min(len(w), len(lp))
            terms.append(-torch.dot(lp[:L], w[:L]))
        loss = torch.stack(terms).mean()
        return loss, {"loss": float(loss.item())}

    return loss_fn


def _selftest():
    import sys
    sys.path.insert(0, "scripts")
    from tinker_cookbook.tokenizer_utils import get_tokenizer
    from tinker_reasoning_data import build_datasets

    tok = get_tokenizer("Qwen/Qwen3-30B-A3B")
    tr, _ = build_datasets(train_path="data/reasoning_dataset/openai_train_relabel_v1.jsonl",
                           batch_size=8)
    batch = tr.get_batch(0)
    d = batch[0]
    tt = np.array(d.loss_fn_inputs["target_tokens"].data, dtype=np.int64)
    w0 = np.array(d.loss_fn_inputs["weights"].data, dtype=np.float64)
    boosted, total = reweight_datum(d, tok, alpha=3.0)
    w1 = np.array(d.loss_fn_inputs["weights"].data, dtype=np.float64)

    assert abs(w1.sum() - 1.0) < 1e-5, f"weights must renormalize to 1, got {w1.sum()}"
    assert 0 < boosted < total, f"boosted span sanity: {boosted}/{total}"
    # the boosted positions must decode to the JSON region (allow small BPE slack)
    boosted_pos = np.where(w1 > w0 + 1e-12)[0]
    dec = tok.decode(tt[boosted_pos].tolist())
    assert JSON_START in dec or dec.strip().startswith("{") or '"overall' in dec or "score" in dec, \
        f"boosted tokens should be the JSON block, got: {dec[:120]!r}"
    # score digits carry more weight than an average prose token now
    frac_weight_on_json = w1[boosted_pos].sum()
    print(f"selftest OK: boosted {boosted}/{total} target tokens "
          f"({100 * boosted / total:.0f}%), JSON now holds {100 * frac_weight_on_json:.0f}% of loss weight")
    print(f"  boosted span decodes to: ...{dec[-90:]!r}")


if __name__ == "__main__":
    _selftest()
