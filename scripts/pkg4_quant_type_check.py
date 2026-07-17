"""Per-tensor quant-type census + shared-expert/DeltaNet precision assertions
for a produced GGUF (PKG4-01 Wave 0, T-27-01/T-27-03 mitigation).

Independently reads the produced GGUF's PER-TENSOR types via gguf.GGUFReader --
does NOT trust llama-quantize's own claim about what it did. Verified against
`~/llama.cpp/src/llama-quant.cpp:288-355` (tensor_allows_quantization()): no
shared-expert special case exists there, so uniform routed==shared precision is
the EXPECTED state -- this script asserts that expectation.

Real tensor names below were derived by reading the actual produced v4 judge
GGUF (models/_gguf/wp-v4-judge-s1.Q8_0.gguf), not guessed:
  routed experts : blk.N.ffn_{gate,up,down}_exps.weight
  shared expert  : blk.N.ffn_{gate,up,down}_shexp.weight
                   (blk.N.ffn_gate_inp_shexp.weight is the shared-expert's own
                   router/gate tensor -- excluded, same as the main router
                   ffn_gate_inp is excluded from the routed-expert set; router
                   tensors are F32 by llama.cpp convention, unrelated to the
                   T-27-01 shared-expert-precision question)
  DeltaNet state : blk.N.ssm_{a,alpha,beta,conv1d,dt,norm,out} -- this v4 judge
                   is a hybrid DeltaNet(30 layers)/full-attention(10 layers)
                   architecture (Phase 25 routing profile); only DeltaNet
                   layers carry ssm_* tensors.

Usage:
    python3 scripts/pkg4_quant_type_check.py <gguf> --expect Q8_0 [--json output/pkg-v4/quant_type_census.json]
    python3 scripts/pkg4_quant_type_check.py --self-check   # no GGUF, no network, no GPU
"""
from __future__ import annotations

import argparse
import datetime
import json
import re
import sys
from pathlib import Path

# ponytail: approximate bits-per-weight for the GGML quant types this project's
# ladder actually touches (PKG4-02: Q8 -> Q6_K -> Q5_K_M). Extend if a new tier
# is added -- an unmapped type raises loudly rather than silently comparing wrong.
BITS_PER_WEIGHT = {
    "F32": 32.0,
    "F16": 16.0,
    "BF16": 16.0,
    "Q8_0": 8.5,
    "Q6_K": 6.5625,
    "Q5_K_M": 5.5,
    "Q5_K": 5.5,
    "Q5_0": 5.5,
    "Q4_K_M": 4.5,
    "Q4_K": 4.5,
    "Q4_0": 4.5,
}

ROUTED_RE = re.compile(r"ffn_(gate|up|down)_exps")
SHARED_RE = re.compile(r"ffn_(gate|up|down)_shexp")
DELTANET_RE = re.compile(r"(^|\.)ssm_")


def bits(type_name: str) -> float:
    if type_name not in BITS_PER_WEIGHT:
        raise KeyError(
            f"UNKNOWN GGML TYPE {type_name!r} -- add it to BITS_PER_WEIGHT before trusting this check"
        )
    return BITS_PER_WEIGHT[type_name]


def check_census(routed_types: set, shared_types: set, deltanet_types: set, expect: str) -> None:
    """Pure assertion body shared by the real read path AND --self-check --
    a self-check that tests a copy of this logic tests nothing (per-plan requirement).

    ponytail: NOT a literal routed_types == {expect} match. llama.cpp's K-quant "M"/"L"
    tiers (e.g. Q5_K_M) are deliberately MIXED-precision per tensor role -- ffn_down_exps
    stays at the next type up (Q6_K) while gate/up drop to the nominal type (Q5_K). The
    real T-27-01 invariant is (a) shared expert precision never diverges from routed expert
    precision and (b) nothing drops BELOW the tier's nominal floor -- not "exactly one type".
    """
    assert routed_types, "ROUTED EXPERT TYPE CENSUS EMPTY -- pattern matched 0 tensors"
    assert shared_types == routed_types, (
        f"SHARED EXPERT TYPE DIVERGENCE: shared={shared_types} routed={routed_types}"
    )
    floor = bits(expect)
    for t in routed_types:
        assert bits(t) >= floor, (
            f"ROUTED/SHARED EXPERT PRECISION BELOW {expect}: {t} ({bits(t)} bits) < floor ({floor} bits)"
        )
    assert deltanet_types, "DELTANET TENSOR PATTERN MATCHED 0 TENSORS -- fix the pattern"
    for t in deltanet_types:
        assert bits(t) >= floor, (
            f"DELTANET PRECISION BELOW {expect}: {t} ({bits(t)} bits) < floor ({floor} bits)"
        )


def _self_check() -> None:
    # Fabricated in-memory census matching a healthy Q8_0 conversion (routed==shared
    # Q8_0, DeltaNet state a mix of F32/Q8_0 -- matches the real GGUF measured on disk).
    check_census({"Q8_0"}, {"Q8_0"}, {"F32", "Q8_0"}, "Q8_0")

    # Deliberately-divergent fake: shared expert dropped to F16 while routed stays Q8_0.
    # A self-check that can't go red proves nothing (per-plan requirement).
    try:
        check_census({"Q8_0"}, {"F16"}, {"F32"}, "Q8_0")
    except AssertionError:
        pass
    else:
        raise AssertionError("self-check FAILED: divergent shared-expert census did not raise")

    print("self-check OK")


def real_run(gguf_path: str, expect: str, json_out: str | None) -> int:
    from gguf import GGUFReader
    import gguf as gguf_mod

    r = GGUFReader(gguf_path)
    routed_types: set = set()
    shared_types: set = set()
    deltanet_types: set = set()
    type_census: dict = {}

    for t in r.tensors:
        tname = gguf_mod.GGMLQuantizationType(t.tensor_type).name
        type_census[tname] = type_census.get(tname, 0) + 1
        if ROUTED_RE.search(t.name):
            routed_types.add(tname)
        if SHARED_RE.search(t.name):
            shared_types.add(tname)
        if DELTANET_RE.search(t.name):
            deltanet_types.add(tname)

    check_census(routed_types, shared_types, deltanet_types, expect)
    print(
        f"[pkg4_quant_type_check] routed={routed_types} shared={shared_types} "
        f"deltanet={deltanet_types}"
    )
    print("[pkg4_quant_type_check] shared-expert uniform: PASS")

    if json_out:
        receipt = {
            "gguf": gguf_path,
            "expect": expect,
            "routed_expert_types": sorted(routed_types),
            "shared_expert_types": sorted(shared_types),
            "deltanet_state_types": sorted(deltanet_types),
            "type_census": type_census,
            "shared_expert_uniform": True,
            "checked_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }
        Path(json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(json_out).write_text(json.dumps(receipt, indent=2))
        print(f"[pkg4_quant_type_check] receipt written to {json_out}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("gguf", nargs="?", help="path to the produced GGUF (required unless --self-check)")
    ap.add_argument("--expect", help="expected uniform GGML quant type, e.g. Q8_0, Q6_K, Q5_K_M")
    ap.add_argument("--json", help="optional path to write the receipt JSON")
    ap.add_argument("--self-check", action="store_true")
    args = ap.parse_args()

    if args.self_check:
        _self_check()
        return 0

    if not args.gguf or not args.expect:
        ap.error("gguf and --expect are required unless --self-check")
    if not Path(args.gguf).is_file():
        print(f"MISSING gguf: {args.gguf}", file=sys.stderr)
        return 2

    return real_run(args.gguf, args.expect, args.json)


if __name__ == "__main__":
    raise SystemExit(main())
