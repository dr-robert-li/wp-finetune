#!/usr/bin/env python
"""Tinker connectivity + loop smoke (P0 of the Tinker pivot).

Run in the Tinker venv with the key exported:
    export TINKER_API_KEY="$(grep '^TINKER_API_KEY=' .env | cut -d= -f2-)"
    .venv-tinker/bin/python scripts/_tinker_smoke.py [--loop]

- default: auth + list server capabilities + assert Qwen3-30B-A3B is accessible (FREE, no compute).
- --loop:  also run one forward_backward + optim_step on Llama-3.2-1B (cheap; validates the loop).
"""
import sys

import tinker


def main():
    sc = tinker.ServiceClient()
    caps = sc.get_server_capabilities()
    models = getattr(caps, "supported_models", None) or getattr(caps, "models", []) or []

    def name_of(m):
        for a in ("model_name", "name", "id"):
            if hasattr(m, a):
                return getattr(m, a)
        return str(m)

    names = [name_of(m) for m in models]
    print(f"AUTH OK — {len(names)} models available")
    target = [n for n in names if "Qwen3-30B-A3B" in str(n)]
    assert target, "Qwen3-30B-A3B NOT accessible to this account"
    print("Qwen3-30B-A3B variants:", target)

    if "--loop" in sys.argv:
        import numpy as np
        tc = sc.create_lora_training_client(base_model="meta-llama/Llama-3.2-1B", rank=8)
        toks = [1, 2, 3, 4, 5, 6, 7, 8]
        d = tinker.Datum(
            model_input=tinker.ModelInput.from_ints(toks),
            loss_fn_inputs={
                "target_tokens": tinker.TensorData.from_numpy(np.array(toks[1:] + [0], dtype=np.int32)),
                "weights": tinker.TensorData.from_numpy(np.array([1] * 7 + [0], dtype=np.float32)),
            },
        )
        fb = tc.forward_backward(data=[d], loss_fn="cross_entropy")
        (fb.result() if hasattr(fb, "result") else fb)
        os_ = tc.optim_step(tinker.AdamParams(learning_rate=1e-4))
        (os_.result() if hasattr(os_, "result") else os_)
        print("LOOP SMOKE PASS (forward_backward + optim_step)")
    print("P0 OK")


if __name__ == "__main__":
    main()
