"""Wrap a checkpoint's weights with the ``module.`` prefix DDP evaluation expects.

valid_predictor loads state dicts into a DistributedDataParallel wrapper, whose
keys are ``module.``-prefixed; EMA extractions (and plain single-GPU
checkpoints) are not. This rewrites the keys so such checkpoints can be
evaluated with the standard DDP validation entrypoint unchanged.
"""

import argparse

import torch


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    args = ap.parse_args()
    ck = torch.load(args.src, map_location="cpu", weights_only=False)
    state = ck["model"] if "model" in ck else ck
    wrapped = {(k if k.startswith("module.") else f"module.{k}"): v for k, v in state.items()}
    torch.save({"model": wrapped}, args.dst)
    print(f"wrote {args.dst}")


if __name__ == "__main__":
    main()
