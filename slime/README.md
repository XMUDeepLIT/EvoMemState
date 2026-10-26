# slime training backend

Runtime source, model definitions and conversion tools vendored from the original project's slime checkout at revision `a3f500977f33b82de6ac0414c0c43abfe656c3e0`. The working-tree change in `slime/utils/arguments.py` is retained. Upstream Apache-2.0 license is preserved in LICENSE. Git history, datasets, checkpoints, caches, external examples and local environment files are excluded.

Install with `pip install -e ./slime`. CUDA-compatible PyTorch, SGLang, Megatron-LM and fused training kernels must be installed separately for your GPU environment.
