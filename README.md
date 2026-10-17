# EvoMemState

EvoMemState is an inference pipeline for personal memory. Dialogue updates a persistent **memory ontology**. For each question, retrieval produces a request-specific **Relational State**: selected memories, their status, scope, and relations such as `SUPERSEDES` and `CONFLICTS_WITH`. A policy model then generates a **Reasoning State**: short rules for how the answer model should use that evidence. The answer model receives both states as context.

The Python package, command names, and environment prefix are `evomemstate` / `EVOMEMSTATE`.

## Install

Use Python 3.12 for the benchmark runners:

```bash
conda env create -f envs/evomemstate.environment.yml
conda activate evomemstate
python -m pip install -e .
cp .env.example .env
```

Fill `.env` with endpoints for ontology construction (`MEMORY_*`), the Reasoning State model (`REASONING_STATE_*`), the answer model (`GEN_B_*`), and the benchmark judge (`JUDGE_*`). The Reasoning State model is an inference endpoint; this repository contains no policy training pipeline or training dataset.

## Inference flow

1. `evomemstate.builder` and `evomemstate.ontology` extract facts from dialogue and update the memory ontology.
2. `evomemstate.retrieval.retrieve(...)` returns a `RelationalState` for a question.
3. `evomemstate.reasoning_state.render_with_reasoning_state(...)` renders the Relational State and asks the policy model for request-specific rules. Set `EVOMEMSTATE_REASONING_STATE_ENABLED=1` to use it in the harness.
4. `evomemstate.prompting.render_context(...)` renders the Relational State and Reasoning State for the answer model.

If the policy endpoint fails, inference falls back to fixed Reasoning State rules and records the error in metadata.

## Run evaluations

MemSyco-Bench: place its checkout at `datasets/MemSyco-Bench`, register the bridge, and run the inference script:

```bash
bash scripts/install_memsyco_adapter.sh
DRY_RUN=1 MODE=smoke bash scripts/run_evomemstate_memsyco.sh
MODE=smoke bash scripts/run_evomemstate_memsyco.sh
```

LoCoMo-Refined: place its checkout beside this repository or set `LOCOMO_BENCHMARK_ROOT`, then run:

```bash
bash scripts/run_locomo_reasoning_state.sh
```

The benchmark runners build or load ontology graphs, retrieve a Relational State, generate a Reasoning State, and score answers. `datasets/` and `outputs/` are local runtime directories and are ignored by Git.

## Offline checks

```bash
python -m unittest discover -s tests -q
```

See [implementation notes](docs/evomemstate_implementation.md) and [environment setup](docs/INSTALL_ENV.md).
