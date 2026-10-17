# EvoMemState environment

Create the inference environment and install the local package:

```bash
conda env create -f envs/evomemstate.environment.yml
conda activate evomemstate
python -m pip install -e .
cp .env.example .env
```

Set `MEMORY_API_KEY`, `MEMORY_BASE_URL`, and `MEMORY_LLM_MODEL` for ontology construction. Set `REASONING_STATE_MODEL`, `REASONING_STATE_BASE_URL`, and `REASONING_STATE_API_KEY` for generated rules. Set `GEN_B_*` for the answer model. Benchmark scoring additionally uses `JUDGE_*`. Optional semantic retrieval uses `EMBEDDING_*`.

The runners source `.env`. Do not commit credentials. The policy and answer models can be served by any compatible OpenAI-style chat endpoint.
