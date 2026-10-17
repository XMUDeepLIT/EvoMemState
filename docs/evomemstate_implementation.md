# EvoMemState inference implementation

The persistent memory ontology is stored by `OntologyStore` in `evomemstate/ontology.py`. `MemoryBuilder` and `OntologyGuidedExtractor` add atomic memories and maintain `SUPERSEDES` and `CONFLICTS_WITH` edges. Serialized ontology files are named `ontology.json` in the benchmark adapters.

`retrieve` in `evomemstate/retrieval.py` returns a `RelationalState`. It contains the memories selected for the current question, relation-complete bundles, selected concepts, ranking signals, and relation audit fields. This state is transient; the ontology remains the persistent store.

`evomemstate/reasoning_state.py` renders those memories and asks a policy endpoint to generate a request-specific Reasoning State. The model is instructed to choose which evidence, preferences, constraints, conflicts, and updates should control the answer without answering the question itself. `evomemstate/prompting.py` combines the Relational State and generated rules in the answer context. A fixed rule set is used if policy generation fails.

The MemSyco and MemTrap adapters use the same harness in `evomemstate/harness/`. LoCoMo uses `evomemstate/run_locomo.py`. The benchmark adapters retain external harness fields such as `retrieved_memories` because those fields are defined by the external harness.
