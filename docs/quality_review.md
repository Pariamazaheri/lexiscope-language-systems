# Engineering review and verification

The initial implementation supplied completed representation, decoding and training experiments. The review identified weaknesses in evaluation isolation, numerical contracts, artifact provenance and reproducibility. Results in this repository were rerun after the corrections; stale notebook outputs are not treated as verified benchmark evidence.

| Finding | Correction | Verification |
|---|---|---|
| The held-out sentiment set was monitored every epoch while model settings were adjustable | Create separate train, validation and final test partitions | Hash-disjoint manifest; test labels checked against original archive source labels |
| Normalized duplicate reviews could cross partitions | Deduplicate before sampling | Seventeen removed; one partition per normalized text hash |
| Early-stopping settings were configured but unused; final weights replaced the best validation model | Implement patience, checkpoint capture and restoration | Contract test simulates worsening validation and verifies exact best-state restoration |
| Averaging batch losses equally misweights a shorter final batch | Sum losses and divide by actual tokens/reviews | Uneven-batch numerical test and independently recomputed classification log loss |
| Token clamping silently changes invalid IDs | Validate vocabulary bounds and reject invalid inputs | Negative/out-of-range IDs fail without substitution |
| Padding detection used EOS token equality | Propagate explicit masks and gather the last valid position | Valid EOS content and padding-invariance tests |
| Transfer reused a mutable LM body; pretraining and downstream evidence could become conflated | Deep-copy the selected body for each downstream run | Frozen-state comparisons and downstream-update isolation test |
| Parameter-count prose did not match actual configurations | Generate named parameter tables directly from models | Independent reconstruction verifies every parameter shape/count |
| Custom decoding omitted EOS stopping and parameter guards | Validated filters, local seeded RNG and EOS-aware loop | EOS, nucleus, top-k and sampling reproducibility contracts; three real greedy parity checks |
| BERT input-layer interpretation omitted positional information | Add matched-position controls and character-offset span pooling | Saved layer vectors reproduce all cosine curves |
| Visual clustering was described as stronger proof than the projection supports | Measure three-seed local-neighbor retention | Original geometry, projection coordinates and stability table retained |
| Persistent notebook state and local paths impeded reproduction | Package modular commands and preserve source manifests/revisions | Public-source reconstruction reproduces exact normalized splits; installed CLI smoke tests |

## Evidence coverage

| Capability | Saved evidence |
|---|---|
| Static neighbors and analogy arithmetic | `static_neighbors.csv` |
| Static geometry and projection stability | `static_geometry.npz`, `static_projection.csv`, `projection_stability.csv` |
| Contextual layers and position controls | `contextual_vectors.npz`, `contextual_similarity.csv` |
| Greedy/beam/stochastic and custom decoding | `decoding_samples.json`, `greedy_parity.csv`, `prompt_uncertainty.csv` |
| LSTM/Transformer next-token models | `language_model_metrics.csv`, `lm_split_manifest.csv` |
| Training curves and selected checkpoints | `training_history.csv`, `checkpoint_selection.csv` |
| Scratch/transfer classifiers and lexical reference | `sentiment_metrics.csv`, `sentiment_predictions.csv` |
| Conditional transfer uncertainty | `paired_transfer.csv` |
| Parameter anatomy | `parameter_breakdown.csv`, `parameter_summary.csv` |
| Operational candidate selection | `validation_candidates.csv`, `selection.json` |
| Source and run provenance | `source_audit.json`, `run_metadata.json`, embedding/generation metadata |

The tests establish contracts for these implementations; they do not establish application-level generalization. The independent result auditor recomputes published metrics, confidence intervals, layer similarities and parameter counts without downloading external weights. GitHub CI runs those checks on Python 3.11 and 3.12 and builds an installable wheel.
