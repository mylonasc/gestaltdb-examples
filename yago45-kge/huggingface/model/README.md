---
license: cc-by-sa-3.0
language:
- en
tags:
- knowledge-graph
- graph-neural-network
- link-prediction
- yago
library_name: tensorflow
---

# YAGO 4.5 inductive graph autoencoder

An independently trained, ID-independent GNN encoder and contextual ComplEx
decoder for structural knowledge-graph link prediction on YAGO 4.5.

**Repository status:** private release preparation. This repository currently
contains documentation and licensing notices only. Model files are not uploaded
yet. It is not an official YAGO release and is not endorsed by the YAGO team.

## Model and intended use

- Encoder: `tf_gnns` mean-message-passing MPNN, three steps, hidden width 128,
  with per-step residual connections.
- Inputs: masked directional relation-count features and observed neighborhoods;
  no learned entity-ID row is required for the GNN.
- Outputs: normalized 200-dimensional contextual entity vectors.
- Decoder: relation-conditioned ComplEx scoring.
- Standalone TensorFlow SavedModel: approximately 1.3 MB, with `encode`,
  `predict_graph`, and `score_embeddings` interfaces.
- Independently trained 32-dimensional shallow ComplEx tables are a separate
  representation, not GNN inputs. They are not included in the standalone GNN.

Intended for research on inductive graph representations and link completion.
An unseen entity needs an observed neighborhood; entity names alone are not
model inputs. Cosine similarity is not validated as a general semantic metric.

## Training and data

Source: YAGO archive `yago-4.5.0.2.zip`, SHA256
`b226ba9ce82e3c13187c501bca39c766cb243d0ed02c6b5137146370fadb8a8a`.
All Turtle members were streamed without an edge or line cap. Literal/text
facts were excluded, yielding 119,045,262 retained entity–entity facts before
temporal exclusion, deduplication and splitting.

The prepared graph contains 49,281,921 entities, 63 relations, 96,428,602 training
facts and 21,783,785 inference-support facts. Training ran 185,111 updates under a
strict two-pass entity-coverage schedule (86,927,656 supervised anchor draws;
every training entity participates as a supervised positive endpoint at least
twice), with mixed global/range negatives and fanouts 8/4/2. Batch size migrated
128 → 512 at step 20,439 and sampling switched to vectorized CSR v3 under an
audited coverage-preserving migration; a forked 4-worker shared-memory producer
pool (per-step deterministic batch RNG) was adopted at step 44,631 under a
second audited migration that preserved the exact anchor schedule. The exported
checkpoint is the final step 185,111 (coverage-complete; validation-best step
185,000 is retained separately). Temporal validation guided checkpoint retention
only, never the coverage schedule.

## Evaluation and limitations

Sampled-filtered tail ranking, 2,000 queries per cohort, up to 50 negatives;
pessimistic model ties. These are **not full-ranking results**.

| Inductive candidates | MRR | Hits@1 | Hits@10 | Frequency MRR (expected ties) |
|---|---:|---:|---:|---:|
| Uniform | 0.974023 | 0.9605 | 0.9980 | 0.964812 |
| Relation-range | 0.886853 | 0.8255 | 0.9845 | 0.779454 |

No-message range MRR is 0.675571. Paired message-passing gain is 0.211282,
with a bootstrap 95% interval [0.195097, 0.226602]. Strong degree/frequency
shortcuts mean high sampled MRR is not evidence of general semantic reasoning.

Temporal test MRR is 0.859868 (uniform) and 0.582621 (range), versus frequency
0.838928 and 0.387878. Both test policies beat their controls here.
Dates represent earliest annotated triple-validity starts: training through
2010, validation 2011–2018, test 2019–2024. Undated static background remains;
this is **valid-time completion, not causal knowledge-acquisition forecasting**.

## Related resources

- Embedding dataset: https://huggingface.co/datasets/mylonasc/yago45-inductive-embeddings
- Code and executed notebook with t-SNE/loss curves:
  https://github.com/mylonasc/gestaltdb-examples/tree/main/yago45-kge
- Reproducible results milestone: Git commit `139eb02` in that repository.

## License and attribution

This release is offered under **Creative Commons Attribution-ShareAlike 3.0
Unported (CC BY-SA 3.0)**, conservatively treating trained artifacts as adaptations
of YAGO 4.5. See [LICENSE](LICENSE) and [NOTICE.md](NOTICE.md). This choice does
not assert that all learned weights necessarily constitute derivative works.
It does not relicense independent training code or third-party libraries.

YAGO 4.5 is licensed by the **YAGO team of Télécom Paris** under CC BY-SA 3.0:
https://yago-knowledge.org/downloads/yago-4-5

Changes by this project include structural filtering, temporal/entity splits,
masked-feature construction, GNN/decoder training, and model export. Outputs
are learned approximations, not authoritative source facts. No warranties are
provided; retain attribution/license notices and identify modifications when
redistributing. Do not imply endorsement by YAGO or its contributors.

### Source citation

Fabian M. Suchanek, Mehwish Alam, Thomas Bonald, Lihu Chen, Pierre-Henri Paris,
and Jules Soria. **YAGO 4.5: A Large and Clean Knowledge Base with a Rich
Taxonomy.** SIGIR, 2024. https://suchanek.name/work/publications/sigir-2024.pdf
