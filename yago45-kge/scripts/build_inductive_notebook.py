"""Build and execute the portable results notebook from committed evidence.

Run from yago45-kge with .venv/bin/python scripts/build_inductive_notebook.py.
Requires nbformat, nbclient and ipykernel; no TensorFlow or source dataset needed.
"""
from argparse import ArgumentParser
from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient


parser = ArgumentParser(__doc__)
parser.add_argument("--report", default="experiments/inductive_full_20k",
                    help="Evidence directory, relative to yago45-kge")
parser.add_argument("--output", default="notebooks/02_inductive_full_results.ipynb",
                    help="Notebook path, relative to yago45-kge")
options = parser.parse_args()


root = Path(__file__).resolve().parents[1]
report_rel = options.report
md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell
notebook = nbf.v4.new_notebook(cells=[
    md("""# Full-graph inductive YAGO results

Executed from committed experiment evidence; embedded PNG outputs render on GitHub.
No local graph, checkpoint or GPU is needed to rerun this notebook.

The encoder uses ID-independent directional relation-count features and a
`tf_gnns` MPNN, with a contextual ComplEx decoder. Shallow ID tables are trained
separately and are **not** inputs to the transferable encoder.

**Protocol:** sampled-filtered tail prediction, 2,000 queries per test cohort,
up to 50 negatives, pessimistic model ties. These are not full-ranking metrics.
Checkpoint selection uses only temporal validation (mean of uniform/range MRR).
"""),
    code("import json\n"
         "from pathlib import Path\n"
         "from IPython.display import display, Markdown, Image\n"
         "\n"
         "root = Path.cwd()\n"
         "if not (root / 'experiments').exists():\n"
         "    root = root.parent\n"
         f"report = root / '{report_rel}'\n"
         "summary = json.loads((report / 'summary.json').read_text())\n"
         "results = summary['results']\n"
         "\n"
         "def table(headers, rows):\n"
         "    display(Markdown('| ' + ' | '.join(headers) + ' |\\n|'\n"
         "                     + '|'.join(['---'] * len(headers)) + '|\\n'\n"
         "                     + '\\n'.join('| ' + ' | '.join(map(str, row)) + ' |' for row in rows)))\n"
         "\n"
         "table(['Dataset / budget', 'Value'], [\n"
         "    ['Mapped entities', f\"{summary['num_nodes']:,}\"],\n"
         "    ['Training facts', f\"{summary['splits']['train']:,}\"],\n"
         "    ['Inference-support facts', f\"{summary['splits']['support']:,}\"],\n"
         "    ['Relations', summary['num_relations']],\n"
         "    ['Updates / selected step', f\"{summary['completed_steps']:,} / {summary['best']['step']:,}\"],\n"
         "    ['Seed-draw-equivalent epochs', f\"{summary['seed_draw_equivalent_epochs']:.6f}\"],\n"
         "])\n"),
    md("""The uncapped archive yielded 119,045,262 structural entity–entity facts.
Temporal exclusion, deduplication and independent splits produce the counts above.
The full graph is available as context; supervised anchor draws cover every
training entity (see the draws/epochs row), which is participation coverage,
**not** an exhaustive supervised pass over every training fact."""),
    md('## Unseen-entity link prediction and controls'),
    code("""rows = []
for policy in ['uniform', 'relation-range']:
    r = results['trained_' + policy]
    rows.append([policy, f"{r['mrr']:.6f}",
                 '–'.join(f'{v:.6f}' for v in r['mrr_ci95']),
                 f"{r['hits1']:.4f}", f"{r['hits10']:.4f}",
                 f"{r['frequency_mrr']:.6f}"])
table(['Candidates', 'GNN MRR', '95% bootstrap CI', 'Hits@1', 'Hits@10',
       'Frequency MRR (expected ties)'], rows)
gain = results['message_passing_gain']
display(Markdown(f"No-message range MRR: **{results['no_message_relation_range']['mrr']:.6f}**. "
                 f"Paired message-passing gain: **{gain['mean_delta_mrr']:.6f}**, "
                 f"95% CI {gain['paired_bootstrap_ci95']}."))"""),
    md("""Frequency is a strong control: global negatives often make this task easy.
Full directional degree features encode powerful predicate-range information;
high sampled MRR alone is not evidence of general semantic reasoning.
The no-message ablation retains structural features but removes messages.
Raw rankings, relation groups and support-degree groups are retained in the report.
Degree-matched candidates are an additional diagnostic, not a tuning objective."""),
    md('## Temporal test: valid-time completion'),
    code("""table(['Candidates', 'GNN MRR', 'Frequency MRR', 'Hits@1', 'Hits@10'], [
    [policy, f"{results['temporal_test_' + policy]['mrr']:.6f}",
     f"{results['temporal_test_' + policy]['frequency_mrr']:.6f}",
     f"{results['temporal_test_' + policy]['hits1']:.4f}",
     f"{results['temporal_test_' + policy]['hits10']:.4f}"]
    for policy in ['uniform', 'relation-range']])"""),
    md("""Dates are earliest triple-level validity starts: training through 2010,
validation 2011–2018, test 2019–2024. Undated static background remains available.
**This is not causal knowledge-acquisition forecasting.** Neither test policy
was used for checkpoint selection or further tuning; compare each GNN number
against its own frequency control in the table above."""),
    md('## Losses and validation curves'),
    code("""display(Image(filename=str(report / 'learning_curves.png')))
table(['Final objective', 'Loss'], [
    [name, f"{summary['training_final'][key]:.6f}"]
    for name, key in [('GNN', 'gnn_loss'), ('Shallow', 'shallow_loss'), ('Joint', 'loss')]])"""),
    md('## Contextual GNN t-SNE — actual full-run vectors'),
    code("""display(Image(filename=str(report / 'gnn_tsne.png')))
v = summary['visualization']
table(['Visualization diagnostic', 'Value'], list(v.items()))"""),
    md("""Sampled contextual vectors include both train and unseen entities (see the
diagnostics table). Colors denote dominant relations in observed neighborhoods,
not held-out semantic classes.
t-SNE uses PCA initialization and a fixed seed. Relation 1-NN agreement and
trustworthiness measure this diagnostic only; they are **not semantic benchmark
accuracy**, and 2-D spacing must not be interpreted as calibrated similarity."""),
    md('## Independent shallow embeddings t-SNE'),
    code("""display(Image(filename=str(report / 'shallow_tsne.png')))
display(Markdown('Final known-entity validation MRR: **' +
                 f"{results['shallow_final_known_valid']['mrr']:.6f}" + '**.'))"""),
    md('## Complete all-node GNN cache and separate evaluation'),
    code("""cache = json.loads((report / 'cache_metadata.json').read_text())
cached_results = json.loads((report / 'cached_link_prediction.json').read_text())
table(['Cache property', 'Value'], [
    ['Shape', cache['matrix']['shape']], ['Dtype', cache['matrix']['dtype']],
    ['Complete', cache['complete']], ['All values finite', cache['matrix']['all_finite']],
    ['Matrix bytes', cache['matrix']['file_bytes']]])
table(['Candidates', 'MRR', 'Hits@1', 'Hits@10', 'Mean candidates', 'Frequency MRR'], [
    [policy, f"{r['mrr']:.6f}", f"{r['hits1']:.4f}", f"{r['hits10']:.4f}",
     f"{r['mean_candidates']:.3f}", f"{r['frequency_mrr']:.6f}"]
    for policy, r in cached_results.items()])"""),
    md("""Every mapped node has a 200-dimensional float16 vector. Before caching,
all held-out queries and same-relation reverses were globally scrubbed from
context and degree features. Static batched contexts differ from per-query
contexts; candidate pools/frequency controls also reflect the scrubbed graph.
These are separate measurements, not exact checkpoint-ranking reproduction.

The degree-matched diagnostic controls incoming-predicate degree with zero and
logarithmic nonzero bins. Its lower MRR reveals the difficulty hidden by easy
global negatives. Some bins have few candidates (30.7 mean versus 51 uniform),
so compare against its own controls, not directly against the other policies.
Frequency here is simply observed incoming-predicate count, with expected ties.
"""),
    md("""## Reusable artifacts and reproduction

- [Complete report and raw rankings](../""" + report_rel + """/README.md)
- [Source manifest and SHA256](../experiments/inductive_full_manifest.json)
- [Standalone GNN SavedModel](../""" + report_rel + """/gnn_saved_model/):
  about 4 MB; excludes entity-ID tables; exposes `encode`, `predict_graph`,
  and `score_embeddings`.
- [Method, training and CLI commands](../docs/09-inductive-autoencoder.md)
- [Tiny mixed-negative control](../experiments/inductive_tiny_mixed_10k/README.md)
- [Original range-only calibration failure](../experiments/inductive_tiny_10k/README.md)

Large optimizer checkpoints, shallow tables and the all-node float16 cache stay
under ignored `data/yago45_full/`. The cache has a separate evaluation because
batched static contexts and float16 vectors differ from per-query inference.
New entities require an observed neighborhood, not a trained entity-ID row.

Rebuild this executed notebook from `yago45-kge`:
```bash
.venv/bin/python -m pip install nbformat nbclient ipykernel
.venv/bin/python scripts/build_inductive_notebook.py --report """ + report_rel + """
```
"""),
])
notebook.metadata.kernelspec = dict(display_name="Python 3", language="python", name="python3")
NotebookClient(notebook, timeout=120, resources={"metadata": {"path": str(root)}}).execute()
destination = root / options.output
nbf.write(notebook, destination)
print(destination)
