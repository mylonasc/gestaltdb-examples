# YAGO 4.5 derived artifacts — attribution and licensing

Upstream work: **YAGO 4.5**, by the **YAGO team of Télécom Paris**.

- Data and specific licensing statement: https://yago-knowledge.org/downloads/yago-4-5
- Source archive: https://yago-knowledge.org/data/yago4.5/yago-4.5.0.2.zip
- License: Creative Commons Attribution-ShareAlike 3.0 Unported
  https://creativecommons.org/licenses/by-sa/3.0/
- Source data combines Wikidata with Schema.org taxonomy/properties; the YAGO
  data page acknowledges these contributors and identifies ShareAlike terms.
- YAGO's generator-code license and the website footer do not replace the
  version-specific data license.

Source citation: Fabian M. Suchanek, Mehwish Alam, Thomas Bonald, Lihu Chen,
Pierre-Henri Paris, and Jules Soria. *YAGO 4.5: A Large and Clean Knowledge Base
with a Rich Taxonomy.* SIGIR, 2024.
https://suchanek.name/work/publications/sigir-2024.pdf

Independent adaptation by the `mylonasc/gestaltdb-examples` project:
structural-fact filtering, temporal/entity splits, masked graph features,
GNN/ComplEx training, lightweight model export, query/reverse scrubbing and
float16 embedding caching. See each repository card for artifact scope.

This release conservatively applies CC BY-SA 3.0 to the derived artifacts and
redistributed mappings. It does not claim exclusive rights over source facts,
assert settled legal status for learned weights, or relicense independent
software. No endorsement by YAGO, Télécom Paris, Wikidata or Schema.org is
implied. The work is provided as-is, without warranties; retain applicable
license and attribution notices. See LICENSE for the complete terms, including
disclaimers and limitations of liability.
