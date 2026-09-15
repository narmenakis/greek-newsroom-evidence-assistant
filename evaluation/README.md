# Evaluation artifacts and benchmark records

These fixtures belong to the current 200-URL development corpus. The corpus is intentionally Tempi-heavy, but also contains non-Tempi articles that are useful for negative and source-diversity cases.

`corpus_manifest.json` records the URL-list identity and expected source distribution. The article CSV and vector database are generated locally and are not committed.

`tempi_questions.jsonl` contains the versioned retrieval cases. Its contract is described by `tempi_questions.schema.json`, and `evaluation_manifest.json` ties the question fixture to `corpus_manifest.json`, records its SHA-256 hash, and records the label-review status. Each case now records task type, topic, difficulty, answerability, language, relevant URL/article-ID pairs, acceptable-evidence notes, and review provenance. All 11 cases received an assistant review against the local article text and project-owner confirmation. They are intentionally separate from generated answers.

Validate the fixture without loading a model or vector store:

```bash
PYTHONPATH=src ./.venv/bin/python -m journalism_rag.evaluation.fixtures
```

Also verify every labeled URL and stable article ID against the ignored local corpus snapshot:

```bash
PYTHONPATH=src ./.venv/bin/python -m journalism_rag.evaluation.fixtures --corpus sample_database.csv
```

The validator rejects malformed JSON, duplicate case IDs, unsupported annotations, invalid URLs or article IDs, inconsistent answerability labels, invalid filters, and URL/ID pairs absent from the selected corpus. The corpus check validates identity only; it is not a semantic relevance score. Content-review judgments are recorded separately in each case's `label_review` and `acceptable_evidence` fields.

Run the provider-free retrieval benchmark against the existing local Chroma
collection:

```bash
HF_HUB_OFFLINE=1 PYTHONPATH=src ./.venv/bin/python -m journalism_rag.evaluation.retrieval_benchmark \
  --output evaluation/retrieval_benchmark.json
```

The command runs dense and hybrid retrieval independently of generation and
records per-case results plus aggregate breakdowns by topic, outlet, language,
and filter behavior. Article-level Recall@K is the fraction of labeled relevant
articles found among the first K unique article results; hit@K records whether
any relevant article was found. MRR and nDCG@10 use the first 10 unique article
results. The no-answer case remains in the per-case report but is excluded from
ranking denominators. The latest local run is recorded in
`retrieval_benchmark.json`; it is a measurement of this corpus/index snapshot,
not a general model-quality claim.

The generation contract is covered separately by
`tests/test_phase4_generation_contracts.py`. Its nine report cases (represented
by eight focused test methods) use mock provider
responses to check language matching, structural citation failures, abstention,
source-conflict instructions, prompt-injection boundaries, and typed provider
failures. Run the report command with:

```bash
PYTHONPATH=src ./.venv/bin/python -m journalism_rag.evaluation.generation_contract_benchmark
```

It writes `generation_contract_benchmark.json` and
`generation_contract_report.md`. It does not call DeepSeek or Ollama and is not
a live model-quality measurement.

The review corrected two seed labels: `tempi-investigation-001` pointed to a URL absent from the validated snapshot and now uses the present article covering the same investigation thread; `tempi-followup-001` had an incomplete slug and now uses the exact canonical corpus URL. Details are recorded in `evaluation_manifest.json`.

`label_review.md` is the completed human-confirmation packet. It presents each question,
the expected evidence, labeled titles and stable IDs, short excerpts from the
frozen local article text, and blank confirm/revise decisions. Completing that
packet is a human judgment. The project-owner review confirmed ten cases as
presented and revised `tempi-fact-001` to remove the older 38-death article;
the fixture now records `human_confirmed` for all 11 cases.

The earlier pre-refactor baseline and longitudinal hardware manifest are kept
in the private development repository, not this public release. The reports
included here are the selected corpus, retrieval, embedding, reranker, and
mocked generation-contract artifacts. Historical model identifiers remain valid
only for the configuration and hardware recorded when each measurement ran;
they are not current provider defaults. Live model-quality evaluation remains
outside the deterministic evaluation scope.

The URL list contains 200 entries, but one EFSYN URL returns HTTP 404. The scraper now rejects that page, so the validated local snapshot contains 199 articles; the rejected URL is recorded in `corpus_manifest.json`.

`embedding_benchmark.json` summarizes the embedding comparison. The detailed
E5 reports contain per-question rankings. The deferred native MPS run for
Qwen3-Embedding-0.6B is recorded in `embedding_benchmark_qwen_mps.json`: it
completed in 161.154 seconds with Recall@5 0.416667 and nDCG@10 0.40595, below the
selected E5-large result (Recall@5 0.483333, nDCG@10 0.545581). E5-large remains the
production embedder, and the Qwen embedding is not used to build a collection.

The active local collection is `journalism_rag__e5-large-instruct`. Production indexing and retrieval use the same E5 passage and query prompt formats measured by the harness. Obsolete incompatible collections are not retained locally; historical reports keep the collection identifiers captured when those measurements were made.

`reranker_benchmark.json` records the original Phase 3 comparison of hybrid retrieval with and without `BAAI/bge-reranker-v2-m3` over the 11-case fixture. It reranks the same 20 hybrid candidates and measures article-level Recall@5/10, MRR, nDCG@10, retrieval time, reranking time, model-load time, and peak memory. With corrected fraction-based Recall, the CPU run raised MRR from 0.481667 to 0.673611 and nDCG@10 from 0.463935 to 0.539404, while Recall@5 changed from 0.45 to 0.416667; its 196.0-second fixture runtime keeps it opt-in.

`reranker_benchmark_qwen.json` records the native arm64/MPS comparison for
`Qwen/Qwen3-Reranker-0.6B`. Its causal-LM yes/no scoring adapter raised
Recall@5/10 to 0.583333/0.7, MRR to 0.761667, and nDCG@10 to 0.638098; 20 candidates
were scored in 16.9 seconds across the fixture, with a 3.6 GB process peak.
`reranker_benchmark_bge_native.json` records the matching native BGE run:
Recall@5/10 0.416667/0.616667, MRR 0.673611, nDCG@10 0.539404, 9.0 seconds of reranking,
and a 2.8 GB process peak. The evaluation-only diversity selector matched the
hybrid baseline exactly, so it remains disabled. Neither reranker is enabled
by default; Qwen is the selected opt-in model for the demonstration, while BGE
remains a modular alternative that is faster and smaller but loses Recall@5.
Run the Qwen report with:

```bash
HF_HUB_OFFLINE=1 RAG_EMBEDDING_DEVICE=mps PYTHONPATH=src \
  ./.venv/bin/python -m journalism_rag.evaluation.reranker_benchmark \
  --reranker-model Qwen/Qwen3-Reranker-0.6B --reranker-device mps \
  --reranker-batch-size 4 --reranker-max-length 256 \
  --output evaluation/reranker_benchmark_qwen.json
```

The Qwen reranker is integrated as an opt-in pipeline stage. The provider
switch and local tool-calling contract are covered by mocked tests; no formal
live research or provider-quality result is claimed by this release.
