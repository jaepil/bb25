# bb25 (Bayesian BM25)

bb25 is a fast, self-contained BM25 + Bayesian calibration implementation with a minimal Python API. It also includes a small reference corpus and experiment suite so you can validate the expected numerical properties.

## Install

```
pip install bb25
```

## Quick start

### Use the built-in corpus and queries

```
import bb25 as bb

corpus = bb.build_default_corpus()
docs = corpus.documents()
queries = bb.build_default_queries()

bm25 = bb.BM25Scorer(corpus, 1.2, 0.75)
score = bm25.score(queries[0].terms, docs[0])
print("score0", score)
```

### Build your own corpus

```
import bb25 as bb

corpus = bb.Corpus()
corpus.add_document("d1", "neural networks for ranking", [0.1] * 8)
corpus.add_document("d2", "bm25 is a strong baseline", [0.2] * 8)
corpus.build_index()  # must be called before creating scorers

bm25 = bb.BM25Scorer(corpus, 1.2, 0.75)
print(bm25.idf("bm25"))
```

### Bayesian calibration + hybrid fusion

```
import bb25 as bb

corpus = bb.build_default_corpus()
docs = corpus.documents()
queries = bb.build_default_queries()

bm25 = bb.BM25Scorer(corpus, 1.2, 0.75)
bayes = bb.BayesianBM25Scorer(bm25, 1.0, 0.5)
vector = bb.VectorScorer()
hybrid = bb.HybridScorer(bayes, vector)

q = queries[0]
prob_or = hybrid.score_or(q.terms, q.embedding, docs[0])
prob_and = hybrid.score_and(q.terms, q.embedding, docs[0])
print("OR", prob_or, "AND", prob_and)
```

## Calibration modes

BayesianBM25Scorer converts raw BM25 scores into probabilities via a sigmoid: `P(relevant) = sigmoid(alpha * (score - beta))`. The `alpha` and `beta` parameters control the sigmoid's steepness and midpoint. There are three ways to set them:

### Fixed (default)

```python
bayes = bb.BayesianBM25Scorer(bm25, alpha=1.0, beta=0.5)
```

Uses the same `alpha` and `beta` for every term and every query. Simple, fast, and works well when the corpus has a homogeneous score distribution. Best overall choice for general retrieval.

### Per-term dynamic

```python
bayes = bb.BayesianBM25Scorer(bm25, alpha=1.0, beta=0.5, dynamic=True)
```

At construction time, computes the median and standard deviation of BM25 scores for each term across the entire corpus. The sigmoid parameters are then adapted per term:

- `alpha_eff = alpha / std_dev` -- terms with tight score distributions get a steeper sigmoid
- `beta_eff = median` -- the sigmoid midpoint shifts to where typical scores are

This normalizes the sigmoid input similarly to a z-score, so each term contributes on a comparable scale.

### Query-level dynamic

```python
results = bayes.score_query(["neural", "network"], corpus)
# returns List[Tuple[str, float]] -- (doc_id, score) pairs
```

Computes per-term median and standard deviation on-the-fly from the provided document set. Unlike per-term dynamic, the statistics come from the actual candidate documents rather than the full corpus.

Best suited for re-ranking a candidate set retrieved by a first-stage retriever, or for adversarial settings where the score distribution varies widely across queries.

### When to use which

| Mode | Best for | Trade-off |
|------|----------|-----------|
| Fixed | General retrieval, stable corpora | Simple and robust; no per-term adaptation |
| Per-term dynamic | Large corpora with diverse term distributions | Adapts per term; upfront cost at construction |
| Query-level dynamic | Re-ranking, adversarial queries | Adapts to each query's candidate set; slower per query |

## Run the experiments

```
import bb25 as bb

results = bb.run_experiments()
print(all(r.passed for r in results))
```

## Sample script

See `docs/sample_usage.py` for an end-to-end example using BM25, Bayesian calibration, and hybrid fusion.

## Benchmarks (BM25 vs Bayesian)

See `benchmarks/README.md` for a lightweight runner that compares BM25 and Bayesian BM25 on your own corpora.

## BEIR Benchmark Results

Evaluated on four BEIR datasets of increasing difficulty for lexical retrieval.

### NDCG@10

| Dataset | Docs | Queries | BM25 | Bayesian (fixed) | Bayesian (query) |
|---------|------|---------|------|-------------------|-------------------|
| SciFact | 5,183 | 300 | 0.6007 | **0.6563** (+9.3%) | 0.4917 |
| NFCorpus | 3,633 | 323 | 0.2932 | **0.3121** (+6.4%) | 0.2432 |
| FiQA | 57,638 | 648 | 0.2073 | **0.2190** (+5.6%) | 0.1043 |
| ArguAna | 8,674 | 1,406 | 0.0962 | 0.0540 | **0.1577** (+64%) |

**Fixed calibration** consistently improves over BM25 on standard retrieval tasks (SciFact, NFCorpus, FiQA). The symmetric norm_prior corrects length bias, and the fixed sigmoid provides stable probability estimates.

**Query-level dynamic** wins on ArguAna, where counter-argument retrieval causes high lexical overlap between relevant and irrelevant documents. Per-query adaptation of the sigmoid midpoint helps distinguish signal from noise in adversarial score distributions.

### Hybrid Search (SQuAD, 100 validation queries)

| Method               | NDCG@10       | MRR@10   | Notes                                |
| -------------------- | ------------ | -------- | ------------------------------------ |
| **WS (BB25+Dense)**  | **0.9149** | **0.8850** | **SOTA!**                |
| WS (BM25+Dense)      | 0.9051       | 0.8717   |                                      |
| RRF (BM25+Dense)     | 0.8874       | 0.8483   | RRF underperforms weighted sum       |

## Conclusion

Bayesian BM25 (bb25) consistently outperforms classic BM25 on standard retrieval benchmarks (+5--9% NDCG@10 on BEIR). For adversarial or re-ranking scenarios, query-level dynamic calibration provides further gains. In hybrid search, the probabilistic scores from bb25 blend more smoothly with vector scores than raw BM25 (less scale mismatch).

Original paper:

```
https://www.researchgate.net/publication/400212695_Bayesian_BM25_A_Probabilistic_Framework_for_Hybrid_Text_and_Vector_Search
```

## Build from source (Rust)

```
make build
```

## PyPI publishing

Build a wheel with maturin:

```
python -m pip install maturin
maturin build --release
```

For Pyodide builds, see `docs/pyodide.md`.
