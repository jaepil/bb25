import unittest

import bb25 as bb


class SmokeTests(unittest.TestCase):
    def test_run_experiments(self):
        results = bb.run_experiments()
        self.assertEqual(len(results), 13)
        self.assertTrue(all(r.passed for r in results))

    def test_default_builders(self):
        corpus = bb.build_default_corpus()
        queries = bb.build_default_queries()
        self.assertEqual(corpus.n, 20)
        self.assertEqual(len(queries), 7)

    def test_scoring(self):
        corpus = bb.build_default_corpus()
        doc = corpus.get_document("d01")
        bm25 = bb.BM25Scorer(corpus, 1.2, 0.75)
        bayes = bb.BayesianBM25Scorer(bm25)
        score = bayes.score(["machine", "learning"], doc)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_dynamic_scoring(self):
        corpus = bb.build_default_corpus()
        bm25 = bb.BM25Scorer(corpus, 1.2, 0.75)
        bayes = bb.BayesianBM25Scorer(bm25, dynamic=True)
        self.assertTrue(bayes.has_dynamic_term_stats)

        bayes_fixed = bb.BayesianBM25Scorer(bm25)
        self.assertFalse(bayes_fixed.has_dynamic_term_stats)

        doc = corpus.get_document("d01")
        score = bayes.score(["machine", "learning"], doc)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_score_query(self):
        corpus = bb.build_default_corpus()
        bm25 = bb.BM25Scorer(corpus, 1.2, 0.75)
        bayes = bb.BayesianBM25Scorer(bm25)

        results = bayes.score_query(["machine", "learning"], corpus)
        self.assertEqual(len(results), corpus.n)
        for doc_id, score in results:
            self.assertIsInstance(doc_id, str)
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)

    def test_prior_weight(self):
        corpus = bb.build_default_corpus()
        bm25 = bb.BM25Scorer(corpus, 1.2, 0.75)

        # prior_weight=0.0 preserves BM25 ordering
        flat = bb.BayesianBM25Scorer(bm25, prior_weight=0.0)
        self.assertAlmostEqual(flat.prior_weight, 0.0)

        terms = ["machine", "learning"]
        docs = corpus.documents()
        bm25_ranking = sorted(docs, key=lambda d: -bm25.score(terms, d))
        bayes_ranking = sorted(docs, key=lambda d: -flat.score(terms, d))
        self.assertEqual(
            [d.id for d in bm25_ranking],
            [d.id for d in bayes_ranking],
        )

        # prior_weight=1.0 (default)
        full = bb.BayesianBM25Scorer(bm25)
        self.assertAlmostEqual(full.prior_weight, 1.0)


if __name__ == "__main__":
    unittest.main()
