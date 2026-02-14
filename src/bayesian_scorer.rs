use std::collections::HashMap;
use std::rc::Rc;

use crate::bm25_scorer::{BM25Scorer, TermScoreStats};
use crate::corpus::Document;
use crate::math_utils::{clamp, median, safe_log, safe_prob, sigmoid, softsign_calibrate, std_dev, EPSILON};

pub struct BayesianBM25Scorer {
    bm25: Rc<BM25Scorer>,
    alpha: f64,
    beta: f64,
    prior_weight: f64,
    term_stats: Option<HashMap<String, TermScoreStats>>,
}

impl BayesianBM25Scorer {
    pub fn new(bm25: Rc<BM25Scorer>, alpha: f64, beta: f64) -> Self {
        Self {
            bm25,
            alpha,
            beta,
            prior_weight: 1.0,
            term_stats: None,
        }
    }

    pub fn with_prior_weight(
        bm25: Rc<BM25Scorer>,
        alpha: f64,
        beta: f64,
        prior_weight: f64,
    ) -> Self {
        Self {
            bm25,
            alpha,
            beta,
            prior_weight: clamp(prior_weight, 0.0, 1.0),
            term_stats: None,
        }
    }

    pub fn with_dynamic_term_stats(bm25: Rc<BM25Scorer>, alpha: f64, beta: f64) -> Self {
        let stats = bm25.compute_term_stats();
        Self {
            bm25,
            alpha,
            beta,
            prior_weight: 1.0,
            term_stats: Some(stats),
        }
    }

    pub fn with_dynamic_term_stats_and_prior_weight(
        bm25: Rc<BM25Scorer>,
        alpha: f64,
        beta: f64,
        prior_weight: f64,
    ) -> Self {
        let stats = bm25.compute_term_stats();
        Self {
            bm25,
            alpha,
            beta,
            prior_weight: clamp(prior_weight, 0.0, 1.0),
            term_stats: Some(stats),
        }
    }

    pub fn has_dynamic_term_stats(&self) -> bool {
        self.term_stats.is_some()
    }

    pub fn prior_weight(&self) -> f64 {
        self.prior_weight
    }

    pub fn likelihood(&self, score: f64) -> f64 {
        sigmoid(self.alpha * (score - self.beta))
    }

    pub fn tf_prior(&self, tf: usize) -> f64 {
        0.2 + 0.7 * (tf as f64 / 10.0).min(1.0)
    }

    pub fn norm_prior(&self, doc_length: usize, avg_doc_length: f64) -> f64 {
        if avg_doc_length < 1.0 {
            return 0.5;
        }
        let ratio = doc_length as f64 / avg_doc_length;
        let deviation = (ratio - 1.0).abs() * 0.5;
        clamp(0.3 + 0.6 * (1.0 - deviation.min(1.0)), 0.1, 0.9)
    }

    pub fn composite_prior(&self, tf: usize, doc_length: usize, avg_doc_length: f64) -> f64 {
        let p_tf = self.tf_prior(tf);
        let p_norm = self.norm_prior(doc_length, avg_doc_length);
        clamp(0.7 * p_tf + 0.3 * p_norm, 0.1, 0.9)
    }

    fn effective_prior(&self, tf: usize, doc_length: usize, avg_doc_length: f64) -> f64 {
        if self.prior_weight == 0.0 {
            return 0.5;
        }
        let composite = self.composite_prior(tf, doc_length, avg_doc_length);
        0.5 + self.prior_weight * (composite - 0.5)
    }

    pub fn posterior(&self, score: f64, prior: f64) -> f64 {
        let mut lik = self.likelihood(score);
        lik = safe_prob(lik);
        let prior = safe_prob(prior);
        let numerator = lik * prior;
        let denominator = numerator + (1.0 - lik) * (1.0 - prior);
        numerator / denominator
    }

    fn effective_params(&self, term: &str) -> (f64, f64) {
        if let Some(stats) = &self.term_stats {
            if let Some(ts) = stats.get(term) {
                if ts.std_dev < EPSILON {
                    return (self.alpha, ts.median);
                }
                return (self.alpha / ts.std_dev, ts.median);
            }
        }
        (self.alpha, self.beta)
    }

    fn likelihood_for_term(&self, term: &str, score: f64) -> f64 {
        let (alpha_eff, beta_eff) = self.effective_params(term);
        sigmoid(alpha_eff * (score - beta_eff))
    }

    fn posterior_with_likelihood(&self, lik: f64, prior: f64) -> f64 {
        let lik = safe_prob(lik);
        let prior = safe_prob(prior);
        let numerator = lik * prior;
        let denominator = numerator + (1.0 - lik) * (1.0 - prior);
        numerator / denominator
    }

    pub fn score_term(&self, term: &str, doc: &Document) -> f64 {
        let raw_score = self.bm25.score_term_standard(term, doc);
        if raw_score == 0.0 {
            return 0.0;
        }
        let tf = *doc.term_freq.get(term).unwrap_or(&0);
        let prior = self.effective_prior(tf, doc.length, self.bm25.avgdl());
        let lik = self.likelihood_for_term(term, raw_score);
        self.posterior_with_likelihood(lik, prior)
    }

    pub fn score(&self, query_terms: &[String], doc: &Document) -> f64 {
        // When prior_weight is 0 (flat prior), apply softsign to the TOTAL
        // BM25 score.  This is a monotonic transform of sum(term_scores),
        // so it preserves BM25 ranking exactly while mapping to (0, 1).
        //
        // We use softsign instead of sigmoid because BM25 totals can range
        // from ~-30 to ~100+. Sigmoid saturates to 1.0 in f64 when the
        // input exceeds ~36, losing ranking information. Softsign never
        // saturates: any two distinct inputs produce distinct outputs.
        if self.prior_weight == 0.0 {
            let total = self.bm25.score(query_terms, doc);
            return softsign_calibrate(total);
        }

        // Full Bayesian path: per-term posterior with document prior,
        // combined via probabilistic OR (log-odds conjunction).
        let mut log_complement_sum = 0.0;
        let mut has_match = false;

        for term in query_terms {
            let p = self.score_term(term, doc);
            if p > 0.0 {
                has_match = true;
                let p = safe_prob(p);
                log_complement_sum += safe_log(1.0 - p);
            }
        }

        if !has_match {
            return 0.0;
        }

        1.0 - log_complement_sum.exp()
    }

    pub fn score_query(&self, query_terms: &[String], docs: &[Document]) -> Vec<f64> {
        let avgdl = self.bm25.avgdl();
        let pw = self.prior_weight;

        // When prior_weight is 0, use query-level dynamic calibration on
        // the TOTAL BM25 score (preserves BM25 ranking).
        if pw == 0.0 {
            let totals: Vec<f64> = docs
                .iter()
                .map(|doc| self.bm25.score(query_terms, doc))
                .collect();

            let positive: Vec<f64> = totals.iter().copied().filter(|&s| s > 0.0).collect();
            let (alpha_eff, beta_eff) = if positive.len() < 2 {
                (self.alpha, self.beta)
            } else {
                let med = median(&positive);
                let sd = std_dev(&positive);
                if sd < EPSILON {
                    (self.alpha, med)
                } else {
                    (self.alpha / sd, med)
                }
            };

            return totals
                .iter()
                .map(|&total| sigmoid(alpha_eff * (total - beta_eff)))
                .collect();
        }

        // Full Bayesian path: per-term dynamic calibration with priors.
        let mut term_params: Vec<(String, f64, f64)> = Vec::new();
        for term in query_terms {
            let scores: Vec<f64> = docs
                .iter()
                .map(|doc| self.bm25.score_term_standard(term, doc))
                .filter(|&s| s > 0.0)
                .collect();

            if scores.len() < 2 {
                term_params.push((term.clone(), self.alpha, self.beta));
            } else {
                let med = median(&scores);
                let sd = std_dev(&scores);
                let alpha_eff = if sd < EPSILON {
                    self.alpha
                } else {
                    self.alpha / sd
                };
                term_params.push((term.clone(), alpha_eff, med));
            }
        }

        docs.iter()
            .map(|doc| {
                let mut log_complement_sum = 0.0;
                let mut has_match = false;

                for (term, alpha_eff, beta_eff) in &term_params {
                    let raw_score = self.bm25.score_term_standard(term, doc);
                    if raw_score == 0.0 {
                        continue;
                    }
                    let tf = *doc.term_freq.get(term.as_str()).unwrap_or(&0);
                    let composite = self.composite_prior(tf, doc.length, avgdl);
                    let prior = 0.5 + pw * (composite - 0.5);
                    let lik = safe_prob(sigmoid(alpha_eff * (raw_score - beta_eff)));
                    let prior = safe_prob(prior);
                    let posterior = lik * prior / (lik * prior + (1.0 - lik) * (1.0 - prior));

                    has_match = true;
                    let p = safe_prob(posterior);
                    log_complement_sum += safe_log(1.0 - p);
                }

                if !has_match {
                    0.0
                } else {
                    1.0 - log_complement_sum.exp()
                }
            })
            .collect()
    }
}
