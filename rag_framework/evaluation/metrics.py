"""Evaluation metrics for RAG system."""

from typing import List, Dict, Set
import re
from collections import Counter
import numpy as np


class RAGEvaluator:
    """Evaluate RAG system performance with multiple metrics."""

    def __init__(self):
        """Initialize evaluator."""
        pass

    def exact_match(self, predicted: str, ground_truth: str) -> float:
        """Calculate exact match score.
        
        Args:
            predicted: Predicted answer
            ground_truth: Ground truth answer
            
        Returns:
            1.0 if exact match, 0.0 otherwise
        """
        return 1.0 if predicted.strip().lower() == ground_truth.strip().lower() else 0.0

    def f1_score(self, predicted: str, ground_truth: str) -> float:
        """Calculate token-level F1 score.
        
        Args:
            predicted: Predicted answer
            ground_truth: Ground truth answer
            
        Returns:
            F1 score (0.0 to 1.0)
        """
        pred_tokens = self._tokenize(predicted)
        truth_tokens = self._tokenize(ground_truth)
        
        if not pred_tokens or not truth_tokens:
            return 0.0
        
        common = Counter(pred_tokens) & Counter(truth_tokens)
        num_common = sum(common.values())
        
        if num_common == 0:
            return 0.0
        
        precision = num_common / len(pred_tokens)
        recall = num_common / len(truth_tokens)
        
        f1 = 2 * (precision * recall) / (precision + recall)
        return f1

    def bleu_score(self, predicted: str, ground_truth: str, n: int = 4) -> float:
        """Calculate BLEU score.
        
        Args:
            predicted: Predicted answer
            ground_truth: Ground truth answer
            n: Maximum n-gram order
            
        Returns:
            BLEU score (0.0 to 1.0)
        """
        pred_tokens = self._tokenize(predicted)
        truth_tokens = self._tokenize(ground_truth)
        
        if not pred_tokens or not truth_tokens:
            return 0.0
        
        # Calculate brevity penalty
        bp = 1.0 if len(pred_tokens) >= len(truth_tokens) else \
             np.exp(1 - len(truth_tokens) / len(pred_tokens))
        
        # Calculate n-gram precisions
        precisions = []
        for i in range(1, min(n + 1, len(pred_tokens) + 1)):
            pred_ngrams = self._get_ngrams(pred_tokens, i)
            truth_ngrams = self._get_ngrams(truth_tokens, i)
            
            if not pred_ngrams:
                continue
            
            common = Counter(pred_ngrams) & Counter(truth_ngrams)
            num_common = sum(common.values())
            precision = num_common / len(pred_ngrams)
            precisions.append(precision)
        
        if not precisions:
            return 0.0
        
        # Geometric mean of precisions
        bleu = bp * np.exp(np.mean([np.log(p) if p > 0 else -np.inf for p in precisions]))
        return bleu if not np.isnan(bleu) and not np.isinf(bleu) else 0.0

    def rouge_l(self, predicted: str, ground_truth: str) -> Dict[str, float]:
        """Calculate ROUGE-L score (Longest Common Subsequence).
        
        Args:
            predicted: Predicted answer
            ground_truth: Ground truth answer
            
        Returns:
            Dictionary with precision, recall, and F1
        """
        pred_tokens = self._tokenize(predicted)
        truth_tokens = self._tokenize(ground_truth)
        
        if not pred_tokens or not truth_tokens:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        
        lcs_length = self._lcs_length(pred_tokens, truth_tokens)
        
        precision = lcs_length / len(pred_tokens) if pred_tokens else 0.0
        recall = lcs_length / len(truth_tokens) if truth_tokens else 0.0
        
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1
        }

    def retrieval_precision_at_k(
        self,
        retrieved_docs: List[str],
        relevant_docs: Set[str],
        k: int = 5
    ) -> float:
        """Calculate precision@k for retrieval.
        
        Args:
            retrieved_docs: List of retrieved document IDs
            relevant_docs: Set of relevant document IDs
            k: Number of top results to consider
            
        Returns:
            Precision@k score
        """
        top_k = retrieved_docs[:k]
        relevant_in_top_k = sum(1 for doc in top_k if doc in relevant_docs)
        return relevant_in_top_k / k if k > 0 else 0.0

    def retrieval_recall_at_k(
        self,
        retrieved_docs: List[str],
        relevant_docs: Set[str],
        k: int = 5
    ) -> float:
        """Calculate recall@k for retrieval.
        
        Args:
            retrieved_docs: List of retrieved document IDs
            relevant_docs: Set of relevant document IDs
            k: Number of top results to consider
            
        Returns:
            Recall@k score
        """
        if not relevant_docs:
            return 0.0
        
        top_k = retrieved_docs[:k]
        relevant_in_top_k = sum(1 for doc in top_k if doc in relevant_docs)
        return relevant_in_top_k / len(relevant_docs)

    def mean_reciprocal_rank(
        self,
        retrieved_docs: List[str],
        relevant_docs: Set[str]
    ) -> float:
        """Calculate Mean Reciprocal Rank (MRR).
        
        Args:
            retrieved_docs: List of retrieved document IDs
            relevant_docs: Set of relevant document IDs
            
        Returns:
            MRR score
        """
        for i, doc in enumerate(retrieved_docs, 1):
            if doc in relevant_docs:
                return 1.0 / i
        return 0.0

    def evaluate_qa(
        self,
        predictions: List[str],
        ground_truths: List[str]
    ) -> Dict[str, float]:
        """Evaluate question answering performance.
        
        Args:
            predictions: List of predicted answers
            ground_truths: List of ground truth answers
            
        Returns:
            Dictionary with average scores for each metric
        """
        if len(predictions) != len(ground_truths):
            raise ValueError("predictions and ground_truths must have same length")
        
        exact_matches = []
        f1_scores = []
        bleu_scores = []
        rouge_scores = []
        
        for pred, truth in zip(predictions, ground_truths):
            exact_matches.append(self.exact_match(pred, truth))
            f1_scores.append(self.f1_score(pred, truth))
            bleu_scores.append(self.bleu_score(pred, truth))
            rouge_scores.append(self.rouge_l(pred, truth)["f1"])
        
        return {
            "exact_match": np.mean(exact_matches),
            "f1": np.mean(f1_scores),
            "bleu": np.mean(bleu_scores),
            "rouge_l": np.mean(rouge_scores)
        }

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into words.
        
        Args:
            text: Input text
            
        Returns:
            List of tokens
        """
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        return text.split()

    def _get_ngrams(self, tokens: List[str], n: int) -> List[tuple]:
        """Get n-grams from tokens.
        
        Args:
            tokens: List of tokens
            n: N-gram size
            
        Returns:
            List of n-grams
        """
        return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]

    def _lcs_length(self, seq1: List[str], seq2: List[str]) -> int:
        """Calculate longest common subsequence length.
        
        Args:
            seq1: First sequence
            seq2: Second sequence
            
        Returns:
            LCS length
        """
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        return dp[m][n]
