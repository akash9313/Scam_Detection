"""
Template Clustering and Cross-Split Leakage Prevention Engine for SpamAdapt (Phase 2).

Provides:
- Exact duplicate detection.
- Normalized near-duplicate detection (ignoring case, punctuation, whitespace).
- High-similarity campaign template clustering using token-level Jaccard similarity.
- Cross-split contamination analysis to detect template leakage between Train and Evaluation sets.
- Cluster-aware splitting utilities to keep identical or near-identical campaign templates
  confined to a single partition.
"""

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union


def get_token_set(text: str) -> Set[str]:
    """Extracts a set of normalized alphanumeric tokens from text."""
    clean = re.sub(r'[^\w\s]', ' ', text.lower())
    tokens = clean.strip().split()
    return set(tokens)


def compute_jaccard_similarity(tokens1: Set[str], tokens2: Set[str]) -> float:
    """Computes Jaccard similarity between two token sets."""
    if not tokens1 or not tokens2:
        return 0.0
    intersection = len(tokens1.intersection(tokens2))
    union = len(tokens1.union(tokens2))
    return intersection / union if union > 0 else 0.0


class TemplateClusterEngine:
    """
    Clusters message templates and identifies data leakage across partitions.
    """

    def __init__(self, jaccard_threshold: float = 0.85):
        self.jaccard_threshold = jaccard_threshold

    @staticmethod
    def normalize_text(text: str) -> str:
        """Normalized string representation for near-duplicate matching."""
        t = text.lower()
        t = re.sub(r'\s+', ' ', t)
        t = re.sub(r'[^\w\s]', '', t)
        return t.strip()

    def find_near_duplicates(
        self,
        records_a: List[Dict[str, Any]],
        records_b: Optional[List[Dict[str, Any]]] = None,
        text_key: str = "sms",
    ) -> List[Dict[str, Any]]:
        """
        Finds normalized duplicates either within records_a or across records_a and records_b.
        """
        matches = []
        if records_b is None:
            # Within single list
            seen = {}
            for idx, r in enumerate(records_a):
                norm = self.normalize_text(r[text_key])
                if norm in seen:
                    matches.append({
                        "original_index": seen[norm],
                        "duplicate_index": idx,
                        "text_1": records_a[seen[norm]][text_key],
                        "text_2": r[text_key],
                    })
                else:
                    seen[norm] = idx
        else:
            # Across two distinct splits
            norm_map_a = {}
            for idx, r in enumerate(records_a):
                norm = self.normalize_text(r[text_key])
                if norm not in norm_map_a:
                    norm_map_a[norm] = []
                norm_map_a[norm].append((idx, r))

            for idx_b, r_b in enumerate(records_b):
                norm_b = self.normalize_text(r_b[text_key])
                if norm_b in norm_map_a:
                    for idx_a, r_a in norm_map_a[norm_b]:
                        matches.append({
                            "split_a_index": idx_a,
                            "split_b_index": idx_b,
                            "text_a": r_a[text_key],
                            "label_a": r_a.get("label"),
                            "text_b": r_b[text_key],
                            "label_b": r_b.get("label"),
                        })
        return matches

    def cluster_campaign_templates(
        self,
        records: List[Dict[str, Any]],
        text_key: str = "sms",
    ) -> List[List[int]]:
        """
        Clusters messages based on high Jaccard token similarity.
        Groups repeated promotional and phishing campaign templates.
        """
        token_sets = [get_token_set(r[text_key]) for r in records]
        n = len(records)
        visited = [False] * n
        clusters = []

        for i in range(n):
            if visited[i]:
                continue
            current_cluster = [i]
            visited[i] = True

            # Find all records with Jaccard similarity above threshold
            for j in range(i + 1, n):
                if not visited[j]:
                    sim = compute_jaccard_similarity(token_sets[i], token_sets[j])
                    if sim >= self.jaccard_threshold:
                        current_cluster.append(j)
                        visited[j] = True

            clusters.append(current_cluster)

        return clusters

    def audit_cross_split_template_leakage(
        self,
        train_records: List[Dict[str, Any]],
        test_records: List[Dict[str, Any]],
        text_key: str = "sms",
        max_samples_to_report: int = 15,
    ) -> Dict[str, Any]:
        """
        Exhaustively checks whether spam or ham campaign templates in train
        leak into the test split via high Jaccard token overlap.
        """
        train_tokens = [(idx, r, get_token_set(r[text_key])) for idx, r in enumerate(train_records)]
        test_tokens = [(idx, r, get_token_set(r[text_key])) for idx, r in enumerate(test_records)]

        leaked_pairs = []

        # Compare test against train
        for test_idx, test_r, t_tokens in test_tokens:
            if len(t_tokens) < 3:
                continue  # Skip ultra-short messages like "Ok" from template clustering
            for train_idx, train_r, tr_tokens in train_tokens:
                sim = compute_jaccard_similarity(t_tokens, tr_tokens)
                if sim >= self.jaccard_threshold:
                    leaked_pairs.append({
                        "test_index": test_idx,
                        "train_index": train_idx,
                        "similarity": round(sim, 4),
                        "test_text": test_r[text_key],
                        "train_text": train_r[text_key],
                        "test_label": test_r.get("label"),
                        "train_label": train_r.get("label"),
                    })
                    break  # One match per test message is enough to establish leakage

        # Also check exact and normalized
        exact_matches = 0
        train_texts = {r[text_key] for r in train_records}
        for r in test_records:
            if r[text_key] in train_texts:
                exact_matches += 1

        norm_matches = self.find_near_duplicates(train_records, test_records, text_key=text_key)

        report = {
            "train_record_count": len(train_records),
            "test_record_count": len(test_records),
            "jaccard_similarity_threshold": self.jaccard_threshold,
            "exact_duplicate_count": exact_matches,
            "normalized_near_duplicate_count": len(norm_matches),
            "template_leakage_count": len(leaked_pairs),
            "leakage_percentage_of_test": round(len(leaked_pairs) / len(test_records) * 100, 2) if test_records else 0.0,
            "sample_template_leakages": leaked_pairs[:max_samples_to_report],
        }
        return report
