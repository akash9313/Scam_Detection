"""
Modular Data Ingestion and Dataset Integration Engine for SpamAdapt (Phase 2).

Provides:
- Configurable ingestion for arbitrary labeled CSV datasets with custom text and label schemas.
- Explicit label mapping and validation (strictly binary: 0 = HAM, 1 = SPAM).
- Provenance and metadata tracking (source, category, language, is_synthetic).
- Minimal, signal-preserving normalization (preserving URLs, numbers, punctuation, currency, capitalization).
- Exact and normalized duplicate detection.
- Export of reproducible dataset manifests and source-level audits.
"""

import csv
import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from .preprocessing import clean_text_minimal


@dataclass
class DatasetSourceConfig:
    """Configuration schema for a specific external or internal dataset source."""
    source_name: str
    filepath: Union[str, Path]
    text_column: str = "sms"
    label_column: str = "label"
    label_mapping: Dict[Any, int] = field(default_factory=lambda: {
        "ham": 0, "spam": 1,
        "0": 0, "1": 1,
        0: 0, 1: 1,
        "legitimate": 0, "fraud": 1,
        "normal": 0, "phishing": 1,
        "smishing": 1
    })
    default_category: str = "general"
    category_column: Optional[str] = None
    default_language: str = "en"
    language_column: Optional[str] = None
    is_synthetic: bool = False


@dataclass
class StandardizedMessageRecord:
    """Standardized record format for all ingested messages in SpamAdapt."""
    sms: str
    label: int  # 0 = HAM, 1 = SPAM
    source: str
    category: str
    language: str = "en"
    is_synthetic: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sms": self.sms,
            "label": self.label,
            "source": self.source,
            "category": self.category,
            "language": self.language,
            "is_synthetic": self.is_synthetic,
        }


class ModularDataIngestionEngine:
    """
    Ingests, cleans, deduplicates, and validates diverse SMS messages from
    arbitrary tabular CSV sources without destroying critical spam signals.
    """

    def __init__(self):
        self._exact_hashes: Set[str] = set()
        self._normalized_hashes: Set[str] = set()
        self.ingested_records: List[StandardizedMessageRecord] = []
        self.ingestion_audit_log: List[Dict[str, Any]] = []

    @staticmethod
    def normalize_for_duplicate_check(text: str) -> str:
        """
        Creates a normalized key purely for duplicate/near-duplicate detection.
        Does not mutate the actual message text used for model training/inference.
        """
        t = text.lower()
        t = re.sub(r'\s+', ' ', t)
        t = re.sub(r'[^\w\s]', '', t)
        return t.strip()

    def ingest_csv_source(
        self,
        config: DatasetSourceConfig,
        deduplicate_against_existing: bool = True,
    ) -> Dict[str, Any]:
        """
        Ingests a CSV dataset, validates schema, applies minimal cleaning,
        and standardizes into the unified SpamAdapt schema.
        """
        path = Path(config.filepath)
        if not path.exists():
            raise FileNotFoundError(f"Source file not found at: {path}")

        initial_rows = 0
        dropped_missing = 0
        dropped_invalid_label = 0
        dropped_exact_duplicates = 0
        dropped_normalized_duplicates = 0
        accepted_records = 0

        source_records: List[StandardizedMessageRecord] = []

        with open(path, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                raise ValueError(f"CSV file is empty or corrupted: {path}")

            if config.text_column not in reader.fieldnames:
                raise KeyError(
                    f"Configured text_column '{config.text_column}' not found in CSV. "
                    f"Available columns: {reader.fieldnames}"
                )
            if config.label_column not in reader.fieldnames:
                raise KeyError(
                    f"Configured label_column '{config.label_column}' not found in CSV. "
                    f"Available columns: {reader.fieldnames}"
                )

            for row in reader:
                initial_rows += 1
                raw_text = row.get(config.text_column)
                raw_label = row.get(config.label_column)

                # Check for missing text or label
                if raw_text is None or raw_label is None:
                    dropped_missing += 1
                    continue

                raw_text_str = str(raw_text).strip()
                if not raw_text_str:
                    dropped_missing += 1
                    continue

                # Map label
                mapped_label = None
                # Check direct match or string lower match
                if raw_label in config.label_mapping:
                    mapped_label = config.label_mapping[raw_label]
                elif str(raw_label).lower().strip() in config.label_mapping:
                    mapped_label = config.label_mapping[str(raw_label).lower().strip()]

                if mapped_label not in (0, 1):
                    dropped_invalid_label += 1
                    continue

                # Apply minimal, spam-signal-preserving cleaning
                clean_text = clean_text_minimal(raw_text_str)
                if not clean_text:
                    dropped_missing += 1
                    continue

                # Duplicate checks
                exact_hash = hashlib.sha256(clean_text.encode("utf-8")).hexdigest()
                norm_key = self.normalize_for_duplicate_check(clean_text)
                norm_hash = hashlib.sha256(norm_key.encode("utf-8")).hexdigest()

                if deduplicate_against_existing:
                    if exact_hash in self._exact_hashes:
                        dropped_exact_duplicates += 1
                        continue
                    if norm_hash in self._normalized_hashes:
                        dropped_normalized_duplicates += 1
                        continue

                # Category and Language handling
                category = config.default_category
                if config.category_column and config.category_column in row:
                    c_val = str(row[config.category_column]).strip()
                    if c_val:
                        category = c_val

                language = config.default_language
                if config.language_column and config.language_column in row:
                    l_val = str(row[config.language_column]).strip()
                    if l_val:
                        language = l_val

                record = StandardizedMessageRecord(
                    sms=clean_text,
                    label=int(mapped_label),
                    source=config.source_name,
                    category=category,
                    language=language,
                    is_synthetic=config.is_synthetic,
                )

                self._exact_hashes.add(exact_hash)
                self._normalized_hashes.add(norm_hash)
                source_records.append(record)
                accepted_records += 1

        self.ingested_records.extend(source_records)

        # Compute summary statistics for this source
        ham_count = sum(1 for r in source_records if r.label == 0)
        spam_count = sum(1 for r in source_records if r.label == 1)

        summary = {
            "source_name": config.source_name,
            "filepath": str(path),
            "initial_rows": initial_rows,
            "accepted_records": accepted_records,
            "ham_count": ham_count,
            "spam_count": spam_count,
            "spam_percentage": round(spam_count / accepted_records * 100, 2) if accepted_records > 0 else 0.0,
            "dropped_missing": dropped_missing,
            "dropped_invalid_label": dropped_invalid_label,
            "dropped_exact_duplicates": dropped_exact_duplicates,
            "dropped_normalized_duplicates": dropped_normalized_duplicates,
            "is_synthetic": config.is_synthetic,
        }

        self.ingestion_audit_log.append(summary)
        return summary

    def export_standardized_csv(self, output_path: Union[str, Path]) -> Path:
        """Exports all ingested records to a standardized CSV format."""
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = ["sms", "label", "source", "category", "language", "is_synthetic"]
        with open(out_p, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in self.ingested_records:
                writer.writerow(r.to_dict())

        return out_p

    def generate_manifest(self) -> Dict[str, Any]:
        """Generates a complete dataset manifest with source breakdown and category distribution."""
        total = len(self.ingested_records)
        ham_count = sum(1 for r in self.ingested_records if r.label == 0)
        spam_count = sum(1 for r in self.ingested_records if r.label == 1)

        categories = {}
        for r in self.ingested_records:
            categories[r.category] = categories.get(r.category, 0) + 1

        sources = {}
        for r in self.ingested_records:
            sources[r.source] = sources.get(r.source, 0) + 1

        languages = {}
        for r in self.ingested_records:
            languages[r.language] = languages.get(r.language, 0) + 1

        manifest = {
            "total_messages": total,
            "class_distribution": {
                "ham_count": ham_count,
                "spam_count": spam_count,
                "ham_percentage": round(ham_count / total * 100, 2) if total > 0 else 0.0,
                "spam_percentage": round(spam_count / total * 100, 2) if total > 0 else 0.0,
                "imbalance_ratio": round(ham_count / spam_count, 2) if spam_count > 0 else 0.0,
            },
            "category_distribution": categories,
            "source_distribution": sources,
            "language_distribution": languages,
            "sources_audit": self.ingestion_audit_log,
        }
        return manifest
