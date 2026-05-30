from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


DEFAULT_TAXONOMY_REFERENCE = (
    Path(__file__).resolve().parent
    / "references"
    / "FINAL_UNIFIED_NK_TAXONOMY_REFERENCE_nolayer.md"
)

TIER_WEIGHTS = {
    "core": 4,
    "support": 2,
    "context": 1,
}


@dataclass
class TaxonomyEntry:
    name: str
    layer: str = "unknown"
    canonical_label: str = ""
    core: list[str] = field(default_factory=list)
    support: list[str] = field(default_factory=list)
    context: list[str] = field(default_factory=list)
    negative: list[str] = field(default_factory=list)

    @property
    def markers(self) -> dict[str, list[str]]:
        return {
            "core": self.core,
            "support": self.support,
            "context": self.context,
            "negative_expected_low": self.negative,
        }


def load_taxonomy_entries(path: str | Path | None = None) -> list[TaxonomyEntry]:
    reference_path = Path(path) if path else DEFAULT_TAXONOMY_REFERENCE
    if not reference_path.exists():
        return []

    entries: list[TaxonomyEntry] = []
    current: TaxonomyEntry | None = None
    current_tier: str | None = None
    current_layer = "unknown"

    for raw_line in reference_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("## "):
            current_layer = classify_layer_header(line)
            current = None
            current_tier = None
            continue
        if line.startswith("### "):
            name = line.removeprefix("### ").strip()
            current = TaxonomyEntry(
                name=name,
                layer=current_layer,
                canonical_label=canonical_taxonomy_label(name),
            )
            entries.append(current)
            current_tier = None
            continue
        if current is None:
            continue
        if line.startswith("**"):
            current_tier = classify_tier_header(line)
            continue
        if "`" not in line or current_tier is None:
            continue
        genes = extract_genes_from_backticks(line)
        if not genes:
            continue
        target = getattr(current, current_tier)
        target.extend(gene for gene in genes if gene not in target)

    return [entry for entry in entries if any(entry.markers.values())]


def load_taxonomy_entries_by_layer(layer: str, path: str | Path | None = None) -> list[TaxonomyEntry]:
    requested = str(layer).strip().lower()
    return [entry for entry in load_taxonomy_entries(path) if entry.layer == requested]


def allowed_nk_subtype_labels(path: str | Path | None = None) -> list[str]:
    labels = ordered_unique(entry.canonical_label for entry in load_taxonomy_entries_by_layer("subtype", path))
    return append_unsure_non_nk(labels or [
        "NK1", "NK2", "adaptive_NK_CMV", "adaptive_NK_nonCMV",
        "trNK", "cNK", "L6_Developmental_immature",
    ])


def allowed_nk_state_labels(path: str | Path | None = None) -> list[str]:
    labels = ordered_unique(entry.canonical_label for entry in load_taxonomy_entries_by_layer("state", path))
    return append_unsure_non_nk(labels or [
        "Chemokine_inflammatory", "Checkpoint_exhausted", "ER_stress_UPR",
        "Metabolic_stress_hypoxia", "Proliferating", "IFN_stimulated",
        "Cytotoxic_activated", "Homeostatic_quiescent",
        "CIML_cytokine_preactivated", "CIMP_cytokine_primed_memory_like",
    ])


def classify_layer_header(line: str) -> str:
    lower = line.lower()
    if "state" in lower:
        return "state"
    if "subtype" in lower or "lineage" in lower or "differentiation" in lower:
        return "subtype"
    return "unknown"


def canonical_taxonomy_label(name: str) -> str:
    label = str(name).strip()
    label = re.sub(r"\s*-->.*$", "", label).strip()
    label = re.sub(r"\s*\([^)]*\)\s*$", "", label).strip()
    return label


def ordered_unique(values) -> list[str]:
    seen = set()
    out = []
    for value in values:
        text = str(value).strip()
        if text and text not in seen:
            seen.add(text)
            out.append(text)
    return out


def append_unsure_non_nk(labels: list[str]) -> list[str]:
    out = ordered_unique(labels)
    for label in ["Unsure", "Non-NK"]:
        if label not in out:
            out.append(label)
    return out


def classify_tier_header(line: str) -> str | None:
    lower = line.lower()
    if "core" in lower:
        return "core"
    if "support" in lower:
        return "support"
    if "context" in lower:
        return "context"
    if "negative" in lower or "expected low" in lower or "expected low/absent" in lower:
        return "negative"
    return None


def extract_genes_from_backticks(line: str) -> list[str]:
    genes: list[str] = []
    for content in re.findall(r"`([^`]*)`", line):
        normalized = content.replace("·", " ").replace(",", " ").replace(";", " ")
        for token in normalized.split():
            gene = clean_gene_token(token)
            if gene and gene not in genes:
                genes.append(gene)
    return genes


def clean_gene_token(token: str) -> str:
    gene = re.sub(r"\(D\d+\)", "", token.strip())
    gene = re.sub(r"[^A-Za-z0-9_.-]", "", gene)
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_.-]*", gene):
        return ""
    return gene.upper()


def taxonomy_marker_hits(
    positive_genes: list[str],
    negative_genes: list[str],
    *,
    entries: list[TaxonomyEntry] | None = None,
    max_matches: int = 8,
) -> dict[str, Any]:
    entries = entries if entries is not None else load_taxonomy_entries()
    if not entries:
        return {
            "reference": str(DEFAULT_TAXONOMY_REFERENCE),
            "top_matches": [],
        }

    positive = {str(gene).upper() for gene in positive_genes}
    negative = {str(gene).upper() for gene in negative_genes}
    matches = []
    for entry in entries:
        core_hits = intersect_ordered(entry.core, positive)
        support_hits = intersect_ordered(entry.support, positive)
        context_hits = intersect_ordered(entry.context, positive)
        negative_expected_low_hits = intersect_ordered(entry.negative, negative)
        negative_contradictions = intersect_ordered(entry.negative, positive)

        weighted_hit_score = (
            TIER_WEIGHTS["core"] * len(core_hits)
            + TIER_WEIGHTS["support"] * len(support_hits)
            + TIER_WEIGHTS["context"] * len(context_hits)
        )
        max_score = max_taxonomy_score(entry)
        percent_of_max = round(100.0 * weighted_hit_score / max_score, 1) if max_score else 0.0
        support_level = taxonomy_support_level(
            core_hits=core_hits,
            support_hits=support_hits,
            context_hits=context_hits,
            negative_expected_low_hits=negative_expected_low_hits,
            negative_contradictions=negative_contradictions,
        )
        if support_level == "none":
            continue
        matches.append(
            {
                "taxonomy_state": entry.name,
                "taxonomy_label": entry.canonical_label,
                "taxonomy_layer": entry.layer,
                "support_level": support_level,
                "percent_of_max_score": percent_of_max,
                "weighted_hit_score": weighted_hit_score,
                "max_possible_score": max_score,
                "core_hits": core_hits,
                "support_hits": support_hits,
                "context_hits": context_hits,
                "negative_expected_low_hits": negative_expected_low_hits,
                "negative_contradictions": negative_contradictions,
            }
        )

    matches = sorted(
        matches,
        key=lambda item: (
            support_level_rank(item["support_level"]),
            item["weighted_hit_score"],
            len(item["core_hits"]),
            item["percent_of_max_score"],
        ),
        reverse=True,
    )
    return {
        "reference": str(DEFAULT_TAXONOMY_REFERENCE),
        "scoring": "CORE=4, SUPPORT=2, CONTEXT=1; percent_of_max_score uses positive taxonomy tiers only",
        "support_level_rule": (
            "strong: >=3 CORE, or >=2 CORE + >=3 SUPPORT, or >=1 CORE + >=5 SUPPORT; "
            "moderate: >=2 CORE, or >=1 CORE + >=2 SUPPORT, or >=4 SUPPORT; "
            "weak: any smaller positive/expected-low hit pattern; contradictory: negative-defining genes are positive"
        ),
        "top_matches": matches[:max_matches],
    }


def max_taxonomy_score(entry: TaxonomyEntry) -> int:
    return (
            TIER_WEIGHTS["core"] * len(entry.core)
            + TIER_WEIGHTS["support"] * len(entry.support)
            + TIER_WEIGHTS["context"] * len(entry.context)
    )


def taxonomy_support_level(
    *,
    core_hits: list[str],
    support_hits: list[str],
    context_hits: list[str],
    negative_expected_low_hits: list[str],
    negative_contradictions: list[str],
) -> str:
    n_core = len(core_hits)
    n_support = len(support_hits)
    n_context = len(context_hits)
    n_expected_low = len(negative_expected_low_hits)
    n_contradictions = len(negative_contradictions)
    n_positive_hits = n_core + n_support + n_context
    if n_contradictions >= 2 or (n_contradictions >= 1 and n_positive_hits < 2):
        return "contradictory"
    if n_core >= 3 or (n_core >= 2 and n_support >= 3) or (n_core >= 1 and n_support >= 5):
        return "strong"
    if n_core >= 2 or (n_core >= 1 and n_support >= 2) or n_support >= 4:
        return "moderate"
    if n_positive_hits > 0 or n_expected_low >= 2:
        return "weak"
    return "none"


def support_level_rank(level: str) -> int:
    return {
        "strong": 4,
        "moderate": 3,
        "weak": 2,
        "contradictory": 1,
        "none": 0,
    }.get(str(level), 0)


def intersect_ordered(markers: list[str], gene_set: set[str]) -> list[str]:
    return [gene for gene in markers if gene.upper() in gene_set]
