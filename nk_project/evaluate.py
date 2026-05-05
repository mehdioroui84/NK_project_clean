from __future__ import annotations

import numpy as np
import pandas as pd
import scarches as sca

from .metrics import filtered_classification_metrics


def evaluate_scanvi_split(
    adata_eval,
    *,
    split_name: str,
    model,
    label_key: str,
    training_classes: set[str],
    unlabeled: str = "Unknown",
    min_class_eval: int = 30,
    extend_categories: bool = False,
):
    adata_scvi = adata_eval.copy()
    sca.models.SCANVI.prepare_query_anndata(adata_scvi, model)
    new_manager = model.adata_manager.transfer_fields(
        adata_scvi, extend_categories=extend_categories
    )
    model._register_manager_for_instance(new_manager)

    proba = model.predict(adata_scvi, soft=True)
    pred = proba.idxmax(axis=1).values
    true = adata_scvi.obs[label_key].astype(str).values

    mask_no_unlab = true != unlabeled
    mask_in_training = np.isin(true, list(training_classes))
    final_mask = mask_no_unlab & mask_in_training

    dropped = sorted(set(true[mask_no_unlab & ~mask_in_training]) - {unlabeled})
    if dropped:
        print(f"[{split_name}] Removed from outputs (not in training): {dropped}")

    true_eval = true[final_mask]
    pred_eval = pred[final_mask]
    proba_eval = proba.loc[final_mask]
    metrics = filtered_classification_metrics(
        true_eval,
        pred_eval,
        split_name=split_name,
        training_classes=training_classes,
        unlabeled=unlabeled,
        min_class_eval=min_class_eval,
    )

    return {
        "adata": adata_scvi,
        "true": true_eval,
        "pred": pred_eval,
        "proba": proba_eval,
        "mask": final_mask,
        "metrics": metrics,
    }


def probability_summary(proba: pd.DataFrame) -> pd.DataFrame:
    p = proba.values + 1e-12
    confidence = p.max(axis=1)
    entropy = -(p * np.log(p)).sum(axis=1)
    certainty = 1.0 - entropy / np.log(p.shape[1])
    return pd.DataFrame(
        {
            "pred_label": proba.idxmax(axis=1).values,
            "confidence": confidence,
            "certainty": certainty,
        },
        index=proba.index,
    )
