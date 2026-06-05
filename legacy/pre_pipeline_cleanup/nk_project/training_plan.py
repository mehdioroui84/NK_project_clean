from __future__ import annotations

import torch.nn.functional as F
from scvi.train import SemiSupervisedTrainingPlan


class WeightedSemiSupervisedTrainingPlan(SemiSupervisedTrainingPlan):
    """SCANVI training plan with weighted classification cross-entropy."""

    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def on_fit_start(self):
        super().on_fit_start()
        if self.class_weights is not None:
            self.class_weights = self.class_weights.to(self.device)

    def loss(self, tensors, inference_outputs, generative_outputs, loss_kwargs=None):
        loss_output = super().loss(
            tensors, inference_outputs, generative_outputs, loss_kwargs=loss_kwargs
        )
        logits = inference_outputs["classification_logits"]
        y = tensors["labels"].long().view(-1)
        ce = F.cross_entropy(logits, y, weight=self.class_weights)
        loss_output.classification_loss = ce
        loss_output.loss = (
            loss_output.reconstruction_loss
            + loss_output.kl_local
            + loss_output.kl_global
            + self.classification_ratio * ce
        )
        return loss_output
