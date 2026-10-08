"""
Transformer encoder with a classification head on the [CLS] representation.
"""

import torch
import torch.nn as nn
from transformers import AutoModel


class ClaimCheckWorthinessClassifier(nn.Module):
    """
    Binary claim classifier (used for both VCD and CCD).
    Optional source-feature fusion is disabled in all reported experiments.
    """

    def __init__(
            self,
            model_name: str,
            use_source_features: bool = False,
            source_feature_dim: int = 6,
            hidden_dropout: float = 0.2,
            classifier_hidden_dim: int = 256,
            feature_hidden_dim: int = 64
    ):
        super().__init__()

        self.model_name = model_name
        self.use_source_features = use_source_features

        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size

        if use_source_features:
            self.feature_fusion = nn.Sequential(
                nn.Linear(source_feature_dim, feature_hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(feature_hidden_dim),
                nn.Dropout(0.1)
            )

            self.classifier = nn.Sequential(
                nn.Linear(hidden_size + feature_hidden_dim, classifier_hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(classifier_hidden_dim),
                nn.Dropout(hidden_dropout),
                nn.Linear(classifier_hidden_dim, classifier_hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(hidden_dropout),
                nn.Linear(classifier_hidden_dim // 2, 2)
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(hidden_size, classifier_hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(classifier_hidden_dim),
                nn.Dropout(hidden_dropout),
                nn.Linear(classifier_hidden_dim, classifier_hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(hidden_dropout),
                nn.Linear(classifier_hidden_dim // 2, 2)
            )

    def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            source_features: torch.Tensor = None
    ):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        cls_output = outputs.last_hidden_state[:, 0, :]

        if self.use_source_features and source_features is not None:
            source_emb = self.feature_fusion(source_features)
            combined = torch.cat([cls_output, source_emb], dim=1)
            logits = self.classifier(combined)
        else:
            logits = self.classifier(cls_output)

        return logits

    def get_embeddings(self, input_ids, attention_mask, source_features=None):
        """Return the [CLS] representation (concatenated with source features if enabled)."""
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        cls_output = outputs.last_hidden_state[:, 0, :]

        if self.use_source_features and source_features is not None:
            source_emb = self.feature_fusion(source_features)
            combined = torch.cat([cls_output, source_emb], dim=1)
            return combined
        return cls_output
