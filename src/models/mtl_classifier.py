import torch
import torch.nn as nn
from transformers import AutoModel


class MTLClaimClassifier(nn.Module):
    """
    Multi-task classifier with shared encoder and two task heads:
      - VCD head: verifiable claim detection (claim text only)
      - CCD head: check-worthiness classification (claim text, optionally + source features)

    use_source_features controls whether the CCD head fuses external source features.
    Default is False: both heads use [CLS] only, enabling clean RSA comparison.
    """

    def __init__(
            self,
            model_name: str,
            use_source_features: bool = False,
            source_feature_dim: int = 6,
            hidden_dropout: float = 0.2,
            classifier_hidden_dim: int = 256,
            feature_hidden_dim: int = 64,
    ):
        super().__init__()

        self.use_source_features = use_source_features
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size

        # VCD head: claim text only
        self.vcd_head = nn.Sequential(
            nn.Linear(hidden_size, classifier_hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(classifier_hidden_dim),
            nn.Dropout(hidden_dropout),
            nn.Linear(classifier_hidden_dim, classifier_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(hidden_dropout),
            nn.Linear(classifier_hidden_dim // 2, 2),
        )

        if use_source_features:
            self.source_feature_fusion = nn.Sequential(
                nn.Linear(source_feature_dim, feature_hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(feature_hidden_dim),
                nn.Dropout(0.1),
            )
            ccd_input_dim = hidden_size + feature_hidden_dim
        else:
            ccd_input_dim = hidden_size

        # CCD head: [CLS] only, or [CLS] + projected source features
        self.ccd_head = nn.Sequential(
            nn.Linear(ccd_input_dim, classifier_hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(classifier_hidden_dim),
            nn.Dropout(hidden_dropout),
            nn.Linear(classifier_hidden_dim, classifier_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(hidden_dropout),
            nn.Linear(classifier_hidden_dim // 2, 2),
        )

    def forward(self, input_ids, attention_mask, source_features=None):
        cls_output = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).last_hidden_state[:, 0, :]

        vcd_logits = self.vcd_head(cls_output)

        if self.use_source_features and source_features is not None:
            source_emb = self.source_feature_fusion(source_features)
            ccd_input = torch.cat([cls_output, source_emb], dim=1)
        else:
            ccd_input = cls_output
        ccd_logits = self.ccd_head(ccd_input)

        return vcd_logits, ccd_logits

    def get_cls_embeddings(self, input_ids, attention_mask):
        """Extract [CLS] representations from the shared encoder for analysis."""
        return self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).last_hidden_state[:, 0, :]
