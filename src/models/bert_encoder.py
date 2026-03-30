"""
BERT Encoder Model for Intent Recognition
Implements BERT-based encoder with classification head
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertConfig
from typing import Dict, Tuple, Optional


class BERTIntentEncoder(nn.Module):
    """BERT-based Intent Recognition Encoder"""

    def __init__(
        self,
        bert_model_name: str = 'bert-base-uncased',
        num_labels: int = 151,
        hidden_size: int = 768,
        dropout: float = 0.1,
        freeze_bert: bool = False
    ):
        """
        Initialize BERT encoder

        Args:
            bert_model_name: Pretrained BERT model name
            num_labels: Number of intent classes (150 ID + 1 OOS)
            hidden_size: Hidden size of BERT
            dropout: Dropout probability
            freeze_bert: Whether to freeze BERT parameters
        """
        super().__init__()

        self.num_labels = num_labels
        self.hidden_size = hidden_size

        # Load pretrained BERT
        self.bert = BertModel.from_pretrained(bert_model_name)

        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_labels)
        )

        # Initialize classifier weights
        self._init_weights()

    def _init_weights(self):
        """Initialize classifier weights"""
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=0.02)
                if module.bias is not None:
                    module.bias.data.zero_()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        return_hidden: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass

        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            return_hidden: Whether to return hidden states

        Returns:
            Dictionary containing:
                - logits: Classification logits [batch_size, num_labels]
                - hidden_states: [CLS] token hidden states [batch_size, hidden_size]
                - pooled_output: Pooled output from BERT
        """
        # BERT encoding
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )

        # Get [CLS] token hidden state
        hidden_states = outputs.last_hidden_state[:, 0, :]  # [batch_size, hidden_size]
        pooled_output = outputs.pooler_output  # [batch_size, hidden_size]

        # Classification
        logits = self.classifier(hidden_states)  # [batch_size, num_labels]

        result = {
            'logits': logits,
            'hidden_states': hidden_states,
            'pooled_output': pooled_output
        }

        return result

    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Encode input to hidden representation

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask

        Returns:
            Hidden states [batch_size, hidden_size]
        """
        with torch.no_grad():
            outputs = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True
            )
            hidden_states = outputs.last_hidden_state[:, 0, :]

        return hidden_states

    def classify(
        self,
        hidden_states: torch.Tensor
    ) -> torch.Tensor:
        """
        Classify from hidden states

        Args:
            hidden_states: Hidden states [batch_size, hidden_size]

        Returns:
            Logits [batch_size, num_labels]
        """
        return self.classifier(hidden_states)

    def get_num_parameters(self) -> Tuple[int, int]:
        """
        Get number of parameters

        Returns:
            Tuple of (total_params, trainable_params)
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total_params, trainable_params


class BERTWithPrototype(nn.Module):
    """BERT Encoder with Prototype Learning Support"""

    def __init__(
        self,
        bert_encoder: BERTIntentEncoder,
        num_intents: int = 150,
        num_sub_prototypes: int = 5,
        hidden_size: int = 768
    ):
        """
        Initialize BERT with prototype learning

        Args:
            bert_encoder: BERT encoder model
            num_intents: Number of in-domain intents
            num_sub_prototypes: Number of sub-prototypes per intent (K)
            hidden_size: Hidden size
        """
        super().__init__()

        self.bert_encoder = bert_encoder
        self.num_intents = num_intents
        self.num_sub_prototypes = num_sub_prototypes
        self.hidden_size = hidden_size

        # Prototype bank (will be updated during training)
        # Shape: [num_intents, num_sub_prototypes, hidden_size]
        self.register_buffer(
            'prototype_bank',
            torch.zeros(num_intents, num_sub_prototypes, hidden_size)
        )

        # Prototype initialized flag
        self.prototypes_initialized = False

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Forward pass"""
        return self.bert_encoder(input_ids, attention_mask)

    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Encode input"""
        return self.bert_encoder.encode(input_ids, attention_mask)

    def update_prototypes(self, prototype_bank: torch.Tensor):
        """
        Update prototype bank

        Args:
            prototype_bank: New prototype bank [num_intents, K, hidden_size]
        """
        self.prototype_bank.copy_(prototype_bank)
        self.prototypes_initialized = True

    def compute_prototype_distances(
        self,
        hidden_states: torch.Tensor,
        distance_metric: str = 'cosine'
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute distances to all prototypes

        Args:
            hidden_states: Hidden states [batch_size, hidden_size]
            distance_metric: Distance metric ('cosine' or 'euclidean')

        Returns:
            Tuple of:
                - min_distances: Minimum distance to any prototype [batch_size]
                - closest_intents: Closest intent ID [batch_size]
        """
        if not self.prototypes_initialized:
            raise RuntimeError("Prototypes not initialized. Call update_prototypes first.")

        batch_size = hidden_states.size(0)

        # Expand dimensions for broadcasting
        # hidden_states: [batch_size, 1, 1, hidden_size]
        # prototype_bank: [1, num_intents, num_sub_prototypes, hidden_size]
        hidden_expanded = hidden_states.unsqueeze(1).unsqueeze(2)
        prototypes_expanded = self.prototype_bank.unsqueeze(0)

        if distance_metric == 'cosine':
            # Cosine distance = 1 - cosine_similarity
            hidden_norm = F.normalize(hidden_expanded, p=2, dim=-1)
            proto_norm = F.normalize(prototypes_expanded, p=2, dim=-1)
            similarities = (hidden_norm * proto_norm).sum(dim=-1)
            distances = 1 - similarities  # [batch_size, num_intents, num_sub_prototypes]
        elif distance_metric == 'euclidean':
            # Euclidean distance
            distances = torch.norm(
                hidden_expanded - prototypes_expanded, p=2, dim=-1
            )  # [batch_size, num_intents, num_sub_prototypes]
        else:
            raise ValueError(f"Unknown distance metric: {distance_metric}")

        # Find minimum distance across sub-prototypes for each intent
        min_distances_per_intent, _ = distances.min(dim=2)  # [batch_size, num_intents]

        # Find minimum distance across all intents
        min_distances, closest_intents = min_distances_per_intent.min(dim=1)  # [batch_size]

        return min_distances, closest_intents


if __name__ == "__main__":
    # Test BERT encoder
    print("Testing BERT Intent Encoder...")

    model = BERTIntentEncoder(
        bert_model_name='bert-base-uncased',
        num_labels=151,
        hidden_size=768,
        dropout=0.1
    )

    # Test forward pass
    batch_size = 4
    seq_len = 64

    input_ids = torch.randint(0, 30522, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)

    outputs = model(input_ids, attention_mask)

    print(f"Logits shape: {outputs['logits'].shape}")
    print(f"Hidden states shape: {outputs['hidden_states'].shape}")

    total_params, trainable_params = model.get_num_parameters()
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    print("\nTesting BERT with Prototype...")
    model_with_proto = BERTWithPrototype(
        bert_encoder=model,
        num_intents=150,
        num_sub_prototypes=5,
        hidden_size=768
    )

    print(f"Prototype bank shape: {model_with_proto.prototype_bank.shape}")
