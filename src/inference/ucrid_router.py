"""
UCRID Stage 2: Dual-Threshold Adaptive Router
Computes uncertainty score s(u) = alpha * H_norm + (1-alpha) * d_norm
and routes to: small model / direct OOS / LLM judge.
"""

from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class UCRIDRouter:
    """
    Dual-threshold router for UCRID cascade system.

    Routing logic:
      s(u) <= tau_accept                    -> small model top-1
      s(u) >= tau_reject AND d_min > delta  -> direct OOS
      else                                  -> LLM judge (top-k candidates)
    """

    def __init__(
        self,
        tau_accept: float = 0.3,
        tau_reject: float = 0.8,
        delta: float = 1.0,
        alpha: float = 0.5,
        top_k: int = 3,
        oos_label: int = 150,
        temperature: float = 1.0,
    ):
        self.tau_accept = tau_accept
        self.tau_reject = tau_reject
        self.delta = delta
        self.alpha = alpha
        self.top_k = top_k
        self.oos_label = oos_label
        self.temperature = max(float(temperature), 1e-3)

        # Running stats for normalization (updated on calibration set)
        self._H_min = 0.0
        self._H_max = 1.0
        self._d_min = 0.0
        self._d_max = 1.0
        self._eps = 1e-8

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def calibrate(self, entropies: np.ndarray, distances: np.ndarray):
        """Fit normalization stats on validation set."""
        entropies = np.asarray(entropies, dtype=np.float32)
        distances = np.asarray(distances, dtype=np.float32)
        self._H_min = float(entropies.min())
        self._H_max = float(entropies.max()) + self._eps
        self._d_min = float(distances.min())
        self._d_max = float(distances.max()) + self._eps

    def fit_temperature(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        max_iter: int = 50,
    ) -> float:
        """
        Fit a single temperature scalar on a validation set by minimizing NLL.
        """
        logits = logits.detach()
        labels = labels.detach().long()

        log_temperature = nn.Parameter(
            torch.log(torch.tensor(self.temperature, device=logits.device))
        )
        optimizer = torch.optim.LBFGS([log_temperature], lr=0.1, max_iter=max_iter)

        def closure():
            optimizer.zero_grad()
            temperature = log_temperature.exp().clamp(min=1e-3, max=100.0)
            loss = F.cross_entropy(logits / temperature, labels)
            loss.backward()
            return loss

        optimizer.step(closure)
        self.temperature = float(log_temperature.detach().exp().clamp(min=1e-3, max=100.0).item())
        return self.temperature

    def scale_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply scalar temperature calibration to logits."""
        return logits / max(self.temperature, 1e-3)

    # ------------------------------------------------------------------
    # Core computation
    # ------------------------------------------------------------------

    def compute_entropy(self, logits: torch.Tensor) -> torch.Tensor:
        """Softmax entropy H(u) = -sum p log p.  Shape: [B]"""
        scaled_logits = self.scale_logits(logits)
        probs = F.softmax(scaled_logits, dim=-1)
        entropy = -(probs * (probs + self._eps).log()).sum(dim=-1)
        return entropy

    def compute_d_min(
        self, hidden: torch.Tensor, prototypes: torch.Tensor
    ) -> torch.Tensor:
        """
        Euclidean distance to nearest intent prototype.
        hidden:     [B, D]
        prototypes: [num_intents, D]
        returns:    [B]
        """
        dists = torch.cdist(hidden, prototypes, p=2)  # [B, num_intents]
        return dists.min(dim=1)[0]

    def uncertainty_score(
        self,
        logits: torch.Tensor,
        hidden: torch.Tensor,
        prototypes: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns (s, H_norm, d_norm) each of shape [B].
        """
        H = self.compute_entropy(logits)
        d = self.compute_d_min(hidden, prototypes)

        H_norm = ((H - self._H_min) / (self._H_max - self._H_min + self._eps)).clamp(0.0, 1.0)
        d_norm = ((d - self._d_min) / (self._d_max - self._d_min + self._eps)).clamp(0.0, 1.0)
        s = self.alpha * H_norm + (1.0 - self.alpha) * d_norm

        return s, H_norm, d_norm

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------

    def route(
        self,
        logits: torch.Tensor,
        hidden: torch.Tensor,
        prototypes: torch.Tensor,
    ) -> Dict:
        """
        Route each sample in the batch.

        Returns dict with keys:
          decisions:   list of 'small_model' | 'direct_oos' | 'llm'
          predictions: tensor [B], filled for small_model/direct_oos, -1 for llm
          top1_intents: tensor [B]
          topk_intents: list of lists (top-k intent indices per sample)
          scores:      tensor [B]  uncertainty scores
          d_min:       tensor [B]
        """
        B = logits.size(0)
        scaled_logits = self.scale_logits(logits)
        probs = F.softmax(scaled_logits, dim=-1)
        top1 = probs.argmax(dim=-1)  # [B]
        topk_vals, topk_idx = probs.topk(self.top_k, dim=-1)  # [B, k]

        d_min_t = self.compute_d_min(hidden, prototypes)  # [B]
        s, H_norm, d_norm = self.uncertainty_score(logits, hidden, prototypes)

        decisions = []
        predictions = torch.full((B,), -1, dtype=torch.long, device=logits.device)

        for i in range(B):
            si = s[i].item()
            di = d_min_t[i].item()

            if si <= self.tau_accept:
                decisions.append("small_model")
                predictions[i] = top1[i]
            elif si >= self.tau_reject and di > self.delta:
                decisions.append("direct_oos")
                predictions[i] = self.oos_label
            else:
                decisions.append("llm")

        return {
            "decisions": decisions,
            "predictions": predictions,
            "top1_intents": top1,
            "topk_intents": topk_idx.tolist(),
            "topk_probs": topk_vals.tolist(),
            "scores": s,
            "entropy_norm": H_norm,
            "distance_norm": d_norm,
            "d_min": d_min_t,
        }

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def routing_stats(self, decisions: List[str]) -> Dict:
        n = len(decisions)
        if n == 0:
            return {"total": 0, "small_model": 0.0, "direct_oos": 0.0, "llm": 0.0}
        return {
            "total": n,
            "small_model": decisions.count("small_model") / n,
            "direct_oos": decisions.count("direct_oos") / n,
            "llm": decisions.count("llm") / n,
        }
