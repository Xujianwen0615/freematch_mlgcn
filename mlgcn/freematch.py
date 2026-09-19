# -*- coding: utf-8 -*-
"""
FreeMatch (Microsoft Semi-supervised-learning / USB) adapted to multi-label classification.

References:
  - FreeMatch: https://github.com/microsoft/Semi-supervised-learning.git  (semilearn.algorithms.freematch)
  - Adapted from the prior working implementation in
    D:\\freematch_mlgcn\\semilearn\\algorithms\\freematch_mlgcn\\.

What FreeMatch does and how it is adapted for multi-label
----------------------------------------------------------
  1. A *self-adaptive per-class threshold* replaces the fixed 0.5 (or class-balancing) threshold.
     For multi-label we keep one threshold per class (each (sample, class) pair is decided
     independently) rather than using the per-sample max (which would always be ~1.0).
  2. Pseudo-labels are generated on a weakly-augmented unlabeled view (via an EMA teacher by
     default) and enforced on a strongly-augmented view with a masked consistency loss.
  3. An entropy regularizer (confidence + fairness terms) sharpens model prediction and prevents
     collapse toward a few dominant classes.

This module is framework-agnostic: it exposes a ``FreeMatchMultiLabel`` class with a
``train_step``, ``evaluate``, ``update`` (EMA), and ``state_dict``. It does not depend on the
semilearn framework, so it is easy to run on a small (4 GB) GPU with a standalone train.py.
"""
import copy

import numpy as np
import torch
import torch.nn as nn

from .metrics import compute_metrics


def entropy_loss(mask, logits_s, prob_model, label_hist, fairness_weight=0.15,
                 class_weight_max=20.0, loss_clamp=10.0):
    """
    Multi-label entropy regularizer (FreeMatch) — confidence + fairness terms.

    All computation is forced to float32 to avoid AMP/float16 underflow → log(0) = NaN.
    """
    mask = mask.bool()
    logits_s = logits_s[mask].float()
    if logits_s.numel() == 0:
        return torch.tensor(0.0, device=logits_s.device)

    prob_s = torch.sigmoid(logits_s)
    prob_s = torch.clamp(prob_s, 1e-7, 1 - 1e-7)

    # 1) Confidence loss: mean binary entropy of the masked predicted probabilities.
    binary_entropy = -(prob_s * torch.log(prob_s) + (1 - prob_s) * torch.log(1 - prob_s))
    confidence_loss = binary_entropy.mean()

    # 2) Fairness loss: match the batch per-class mean prob to the historical label distribution.
    hist_s = prob_s.mean(dim=0)
    prob_model = prob_model.reshape(-1).float()
    label_hist = label_hist.reshape(-1).float().clamp(min=1e-7)

    class_weight = (1.0 / label_hist).clamp(max=class_weight_max)  # capped so rare classes do not dominate
    class_weight = class_weight / (class_weight.sum() + 1e-7)

    hist_s_norm = hist_s / (hist_s.sum() + 1e-7)
    hist_s_norm = hist_s_norm.clamp(min=1e-7)
    prob_model_norm = prob_model / (prob_model.sum() + 1e-7)
    prob_model_norm = prob_model_norm.clamp(min=1e-7)

    fairness_loss = -(class_weight * prob_model_norm * torch.log(hist_s_norm)).sum()

    loss = confidence_loss + fairness_weight * fairness_loss
    return torch.clamp(loss, 0.0, loss_clamp)


class FreeMatchMultiLabel:
    """Self-contained FreeMatch + multi-label trainer (student + EMA teacher)."""

    def __init__(self, model, num_classes, criterion, consistency_criterion, loss_type="asl",
                 T=0.5, hard_label=False, ema_p=0.999, ema_m=0.999, use_ema_teacher=True,
                 use_quantile=True, clip_thresh=True, time_p_init=0.3,
                 lambda_u=1.0, lambda_e=0.02, use_class_balancing=True,
                 asl_warmup_ratio=0.03, label_counts=None, cb_beta=0.999,
                 thresh_q=0.7, thresh_mod_min=0.15, thresh_lo=0.15, thresh_hi=0.85,
                 clip_lo=0.4, clip_hi=0.90,
                 conf_pow=2.0, unsup_class_balance=True,
                 fairness_weight=0.15, fairness_class_weight_max=20.0,
                 device="cuda", use_amp=False, logger=print):
        self.model = model
        self.ema_model = copy.deepcopy(model)
        self.num_classes = num_classes
        self.criterion = criterion
        self.consistency_criterion = consistency_criterion

        # ==============================
        # 新增
        # ==============================
        self.loss_type = str(loss_type).lower()
        if self.loss_type not in ("asl", "bce"):
            raise ValueError(
                f"Unsupported loss_type={loss_type}. Expected 'asl' or 'bce'."
            )

        self.T = T
        self.use_hard_label = hard_label
        self.ema_p = ema_p  # threshold EMA momentum
        self.ema_m = ema_m  # weight EMA momentum
        self.use_ema_teacher = use_ema_teacher
        self.use_quantile = use_quantile
        self.clip_thresh = clip_thresh

        # ---- P1-1: relaxed / configurable pseudo-label thresholds ----
        self.thresh_q = thresh_q  # quantile of unlabeled probs used to update time_p
        self.thresh_mod_min = thresh_mod_min
        self.thresh_lo = thresh_lo
        self.thresh_hi = thresh_hi
        self.clip_lo = clip_lo  # clamp bounds applied to time_p when clip_thresh=True
        self.clip_hi = clip_hi
        # ---- P1-2: confidence-soft-weighted unsupervised consistency loss ----
        self.conf_pow = conf_pow  # 0 => plain binary mask (old behaviour)
        self.unsup_class_balance = unsup_class_balance  # tail-aware re-weighting of pseudo-labels
        # ---- P1-3: strengthened entropy fairness term ----
        self.fairness_weight = fairness_weight
        self.fairness_class_weight_max = fairness_class_weight_max

        self.lambda_u = lambda_u
        self.lambda_e = lambda_e
        self.use_class_balancing = use_class_balancing
        self.cb_beta = cb_beta
        # True per-class positive counts of the *labeled* pool (not the model-estimated
        # ``label_hist``) — used for effective-number class balancing so that rare classes get a
        # large, stable re-balancing weight instead of tracking the model's own long-tail bias.
        if label_counts is not None:
            self.label_counts = torch.as_tensor(label_counts, dtype=torch.float32, device=device)
        else:
            self.label_counts = torch.ones((num_classes,), dtype=torch.float32, device=device)
        self.asl_warmup_ratio = asl_warmup_ratio
        self.device = device
        self.use_amp = use_amp
        self.log = logger

        # AMP autocast context manager (torch 2.x API).
        self.amp_cm = torch.amp.autocast(device_type="cuda", enabled=use_amp)

        # FreeMatch running statistics (per class).
        self.p_model = torch.full((num_classes,), 0.5, device=device)
        self.label_hist = torch.full((num_classes,), 1.0 / num_classes, device=device)
        self.time_p = torch.tensor(time_p_init, device=device)
        self.it = 0

    # ------------------------------------------------------------------ utils
    def _to_device(self, *tensors):
        return [t.to(self.device) for t in tensors]

    def _update_loss_schedule(self):
        """
        Update loss-specific training schedule.
        ASL:
            progressively enables gamma / clipping
            during the first asl_warmup_ratio of training.
        BCE:
            no warmup is required.
        """

        if self.loss_type != "asl":
            return

        if not hasattr(self.criterion, "set_progress"):
            return

        warmup_iters = max(1, int(self.num_train_iter * self.asl_warmup_ratio))
        progress = min(1.0,self.it / warmup_iters)
        self.criterion.set_progress(progress)

    def _per_class_threshold(self):
        """per_class_thresh[c] = time_p * (p_model[c] / max(p_model)), clamped."""
        mod = self.p_model / self.p_model.max()

        # 改后（更低阈值 → 更多伪标签通过）
        mod = torch.clamp(mod, self.thresh_mod_min, 1.0)  # 0.25 → 0.15
        thresh = torch.clamp(self.time_p * mod, self.thresh_lo, self.thresh_hi)  # 0.25/0.90 → 0.15/0.85
        return thresh

    @torch.no_grad()
    def _update_threshold(self, probs_ulb):
        """EMA-update time_p, p_model and label_hist from unlabeled probabilities [U, C]."""
        max_probs = probs_ulb.max(dim=-1)[0]
        if self.use_quantile:
            flat = probs_ulb.flatten().float()
            q = torch.quantile(flat, self.thresh_q)
        else:
            q = torch.quantile(max_probs.float(), self.thresh_q)
        self.time_p = self.time_p * self.ema_p + (1 - self.ema_p) * q
        if self.clip_thresh:
            self.time_p = torch.clamp(self.time_p, self.clip_lo, self.clip_hi)

        self.p_model = self.p_model * self.ema_p + (1 - self.ema_p) * probs_ulb.float().mean(dim=0)

        pos_count = (probs_ulb > 0.5).sum(dim=0).to(self.p_model.dtype)
        pos_sum = pos_count.sum()
        if pos_sum > 0:
            self.label_hist = self.label_hist * self.ema_p + (1 - self.ema_p) * (pos_count / pos_sum)

    def _ema_update(self):
        """Exponential moving average of the student weights into the teacher."""
        with torch.no_grad():
            for p, tp in zip(self.model.parameters(), self.ema_model.parameters()):
                tp.mul_(self.ema_m).add_(p.detach(), alpha=1 - self.ema_m)
            for bp, tbp in zip(self.model.buffers(), self.ema_model.buffers()):
                tbp.copy_(bp)

    # ------------------------------------------------------------------ train
    def _class_balancing_weight(self):
        """Effective-number (class-balanced) per-class weight from the *labeled* class counts.

        ``label_counts[c]`` = number of labeled images containing class c. Using the effective
        number ``(1-beta)/(1-beta^{n_c})`` (Cui et al., CVPR 2019) gives rare classes a much larger
        weight than the old linear ``1/n`` term capped at 5.0. Weights are mean-normalised and then
        clamped (0.05..20) so no single class dominates a batch.
        """
        counts = self.label_counts.clamp(min=1.0)
        beta = float(self.cb_beta)
        beta_t = torch.tensor(beta, device=counts.device)
        en = (1.0 - beta_t.pow(counts)) / (1.0 - beta + 1e-7)  # effective number E_n
        w = 1.0 / en.clamp(min=1e-7)  # class-balanced weight = 1 / E_n
        w = w / (w.mean().clamp(min=1e-7))
        return w.clamp(0.05, 20.0)

    def _sup_loss(self, logits_x_lb, y_lb):
        """Supervised (ASL / BCE) loss with effective-number (class-balanced) re-weighting."""
        per_elem = self.criterion(logits_x_lb, y_lb)  # [B, C]
        # 防止错误使用 reduction='mean'
        if per_elem.ndim == 0:
            raise RuntimeError(
                "Supervised criterion returned a scalar. "
                "FreeMatch requires reduction='none' "
                "so that the loss shape is [B, C]."
            )
        if per_elem.shape != y_lb.shape:
            raise RuntimeError(
                f"Loss shape mismatch: "
                f"loss={tuple(per_elem.shape)}, "
                f"target={tuple(y_lb.shape)}"
            )

        if self.use_class_balancing:
            with torch.no_grad():
                class_weight = self._class_balancing_weight()
                weight = torch.where(
                    y_lb > 0.5,
                    class_weight.unsqueeze(0),
                    torch.tensor(1.0, device=y_lb.device),
                )
                normalizer = weight.sum().clamp(min=1.0)
            return (per_elem * weight).sum() / normalizer
        return per_elem.mean()

    #     def _sup_loss(self, logits_x_lb, y_lb):
    #         """Supervised (ASL / BCE) loss with optional class re-balancing. Returns scalar."""
    #         per_elem = self.criterion(logits_x_lb, y_lb)  # [B, C]
    #         if self.use_class_balancing:
    #             with torch.no_grad():
    #                 class_weight = 1.0 / (self.label_hist.to(y_lb.device) * self.num_classes + 0.1)
    #                 class_weight = class_weight.clamp(0.5, 5.0)

    #                 # # B) Effective-number (class-balanced) reweighting，β 可调（图省事就用 A）
    #                 # beta = 0.999
    #                 # n_c = self.label_hist.to(y_lb.device) * self.num_classes          # 各类有效计数
    #                 # class_weight = (1.0 - beta) / (1.0 - pow(beta, n_c) + 1e-7)
    #                 # class_weight = class_weight / class_weight.sum() * self.num_classes  # 归一化回均值

    #                 weight = torch.where(
    #                     y_lb > 0.5,
    #                     class_weight.unsqueeze(0),
    #                     torch.tensor(1.0, device=y_lb.device),
    #                 )
    #                 normalizer = weight.sum().clamp(min=1.0)
    #             return (per_elem * weight).sum() / normalizer
    #         return per_elem.mean()

    def sup_train_step(self, x_lb, y_lb):
        """Supervised-only step (used for full-supervision baselines / no-unlabeled runs).
        Compatible with:ASL     BCE
        """
        x_lb, y_lb = self._to_device(x_lb, y_lb)
        self.it += 1
        # if hasattr(self.criterion, "set_progress"):
        #     progress = min(1.0, self.it / max(1, int(self.num_train_iter * self.asl_warmup_ratio)))
        #     self.criterion.set_progress(progress)
        # ASL warmup; BCE automatically skips
        self._update_loss_schedule()
        with self.amp_cm:
            logits_x_lb = self.model(x_lb)
            sup_loss = self._sup_loss(logits_x_lb, y_lb)
        return {"sup_loss": sup_loss.item(), "unsup_loss": 0.0,
                "ent_loss": 0.0, "total_loss": sup_loss.item(), "util_ratio": 0.0}, sup_loss

    def train_step(self, x_lb, y_lb, x_ulb_w, x_ulb_s):
        """
        Args:
            x_lb:    [B, 3, H, W] labeled images
            y_lb:    [B, C]      labeled targets in {0, 1}
            x_ulb_w: [U, 3, H, W] weakly-augmented unlabeled images
            x_ulb_s: [U, 3, H, W] strongly-augmented unlabeled images
        Returns: dict of scalar losses.
        """
        x_lb, y_lb, x_ulb_w, x_ulb_s = self._to_device(x_lb, y_lb, x_ulb_w, x_ulb_s)
        self.it += 1

        # # ASL warmup: first ``asl_warmup_ratio`` of training behaves like BCE.
        # if hasattr(self.criterion, "set_progress"):
        #     progress = min(1.0, self.it / max(1, int(self.num_train_iter * self.asl_warmup_ratio)))
        #     self.criterion.set_progress(progress)
        # ================================
        # ASL → warmup
        # BCE → no-op
        # ================================
        self._update_loss_schedule()

        # ---- forward (student) ----
        with self.amp_cm:
            logits_x_lb = self.model(x_lb)
            logits_x_ulb_s = self.model(x_ulb_s)

            # ---- supervised loss (ASL or BCE, with optional class re-balancing) ----
            sup_loss = self._sup_loss(logits_x_lb, y_lb)

            # ---- pseudo-label from the EMA teacher on the weak view ----
            with torch.no_grad():
                teacher = self.ema_model if self.use_ema_teacher else self.model
                logits_x_ulb_w = teacher(x_ulb_w)
                probs_x_ulb_w = torch.sigmoid(logits_x_ulb_w.float())
                self._update_threshold(probs_x_ulb_w)

                thresh = self._per_class_threshold()  # [C]
                mask = probs_x_ulb_w > thresh.unsqueeze(0)  # [U, C] bool
                # pos_mask = prob >= tau_pos
                # neg_mask = prob <= tau_neg
                # mask = pos_mask | neg_mask
                if self.use_hard_label:
                    pseudo_label = (probs_x_ulb_w > thresh.unsqueeze(0)).float()
                else:
                    pseudo_label = torch.sigmoid(logits_x_ulb_w.float() / self.T)  # sharpened soft labels
                    pseudo_label = pseudo_label.clamp(1e-6,1.0 - 1e-6)
                pseudo_label = pseudo_label.float()

            # ---- unsupervised consistency loss (masked) ----
            per_elem_unsup = self.consistency_criterion(logits_x_ulb_s, pseudo_label)  # [U, C]

            if per_elem_unsup.ndim == 0:
                raise RuntimeError("Consistency criterion must use reduction='none'.")
            if per_elem_unsup.shape != pseudo_label.shape:
                raise RuntimeError(f"Consistency loss shape mismatch: {tuple(per_elem_unsup.shape)} vs {tuple(pseudo_label.shape)}")

            # 无标签一致性用置信度软权重
            # if mask.sum() > 0:
            #     unsup_loss = (per_elem_unsup * mask.float()).sum() / mask.float().sum()
            # else:
            #     unsup_loss = torch.tensor(0.0, device=x_ulb_s.device)

            if mask.sum() > 0:
                # P1-2: keep only above-threshold entries, then weight them softly.
                w_u = mask.float()
                if self.conf_pow > 0:
                    # Down-weight low-confidence pseudo-labels (they are the noisy ones).
                    w_u = w_u * probs_x_ulb_w.float().pow(self.conf_pow)
                if self.unsup_class_balance:
                    # Confidence weighting alone would further suppress rare classes (they are
                    # less confident), so compensate with the effective-number class weights.
                    w_u = w_u * self._class_balancing_weight().unsqueeze(0)
                unsup_loss = (per_elem_unsup * w_u).sum() / w_u.sum().clamp(min=1e-7)
            else:
                unsup_loss = torch.tensor(0.0, device=x_ulb_s.device)

            # ---- entropy regularizer ----
            if mask.sum() > 0:
                ent_loss = entropy_loss(mask, logits_x_ulb_s, self.p_model, self.label_hist,
                                        fairness_weight=self.fairness_weight,
                                        class_weight_max=self.fairness_class_weight_max)
                if torch.isnan(ent_loss):
                    ent_loss = torch.tensor(0.0, device=x_ulb_s.device)
            else:
                ent_loss = torch.tensor(0.0, device=x_ulb_s.device)

            total_loss = sup_loss + self.lambda_u * unsup_loss + self.lambda_e * ent_loss

        return {
            "sup_loss": sup_loss.item(),
            "unsup_loss": unsup_loss.item(),
            "ent_loss": ent_loss.item(),
            "total_loss": total_loss.item(),
            "util_ratio": mask.float().mean().item() if mask is not None else 0.0,
        }, total_loss

    # ------------------------------------------------------------------ eval
    @torch.no_grad()
    def evaluate(self, loader, use_ema=True, threshold=0.5):
        """Compute multi-label metrics over a val/test loader."""
        self.model.eval()
        eval_model = self.ema_model if (use_ema and self.ema_model is not None) else self.model
        eval_model.eval()

        all_logits, all_targets = [], []
        for batch in loader:
            x, y = batch
            x = x.to(self.device)
            logits = eval_model(x)
            all_logits.append(logits.detach().cpu().numpy())
            all_targets.append(y.numpy())

        all_logits = np.concatenate(all_logits, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        all_probs = 1.0 / (1.0 + np.exp(-all_logits))

        metrics = compute_metrics(all_targets, all_probs, threshold=threshold,
                                  class_names=getattr(self, "class_names", None), eval_name="eval")
        self.model.train()
        return metrics, all_logits, all_targets

    # ------------------------------------------------------------------ io
    def update(self):
        self._ema_update()

    def state_dict(self):
        return {
            "model": self.model.state_dict(),
            "ema_model": self.ema_model.state_dict(),
            "p_model": self.p_model.cpu(),
            "label_hist": self.label_hist.cpu(),
            "time_p": self.time_p.cpu(),
            "it": self.it,
        }

    def load_state_dict(self, sd):
        self.model.load_state_dict(sd["model"])
        self.ema_model.load_state_dict(sd["ema_model"])
        self.p_model = sd["p_model"].to(self.device)
        self.label_hist = sd["label_hist"].to(self.device)
        self.time_p = sd["time_p"].to(self.device)
        self.it = sd.get("it", self.it)


if __name__ == "__main__":
    # Smoke test with random data (no dataset needed).
    from losses import build_loss
    from models import build_model

    torch.manual_seed(0)
    model = build_model(15, backbone="resnet50", variant="gcn", pretrained=False,
                        adj_file=None, inp=None)
    algo = FreeMatchMultiLabel(model, 15, build_loss("asl", reduction="none"),
                               build_loss("asl_soft", reduction="none"),
                               device="cpu", use_ema_teacher=True)
    algo.num_train_iter = 1000
    x_lb = torch.randn(8, 3, 224, 224)
    y_lb = (torch.rand(8, 15) > 0.6).float()
    x_u_w = torch.randn(16, 3, 224, 224)
    x_u_s = torch.randn(16, 3, 224, 224)
    log_dict, loss = algo.train_step(x_lb, y_lb, x_u_w, x_u_s)
    print("loss:", loss.item(), log_dict)
