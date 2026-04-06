import math

from transformers import TrainerCallback


class BBoxResidualWeightWarmupCallback(TrainerCallback):
    def __init__(self, target_weight: float, warmup_ratio: float) -> None:
        self.target_weight = float(target_weight)
        self.warmup_ratio = max(0.0, float(warmup_ratio))

    def _resolve_model(self, model):
        return getattr(model, "module", model)

    def _compute_current_weight(self, state, args) -> float:
        if self.target_weight <= 0 or self.warmup_ratio <= 0:
            return self.target_weight

        max_steps = int(getattr(state, "max_steps", 0) or 0)
        if max_steps <= 0:
            max_steps = int(getattr(args, "max_steps", 0) or 0)
        if max_steps <= 0:
            return self.target_weight

        warmup_steps = max(1, int(math.ceil(max_steps * self.warmup_ratio)))
        progress = min(max(int(state.global_step), 0), warmup_steps) / warmup_steps
        return self.target_weight * progress

    def _sync_weight(self, model, state, args) -> None:
        if model is None:
            return
        model = self._resolve_model(model)
        current_weight = self._compute_current_weight(state, args)
        setattr(model.config, "bbox_residual_loss_weight_target", self.target_weight)
        setattr(model.config, "bbox_residual_loss_weight_warmup_ratio", self.warmup_ratio)
        setattr(model.config, "bbox_residual_loss_weight_current", float(current_weight))

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        self._sync_weight(model, state, args)
        return control

    def on_step_begin(self, args, state, control, model=None, **kwargs):
        self._sync_weight(model, state, args)
        return control
