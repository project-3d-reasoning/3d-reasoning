import math

from transformers import TrainerCallback


class BBoxResidualWeightWarmupCallback(TrainerCallback):
    def __init__(
        self,
        target_weight: float,
        warmup_ratio: float,
        target_loss_ratio: float = -1.0,
        target_loss_ratio_start: float = 0.5,
        max_dynamic_weight: float = 10.0,
    ) -> None:
        self.target_weight = float(target_weight)
        self.warmup_ratio = max(0.0, float(warmup_ratio))
        self.target_loss_ratio = float(target_loss_ratio)
        self.target_loss_ratio_start = min(max(float(target_loss_ratio_start), 0.0), 1.0)
        self.max_dynamic_weight = max(0.0, float(max_dynamic_weight))

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

    def _compute_training_progress(self, state, args) -> float:
        max_steps = int(getattr(state, "max_steps", 0) or 0)
        if max_steps <= 0:
            max_steps = int(getattr(args, "max_steps", 0) or 0)
        if max_steps <= 0:
            return 0.0
        return min(max(int(state.global_step), 0), max_steps) / max_steps

    def _sync_weight(self, model, state, args) -> None:
        if model is None:
            return
        model = self._resolve_model(model)
        current_weight = self._compute_current_weight(state, args)
        training_progress = self._compute_training_progress(state, args)
        ratio_active = (
            0.0 < self.target_loss_ratio < 1.0
            and training_progress >= self.target_loss_ratio_start
        )
        setattr(model.config, "bbox_residual_loss_weight_target", self.target_weight)
        setattr(model.config, "bbox_residual_loss_weight_warmup_ratio", self.warmup_ratio)
        setattr(model.config, "bbox_residual_loss_weight_current", float(current_weight))
        setattr(model.config, "bbox_residual_loss_ratio_target", self.target_loss_ratio)
        setattr(model.config, "bbox_residual_loss_ratio_start", self.target_loss_ratio_start)
        setattr(model.config, "bbox_residual_loss_ratio_active", bool(ratio_active))
        setattr(model.config, "bbox_residual_loss_weight_max", self.max_dynamic_weight)

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        self._sync_weight(model, state, args)
        return control

    def on_step_begin(self, args, state, control, model=None, **kwargs):
        self._sync_weight(model, state, args)
        return control
