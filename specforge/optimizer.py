import logging
import math

import torch
import torch.distributed as dist

from specforge.lr_scheduler import CosineAnnealingWarmupLR
from specforge.utils import print_on_rank0

logger = logging.getLogger(__name__)

SPIKE_MODES = ("off", "observe", "on")


class GradNormSpikeGuard:
    """Control limit on the gradient norm, as a multiple of its typical value.

    Global norm clipping bounds the *norm* of an update but not its direction,
    and AdamW's per-parameter step is very nearly invariant to a global rescale
    of the gradient -- so a clipped outlier still moves every parameter by about
    one learning rate, in whatever direction the outlier pointed. This guard
    exists to discard those steps outright instead of scaling them down.

    Gradient norms are approximately log-normal, so the limit lives in log space
    and is expressed as a dimensionless multiple of the running geometric mean
    rather than an absolute value. The absolute scale is not a constant to
    calibrate once: on the run this was designed against it drifted from 0.80 to
    0.25 within a single training run, and it moves further with the dataset,
    the learning rate and the batch size. The multiple is what transfers.

    Only the geometric mean is learned. An earlier design also tracked the
    log-space standard deviation and used ``exp(mu + k*sigma)``; that form is
    actively worse, because sigma enters the limit exponentially and is inflated
    by exactly the events it must detect -- the spike itself, and the elevated
    gradients of the damage it causes. Replayed against a real run containing
    four spikes, ``mu + 5*sigma`` caught one of four (its limit had been pushed
    to 525 by the first event); a plain multiple of the geometric mean caught
    four of four with no false positive in 858 healthy steps.

    Accepted observations are winsorized at the limit, so nothing that survives
    the check -- which is everything during warmup, when the check is not yet
    active -- can drag the estimate up by more than one bounded increment.
    """

    def __init__(
        self,
        *,
        mode: str = "off",
        ratio: float = 10.0,
        warmup_steps: int = 500,
        min_observations: int = 50,
        decay: float = 0.99,
        max_consecutive_skips: int = 10,
    ) -> None:
        if mode not in SPIKE_MODES:
            raise ValueError(
                f"grad spike mode must be one of {SPIKE_MODES}, got {mode!r}"
            )
        if ratio <= 1.0:
            raise ValueError(f"grad spike ratio must be > 1, got {ratio}")
        if not 0.0 < decay < 1.0:
            raise ValueError(f"grad spike decay must be in (0, 1), got {decay}")
        self.mode = mode
        self.ratio = float(ratio)
        self.warmup_steps = int(warmup_steps)
        self.min_observations = int(min_observations)
        self.decay = float(decay)
        self.max_consecutive_skips = int(max_consecutive_skips)
        self.log_mean = None
        self.steps = 0
        self.observations = 0
        self.skipped = 0
        self.flagged = 0
        self.consecutive_skips = 0

    @property
    def enabled(self) -> bool:
        return self.mode != "off"

    @property
    def typical_norm(self):
        """Running geometric mean of the gradient norm, or ``None`` while cold."""
        return None if self.log_mean is None else math.exp(self.log_mean)

    @property
    def limit(self):
        """Current control limit, or ``None`` while the estimate is not usable."""
        if self.log_mean is None or self.observations < self.min_observations:
            return None
        return math.exp(self.log_mean + math.log(self.ratio))

    def consider(self, grad_norm: float) -> bool:
        """Advance the guard by one optimizer step; return whether to discard it.

        The caller must pass the norm that has ALREADY been reduced across
        ranks, and must call this exactly once per optimizer step on every rank.
        Every rank then walks the same state machine over the same inputs and
        reaches the same verdict, so no extra collective is needed to agree.
        A rank-local norm would let ranks disagree about whether a step
        happened, which desynchronizes the optimizer and the LR schedule.
        """
        if self.mode == "off":
            # A complete no-op, so an off guard cannot report activity it never
            # had. ``step`` also gates on ``enabled`` and never calls this.
            return False
        self.steps += 1
        finite = math.isfinite(grad_norm)
        limit = self.limit
        flagged = (not finite) or (limit is not None and grad_norm > limit)
        if not flagged:
            self._observe(grad_norm)
            self.consecutive_skips = 0
            return False

        self.flagged += 1
        if self.mode != "on":
            # observe: report what would have happened without changing training.
            return False
        if self.consecutive_skips >= self.max_consecutive_skips:
            # Refusing every step forever is worse than taking a bad one: the run
            # would make no progress and say nothing about why. Let this one
            # through, complain loudly, and re-baseline on it so a genuine shift
            # in gradient scale recalibrates instead of deadlocking the guard.
            logger.error(
                "grad spike guard skipped %d consecutive steps (norm=%.6g, "
                "limit=%.6g); accepting this step and re-baselining. Either the "
                "run is genuinely diverging or grad_spike_ratio is too tight",
                self.consecutive_skips,
                grad_norm,
                float("nan") if limit is None else limit,
            )
            self.consecutive_skips = 0
            self._observe(grad_norm)
            return False

        self.skipped += 1
        self.consecutive_skips += 1
        logger.warning(
            "grad spike guard skipping step %d: grad_norm=%.6g exceeds limit "
            "%.6g (%.1fx the typical %.6g)",
            self.steps,
            grad_norm,
            float("nan") if limit is None else limit,
            (
                float("nan")
                if not finite or not self.typical_norm
                else grad_norm / self.typical_norm
            ),
            float("nan") if self.typical_norm is None else self.typical_norm,
        )
        return True

    def _observe(self, grad_norm: float) -> None:
        """Fold one accepted norm into the geometric mean."""
        if self.steps <= self.warmup_steps:
            # Freshly initialized weights produce legitimately huge, wildly
            # varying norms (52.3 at step 10 on the reference run). Letting them
            # into the estimate raises the limit for thousands of steps.
            return
        if not math.isfinite(grad_norm) or grad_norm <= 0.0:
            # A zero norm is legitimate (an accumulation window with no grads);
            # its log is not. Skip it rather than poisoning the mean.
            return
        value = math.log(grad_norm)
        if self.log_mean is None:
            self.log_mean = value
        else:
            value = min(value, self.log_mean + math.log(self.ratio))
            self.log_mean += (1.0 - self.decay) * (value - self.log_mean)
        self.observations += 1

    def metrics(self) -> dict:
        if not self.enabled:
            return {}
        out = {
            "grad_spike_skipped": float(self.skipped),
            "grad_spike_flagged": float(self.flagged),
        }
        if self.typical_norm is not None:
            out["grad_norm_typical"] = float(self.typical_norm)
        if self.limit is not None:
            out["grad_norm_limit"] = float(self.limit)
        return out

    def state_dict(self) -> dict:
        return {
            "log_mean": self.log_mean,
            "steps": self.steps,
            "observations": self.observations,
            "skipped": self.skipped,
            "flagged": self.flagged,
            "consecutive_skips": self.consecutive_skips,
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore the estimate; tolerate a checkpoint written without one.

        Deliberately does NOT validate the configuration against the checkpoint.
        Unlike ``max_grad_norm``, this guard is a runtime policy rather than an
        objective semantic, and the whole point is to be able to switch it on
        when resuming a run that spiked without it.
        """
        self.log_mean = state.get("log_mean")
        self.steps = int(state.get("steps", 0))
        self.observations = int(state.get("observations", 0))
        self.skipped = int(state.get("skipped", 0))
        self.flagged = int(state.get("flagged", 0))
        self.consecutive_skips = int(state.get("consecutive_skips", 0))


class BF16Optimizer:
    """AdamW over fp32 master copies of the bf16 trainable params, with grad
    clipping and cosine warmup scheduling."""

    def __init__(
        self,
        model,
        lr,
        weight_decay=0.0,
        max_grad_norm=0.5,
        total_steps=800_000,
        warmup_ratio=0.015,
        offload_master=False,
        betas=(0.9, 0.999),
        grad_spike_skip="off",
        grad_spike_ratio=10.0,
        grad_spike_warmup_steps=500,
    ):
        # defaults copied from EAGLE traineagle3 ds_config.json
        self.model = model
        self.model_params = [p for p in model.parameters() if p.requires_grad]
        self.max_grad_norm = max_grad_norm
        self.offload_master = bool(offload_master)
        self.spike_guard = GradNormSpikeGuard(
            mode=grad_spike_skip,
            ratio=grad_spike_ratio,
            warmup_steps=grad_spike_warmup_steps,
        )
        self.fp32_params = [
            (
                p.detach().to(device="cpu", dtype=torch.float32).clone()
                if self.offload_master
                else p.detach().clone().to(torch.float32)
            )
            for p in self.model_params
        ]
        for mp in self.fp32_params:
            mp.requires_grad = True
        self.optimizer = torch.optim.AdamW(
            self.fp32_params, lr=lr, weight_decay=weight_decay, betas=tuple(betas)
        )
        self.last_grad_norm = None
        self._grad_norm_process_group = None
        self._reduce_grad_norm_across_ranks = True
        self.scheduler = CosineAnnealingWarmupLR(
            self.optimizer,
            total_steps=total_steps,
            warmup_steps=int(warmup_ratio * total_steps),
        )

    def configure_grad_norm_reduction(
        self, *, process_group=None, enabled: bool = True
    ) -> None:
        """Configure the group that owns disjoint gradient shards.

        FSDP backends disable the reduction for replicated/NO_SHARD parameters.
        """
        self._grad_norm_process_group = process_group
        self._reduce_grad_norm_across_ranks = enabled

    def _reduce_grad_norm(self, total_norm_sq):
        """All-reduce the squared L2 norm across shard ranks and derive the
        clip coefficient.

        ``total_norm_sq`` must already live on a device the process group can
        reduce (e.g. CUDA for NCCL). Returns ``(total_norm, clip_coef)``.
        """
        if (
            self._reduce_grad_norm_across_ranks
            and dist.is_available()
            and dist.is_initialized()
        ):
            dist.all_reduce(
                total_norm_sq,
                op=dist.ReduceOp.SUM,
                group=self._grad_norm_process_group,
            )
        total_norm = total_norm_sq.sqrt()
        clip_coef = torch.clamp(self.max_grad_norm / (total_norm + 1e-6), max=1.0)
        return total_norm, clip_coef

    def _grad_norm_and_clip_coefficient(self):
        """Compute the global grad norm from the model params on their own
        device, where NCCL can reduce it safely, without materialising master
        gradients first."""
        grads = [p.grad.detach() for p in self.model_params if p.grad is not None]
        if grads:
            total_norm_sq = torch.stack(
                [grad.float().square().sum() for grad in grads]
            ).sum()
        else:
            device = self.model_params[0].device if self.model_params else "cpu"
            total_norm_sq = torch.zeros((), dtype=torch.float32, device=device)
        return self._reduce_grad_norm(total_norm_sq)

    def _clip_grad_norm(self):
        """Clip already-populated FP32 master gradients in place.

        Convenience entry point for optimizer tests and custom loops. When
        masters are CPU-offloaded, only the scalar norm is moved to the model
        device so a NCCL process group can still participate in the reduction.
        """
        grads = [master.grad for master in self.fp32_params if master.grad is not None]
        if grads:
            local_norm_sq = torch.stack(
                [grad.float().square().sum() for grad in grads]
            ).sum()
        else:
            master_device = self.fp32_params[0].device if self.fp32_params else "cpu"
            local_norm_sq = torch.zeros((), dtype=torch.float32, device=master_device)

        reduction_device = (
            self.model_params[0].device if self.model_params else local_norm_sq.device
        )
        total_norm, clip_coef = self._reduce_grad_norm(
            local_norm_sq.to(reduction_device)
        )
        for grad in grads:
            coefficient = (
                clip_coef
                if clip_coef.device == grad.device
                else float(clip_coef.item())
            )
            grad.mul_(coefficient)
        return total_norm

    def _discard_gradients(self) -> None:
        """Drop this window's gradients without touching the Adam moments.

        Zeroing the gradient and stepping anyway is NOT equivalent: AdamW would
        still decay ``exp_avg``/``exp_avg_sq`` and still apply decoupled weight
        decay. The whole point of a skip is that the outlier never enters the
        moments, because a contaminated ``exp_avg_sq`` is what turns a bad step
        into hundreds of steps of suppressed learning rate.
        """
        self.optimizer.zero_grad()
        for p in self.model_params:
            p.grad = None

    def step(self):
        grad_norm, clip_coefficient = self._grad_norm_and_clip_coefficient()
        if self.spike_guard.enabled:
            # The only device->host sync this guard adds, and only when it is
            # switched on. The training loop already synchronizes at least once
            # per micro-batch (the anchor-count read in the DFlash-family
            # forward), so one more per optimizer step is strictly smaller than
            # what the step already pays.
            if self.spike_guard.consider(float(grad_norm)):
                self.last_grad_norm = grad_norm.detach()
                self._discard_gradients()
                # The schedule still advances: it is keyed to the optimizer step
                # count that the trainer contract and every rank agree on, and
                # freezing it here would desynchronize the LR from global_step.
                self.scheduler.step()
                return self.last_grad_norm
        cpu_clip_coefficient = (
            float(clip_coefficient.item()) if self.offload_master else None
        )
        with torch.no_grad():
            for p, mp in zip(self.model_params, self.fp32_params):
                if p.grad is None:
                    mp.grad = None
                    continue
                master_grad = p.grad.detach().to(
                    device=mp.device,
                    dtype=torch.float32,
                )
                master_grad.mul_(
                    cpu_clip_coefficient
                    if cpu_clip_coefficient is not None
                    else clip_coefficient
                )
                mp.grad = master_grad
        self.last_grad_norm = grad_norm.detach()
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.scheduler.step()
        with torch.no_grad():
            for p, mp in zip(self.model_params, self.fp32_params):
                p.data.copy_(mp.data.to(device=p.device, dtype=p.dtype))
                p.grad = None
        return self.last_grad_norm

    def load_state_dict(self, state_dict):
        """Restore optimizer/scheduler state and, when present, the rank-local
        fp32 master params; without them the masters are re-cloned from the
        bf16 weights and the resume is not numerically faithful."""
        saved_max_grad_norm = state_dict.get("max_grad_norm")
        if saved_max_grad_norm is not None and float(saved_max_grad_norm) != float(
            self.max_grad_norm
        ):
            raise ValueError(
                "checkpoint optimizer used max_grad_norm="
                f"{saved_max_grad_norm} but this run has "
                f"max_grad_norm={self.max_grad_norm}"
            )
        # offload_master is a pure device-placement choice: restored fp32
        # masters and Adam moments are relocated to the current master device,
        # so toggling it on resume is safe and intentionally not gated here.
        self.optimizer.load_state_dict(state_dict["optimizer_state_dict"])
        print_on_rank0("Successfully loaded optimizer state_dict.")
        self.scheduler.load_state_dict(state_dict["scheduler_state_dict"])
        print_on_rank0("Successfully loaded scheduler state_dict.")
        # .get(): checkpoints written before the guard existed carry no entry,
        # and must stay loadable. Their estimate simply starts cold and
        # re-warms; the guard is idle until it has one either way.
        self.spike_guard.load_state_dict(state_dict.get("grad_spike_guard") or {})
        saved_fp32 = state_dict.get("fp32_params")
        if saved_fp32 is not None:
            if len(saved_fp32) != len(self.fp32_params):
                raise ValueError(
                    f"checkpoint carries {len(saved_fp32)} fp32 master params "
                    f"but this rank has {len(self.fp32_params)}"
                )
            with torch.no_grad():
                for i, (saved, mp) in enumerate(zip(saved_fp32, self.fp32_params)):
                    if saved.shape != mp.shape:
                        raise ValueError(
                            f"fp32 master param {i} shape mismatch: checkpoint "
                            f"{tuple(saved.shape)} vs current {tuple(mp.shape)}"
                        )
                    mp.data.copy_(saved.to(mp.device, mp.dtype))
        else:
            logger.warning(
                "checkpoint has no fp32_params; re-cloning master params from "
                "bf16 weights — resume will not be numerically faithful"
            )
            with torch.no_grad():
                for p, mp in zip(self.model_params, self.fp32_params):
                    mp.data.copy_(p.detach().to(device=mp.device, dtype=mp.dtype))

    def state_dict(self):
        return {
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "max_grad_norm": self.max_grad_norm,
            # rank-local fp32 masters; without them a resume re-quantizes from bf16
            "fp32_params": [t.detach().cpu() for t in self.fp32_params],
            "grad_spike_guard": self.spike_guard.state_dict(),
        }

    def get_learning_rate(self):
        return self.optimizer.param_groups[0]["lr"]

    def get_diagnostics(self):
        """Optimizer-owned scalars for the training log (empty when idle)."""
        return self.spike_guard.metrics()
