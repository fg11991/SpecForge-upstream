import logging
import math
import os

import torch
import torch.distributed as dist

from specforge.lr_scheduler import ConstantWarmupLR, CosineAnnealingWarmupLR
from specforge.utils import print_on_rank0

logger = logging.getLogger(__name__)


def _nan_diagnostics_enabled() -> bool:
    """Whether to spend device syncs explaining a non-finite gradient norm.

    Off by default: the check needs the norm on the host every step, and the
    optimizer deliberately keeps its only sync behind the spike guard. It rides
    on the same switch as anomaly detection, since both answer "the loss is
    finite but the gradients are not".
    """
    from specforge.training.backend import detect_anomaly_enabled

    return detect_anomaly_enabled()

def _rank_local_parameter_ids(model) -> frozenset:
    """Ids of parameters each rank owns a disjoint slice of.

    A module opts its own parameters in by setting
    ``_specforge_rank_local_parameters = True``; everything else in the model
    is treated as replicated with identical gradients on every rank.
    """
    if model is None:
        return frozenset()
    ids = set()
    for module in model.modules():
        if getattr(module, "_specforge_rank_local_parameters", False):
            ids.update(id(parameter) for parameter in module.parameters())
    return frozenset(ids)


# AdamW's multi-tensor path allocates one whole-list temporary per operation:
# _foreach_sqrt copies every exp_avg_sq, then the divide and add copy again.
# On the DSpark drafter that is 9.2 GiB per temporary against 3.8 GiB of
# headroom, and it is where the on-device optimizer died -- inside the Adam
# math, after every master and moment had already been allocated. The
# single-tensor path bounds temporaries at one parameter, 33 MiB here, and
# measured faster on CPU as well (139 ms against 177 ms for 210 M parameters),
# so nothing is traded for it. PyTorch already defaults CPU tensors to this
# path; only the accelerator list opts in.
_ADAMW_FOREACH = False


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
    clipping and configurable warmup scheduling."""

    def __init__(
        self,
        model,
        lr,
        weight_decay=0.0,
        max_grad_norm=0.5,
        total_steps=800_000,
        warmup_ratio=0.015,
        lr_scheduler="cosine",
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
            self.fp32_params,
            lr=lr,
            weight_decay=weight_decay,
            betas=tuple(betas),
            foreach=_ADAMW_FOREACH,
        )
        self.last_grad_norm = None
        self._grad_norm_process_group = None
        self._reduce_grad_norm_across_ranks = True
        self._partition_replicated = False
        self._rank_local_param_ids = frozenset()
        # fp32_params[k] is the master for model_params[self._master_index[k]].
        # They are one-to-one until expert-parallel state sharding drops the
        # replicated parameters this rank does not own.
        self._master_index = list(range(len(self.model_params)))
        self._replicated_owner = {}
        # 256 Mi elements is 512 MiB of bf16 or 1 GiB of fp32 in flight, small
        # against the drafter's footprint and large enough that a bucket is
        # bandwidth-bound rather than latency-bound.
        self._transfer_bucket_elements = 256 * 1024 * 1024
        scheduler_types = {
            "constant": ConstantWarmupLR,
            "cosine": CosineAnnealingWarmupLR,
        }
        if lr_scheduler not in scheduler_types:
            raise ValueError(
                f"unsupported lr_scheduler={lr_scheduler!r}; "
                f"expected one of {sorted(scheduler_types)}"
            )
        self.lr_scheduler_type = lr_scheduler
        # Kept so the scheduler can be rebuilt alongside the optimizer when
        # expert-parallel state sharding replaces the parameter group.
        self._scheduler_types = scheduler_types
        self._scheduler_total_steps = total_steps
        self._scheduler_warmup_steps = int(warmup_ratio * total_steps)
        self.scheduler = self._build_scheduler()

    def _build_scheduler(self):
        return self._scheduler_types[self.lr_scheduler_type](
            self.optimizer,
            total_steps=self._scheduler_total_steps,
            warmup_steps=self._scheduler_warmup_steps,
        )

    def configure_grad_norm_reduction(
        self, *, process_group=None, enabled: bool = True,
        partition_replicated: bool = False,
    ) -> None:
        """Configure the group that owns disjoint gradient shards.

        FSDP backends disable the reduction for replicated/NO_SHARD parameters.

        ``partition_replicated`` says the group holds a *mix*: some parameters
        are rank-local (each rank owns a disjoint slice) and the rest are
        replicated with identical gradients on every rank.  Expert parallelism
        is exactly that -- summing every square over the group would count the
        replicated parameters once per rank and inflate the norm by up to
        ``sqrt(group_size)``.  Modules mark their rank-local parameters by
        setting ``_specforge_rank_local_parameters = True`` on themselves.
        """
        self._grad_norm_process_group = process_group
        self._reduce_grad_norm_across_ranks = enabled
        self._partition_replicated = partition_replicated
        self._rank_local_param_ids = (
            _rank_local_parameter_ids(self.model) if partition_replicated else frozenset()
        )
        if partition_replicated:
            self._shard_replicated_state()

    def _shard_replicated_state(self) -> None:
        """Keep master and Adam state for only this rank's share of the replicas.

        Under expert parallelism the routed experts are already disjoint, but
        every rank holds an identical copy of the attention, mHC, router and
        shared-expert parameters -- and an identical copy of their fp32 master
        and both Adam moments, which it updates identically. Assigning each
        replicated parameter to one rank recovers ``(group - 1) / group`` of
        that state; the owner broadcasts the updated weights after each step so
        every rank ends the step with the same parameters it would have had.

        The split is by parameter count, greedy and deterministic, so every rank
        computes the same assignment without communicating.
        """
        if not (dist.is_available() and dist.is_initialized()):
            return
        try:
            group_size = dist.get_world_size(self._grad_norm_process_group)
            group_rank = dist.get_rank(self._grad_norm_process_group)
        except Exception:
            return
        if group_size <= 1:
            return

        local_ids = self._rank_local_param_ids
        replicated = [
            index
            for index, parameter in enumerate(self.model_params)
            if id(parameter) not in local_ids
        ]
        if not replicated:
            return
        load = [0] * group_size
        owner = {}
        # Largest first so the greedy assignment balances instead of trailing
        # one huge tensor onto whichever rank happens to come last.
        for index in sorted(
            replicated, key=lambda i: self.model_params[i].numel(), reverse=True
        ):
            target = min(range(group_size), key=lambda r: (load[r], r))
            owner[index] = target
            load[target] += self.model_params[index].numel()

        keep = sorted(
            index
            for index in range(len(self.model_params))
            if index not in owner or owner[index] == group_rank
        )
        self._replicated_owner = owner
        self._rebuild_masters(keep)

    def _rebuild_masters(self, keep) -> None:
        """Restrict the fp32 masters to ``keep`` and rebuild AdamW over them.

        Safe only before the first step: AdamW state is empty at configure time,
        so nothing is discarded by replacing the parameter group.
        """
        self._master_index = list(keep)
        self.fp32_params = [
            (
                self.model_params[index]
                .detach()
                .to(device="cpu", dtype=torch.float32)
                .clone()
                if self.offload_master
                else self.model_params[index].detach().clone().to(torch.float32)
            )
            for index in self._master_index
        ]
        for master in self.fp32_params:
            master.requires_grad = True
        defaults = self.optimizer.defaults
        self.optimizer = torch.optim.AdamW(
            self.fp32_params,
            lr=defaults["lr"],
            weight_decay=defaults["weight_decay"],
            betas=defaults["betas"],
            foreach=_ADAMW_FOREACH,
        )
        # Rebuild rather than re-point: the scheduler wrote initial_lr into the
        # old parameter groups and applied its first value there, so a swapped
        # optimizer would start at the raw constructor learning rate and be one
        # warmup step out for the whole run.
        self.scheduler = self._build_scheduler()

    def _broadcast_replicated(self) -> None:
        """Publish each replicated parameter from the rank that updated it."""
        if not self._replicated_owner:
            return
        group = self._grad_norm_process_group
        by_owner = {}
        for index, rank in self._replicated_owner.items():
            by_owner.setdefault(rank, []).append(index)
        for rank in sorted(by_owner):
            indices = sorted(by_owner[rank])
            flat = torch.cat(
                [self.model_params[index].data.reshape(-1) for index in indices]
            )
            source = rank
            resolve = getattr(dist, "get_global_rank", None)
            if callable(resolve) and group is not None:
                try:
                    source = resolve(group, rank)
                except Exception:
                    source = rank
            # One collective per owner rather than one per parameter; every rank
            # issues them in the same order, which the collective requires.
            dist.broadcast(flat, src=source, group=group)
            offset = 0
            for index in indices:
                parameter = self.model_params[index]
                count = parameter.numel()
                parameter.data.copy_(
                    flat[offset : offset + count].view_as(parameter)
                )
                offset += count

    def _reduce_grad_norm(self, total_norm_sq, replicated_norm_sq=None):
        """All-reduce the squared L2 norm across shard ranks and derive the
        clip coefficient.

        ``total_norm_sq`` must already live on a device the process group can
        reduce (e.g. CUDA for NCCL). When ``replicated_norm_sq`` is given it
        carries the squares of the parameters every rank holds a copy of; those
        are summed and divided back by the group size so they contribute
        exactly once, while ``total_norm_sq`` (the rank-local slices) is summed
        normally. Returns ``(total_norm, clip_coef)``.
        """
        if (
            self._reduce_grad_norm_across_ranks
            and dist.is_available()
            and dist.is_initialized()
        ):
            if replicated_norm_sq is None:
                dist.all_reduce(
                    total_norm_sq,
                    op=dist.ReduceOp.SUM,
                    group=self._grad_norm_process_group,
                )
            else:
                # One collective for both halves keeps the collective count and
                # ordering identical to the un-partitioned path.
                packed = torch.stack([total_norm_sq, replicated_norm_sq])
                dist.all_reduce(
                    packed, op=dist.ReduceOp.SUM,
                    group=self._grad_norm_process_group,
                )
                group_size = dist.get_world_size(self._grad_norm_process_group)
                total_norm_sq = packed[0] + packed[1] / group_size
        elif replicated_norm_sq is not None:
            total_norm_sq = total_norm_sq + replicated_norm_sq
        total_norm = total_norm_sq.sqrt()
        clip_coef = torch.clamp(self.max_grad_norm / (total_norm + 1e-6), max=1.0)
        return total_norm, clip_coef

    def _grad_norm_and_clip_coefficient(self):
        """Compute the global grad norm from the model params on their own
        device, where NCCL can reduce it safely, without materialising master
        gradients first."""
        device = self.model_params[0].device if self.model_params else "cpu"
        zero = torch.zeros((), dtype=torch.float32, device=device)
        local_ids = self._rank_local_param_ids
        rank_local, replicated = [], []
        for parameter in self.model_params:
            if parameter.grad is None:
                continue
            square = parameter.grad.detach().float().square().sum()
            if not local_ids or id(parameter) in local_ids:
                rank_local.append(square)
            else:
                replicated.append(square)
        total_norm_sq = torch.stack(rank_local).sum() if rank_local else zero
        replicated_norm_sq = (
            torch.stack(replicated).sum() if replicated else (zero if local_ids else None)
        )
        # Keep the pre-reduction value: when the reduced norm comes back
        # non-finite, it is the one measurement that says whether this rank's
        # own gradients were already bad or the collective produced it.
        self._last_local_norm_sq = (
            total_norm_sq
            if replicated_norm_sq is None
            else total_norm_sq + replicated_norm_sq
        ).detach().clone()
        return self._reduce_grad_norm(total_norm_sq, replicated_norm_sq)

    def diagnose_non_finite_norm(self, total_norm) -> str:
        """Describe a non-finite gradient norm well enough to act on it.

        Answers the only question that matters first: were this rank's own
        gradients already non-finite, or did the cross-rank reduction produce
        the NaN?  Costs several device syncs and is called only once the run is
        already broken.
        """
        local_sq = getattr(self, "_last_local_norm_sq", None)
        local_text = "unknown"
        if local_sq is not None:
            local_value = float(local_sq.item())
            local_text = (
                f"{local_value:.6g}"
                if math.isfinite(local_value)
                else f"NON-FINITE ({local_value})"
            )
        bad = []
        total = 0
        for index, parameter in enumerate(self.model_params):
            if parameter.grad is None:
                continue
            total += 1
            if not bool(torch.isfinite(parameter.grad).all().item()):
                bad.append(index)
        return (
            f"grad_norm={float(total_norm):.6g} is not finite; "
            f"local_norm_sq={local_text}; "
            f"{len(bad)} of {total} parameter gradients non-finite "
            f"(first indices {bad[:8]}); "
            f"reduction_enabled={self._reduce_grad_norm_across_ranks}"
        )

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
        from specforge.training.step_profile import phase

        with phase("grad_norm"):
            grad_norm, clip_coefficient = self._grad_norm_and_clip_coefficient()
        if _nan_diagnostics_enabled():
            value = float(grad_norm.item())
            if not math.isfinite(value):
                logger.error("%s", self.diagnose_non_finite_norm(value))
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
        from specforge.training.step_profile import phase

        cpu_clip_coefficient = (
            float(clip_coefficient.item()) if self.offload_master else None
        )
        coefficient = (
            cpu_clip_coefficient
            if cpu_clip_coefficient is not None
            else clip_coefficient
        )
        with phase("grads_to_master"), torch.no_grad():
            self._gradients_to_master(coefficient)
        self.last_grad_norm = grad_norm.detach()
        with phase("adamw"):
            self.optimizer.step()
            self.optimizer.zero_grad()
        self.scheduler.step()
        with phase("params_to_device"), torch.no_grad():
            self._master_to_parameters()
        return self.last_grad_norm

    def _mastered(self):
        """Iterate ``(model parameter, its fp32 master)`` for the masters kept."""
        for position, index in enumerate(self._master_index):
            yield self.model_params[index], self.fp32_params[position]

    def _transfer_buckets(self):
        """Group master positions into same-signature runs under a byte budget.

        Offload turns each parameter into a separate cross-device copy, and the
        routed experts make that 308 of them per step -- 9.5 s of a 37 s
        optimizer step, nearly all of it per-transfer latency rather than
        bandwidth. Concatenating into a few large transfers pays one latency per
        bucket instead. The budget bounds the temporary the concatenation needs.
        """
        buckets, current, current_elements, signature = [], [], 0, None
        for position, (p, mp) in enumerate(self._mastered()):
            key = (p.device, p.dtype, mp.device)
            if key != signature or current_elements >= self._transfer_bucket_elements:
                if current:
                    buckets.append(current)
                current, current_elements, signature = [], 0, key
            current.append(position)
            current_elements += p.numel()
        if current:
            buckets.append(current)
        return buckets

    def _gradients_to_master(self, coefficient) -> None:
        if not self.offload_master:
            # Same device: the copies are cheap and the concatenation buffer
            # would be pure overhead.
            for p, mp in self._mastered():
                if p.grad is None:
                    mp.grad = None
                    continue
                master_grad = p.grad.detach().to(device=mp.device, dtype=torch.float32)
                master_grad.mul_(coefficient)
                mp.grad = master_grad
            return
        for bucket in self._transfer_buckets():
            present = [
                position
                for position in bucket
                if self.model_params[self._master_index[position]].grad is not None
            ]
            for position in bucket:
                if self.model_params[self._master_index[position]].grad is None:
                    self.fp32_params[position].grad = None
            if not present:
                continue
            flat = torch.cat(
                [
                    self.model_params[self._master_index[position]]
                    .grad.detach()
                    .reshape(-1)
                    for position in present
                ]
            )
            # Move at the parameter dtype and widen on the host: casting first
            # would double the bytes crossing the link for no gain, since
            # bf16 -> fp32 is exact either way.
            master = flat.to(device=self.fp32_params[present[0]].device).to(torch.float32)
            master.mul_(coefficient)
            offset = 0
            for position in present:
                master_parameter = self.fp32_params[position]
                count = master_parameter.numel()
                master_parameter.grad = master[offset : offset + count].view_as(
                    master_parameter
                )
                offset += count

    def _master_to_parameters(self) -> None:
        """Write the updated masters back, one parameter at a time.

        Deliberately not bucketed. The gradient direction concatenates on the
        device, where the copy runs at device bandwidth and buys a 3x cut in
        transfer latency. This direction would have to concatenate on the host:
        2.9 B fp32 masters is an 11.7 GiB temporary plus a bf16 cast, and
        measuring it made this phase worse -- 0.8-1.4 s per step became
        2.2-4.6 s. Per-tensor casts write the same bytes without the temporary.
        """
        for p, mp in self._mastered():
            p.data.copy_(mp.data.to(device=p.device, dtype=p.dtype))
            p.grad = None
        # Parameters this rank does not own were not updated here; their owner
        # publishes them. Ranks that own nothing replicated still take part --
        # the collective requires every rank in the group.
        self._broadcast_replicated()
        for parameter in self.model_params:
            parameter.grad = None

    def load_state_dict(self, state_dict):
        """Restore optimizer/scheduler state and, when present, the rank-local
        fp32 master params; without them the masters are re-cloned from the
        bf16 weights and the resume is not numerically faithful."""
        saved_scheduler_type = state_dict.get("lr_scheduler_type", "cosine")
        if saved_scheduler_type != self.lr_scheduler_type:
            raise ValueError(
                "checkpoint optimizer used lr_scheduler="
                f"{saved_scheduler_type!r} but this run has "
                f"lr_scheduler={self.lr_scheduler_type!r}"
            )
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
                for p, mp in self._mastered():
                    mp.data.copy_(p.detach().to(device=mp.device, dtype=mp.dtype))

    def state_dict(self):
        return {
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "lr_scheduler_type": self.lr_scheduler_type,
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
