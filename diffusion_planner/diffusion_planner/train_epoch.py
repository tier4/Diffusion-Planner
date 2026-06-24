import numpy as np
import torch
import wandb
from torch import nn
from tqdm import tqdm

from diffusion_planner.model.module.decoder import compute_training_loss
from diffusion_planner.utils import ddp
from diffusion_planner.utils.data_augmentation import StatePerturbation
from diffusion_planner.utils.train_utils import get_epoch_mean_loss


def heading_to_cos_sin(x):
    """
    Convert heading angle to cosine and sine.
    Args:
        x: [B, T, 3] where last dimension is (x, y, heading)
    Output:
        x: [B, T, 4] where last dimension is (x, y, cos(heading), sin(heading))

    Idempotent: a [..., 4] input that is already (x, y, cos, sin) is returned
    unchanged. This guards against double-conversion (cos(cos)) now that scene-gen
    emits 4-col futures — callers can hand it either layout safely.
    """
    if x.shape[-1] == 4:
        return x
    return torch.cat(
        [
            x[..., :2],
            x[..., 2:3].cos(),
            x[..., 2:3].sin(),
        ],
        dim=-1,
    )


def _prep_inputs(inputs, args, aug):
    """Shared per-frame prep: heading->cos/sin, optional augmentation, normalise.
    Returns (inputs, futures=(ego_future, neighbors_future, mask))."""
    inputs["ego_agent_past"] = heading_to_cos_sin(inputs["ego_agent_past"])
    inputs["goal_pose"] = heading_to_cos_sin(inputs["goal_pose"])
    ego_future = inputs["ego_agent_future"]
    neighbors_future = inputs["neighbor_agents_future"]
    if aug is not None:
        inputs, ego_future, neighbors_future = aug(inputs, ego_future, neighbors_future)
    ego_future = heading_to_cos_sin(ego_future)
    mask = torch.sum(torch.ne(neighbors_future[..., :3], 0), dim=-1) == 0
    neighbors_future = heading_to_cos_sin(neighbors_future)
    neighbors_future[mask] = 0.0
    inputs = args.observation_normalizer(inputs)
    return inputs, (ego_future, neighbors_future, mask)


def _propagate_plan(plan_t, rp, rh, g, T):
    """frame_t plan [T,4]=(x,y,cos,sin) in frame-t -> prior [T,4] in frame-(t+g).
    Overlap (steps g:) via R(-theta)(q-p); tail (last g steps, no prior info) =
    constant-velocity extrapolation. Mirrors scratchpad/sdedit_probe.py::propagate."""
    L = T - g
    th = float(rh); c, s = np.cos(th), np.sin(th); p = np.asarray(rp).reshape(2)
    a = plan_t[g:g + L]
    d = a[:, :2] - p
    x = c * d[:, 0] + s * d[:, 1]
    y = -s * d[:, 0] + c * d[:, 1]
    phi = np.arctan2(a[:, 3], a[:, 2]) - th
    pri = np.zeros((T, 4), dtype=np.float32)
    pri[:L, 0] = x; pri[:L, 1] = y; pri[:L, 2] = np.cos(phi); pri[:L, 3] = np.sin(phi)
    if L >= 2:
        v = pri[L - 1, :2] - pri[L - 2, :2]
        for i in range(L, T):
            pri[i, :2] = pri[L - 1, :2] + (i - L + 1) * v
            pri[i, 2:] = pri[L - 1, 2:]
    elif L >= 1:
        pri[L:] = pri[L - 1]
    return pri


def _build_sdedit_prior(ego_a, rel_pos, rel_h, fut_b, args):
    """Propagate frame_t's plan (ego_a, [B,T,4] un-normalized) into frame_{t+g} and
    normalize into the x_start space. Returns [B,P,1+T,4]; only the ego future rows
    ([:,0,1:,:], the part the training hook reads) are meaningful — they hold the
    normalized propagated prior, in the SAME space as norm(gt_future)."""
    B, T, _ = ego_a.shape
    g = int(args.tc_step_g)
    ego_np = ego_a.detach().float().cpu().numpy()
    rp_np = rel_pos.detach().float().cpu().numpy().reshape(B, -1)
    rh_np = rel_h.detach().float().cpu().numpy().reshape(B)
    pri = np.stack([_propagate_plan(ego_np[i], rp_np[i], rh_np[i], g, T) for i in range(B)])
    pri_t = torch.as_tensor(pri, device=ego_a.device, dtype=ego_a.dtype)  # [B,T,4]
    _, nbr_fut, _ = fut_b
    P = 1 + nbr_fut.shape[1]
    prior_future = torch.zeros(B, P, T, 4, device=ego_a.device, dtype=ego_a.dtype)
    prior_future[:, 0] = pri_t
    prior_norm = args.state_normalizer(prior_future)  # matches all_gt's norm(gt_future)
    prior_tensor = torch.zeros(B, P, T + 1, 4, device=ego_a.device, dtype=ego_a.dtype)
    prior_tensor[:, :, 1:, :] = prior_norm
    return prior_tensor


def _temporal_consistency_term(batch, model, args, aug):
    """Paired-batch (frame_t + frame_{t+g}) temporal training. Two modes:

    - args.sdedit_train: SDEdit-aware training. frame_t gets STANDARD planning loss
      (base capability + the cold-start / no-prior case); frame_{t+g}'s ego diffusion
      input is noised from frame_t's PROPAGATED plan (the prior) while supervised toward
      GT — teaching the model to correct a prior-init toward the scene-appropriate plan
      (keep it where the scene is unchanged, fix it where it changed).
    - else: the cross-frame consistency-loss A/B (frame_t planning + a soft consistency
      term between near-clean predictions of both frames).

    STAGED BACKWARD throughout: each planning loss is backwarded immediately so its graph
    frees before the next forward (bounds peak memory to one graph at 64/GPU; the two
    backwards accumulate into .grad and DDP all-reduces each, identical to one summed
    backward). Returns a DETACHED loss dict; the caller must NOT backward again.
    """
    from planner_metrics.replan_consistency import temporal_consistency_loss

    dev = args.device
    rel_pos = batch.pop("tc_rel_pos").to(dev)
    rel_h = batch.pop("tc_rel_h").to(dev)
    a = {k: v.to(dev) for k, v in batch.items() if not k.startswith("b__")}
    b = {k[len("b__"):]: v.to(dev) for k, v in batch.items() if k.startswith("b__")}

    inp_a, fut_a = _prep_inputs(a, args, aug)
    inp_b, fut_b = _prep_inputs(b, args, aug)

    def planning_total(loss):
        return (
            args.alpha_neighbor_loss * loss["neighbor_prediction_loss"]
            + args.alpha_planning_loss * loss["ego_planning_loss"]
            + loss["turn_indicator_loss"]
            + args.coeff_road_border_loss * loss["road_border_loss"]
            + args.coeff_neighbor_collision_loss * loss["neighbor_collision_loss"]
        )

    if getattr(args, "history_cond_train", False):
        # HISTORY-CONTEXT conditioning (pivot after prior-conditioning = clean negative: feeding
        # the previous PLAN reduces flicker only by COPYING). Here we feed the previous frame's
        # POOLED SCENE ENCODING (temporal context, NO trajectory to echo) to frame_{t+g}.
        #   frame_t   : standard planning loss (anchor) + provides its pooled encoding as hist_ctx.
        #   frame_{t+g}: planning with hist_ctx fed as cross-attention tokens (50% dropout) +
        #               optional consistency loss (coeff>0) as the incentive to USE the context
        #               for stability. The model must PRODUCE the plan from context (can't copy a
        #               plan) -> the hope is real stability without copying. Detached hist_ctx
        #               (frame_t's encoder is trained by its own anchor, not the history path).
        m = model.module if hasattr(model, "module") else model
        with torch.no_grad():
            hist_ctx = m.encoder(inp_a).mean(dim=1)  # [B, D] frame_t pooled scene encoding
        loss_a = compute_training_loss(model, inp_a, fut_a, args)
        loss_a.pop("ego_pred_world", None)
        total_a = planning_total(loss_a)
        total_a.backward()
        Bsz = hist_ctx.shape[0]
        keep = torch.rand(Bsz, device=dev) < 0.5  # 50% dropout (anti-over-reliance + context-free capability)
        inp_b["hist_ctx"] = hist_ctx
        inp_b["hist_keep"] = keep
        loss_b = compute_training_loss(model, inp_b, fut_b, args)
        loss_b.pop("ego_pred_world", None)
        total_b = planning_total(loss_b)
        total_b.backward()
        cons_val = torch.tensor(0.0, device=dev)
        if args.coeff_temporal_consistency > 0 and bool(keep.any()):
            args._return_ego_pred_world = True
            args._fixed_diffusion_t = args.tc_fixed_t
            with torch.no_grad():
                ego_a = compute_training_loss(model, inp_a, fut_a, args)["ego_pred_world"]
            ego_b_clean = compute_training_loss(model, inp_b, fut_b, args)["ego_pred_world"]
            args._fixed_diffusion_t = None
            cons = temporal_consistency_loss(
                ego_a[keep], ego_b_clean[keep], args.tc_step_g, rel_pos[keep], rel_h[keep],
                w_heading=args.tc_w_heading, stop_grad_a=True,
            ) / args.tc_cons_scale
            (args.coeff_temporal_consistency * cons).backward()
            cons_val = cons.detach()
            if not getattr(args, "_hist_sanity_done", False):
                print(f"[hist-sanity] cons={float(cons_val):.4f} kept={int(keep.sum())}/{Bsz} "
                      f"hist_ctx={tuple(hist_ctx.shape)} (history-context, no plan to copy)", flush=True)
                args._hist_sanity_done = True
        inp_b.pop("hist_ctx", None)
        inp_b.pop("hist_keep", None)
        out = {kk: (vv.detach() if torch.is_tensor(vv) else vv) for kk, vv in loss_b.items()}
        out["temporal_consistency_loss"] = cons_val
        out["loss"] = total_a.detach() + total_b.detach() + args.coeff_temporal_consistency * cons_val
        return out

    if getattr(args, "prior_cond_train", False):
        # SYNTHESIS = explicit prior-conditioning INPUT + consistency LOSS.
        #   frame_t   : STANDARD planning loss, no prior (the no-prior anchor; base capability)
        #               + its clean (fixed-t, no-grad) plan -> propagated -> the prior.
        #   frame_{t+g}: (a) planning loss with the prior fed as cross-attention tokens, 50%
        #               per-sample dropout (anti-copy + prior-free capability);
        #               (b) on the KEPT (prior-given) samples, a clean (fixed-t) forward WITH
        #               the prior -> a CONSISTENCY loss vs frame_t's propagated clean plan,
        #               coeff_temporal_consistency-weighted. (a) keeps it accurate + un-reliant,
        #               (b) is the INCENTIVE to actually USE the prior for stability.
        # The conditioning is proven not to copy (Step 2); the loss makes it get used.
        args._return_ego_pred_world = True
        args._fixed_diffusion_t = args.tc_fixed_t
        with torch.no_grad():
            ego_a = compute_training_loss(model, inp_a, fut_a, args)["ego_pred_world"]
        args._fixed_diffusion_t = None
        # [B,T,4] normalized ego-future prior (ego row of the propagated, normalized plan)
        prior_traj = _build_sdedit_prior(ego_a, rel_pos, rel_h, fut_b, args)[:, 0, 1:, :]
        loss_a = compute_training_loss(model, inp_a, fut_a, args)
        loss_a.pop("ego_pred_world", None)
        total_a = planning_total(loss_a)
        total_a.backward()
        Bsz = prior_traj.shape[0]
        keep = torch.rand(Bsz, device=dev) < 0.5  # 50% keep -> strong dropout
        inp_b["prior_traj"] = prior_traj
        inp_b["prior_keep"] = keep
        # (a) random-t planning loss with the prior conditioning (dropout applied in-model)
        loss_b = compute_training_loss(model, inp_b, fut_b, args)
        loss_b.pop("ego_pred_world", None)
        total_b = planning_total(loss_b)
        total_b.backward()
        # (b) consistency loss on the KEPT (prior-given) samples: clean (fixed-t) forward of
        # frame_{t+g} WITH the prior -> align to frame_t's propagated clean plan.
        cons_val = torch.tensor(0.0, device=dev)
        if args.coeff_temporal_consistency > 0 and bool(keep.any()):
            args._return_ego_pred_world = True
            args._fixed_diffusion_t = args.tc_fixed_t
            ego_b_clean = compute_training_loss(model, inp_b, fut_b, args)["ego_pred_world"]
            args._fixed_diffusion_t = None
            # Scene-aware gate (tc_scene_gate): weight each kept sample's consistency by how
            # STALE the prior is vs the new GT. gt_dev = ||propagated previous PLAN - GT_{t+g}||
            # (ego_a, the prior source, vs frame_{t+g}'s GT). NB: GT-vs-GT is structurally ~0
            # (GT_t's overlap IS the same physical trajectory as GT_{t+g}), so we measure the
            # MODEL's frame_t plan, not GT_t. gt_dev has a non-zero normal baseline (~the model's
            # typical plan-vs-GT gap), so a plain exp(-gt_dev/tau) over-gates everything; instead
            # we use an ADAPTIVE FLOOR = the batch quantile tc_gate_q of gt_dev: frames at/below
            # the normal error keep w=1 (full consistency, stability), and only the stale-prior
            # tail decays w=exp(-(gt_dev-floor)/tau) (so GT wins => responsiveness). Decouples the
            # flicker-gain/copying trade-off the global coeff could only slide along (synthesis
            # @coeff0.5: flk -34% but responsiveness 0.29 vs base 0.62). Default-off => uniform.
            sw = None
            if getattr(args, "tc_scene_gate", False):
                with torch.no_grad():
                    gt_dev = temporal_consistency_loss(
                        ego_a[keep], fut_b[0][keep], args.tc_step_g, rel_pos[keep], rel_h[keep],
                        w_heading=args.tc_w_heading, stop_grad_a=True, per_sample=True,
                    )
                    d0 = torch.quantile(gt_dev, args.tc_gate_q)
                    sw = torch.exp(-torch.relu(gt_dev - d0) / args.tc_gate_tau)
            cons = temporal_consistency_loss(
                ego_a[keep], ego_b_clean[keep], args.tc_step_g, rel_pos[keep], rel_h[keep],
                w_heading=args.tc_w_heading, stop_grad_a=True, sample_weight=sw,
            ) / args.tc_cons_scale
            (args.coeff_temporal_consistency * cons).backward()
            cons_val = cons.detach()
            if not getattr(args, "_synth_sanity_done", False):  # one-time correctness sanity
                gate_msg = (f"scene_gate ON tau={args.tc_gate_tau} w_mean={float(sw.mean()):.3f}"
                            if sw is not None else "scene_gate OFF (uniform)")
                print(f"[synth-sanity] cons={float(cons_val):.4f} kept={int(keep.sum())}/{Bsz} "
                      f"{gate_msg}", flush=True)
                args._synth_sanity_done = True
        inp_b.pop("prior_traj", None)
        inp_b.pop("prior_keep", None)
        out = {kk: (vv.detach() if torch.is_tensor(vv) else vv) for kk, vv in loss_b.items()}
        out["temporal_consistency_loss"] = cons_val
        out["loss"] = total_a.detach() + total_b.detach() + args.coeff_temporal_consistency * cons_val
        return out

    if getattr(args, "sdedit_train", False):
        args._return_ego_pred_world = True
        args._sdedit_train_prior = None
        # frame_t clean plan (no-grad) -> the prior source for frame_{t+g}
        args._fixed_diffusion_t = args.tc_fixed_t
        with torch.no_grad():
            ego_a = compute_training_loss(model, inp_a, fut_a, args)["ego_pred_world"]
        args._fixed_diffusion_t = None
        prior = _build_sdedit_prior(ego_a, rel_pos, rel_h, fut_b, args)
        # frame_t STANDARD planning loss (no prior) -> backward
        loss_a = compute_training_loss(model, inp_a, fut_a, args)
        loss_a.pop("ego_pred_world", None)
        total_a = planning_total(loss_a)
        total_a.backward()
        # frame_{t+g} planning loss with ego diffusion input noised FROM THE PRIOR -> backward
        args._sdedit_train_prior = prior
        loss_b = compute_training_loss(model, inp_b, fut_b, args)
        loss_b.pop("ego_pred_world", None)
        args._sdedit_train_prior = None
        total_b = planning_total(loss_b)
        total_b.backward()
        out = {k: (v.detach() if torch.is_tensor(v) else v) for k, v in loss_b.items()}
        out["temporal_consistency_loss"] = torch.tensor(0.0, device=dev)
        out["loss"] = total_a.detach() + total_b.detach()
        return out

    # --- cross-frame consistency-loss path (A/B; default when sdedit_train is off) ---
    args._return_ego_pred_world = True
    args._fixed_diffusion_t = None
    loss = compute_training_loss(model, inp_a, fut_a, args)
    loss.pop("ego_pred_world", None)  # transient tensor; keep out of epoch-mean reduction
    pt = planning_total(loss)
    pt.backward()

    args._fixed_diffusion_t = args.tc_fixed_t
    with torch.no_grad():
        ego_a = compute_training_loss(model, inp_a, fut_a, args)["ego_pred_world"]
    ego_b = compute_training_loss(model, inp_b, fut_b, args)["ego_pred_world"]
    args._fixed_diffusion_t = None

    cons = temporal_consistency_loss(
        ego_a, ego_b, args.tc_step_g, rel_pos, rel_h,
        w_heading=args.tc_w_heading, stop_grad_a=True,
    ) / args.tc_cons_scale
    (args.coeff_temporal_consistency * cons).backward()

    out = {k: (v.detach() if torch.is_tensor(v) else v) for k, v in loss.items()}
    out["temporal_consistency_loss"] = cons.detach()
    out["loss"] = pt.detach() + args.coeff_temporal_consistency * cons.detach()
    return out


def train_epoch(data_loader, model, optimizer, args, ema, aug: StatePerturbation = None):
    epoch_loss = []
    tc_coeff = getattr(args, "coeff_temporal_consistency", 0.0)
    step_log = getattr(args, "wandb_step_log_interval", 0)  # 0 = off (per-epoch only)
    log_step = args.use_wandb and step_log > 0 and ddp.get_rank() == 0
    if not hasattr(args, "_wandb_global_step"):
        args._wandb_global_step = 0

    model.train()

    if args.ddp:
        torch.cuda.synchronize()

    if ddp.get_rank() == 0:
        data_loader = tqdm(data_loader, desc="Training", unit="batch")

    for inputs in data_loader:
        optimizer.zero_grad()

        if tc_coeff > 0:
            # paired consecutive-frame batch -> planning(frame_t) + temporal consistency.
            # Staged backward happens INSIDE (bounds peak memory); returns a detached loss
            # dict with loss["loss"] already populated — do NOT backward again here.
            loss = _temporal_consistency_term(inputs, model, args, aug)
        else:
            inputs = {key: value.to(args.device) for key, value in inputs.items()}
            inputs, futures = _prep_inputs(inputs, args, aug)
            loss = compute_training_loss(model, inputs, futures, args)
            loss["temporal_consistency_loss"] = torch.tensor(0.0, device=args.device)
            loss["loss"] = (
                args.alpha_neighbor_loss * loss["neighbor_prediction_loss"]
                + args.alpha_planning_loss * loss["ego_planning_loss"]
                + loss["turn_indicator_loss"]
                + args.coeff_road_border_loss * loss["road_border_loss"]
                + args.coeff_neighbor_collision_loss * loss["neighbor_collision_loss"]
                + getattr(args, "coeff_jepa_consistency_loss", 0.0)
                * loss["jepa_consistency_loss"]
                + tc_coeff * loss["temporal_consistency_loss"]
            )
            loss["loss"].backward()

        nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()

        ema.update(model)

        args._wandb_global_step += 1
        if log_step and args._wandb_global_step % step_log == 0:
            # commit=True + auto-step => each call flushes immediately (avoids the
            # explicit-step buffering that hides per-epoch logs until the next epoch).
            wandb.log(
                {
                    "train_step/loss": loss["loss"].item(),
                    "train_step/ego_planning_loss": loss["ego_planning_loss"].item(),
                    "train_step/neighbor_prediction_loss": loss["neighbor_prediction_loss"].item(),
                    "train_step/temporal_consistency_loss": loss["temporal_consistency_loss"].item(),
                    "train_step/global_step": args._wandb_global_step,
                },
                commit=True,
            )

        if args.ddp:
            torch.cuda.synchronize()
        epoch_loss.append(loss)

    epoch_mean_loss = get_epoch_mean_loss(epoch_loss)

    if args.ddp:
        epoch_mean_loss = ddp.reduce_and_average_losses(epoch_mean_loss, torch.device(args.device))

    if ddp.get_rank() == 0:
        print(f"{epoch_mean_loss['loss']=:.4f}")
        print(f"{epoch_mean_loss['turn_indicator_accuracy']=:.4f}")

    return epoch_mean_loss, epoch_mean_loss["loss"]
