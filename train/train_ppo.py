from __future__ import annotations
import os
import time
import csv
import traceback
from dataclasses import dataclass
from typing import List, Dict, Tuple

import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

# === plotting (자동 저장) ===
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_OK = True
except Exception as _e:
    MATPLOTLIB_OK = False

# --- add project root to sys.path (for direct script execution) ---
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent  # .../projects
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
# -----------------------------------------------------------------


# Local imports
from envs.container_sim import PackingEnv, EnvConfig
from utils.preprocess import compute_plane_features, downsample_patches, flatten_for_encoder
from utils.logger import resolve_resume
from utils.rng_state import get_rng_state, set_rng_state
from utils.plotting import save_packing_3d
from agents.backbone import PolicyBackbone, EncoderConfig
from agents.heads import (
    PositionDecoder, SelectionDecoder, OrientationDecoder,
    PositionEmbeddingBuilder, OrientationEmbedder,
)
from agents.value_head import ValueNet

# ------------------------------------------------------------
# 디렉터리 안전가드
# ------------------------------------------------------------
os.makedirs(os.path.join("results", "ckpt"), exist_ok=True)
os.makedirs(os.path.join("results", "logs"), exist_ok=True)
os.makedirs(os.path.join("results", "plots"), exist_ok=True)

# ------------------------------------------------------------
# DEBUG 도우미
# ------------------------------------------------------------
DEBUG = True
def dprint(*args):
    if DEBUG:
        print(*args, flush=True)

# ------------------------------------------------------------
# Configs
# ------------------------------------------------------------

@dataclass
class TrainConfig:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    # PPO
    gamma: float = 0.99
    gae_lambda: float = 0.96
    ppo_clip: float = 0.12
    entropy_coef: float = 1e-3
    # LR
    lr_actor: float = 1e-5
    lr_critic: float = 1e-4
    # Rollout / Schedule
    num_updates: int = 2          # 데모값 (실험 시 늘리세요)
    epochs_per_update: int = 4
    batch_size: int = 1           # 단일 환경
    # Logging / Checkpoint
    ckpt_dir: str = os.path.join("results", "ckpt")
    log_interval: int = 1
    save_interval: int = 1        # train.yaml의 save_interval 반영 (기본 1)

# ------------------------------------------------------------
# YAML 로더 유틸
# ------------------------------------------------------------
def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

# ------------------------------------------------------------
# Policy wrapper (Actor)
# ------------------------------------------------------------

class PolicyNet(nn.Module):
    def __init__(self, d_model: int = 128):
        super().__init__()
        enc_cfg = EncoderConfig(d_model=d_model, nhead=4, num_layers=2)
        self.backbone = PolicyBackbone(enc_cfg)
        self.pos_dec = PositionDecoder(d_model=d_model, nhead=8, num_layers=2)
        self.pos_emb_builder = PositionEmbeddingBuilder(d_model=d_model)
        self.sel_dec = SelectionDecoder(d_model=d_model, nhead=8, num_layers=2)
        self.orient_embed = OrientationEmbedder(d_model=d_model)
        self.orient_dec = OrientationDecoder(d_model=d_model, nhead=8, num_layers=2)

    @staticmethod
    def _pos_index_to_xy(idx: torch.Tensor, patch: int = 10) -> torch.Tensor:
        # idx: (B,) in [0, 99]; map to (x,y) in container grid by *patch* (10)
        ix = (idx % 10) * patch
        iy = (idx // 10) * patch
        return torch.stack([ix, iy], dim=1)  # (B, 2)

    @staticmethod
    def _num_valid_orients(lwh: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """
        lwh: (B,3) for selected boxes
        Returns: (B,) in {1,3,6}
        rule:
          - all equal -> 1
          - any pair equal -> 3
          - all distinct -> 6
        """
        l, w, h = lwh[..., 0], lwh[..., 1], lwh[..., 2]
        eq_lw = torch.isclose(l, w, atol=eps)
        eq_wh = torch.isclose(w, h, atol=eps)
        eq_lh = torch.isclose(l, h, atol=eps)
        all_eq = eq_lw & eq_wh  # l==w==h
        any_pair = eq_lw | eq_wh | eq_lh
        out = torch.where(all_eq, torch.tensor(1, device=l.device),
              torch.where(any_pair, torch.tensor(3, device=l.device),
                          torch.tensor(6, device=l.device)))
        return out

    def forward_decode(self,
                       boxes: torch.Tensor,          # (B,N,3)
                       cont_flat: torch.Tensor,      # (B,100,7)
                       raw_flat: torch.Tensor,       # (B,100,7) same as cont_flat before proj
                       used_mask: torch.Tensor       # (B,N) bool
                       ) -> Dict[str, torch.Tensor]:
        # Encoders
        box_enc, cont_enc = self.backbone(boxes, cont_flat)  # (B,N,d), (B,100,d)

        # 1) Position
        logits_pos, ctx = self.pos_dec(cont_enc, box_enc)  # (B,100), (B,100,d)

        # 2) Sample position index
        dist_p = Categorical(logits=logits_pos)
        pos_idx = dist_p.sample()                   # (B,)
        pos_logp = F.log_softmax(logits_pos, dim=-1).gather(1, pos_idx.view(-1,1)).squeeze(1)
        pos_entropy = dist_p.entropy()

        # 3) Position embedding
        pos_emb = self.pos_emb_builder(ctx, raw_flat, pos_idx)   # (B,1,d)

        # 4) Selection (mask out used boxes)
        logits_sel, _ = self.sel_dec(box_enc, pos_emb)   # (B,N)
        logits_sel = logits_sel.masked_fill(used_mask, float('-inf'))
        dist_s = Categorical(logits=logits_sel)
        sel_idx = dist_s.sample()                 # (B,)
        sel_logp = F.log_softmax(logits_sel, dim=-1).gather(1, sel_idx.view(-1,1)).squeeze(1)
        sel_entropy = dist_s.entropy()

        # 5) Orientation for the selected box (with VALID-CLASS MASKING)
        B, N, _ = boxes.shape
        gather_idx = sel_idx.view(B,1,1).expand(-1,1,3)  # (B,1,3)
        picked_lwh = boxes.gather(1, gather_idx).squeeze(1)  # (B,3)
        orient_emb = self.orient_embed(picked_lwh)  # (B,6,d)
        logits_or, _ = self.orient_dec(orient_emb, pos_emb)  # (B,6)

        # ---- 유효 오리엔테이션 개수(1/3/6)로 마스킹 ----
        n_valid = self._num_valid_orients(picked_lwh)          # (B,)
        ar6 = torch.arange(6, device=logits_or.device).view(1, 6).expand(B, -1)
        valid_mask = ar6 < n_valid.unsqueeze(1)                 # True=valid
        logits_or = logits_or.masked_fill(~valid_mask, float('-inf'))

        dist_o = Categorical(logits=logits_or)
        orient_idx = dist_o.sample()                            # (B,)
        orient_logp = F.log_softmax(logits_or, dim=-1).gather(1, orient_idx.view(-1,1)).squeeze(1)
        orient_entropy = dist_o.entropy()
        # -----------------------------------------------

        # 6) Compose
        total_logp = pos_logp + sel_logp + orient_logp
        total_entropy = pos_entropy + sel_entropy + orient_entropy

        xy = self._pos_index_to_xy(pos_idx)   # (B,2) int-like

        return {
            'pos_idx': pos_idx, 'sel_idx': sel_idx, 'orient_idx': orient_idx,
            'xy': xy,
            'logp_pos': pos_logp, 'logp_sel': sel_logp, 'logp_or': orient_logp,
            'logp': total_logp, 'entropy': total_entropy,
        }

    def evaluate_actions(self,
                         boxes: torch.Tensor,
                         cont_flat: torch.Tensor,
                         raw_flat: torch.Tensor,
                         used_mask: torch.Tensor,
                         pos_idx: torch.Tensor,
                         sel_idx: torch.Tensor,
                         orient_idx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        box_enc, cont_enc = self.backbone(boxes, cont_flat)
        logits_pos, ctx = self.pos_dec(cont_enc, box_enc)
        pos_logp = F.log_softmax(logits_pos, dim=-1).gather(1, pos_idx.view(-1,1)).squeeze(1)
        dist_p = Categorical(logits=logits_pos)
        pos_ent = dist_p.entropy()

        pos_emb = self.pos_emb_builder(ctx, raw_flat, pos_idx)

        logits_sel, _ = self.sel_dec(box_enc, pos_emb)
        logits_sel = logits_sel.masked_fill(used_mask, float('-inf'))
        sel_logp = F.log_softmax(logits_sel, dim=-1).gather(1, sel_idx.view(-1,1)).squeeze(1)
        dist_s = Categorical(logits=logits_sel)
        sel_ent = dist_s.entropy()

        # 선택된 박스에서 유효 오리엔테이션 개수 산출 → 동일 마스킹
        B, N, _ = boxes.shape
        gather_idx = sel_idx.view(B,1,1).expand(-1,1,3)
        picked_lwh = boxes.gather(1, gather_idx).squeeze(1)     # (B,3)
        orient_emb = self.orient_embed(picked_lwh)
        logits_or, _ = self.orient_dec(orient_emb, pos_emb)

        n_valid = self._num_valid_orients(picked_lwh)          # (B,)
        ar6 = torch.arange(6, device=logits_or.device).view(1, 6).expand(B, -1)
        valid_mask = ar6 < n_valid.unsqueeze(1)
        logits_or = logits_or.masked_fill(~valid_mask, float('-inf'))

        orient_logp = F.log_softmax(logits_or, dim=-1).gather(1, orient_idx.view(-1,1)).squeeze(1)
        dist_o = Categorical(logits=logits_or)
        orient_ent = dist_o.entropy()

        total_logp = pos_logp + sel_logp + orient_logp
        total_entropy = pos_ent + sel_ent + orient_ent
        return total_logp, total_entropy


# ------------------------------------------------------------
# Advantage (GAE) helper
# ------------------------------------------------------------

def compute_gae(rewards: torch.Tensor, values: torch.Tensor, dones: torch.Tensor,
                gamma: float, lam: float, device: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """rewards, values, dones: shape (T,)
    returns, advantages: shape (T,)
    """
    T = rewards.size(0)
    adv = torch.zeros(T, device=device)
    last_gae = 0.0
    for t in reversed(range(T)):
        next_val = values[t+1] if t+1 < T else torch.tensor(0.0, device=device)
        next_nonterminal = 1.0 - float(dones[t].item())
        delta = rewards[t] + gamma * next_val * next_nonterminal - values[t]
        last_gae = delta + gamma * lam * next_nonterminal * last_gae
        adv[t] = last_gae
    returns = adv + values
    return returns, adv

# ------------------------------------------------------------
# CSV 초기화(누적-append) & 오프셋 계산
# ------------------------------------------------------------
def init_csv_and_offset(run_name: str) -> Tuple[csv.writer, any, str, int]:
    """
    - results/logs/{run_name}.csv 에 append
    - 없으면 헤더 작성 후 offset=0
    """
    log_dir = os.path.join("results", "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{run_name}.csv")

    header = ["run","ts","update","T","return_sum","mean_UR","actor_loss","critic_loss","entropy"]
    offset = 0
    file_exists = os.path.exists(log_path) and os.path.getsize(log_path) > 0

    if file_exists:
        last_update = 0
        with open(log_path, "r", newline="") as f:
            rd = csv.DictReader(f)
            for row in rd:
                try:
                    last_update = int(row.get("update") or last_update)
                except Exception:
                    continue
        offset = (last_update or 0)
        f = open(log_path, "a", newline="")
        w = csv.writer(f)
    else:
        f = open(log_path, "a", newline="")
        w = csv.writer(f)
        w.writerow(header)

    return w, f, log_path, offset

# ------------------------------------------------------------
# Training loop
# ------------------------------------------------------------

def train(cfg: TrainConfig, env_cfg: EnvConfig, run_name: str):
    dprint(f"[init] num_updates={cfg.num_updates}, epochs_per_update={cfg.epochs_per_update}, batch_size={cfg.batch_size}")
    os.makedirs(cfg.ckpt_dir, exist_ok=True)
    
    # --- RNG 초기화 (처음 한 번만) ---
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device)

    # CSV 로거 준비 (누적-append)
    csv_w, csv_f, log_path, base_offset = init_csv_and_offset(run_name)
    session_tag = time.strftime("%Y%m%d-%H%M%S")
    dprint(f"[log] {log_path} (base_offset={base_offset})")

    # Env & Nets
    env = PackingEnv(env_cfg)
    dprint("[env] PackingEnv initialized")
    policy = PolicyNet(d_model=128).to(device)
    value = ValueNet(EncoderConfig(d_model=128, nhead=4, num_layers=2), d_model=128, nhead=8, num_layers=2).to(device)

    optim_actor = torch.optim.Adam(policy.parameters(), lr=cfg.lr_actor)
    optim_critic = torch.optim.Adam(value.parameters(), lr=cfg.lr_critic)


    # === Resume-safe 학습 초기화 ===
    last_milestone, ckpt_path = resolve_resume(cfg, run_name, log_path)
    if last_milestone > 0 and ckpt_path:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        policy.load_state_dict(ckpt['policy'])
        value.load_state_dict(ckpt['value'])
        optim_actor.load_state_dict(ckpt['optim_actor'])
        optim_critic.load_state_dict(ckpt['optim_critic'])
        set_rng_state(ckpt['rng_state'])
        start_update = last_milestone + 1
        dprint(f"[resume] from {ckpt_path}, start_update={start_update}")

    else:
        start_update = 1
        dprint("[resume] no checkpoint found — start from scratch")
    
    dprint("[start] training loop")
    try:
        for update in range(start_update, cfg.num_updates + 1):
            dprint(f"[loop] update {update}/{cfg.num_updates} start")

            # Rollout buffers (episode length ≤ N)
            traj: Dict[str, List] = {k: [] for k in [
                'boxes','cont_flat','raw_flat','used_mask',
                'pos_idx','sel_idx','orient_idx','logp_old','rewards','dones','values'
            ]}

            obs = env.reset()
            done = False
            t = 0

            while not done:
                # ----- 컨테이너 상태 7채널 -----
                if hasattr(env, "container_state7"):
                    s7 = env.container_state7()  # (100,100,7)
                else:
                    hmap = getattr(env, "heightmap", getattr(env, "height", None))
                    if hmap is None:
                        raise RuntimeError("Env has no heightmap/height; and no container_state7().")
                    height = hmap.astype(np.float32)
                    s7 = compute_plane_features(height)                 # (100,100,7)

                ds7 = downsample_patches(s7, patch=10)                  # (10,10,7)
                raw_flat_np = flatten_for_encoder(ds7)[None, ...]       # (1,100,7)
                raw_flat = torch.from_numpy(raw_flat_np).float().to(device)   # (1,100,7)
                cont_flat = raw_flat  # 동일 텐서

                # Boxes (remaining & used mask)
                boxes_np = np.array(env.boxes, dtype=np.float32)           # (N,3)
                boxes = torch.from_numpy(boxes_np)[None, ...].to(device)   # (1,N,3)
                used_mask_np = env.used.astype(np.bool_)                   # (N,)
                used_mask = torch.from_numpy(used_mask_np)[None, ...].to(device)  # (1,N)

                # Policy step (sample actions)
                with torch.no_grad():
                    out = policy.forward_decode(boxes, cont_flat, raw_flat, used_mask)
                    pos_idx = out['pos_idx']           # (1,)
                    sel_idx = out['sel_idx']           # (1,)
                    orient_idx = out['orient_idx']     # (1,)
                    logp = out['logp']                 # (1,)
                    # Map pos index (0..99) → (x,y) in 100x100 grid (upper-left of the 10x10 patch)
                    x, y = map(int, out['xy'].cpu().numpy()[0])

                # Step env
                obs, reward, done, info = env.step((x, y, int(sel_idx.item()), int(orient_idx.item())))

                # Value (no grad in collect)
                with torch.no_grad():
                    V = value(boxes, cont_flat).squeeze(-1)  # (1,)

                # Store
                traj['boxes'].append(boxes.detach().cpu())
                traj['cont_flat'].append(cont_flat.detach().cpu())
                traj['raw_flat'].append(raw_flat.detach().cpu())
                traj['used_mask'].append(used_mask.detach().cpu())
                traj['pos_idx'].append(pos_idx.detach().cpu())
                traj['sel_idx'].append(sel_idx.detach().cpu())
                traj['orient_idx'].append(orient_idx.detach().cpu())
                traj['logp_old'].append(logp.detach().cpu())
                traj['rewards'].append(torch.tensor([reward], dtype=torch.float32))
                traj['dones'].append(torch.tensor([done], dtype=torch.float32))
                traj['values'].append(V.detach().cpu())

                t += 1
            
            if update % cfg.log_interval == 0:
                save_packing_3d(
                    env.placed_boxes,
                    container=(env.L, env.W, env.current_max_height()),
                    out_path=f"results/plots/packing_ep{update:05d}.png"
                )

            # 에피소드 통계
            return_sum = float(sum([r.item() for r in traj['rewards']]))
            try:
                ur = env.utilization_rate() if callable(getattr(env, "utilization_rate", None)) else float(getattr(env, "utilization_rate", 0.0))
                final_ur = float(ur)
            except Exception:
                final_ur = 0.0

            dprint(f"[collect] trajectories ready | T={t} | return={return_sum:.2f} | UR={final_ur:.4f}")

            # 빈 롤아웃 방지
            if t == 0:
                dprint("[warn] empty rollout (T=0) — skipping update")
                continue

            # Stack rollout tensors (T, ...)
            T = t
            def _stack(key):
                return torch.cat(traj[key], dim=0)  # concat along batch=1 → (T, ...)

            boxes_T      = _stack('boxes').to(device)       # (T,1,N,3)
            cont_flat_T  = _stack('cont_flat').to(device)   # (T,1,100,7)
            raw_flat_T   = _stack('raw_flat').to(device)
            used_mask_T  = _stack('used_mask').to(device)
            pos_idx_T    = _stack('pos_idx').squeeze(-1).to(device)     # (T,)
            sel_idx_T    = _stack('sel_idx').squeeze(-1).to(device)     # (T,)
            orient_idx_T = _stack('orient_idx').squeeze(-1).to(device)  # (T,)
            logp_old_T   = _stack('logp_old').squeeze(-1).to(device)    # (T,)
            rewards_T    = torch.cat(traj['rewards'], dim=0).to(device).squeeze(-1)  # (T,)
            dones_T      = torch.cat(traj['dones'], dim=0).to(device).squeeze(-1)    # (T,)
            values_T     = torch.cat(traj['values'], dim=0).to(device).squeeze(-1)   # (T,)

            # Compute GAE
            returns_T, adv_T = compute_gae(rewards_T, values_T, dones_T, cfg.gamma, cfg.gae_lambda, cfg.device)
            adv_T = (adv_T - adv_T.mean()) / (adv_T.std() + 1e-8)

            # PPO Update
            for epoch in range(cfg.epochs_per_update):
                new_logp_list = []
                entropy_list = []
                new_values = []

                for ti in range(T):
                    # ---- 형상 가드(Shape Guard): (B,N,3), (B,100,7), (B,N)로 통일 ----
                    boxes_b = boxes_T[ti]        # 예상 (1,N,3) or (N,3)
                    cont_b  = cont_flat_T[ti]    # 예상 (1,100,7) or (100,7)
                    raw_b   = raw_flat_T[ti]     # 예상 (1,100,7) or (100,7)
                    used_b  = used_mask_T[ti]    # 예상 (1,N) or (N,)

                    if boxes_b.dim() == 2 and boxes_b.size(-1) == 3:   # (N,3) -> (1,N,3)
                        boxes_b = boxes_b.unsqueeze(0)
                    elif boxes_b.dim() != 3 or boxes_b.size(-1) != 3:
                        raise RuntimeError(f"bad boxes shape: {boxes_b.shape}")

                    if cont_b.dim() == 2 and cont_b.size(-1) == 7:     # (100,7) -> (1,100,7)
                        cont_b = cont_b.unsqueeze(0)
                    elif cont_b.dim() != 3 or cont_b.size(-1) != 7:
                        raise RuntimeError(f"bad cont shape: {cont_b.shape}")

                    if raw_b.dim() == 2 and raw_b.size(-1) == 7:       # (100,7) -> (1,100,7)
                        raw_b = raw_b.unsqueeze(0)
                    elif raw_b.dim() != 3 or raw_b.size(-1) != 7:
                        raise RuntimeError(f"bad raw shape: {raw_b.shape}")

                    if used_b.dim() == 1:                               # (N,) -> (1,N)
                        used_b = used_b.unsqueeze(0)
                    elif used_b.dim() != 2:
                        raise RuntimeError(f"bad used_mask shape: {used_b.shape}")

                    # 액션 인덱스 (1,)
                    pos_b = pos_idx_T[ti:ti+1]
                    sel_b = sel_idx_T[ti:ti+1]
                    or_b  = orient_idx_T[ti:ti+1]

                    # 정책 재평가 / 엔트로피 (forward와 동일 마스킹 적용)
                    logp_new, ent = policy.evaluate_actions(
                        boxes_b, cont_b, raw_b, used_b,
                        pos_b, sel_b, or_b
                    )  # (1,), (1,)
                    new_logp_list.append(logp_new.squeeze(0))
                    entropy_list.append(ent.squeeze(0))

                    # 가치 예측
                    v = value(boxes_b, cont_b).squeeze()
                    new_values.append(v)

                new_logp = torch.stack(new_logp_list)   # (T,)
                entropy  = torch.stack(entropy_list)    # (T,)
                V_pred   = torch.stack(new_values)      # (T,)

                ratio = torch.exp(new_logp - logp_old_T)  # (T,)
                surr1 = ratio * adv_T
                surr2 = torch.clamp(ratio, 1.0 - cfg.ppo_clip, 1.0 + cfg.ppo_clip) * adv_T
                actor_loss = -torch.min(surr1, surr2).mean()

                critic_loss = F.mse_loss(V_pred, returns_T)
                entropy_bonus = entropy.mean()

                loss_actor_total = actor_loss - cfg.entropy_coef * entropy_bonus

                optim_actor.zero_grad()
                loss_actor_total.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                optim_actor.step()

                optim_critic.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(value.parameters(), 1.0)
                optim_critic.step()

            # 콘솔 요약
            if update % cfg.log_interval == 1:
                dprint(f"[update {update}] T={T} | Return={returns_T.sum().item():.1f} | "
                       f"ActorLoss={actor_loss.item():.4f} | CriticLoss={critic_loss.item():.4f} | Entropy={entropy_bonus.item():.4f}")

            # CSV 로그 기록 (누적)
            try:
                csv_w.writerow([
                    run_name, session_tag, update, T,
                    f"{return_sum:.4f}", f"{final_ur:.6f}",
                    f"{actor_loss.item():.6f}", f"{critic_loss.item():.6f}", f"{entropy_bonus.item():.6f}"
                ])
                csv_f.flush()
            except Exception as e:
                dprint("[warn] CSV write failed:", e)

            # Save minimal checkpoint (save_interval 반영)
            if update % cfg.save_interval == 0:
                ckpt_path = os.path.join(cfg.ckpt_dir, f'{run_name}_u{update:05d}.pt')
                torch.save({
                    'policy': policy.state_dict(),
                    'value': value.state_dict(),
                    'optim_actor': optim_actor.state_dict(),
                    'optim_critic': optim_critic.state_dict(),
                    "rng_state": get_rng_state(),
                    'cfg': cfg.__dict__,
                    'env_cfg': env_cfg.__dict__,
                }, ckpt_path)
                dprint(f"[save] ckpt saved to: {ckpt_path}")

        dprint("[done] training loop complete")

    except KeyboardInterrupt:
        dprint("[interrupt] Training interrupted by user (Ctrl+C)")
        dprint(f"[save] interrupted ckpt saved: {ckpt_path}")

    except Exception:
        dprint("[fatal] exception in training loop")
        traceback.print_exc()
        raise
    finally:
        try:
            csv_f.close()
        except Exception:
            pass

        # === 학습 종료 시 자동 플롯 저장 ===
        if MATPLOTLIB_OK:
            try:
                updates, rets, urs = [], [], []
                with open(log_path, "r", newline="") as f:
                    rd = csv.DictReader(f)
                    for row in rd:
                        try:
                            x = int(row.get("update"))
                            updates.append(x)
                            rets.append(float(row["return_sum"]))
                            urs.append(float(row["mean_UR"]))
                        except Exception:
                            pass

                out_dir = os.path.join("results", "plots")
                os.makedirs(out_dir, exist_ok=True)

                plt.figure()
                plt.plot(updates, rets)
                plt.title(f"Return (sum per update) — {run_name}")
                plt.xlabel("update"); plt.ylabel("return_sum")
                plt.tight_layout()
                ret_png = os.path.join(out_dir, f"{run_name}_return.png")
                plt.savefig(ret_png, dpi=150)
                plt.close()

                plt.figure()
                plt.plot(updates, urs)
                plt.title(f"Utilization Rate — {run_name}")
                plt.xlabel("update"); plt.ylabel("UR")
                plt.tight_layout()
                ur_png = os.path.join(out_dir, f"{run_name}_ur.png")
                plt.savefig(ur_png, dpi=150)
                plt.close()

                dprint(f"[plot] saved: {ret_png}")
                dprint(f"[plot] saved: {ur_png}")
            except Exception as e:
                dprint("[warn] plotting failed:", e)
        else:
            dprint("[plot] matplotlib not available; skip saving plots")

# ------------------------------------------------------------
# main
# ------------------------------------------------------------
if __name__ == "__main__":
    # 기본값
    tcfg = TrainConfig()
    ecfg = EnvConfig(L=100, W=100, N=30, seed=42)

    # ---------- YAML 설정 연결 ----------
    # configs/env.yaml 로드
    try:
        env_y = load_yaml(os.path.join("configs", "env.yaml"))
        L = int(env_y["container"]["length"])
        W = int(env_y["container"]["width"])
        seed = int(env_y["container"]["seed"])
        N = int(env_y["boxes"]["count"])
        ecfg = EnvConfig(L=L, W=W, N=N, seed=seed)
        dprint(f"[cfg] env.yaml loaded → L={L}, W={W}, N={N}, seed={seed}")
    except Exception as e:
        dprint("[cfg] env.yaml not loaded, using defaults:", e)

    # configs/train.yaml 로드
    try:
        train_y = load_yaml(os.path.join("configs", "train.yaml"))
        ppo = train_y.get("ppo", {})
        opt = train_y.get("optimizer", {})
        sch = train_y.get("schedule", {})

        tcfg.gamma        = float(ppo.get("gamma", tcfg.gamma))
        tcfg.gae_lambda   = float(ppo.get("gae_lambda", tcfg.gae_lambda))
        tcfg.ppo_clip     = float(ppo.get("clip_epsilon", tcfg.ppo_clip))
        tcfg.entropy_coef = float(ppo.get("entropy_coef", tcfg.entropy_coef))

        tcfg.lr_actor     = float(opt.get("policy_lr", tcfg.lr_actor))
        tcfg.lr_critic    = float(opt.get("value_lr", tcfg.lr_critic))

        tcfg.num_updates       = int(sch.get("num_updates", tcfg.num_updates))
        tcfg.epochs_per_update = int(sch.get("epochs_per_update", tcfg.epochs_per_update))
        tcfg.batch_size        = int(sch.get("batch_size", tcfg.batch_size))
        tcfg.log_interval      = int(sch.get("log_interval", tcfg.log_interval))
        tcfg.save_interval     = int(sch.get("save_interval", tcfg.save_interval))

        dprint(f"[cfg] train.yaml loaded → updates={tcfg.num_updates}, epochs={tcfg.epochs_per_update}, "
               f"batch={tcfg.batch_size}, log_interval={tcfg.log_interval}, save_interval={tcfg.save_interval} | "
               f"gamma={tcfg.gamma}, clip={tcfg.ppo_clip}, entropy_coef={tcfg.entropy_coef}, "
               f"lr_actor={tcfg.lr_actor}, lr_critic={tcfg.lr_critic}")
    except Exception as e:
        dprint("[cfg] train.yaml not loaded, using defaults:", e)
    # -----------------------------------

    # 누적 CSV 파일명 (환경변수로 지정 가능)
    RUN_NAME = os.getenv("RUN_NAME", "3dpp_experiment1")

    dprint("[boot] entering main()")
    train(tcfg, ecfg, run_name=RUN_NAME)
    dprint("[boot] main() returned normally")
