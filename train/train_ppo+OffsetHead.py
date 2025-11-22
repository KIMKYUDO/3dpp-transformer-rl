import warnings
warnings.filterwarnings("ignore", message="Please use the new API settings to control TF32 behavior")
# ---- 환경변수/레거시 정리 ----
import os, sys
cores = os.cpu_count() or 8

# 과거/중복 변수 제거
os.environ.pop("CUDA_ALLOC_CONF", None)  # ← 이게 남아있으면 torch가 그 값을 출력합니다.
os.environ.pop("PYTORCH_ALLOC_CONF", None)

# 정식 변수만 사용
os.environ["PYTORCH_ALLOC_CONF"] = (
    "expandable_segments:True,garbage_collection_threshold:0.8,max_split_size_mb:128"
)

os.environ["OMP_NUM_THREADS"] = str(max(1, cores // 3))
os.environ["MKL_NUM_THREADS"] = str(max(1, cores // 3))
os.environ.setdefault("MPLBACKEND", "Agg")

# ---------- (2) 라이브러리 import ----------
import torch

# 스레드
torch.set_num_threads(max(1, cores // 4))
torch.set_num_interop_threads(2)

# 새 API로만 설정 (성능 우선: TF32)
torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.fp32_precision = "tf32"
torch.backends.cudnn.conv.fp32_precision  = "tf32"

# --- backend 선택 & GUI import 가드 ---
USE_GUI = False  # 훈련 중엔 False 유지

if USE_GUI:
    # GUI 쓸 때만 TkAgg + tkinter 가져오기
    os.environ.pop("MPLBACKEND", None)  # 환경변수 강제값 제거
    import matplotlib
    matplotlib.use("TkAgg", force=True)
    import tkinter as tk
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
else:
    # 기본: 창 없이 파일 저장만 하는 Agg
    os.environ.setdefault("MPLBACKEND", "Agg")
    import matplotlib
    matplotlib.use("Agg", force=True)

# === plotting (자동 저장) ===
try:
    import matplotlib.pyplot as plt
    if not USE_GUI:
        plt.ioff()                              # 인터랙티브 모드 off
    
    import atexit
    atexit.register(lambda: plt.close('all'))
    MATPLOTLIB_OK = True
except Exception:
    MATPLOTLIB_OK = False
    plt = None

import time
import csv
import traceback
from dataclasses import dataclass
from typing import List, Dict, Tuple, cast
from typing import Tuple as _TupleForObs
from collections import Counter

import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.amp import autocast, GradScaler

# --- add project root to sys.path (for direct script execution) ---
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent  # .../projects
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
# -----------------------------------------------------------------


# Local imports
from envs.container_sim import PackingEnv, EnvConfig
from utils.preprocess import compute_plane_features, downsample_patches, flatten_for_encoder, preprocess_batch_gpu
from utils.logger import resolve_resume
from utils.rng_state import get_rng_state, set_rng_state
from utils.plotting import save_packing_3d, save_packing_3d_interactive
from utils.curriculum_scheduler import CurriculumScheduler
from agents.backbone import PolicyBackbone, EncoderConfig
from agents.heads import (
    PositionDecoder, SelectionDecoder, OrientationDecoder,
    PositionEmbeddingBuilder, OrientationEmbedder, OffsetHead
)
from agents.value_head import ValueNet

# ==== Offset options (간단 상수) ====
OFFSET_ENABLED   = True     # 오프셋 헤드 사용 여부
OFFSET_SIGMA     = 0.2      # Δ 가우시안 초기 sigma
OFFSET_QUANT     = 0.1      # 0 또는 None이면 양자화 없음 / 1.0이면 정수화 / 0.1이면 소수1자리
OFFSET_CLAMP_ABS = 0.6      # Δ 클리핑 한계 (패치 반경 대비)
PATCH_SIZE       = 10       # 100→10×10 다운샘플 패치 크기
GRID_H = GRID_W  = 10       # 다운샘플 그리드 크기

print("CUDA_ALLOC_CONF =", os.environ.get("PYTORCH_CUDA_ALLOC_CONF"))

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
    value_clip_eps: float = 0.2
    critic_loss_type: str = "clipped_mse"
    # LR
    lr_actor: float = 1e-5
    lr_critic: float = 1e-4
    # Rollout / Schedule
    n_steps: int = 128
    num_updates: int = 2          # 데모값 (실험 시 늘리세요)
    epochs_per_update: int = 4
    batch_size: int = 1           # 단일 환경
    # Logging / Checkpoint
    ckpt_dir: str = os.path.join("results", "ckpt")
    log_interval: int = 1
    save_interval: int = 1        # train.yaml의 save_interval 반영 (기본 1)
    # AMP
    grad_accum_steps: int = 1   # 그래디언트 누적 스텝 (메모리 부족 시 >1로 설정)
    # Debug / Reason logging
    log_done_reasons_every: int = 1   # 매 몇 update마다 요약 출력할지
    print_first_k_done: int = 256      # 조기 종료 샘플 상세 로그 개수 제한

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
        # === Offset head ===
        if OFFSET_ENABLED:
            self.offset_head = OffsetHead(d_model=d_model, sigma_init=OFFSET_SIGMA)
        else:
            self.offset_head = None
    
    @staticmethod
    def _gather_patch_token(patch_tokens: torch.Tensor, px: torch.Tensor, py: torch.Tensor,
                            grid_h: int = GRID_H, grid_w: int = GRID_W) -> torch.Tensor:
        """
        patch_tokens: (B, grid_h*grid_w, d)
        px, py: (B,)  # 선택된 패치 인덱스
        return: (B, d)
        """
        B, N, d = patch_tokens.shape
        idx = py * grid_w + px                # (B,)
        idx = idx.view(-1, 1).expand(-1, d)   # (B, d)
        return patch_tokens.gather(1, idx.unsqueeze(1)).squeeze(1)  # (B, d)

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
    
    def _compute_valid_position_mask(self,
                                     boxes: torch.Tensor,      # (B,N,3)
                                     used_mask: torch.Tensor,  # (B,N) bool
                                     container_L: int = 100,
                                     container_W: int = 100,
                                     patch: int = 10) -> torch.Tensor:
        B, N, _ = boxes.shape
        device = boxes.device

        # 1) 100개 위치 그리드 생성 (x, y 순서로 통일)
        x_coords = torch.arange(0, container_L, patch, device=device)  # 10개
        y_coords = torch.arange(0, container_W, patch, device=device)  # 10개
        
        # (100, 2) with (x, y) 순서
        grid = torch.stack([
            x_coords.repeat_interleave(len(y_coords)),
            y_coords.repeat(len(x_coords))
        ], dim=1)  # (100, 2) - (x, y)

        remaining_L = (container_L - grid[:, 1]).view(1, 1, 1, 100)  # (1,1,1,100)
        remaining_W = (container_W - grid[:, 0]).view(1, 1, 1, 100)  # (1,1,1,100)

        # 2) 6가지 회전의 2D 풋프린트 (l,w) 쌍을 전부 생성
        l, w, h = boxes[..., 0], boxes[..., 1], boxes[..., 2]          # (B,N)
        pairs = torch.stack([
            torch.stack([l, w], dim=-1),
            torch.stack([l, h], dim=-1),
            torch.stack([w, l], dim=-1),
            torch.stack([w, h], dim=-1),
            torch.stack([h, l], dim=-1),
            torch.stack([h, w], dim=-1),
        ], dim=2)  # (B,N,6,2)

        # 3) 이미 사용된 박스는 아주 크게 만들어 제외
        if used_mask is not None:
            mask = used_mask.unsqueeze(-1).unsqueeze(-1)          # (B,N,1,1)
            pairs = pairs.masked_fill(mask, 99999)

        # 4) 위치별 적합성: (pl<=remL & pw<=remW)
        pL = pairs[..., 0].unsqueeze(-1)   # (B,N,6,1)
        pW = pairs[..., 1].unsqueeze(-1)   # (B,N,6,1)

        fits = (pL <= remaining_L) & (pW <= remaining_W)  # (B,N,6,100)

        # 5) 박스/회전 중 하나라도 맞으면 그 위치는 유효
        valid_mask = fits.any(dim=2).any(dim=1)  # (B,100)

        # 6) 모든 박스 사용된 배치는 False
        all_used = used_mask.all(dim=-1)  # (B,)
        valid_mask[all_used] = False

        return valid_mask
    
    def _compute_valid_selection_mask(self,
                                      boxes: torch.Tensor,       # (B, N, 3)
                                      used_mask: torch.Tensor,   # (B, N)
                                      pos_xy: torch.Tensor,      # (B, 2) - 선택된 위치
                                      container_L: int = 100,
                                      container_W: int = 100) -> torch.Tensor:
        """선택 유효성: pos_xy 위치에 들어갈 수 있는 박스만
        
        Returns: (B, N) bool mask (True = 선택 가능)
        """
        B, N, _ = boxes.shape
        device = boxes.device
        
        valid_mask = ~used_mask  # 기본: 미사용 박스만
        
        x = pos_xy[:, 0]  # (B,)
        y = pos_xy[:, 1]  # (B,)
        
        # 남은 공간 계산
        remaining_L = container_L - x  # (B,)
        remaining_W = container_W - y  # (B,)
        
        for b in range(B):
            if remaining_L[b] <= 0 or remaining_W[b] <= 0:
                # 경계 밖이면 모든 박스 무효
                valid_mask[b, :] = False
                continue

            for n in range(N):
                if used_mask[b, n]:
                    continue
                
                l, w, h = boxes[b, n]
                
                # 6가지 회전 중 **하나라도** 들어가면 OK
                can_fit = False
                for perm in [(l,w), (l,h), (w,l), (w,h), (h,l), (h,w)]:
                    pl, pw = perm[0], perm[1]
                    if pl <= remaining_L[b] and pw <= remaining_W[b]:
                        can_fit = True
                        break
                
                if not can_fit:
                    valid_mask[b, n] = False
        
        return valid_mask
    
    def _compute_valid_orientation_mask(self,
                                        box_lwh: torch.Tensor,     # (B, 3)
                                        pos_xy: torch.Tensor,      # (B, 2)
                                        container_L: int = 100,
                                        container_W: int = 100) -> torch.Tensor:
        """회전 유효성: 경계 체크만 수행 (중복 제거 안 함)
        
        중복 회전은 동일한 결과이므로 제거할 필요 없음.
        오히려 All -inf 방지에 도움.
        
        Returns: (B, 6) bool mask
        """
        B = box_lwh.shape[0]
        device = box_lwh.device
        
        l, w, h = box_lwh[:, 0], box_lwh[:, 1], box_lwh[:, 2]
        x, y = pos_xy[:, 0], pos_xy[:, 1]
        
        remaining_L = container_L - x
        remaining_W = container_W - y
        
        # 6가지 회전 모두 경계 체크 (중복 제거 생략)
        perms = [
            torch.stack([l, w, h], dim=1),  # 0: (l,w,h)
            torch.stack([l, h, w], dim=1),  # 1: (l,h,w)
            torch.stack([w, l, h], dim=1),  # 2: (w,l,h)
            torch.stack([w, h, l], dim=1),  # 3: (w,h,l)
            torch.stack([h, l, w], dim=1),  # 4: (h,l,w)
            torch.stack([h, w, l], dim=1),  # 5: (h,w,l)
        ]
        
        valid_mask = torch.zeros(B, 6, dtype=torch.bool, device=device)
        for i, perm in enumerate(perms):
            pl, pw = perm[:, 0], perm[:, 1]
            # 경계 내부 체크
            fits_L = (pl <= remaining_L) & (remaining_L > 0)
            fits_W = (pw <= remaining_W) & (remaining_W > 0)
            valid_mask[:, i] = fits_L & fits_W

        
        return valid_mask

    def forward_decode(self,
                       boxes: torch.Tensor,          # (B,N,3)
                       cont_flat: torch.Tensor,      # (B,100,7)
                       raw_flat: torch.Tensor,       # (B,100,7) same as cont_flat before proj
                       used_mask: torch.Tensor,      # (B,N) bool
                       env_config: EnvConfig = EnvConfig(),
                       patch: int = 10) -> Dict[str, torch.Tensor]:
        """
        use_heuristic_masking=True: Position/Selection 단계에서도 휴리스틱 마스킹 적용
        """
        B, N, _ = boxes.shape

        box_enc, cont_enc = self.backbone(boxes, cont_flat)
        container_tokens = cont_enc
        
        # Position
        logits_pos, ctx = self.pos_dec(cont_enc, box_enc)
        # 최소 박스 크기 기반 경계 마스킹
        valid_pos_mask = self._compute_valid_position_mask(
            boxes, used_mask, env_config.L, env_config.W, patch
        )
        
        logits_pos = logits_pos.masked_fill(~valid_pos_mask, float('-inf'))

        dist_p = Categorical(logits=logits_pos)
        pos_idx = dist_p.sample()

        pos_logp = dist_p.log_prob(pos_idx)
        pos_entropy = dist_p.entropy()

        xy = self._pos_index_to_xy(pos_idx, patch)
        
        # Selection
        pos_emb = self.pos_emb_builder(ctx, raw_flat, pos_idx)
        logits_sel, _ = self.sel_dec(box_enc, pos_emb)

        # 선택한 위치에 들어갈 박스만 마스킹
        valid_sel_mask = self._compute_valid_selection_mask(
            boxes, used_mask, xy, env_config.L, env_config.W
        )
        logits_sel = logits_sel.masked_fill(~valid_sel_mask, float('-inf'))

        # ===== 핵심 수정: All -inf 체크 BEFORE sampling =====
        all_invalid_sel = torch.all(torch.isinf(logits_sel) & (logits_sel < 0), dim=-1)
        
        if all_invalid_sel.any():
            print(f"[DEBUG] All -inf in selection for batch: {torch.where(all_invalid_sel)[0].tolist()}")
            for b in range(B):
                if all_invalid_sel[b]:
                    # 디버깅 정보 출력
                    print(f"  Batch {b}: xy={xy[b].tolist()}, "
                        f"used={used_mask[b].tolist()}, "
                        f"boxes={boxes[b].tolist()}, "
                        f"remaining_boxes={(~used_mask[b]).sum().item()}")
                            
        dist_s = Categorical(logits=logits_sel)
        sel_idx = dist_s.sample()
        sel_logp = dist_s.log_prob(sel_idx)
        sel_entropy = dist_s.entropy()
        
        # Orientation (중복 제거 + 경계 체크)
        B, N, _ = boxes.shape
        gather_idx = sel_idx.view(B, 1, 1).expand(-1, 1, 3)
        picked_lwh = boxes.gather(1, gather_idx).squeeze(1)
        
        orient_emb = self.orient_embed(picked_lwh)
        logits_or, _ = self.orient_dec(orient_emb, pos_emb)
        
        valid_or_mask = self._compute_valid_orientation_mask(
            picked_lwh, xy, env_config.L, env_config.W
        )
        logits_or = logits_or.masked_fill(~valid_or_mask, float('-inf'))
        
        dist_o = Categorical(logits=logits_or)
        orient_idx = dist_o.sample()
        orient_logp = dist_o.log_prob(orient_idx)
        orient_entropy = dist_o.entropy()
        
        # 최종 출력
        total_logp = pos_logp + sel_logp + orient_logp
        total_entropy = pos_entropy + sel_entropy + orient_entropy
        
        return {
            'pos_idx': pos_idx, 'sel_idx': sel_idx, 'orient_idx': orient_idx,
            'xy': xy,
            'logp_pos': pos_logp, 'logp_sel': sel_logp, 'logp_or': orient_logp,
            'logp': total_logp, 'entropy': total_entropy,
            'logits_pos': logits_pos,
            'logits_sel': logits_sel,
            'logits_or': logits_or,
        }


    def evaluate_actions(self,
                         boxes: torch.Tensor,
                         cont_flat: torch.Tensor,
                         raw_flat: torch.Tensor,
                         used_mask: torch.Tensor,
                         pos_idx: torch.Tensor,
                         sel_idx: torch.Tensor,
                         orient_idx: torch.Tensor,
                         env_config: EnvConfig = EnvConfig(),
                         patch: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
        
        B, N, _ = boxes.shape
        
        box_enc, cont_enc = self.backbone(boxes, cont_flat)

        # ===== 1) POSITION (휴리스틱 마스킹 동일 적용) =====
        logits_pos, ctx = self.pos_dec(cont_enc, box_enc)

        valid_pos_mask = self._compute_valid_position_mask(
            boxes, used_mask, env_config.L, env_config.W, patch
        )  # (B,100) True=유효
        logits_pos = logits_pos.masked_fill(~valid_pos_mask, float('-inf'))
        

        dist_p = Categorical(logits=logits_pos)
        pos_logp = dist_p.log_prob(pos_idx)
        pos_ent = dist_p.entropy()

        # pos_idx -> xy (B,2)
        xy = self._pos_index_to_xy(pos_idx, patch)

        pos_emb = self.pos_emb_builder(ctx, raw_flat, pos_idx)

        # ===== 2) SELECTION (위치 적합성 마스킹 적용) =====
        logits_sel, _ = self.sel_dec(box_enc, pos_emb)

        valid_sel_mask = self._compute_valid_selection_mask(
            boxes, used_mask, xy, env_config.L, env_config.W
        )  # (B,N) True=선택 가능
        logits_sel = logits_sel.masked_fill(~valid_sel_mask, float('-inf'))

        dist_s = Categorical(logits=logits_sel)
        sel_logp = dist_s.log_prob(sel_idx)
        sel_ent = dist_s.entropy()

        # 선택 박스 차원 추출
        gather_idx = sel_idx.view(B,1,1).expand(-1,1,3)
        picked_lwh = boxes.gather(1, gather_idx).squeeze(1)     # (B,3)

        # ===== 3) ORIENTATION (경계 기반 마스킹 적용) =====
        orient_emb = self.orient_embed(picked_lwh)
        logits_or, _ = self.orient_dec(orient_emb, pos_emb)

        valid_or_mask = self._compute_valid_orientation_mask(
            picked_lwh, xy, env_config.L, env_config.W
        )  # (B,6)
        logits_or = logits_or.masked_fill(~valid_or_mask, float('-inf'))

        dist_o = Categorical(logits=logits_or)
        orient_logp = dist_o.log_prob(orient_idx)
        orient_ent = dist_o.entropy()

        # ===== 최종 =====
        total_logp = pos_logp + sel_logp + orient_logp
        total_entropy = pos_ent + sel_ent + orient_ent
        return total_logp, total_entropy
    
    def forward_greedy(self,
                   boxes: torch.Tensor,          # (B,N,3)
                   cont_flat: torch.Tensor,      # (B,100,7)
                   raw_flat: torch.Tensor,       # (B,100,7)
                   used_mask: torch.Tensor,      # (B,N) bool
                   env_config = None,
                   patch: int = 10) -> Dict[str, torch.Tensor]:
        """
        Greedy evaluation 전용 forward pass
        
        Training의 forward_decode와 차이점:
        1. Sampling 대신 argmax 사용
        2. 각 단계마다 masking을 **선택 후** 재계산
        3. Log prob 계산 생략 (불필요)
        4. 더 빠른 실행 (gradient 계산 없음)
        
        Returns:
        --------
        Dict with keys:
            - 'pos_idx': (B,) selected position indices
            - 'sel_idx': (B,) selected box indices  
            - 'orient_idx': (B,) selected orientation indices
            - 'logits_pos': (B, 100) position logits (for debugging)
            - 'logits_sel': (B, N) selection logits (for debugging)
            - 'logits_or': (B, 6) orientation logits (for debugging)
        """
        
        B, N, _ = boxes.shape
        device = boxes.device
        
        # ===== Encoding (공통) =====
        box_enc, cont_enc = self.backbone(boxes, cont_flat)
        
        # ===== 1단계: POSITION (Greedy) =====
        logits_pos, ctx = self.pos_dec(cont_enc, box_enc)
        
        # Position masking (최소 박스 크기 기반)
        valid_pos_mask = self._compute_valid_position_mask(
            boxes, used_mask, env_config.L, env_config.W, patch
        )
        logits_pos = logits_pos.masked_fill(~valid_pos_mask, float('-inf'))
        
        # Greedy selection (argmax)
        pos_idx = logits_pos.argmax(dim=-1)  # (B,)
        xy = self._pos_index_to_xy(pos_idx, patch)  # (B, 2)
        
        # Position embedding (선택된 위치 기반)
        pos_emb = self.pos_emb_builder(ctx, raw_flat, pos_idx)
        
        # ===== 2단계: SELECTION (Greedy, 선택된 위치 기반 masking) =====
        logits_sel, _ = self.sel_dec(box_enc, pos_emb)
        
        # ⭐ 핵심: 선택된 xy 위치에 맞는 박스만 필터링
        valid_sel_mask = self._compute_valid_selection_mask(
            boxes, used_mask, xy, env_config.L, env_config.W
        )
        logits_sel = logits_sel.masked_fill(~valid_sel_mask, float('-inf'))
        
        # Greedy selection (argmax)
        sel_idx = logits_sel.argmax(dim=-1)  # (B,)
        
        # 선택된 박스 차원 추출
        gather_idx = sel_idx.view(B, 1, 1).expand(-1, 1, 3)
        picked_lwh = boxes.gather(1, gather_idx).squeeze(1)  # (B, 3)
        
        # ===== 3단계: ORIENTATION (Greedy, 선택된 박스+위치 기반 masking) =====
        orient_emb = self.orient_embed(picked_lwh)
        logits_or, _ = self.orient_dec(orient_emb, pos_emb)
        
        # ⭐ 핵심: 선택된 박스와 위치에 맞는 회전만 필터링
        valid_or_mask = self._compute_valid_orientation_mask(
            picked_lwh, xy, env_config.L, env_config.W
        )
        logits_or = logits_or.masked_fill(~valid_or_mask, float('-inf'))
        
        # Greedy selection (argmax)
        orient_idx = logits_or.argmax(dim=-1)  # (B,)
        
        return {
            'pos_idx': pos_idx,
            'sel_idx': sel_idx,
            'orient_idx': orient_idx,
            'logits_pos': logits_pos,  # 디버깅용
            'logits_sel': logits_sel,
            'logits_or': logits_or,
        }


# ------------------------------------------------------------
# Advantage (GAE) helper
# ------------------------------------------------------------

def compute_gae(rewards: torch.Tensor, values: torch.Tensor, dones: torch.Tensor,
                gamma: float, lam: float, device: str,
                bootstrap_value: torch.Tensor | float | None = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    rewards, values, dones: (T,)
    bootstrap_value: s_{T}의 V(s_T) (마지막 next state의 value)  # 미종료 시 사용
    """
    T = rewards.size(0)
    adv = torch.zeros(T, device=device)
    last_val = torch.as_tensor(bootstrap_value, device=device, dtype=values.dtype) if bootstrap_value is not None else torch.tensor(0.0, device=device, dtype=values.dtype)
    last_gae = torch.tensor(0.0, device=device, dtype=values.dtype)
    for t in reversed(range(T)):
        next_val = values[t+1] if t+1 < T else last_val
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

    header = ["run","ts","update","T",
              "return_sum","mean_UR_last","mean_UR_best",
              "actor_loss","critic_loss","entropy",
              "mean_step","mean_invalid",
              "return_eval","UR_eval"]
    offset = 0
    file_exists = os.path.exists(log_path) and os.path.getsize(log_path) > 0

    if file_exists:
        last_update = 0
        with open(log_path, "rb") as f:
            f.seek(-2, os.SEEK_END)     # EOF 직전으로 이동
            pos = f.tell() - 1
            while pos > 0:
                f.seek(pos, os.SEEK_SET)
                if f.read(1) == b"\n":
                    break
                pos -= 1
            f.seek(pos + 1, os.SEEK_SET)
            last_line = f.readline().decode().strip()
            
        try:
            last_update = int(last_line.split(",")[2])
            offset = last_update
        except Exception:
            offset = 0
        f = open(log_path, "a", newline="")
        w = csv.writer(f)
    else:
        f = open(log_path, "a", newline="")
        w = csv.writer(f)
        w.writerow(header)

    return w, f, log_path, offset

def make_obs_tensor(env, device):
    """
    _make_obs_from_env와 동일한 로직 (단일 환경용)
    """

    # Heightmap (NumPy)
    hmap = env.height  # NumPy array
    
    # GPU로 전송 + 전처리
    hmap_gpu = torch.from_numpy(hmap).float().to(device)
    s7 = compute_plane_features(hmap_gpu)
    ds7 = downsample_patches(s7, patch=10)
    flat = flatten_for_encoder(ds7)
    
    # (1, 100, 7)로 확장
    raw_flat = flat.unsqueeze(0)
    cont_flat = raw_flat
    
    # Boxes & masks
    boxes = torch.from_numpy(np.array(env.boxes, dtype=np.float32)).unsqueeze(0).to(device)
    used_mask = torch.from_numpy(env.used.astype(np.bool_)).unsqueeze(0).to(device)
    
    return raw_flat, cont_flat, boxes, used_mask

    # BUGFIX: max_eval_steps 지역변수 참조 오류 → 인자화
def render_eval(policy : PolicyNet, env : PackingEnv, device : torch.device,
                out_base : str, max_eval_steps: int | None = None):
    """
    Greedy evaluation with proper cascading masking
    
    Parameters:
    -----------
    policy : PolicyNet
        Must have forward_greedy() method
    env : PackingEnv
        Existing environment (training env를 직접 전달)
    device : torch.device
    out_base : str
    max_eval_steps : int, optional
    
    Returns:
    --------
    total_reward : float
    ur_eval : float (utilization rate)
    """
    
    # 현재 박스 저장 후 reset
    current_boxes = env.boxes
    env.reset(boxes=current_boxes)
    
    policy.eval()
    
    if max_eval_steps is None:
        max_eval_steps = int(getattr(env.cfg, "max_steps", 300) or 300) * 5

    with torch.inference_mode():
        done = False
        steps = 0
        last_placed = -1
        stagnation = 0
        total_reward = 0.0
        
        while (not done) and (steps < max_eval_steps):
            # Observation
            raw_flat, cont_flat, boxes, used_mask = make_obs_tensor(env, device)
            
            # 모든 박스 사용됨 체크
            if used_mask[0].all():
                dprint(f"[eval] ✓ All {len(env.boxes)} boxes placed at step {steps}")
                break
            
            # ⭐ Greedy forward pass (cascading masking 적용)
            out = policy.forward_greedy(
                boxes, cont_flat, raw_flat, used_mask,
                env_config=env.cfg
            )
            
            # Action 추출
            pos_idx = out['pos_idx'].item()
            sel_idx = out['sel_idx'].item()
            orient_idx = out['orient_idx'].item()
            
            # Position index를 xy로 변환
            x_i = (pos_idx % 10) * 10
            y_i = (pos_idx // 10) * 10
            
            # Environment step
            _, reward, done, info = env.step((x_i, y_i, sel_idx, orient_idx))
            total_reward += reward
            
            # Progress tracking
            placed_now = len(env.placed_boxes)
            if placed_now == last_placed:
                stagnation += 1
                if stagnation % 25 == 0:
                    dprint(f"[eval] stagnation={stagnation}/150, "
                           f"placed={placed_now}/{len(env.boxes)}, "
                           f"action=({x_i},{y_i},{sel_idx},{orient_idx})")
            else:
                if stagnation > 0:
                    dprint(f"[eval] ✓ Progress: {placed_now}/{len(env.boxes)} boxes")
                stagnation = 0
                
            last_placed = placed_now
            
            if stagnation >= 150:
                dprint(f"[eval] ✗ Stagnation limit: {placed_now}/{len(env.boxes)} placed")
                break

            steps += 1

        # Results
        boxes = env.placed_boxes
        ur_eval = env.utilization_rate()
        
        dprint(f"[eval] Completed: {len(boxes)}/{len(env.boxes)} boxes, "
               f"UR={ur_eval:.2%}, steps={steps}, reward={total_reward:.2f}")
        
        save_packing_3d(boxes, 
                       container=(env.cfg.L, env.cfg.W, env.current_max_height()), 
                       out_path=out_base+".png")
        save_packing_3d_interactive(boxes, 
                                   container=(env.cfg.L, env.cfg.W, env.current_max_height()), 
                                   out_path=out_base+".html")

    return total_reward, ur_eval

# ------------------------------------------------------------
# Training loop
# ------------------------------------------------------------

def train(cfg: TrainConfig, env_cfg: EnvConfig, run_name: str):
    dprint(f"[init] num_updates={cfg.num_updates}, epochs_per_update={cfg.epochs_per_update}, batch_size={cfg.batch_size}")
    os.makedirs(cfg.ckpt_dir, exist_ok=True)

    curriculum = CurriculumScheduler()
    dprint("[curriculum] Initialized with stages:")
    for i, stage in enumerate(curriculum.stages):
        dprint(f"  Stage {i+1}: {stage.name} | N={stage.box_count_range} | "
               f"Duration={stage.duration_updates}")
    
    # --- RNG 초기화 (처음 한 번만) ---
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device)

    # CSV 로거 준비 (누적-append)
    csv_w, csv_f, log_path, base_offset = init_csv_and_offset(run_name)
    session_tag = time.strftime("%Y%m%d-%H%M%S")
    dprint(f"[log] {log_path} (base_offset={base_offset})")

    # Env & Nets
    n_steps = cfg.n_steps
    num_envs = getattr(env_cfg, "num_envs", 1)
    envs = [PackingEnv(env_cfg) for _ in range(num_envs)]
    dprint(f"[env] {num_envs} PackingEnv initialized")

    policy = PolicyNet(d_model=128).to(device)
    value = ValueNet(EncoderConfig(d_model=128, nhead=4, num_layers=2), d_model=128, nhead=8, num_layers=2).to(device)

    optim_actor = torch.optim.Adam(policy.parameters(), lr=cfg.lr_actor)
    optim_critic = torch.optim.Adam(value.parameters(), lr=cfg.lr_critic)

    # === AMP scaler 정의 ===
    scaler_actor = GradScaler(enabled=(device.type == "cuda"))
    scaler_critic = GradScaler(enabled=(device.type == "cuda"))

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
            eval_ret, eval_ur = 0.0, 0.0
            dprint(f"[loop] update {update}/{cfg.num_updates} start")

            # === 여기에 추가 (기존 done_reason_counts 선언 전) ===
            stage_info = curriculum.get_stage_info(update)
            current_stage = curriculum.get_stage(update)

            # === 모든 환경에 같은 박스 개수 적용 (수정) ===
            N_sampled = curriculum.sample_box_count(update)  # 한 번만 샘플링
    
            # 단계 전환 로그
            if update == start_update or (update > start_update and 
                curriculum.get_stage(update - 1) != current_stage):
                dprint(f"\n{'='*60}")
                dprint(f"[CURRICULUM] Advancing to {stage_info['stage_name']}")
                dprint(f"  Box Count: {current_stage.box_count_range}")
                dprint(f"  Size Range: {current_stage.size_range}")
                dprint(f"{'='*60}\n")

            done_reason_counts = Counter()
            step_reason_counts = Counter()   # ← 모든 스텝의 reason 집계(종료/비종료 포함)            
            done_reason_samples = []  # (t, env_id, reason, detail)

            # Rollout buffers (episode length ≤ N)
            trajs = [
            {k: [] for k in [
                'boxes','cont_flat','raw_flat','used_mask',
                'pos_idx','sel_idx','orient_idx','logp_old',
                'rewards','dones','values'
            ]}
            for _ in range(num_envs)
            ]

            # GPU buffers (GPU - 임시 누적용) ← 추가!
            gpu_buffers = [
                {k: [] for k in [
                    'boxes','cont_flat','raw_flat','used_mask',
                    'pos_idx','sel_idx','orient_idx','logp_old','values'
                ]}
                for _ in range(num_envs)
            ]  

            # 초기화
            for env in envs:
                env.cfg.N = N_sampled
                env.cfg.l_range = current_stage.size_range
                env.cfg.w_range = current_stage.size_range
                env.cfg.h_range = current_stage.size_range
                env.reset()  # 새 설정으로 리셋
            # ===================

            last_done = [False] * num_envs

            # 고정 길이 롤아웃: 정확히 n_steps 스텝 수집
            # --- 최고 UR 스냅샷 추적기 초기화 ---
            best_ur_per_env = [-1.0 for _ in range(num_envs)]
            last_ur_per_env = [-1.0 for _ in range(num_envs)]
            # (boxes_snapshot(list of tuples), H_tilde(int))
            best_snap_per_env: List[Tuple[List[Tuple[int,int,int,int,int,int]], int] | None] = [None for _ in range(num_envs)]
            last_complete_state: List[Tuple[List[Tuple[int,int,int,int,int,int]], int] | None] = [None for _ in range(num_envs)]

            for t in range(n_steps):
                active_envs = list(range(num_envs))

                # ----- GPU 배치 전처리 (ThreadPool 제거!) -----
                
                # 1. NumPy로 수집
                heightmaps = np.stack([envs[i].height for i in active_envs])  # (B, H, W)
                boxes_list = [envs[i].boxes for i in active_envs]
                used_list = [envs[i].used for i in active_envs]
                
                # 2. GPU에서 배치 전처리
                raw_flat = preprocess_batch_gpu(heightmaps, patch=10, device=device)  # (B, 100, 7) on GPU
                cont_flat = raw_flat
                
                # 3. Boxes & masks → GPU
                boxes_arr = np.array([list(b) for b in boxes_list], dtype=np.float32)  # (B, N, 3)
                used_arr = np.stack(used_list)  # (B, N)
                
                boxes = torch.from_numpy(boxes_arr).float().to(device)
                used_mask = torch.from_numpy(used_arr).to(device)

                # --- Policy batch forward ---
                with torch.inference_mode():
                    out = policy.forward_decode(boxes, cont_flat, raw_flat, used_mask, env_cfg)
                    pos_idx = out['pos_idx']      # (B,)
                    sel_idx = out['sel_idx']      # (B,)
                    orient_idx = out['orient_idx']# (B,)
                    logp = out['logp']            # (B,)
                    xy = out['xy'].cpu().numpy()  # (B,2)
                    
                    V: torch.Tensor = cast(torch.Tensor, value(boxes, cont_flat).squeeze(-1))

                                # --- 각 env에 step 적용 ---
                
                # 최적화 1: 배치 변환 (루프 전에 한번만)
                sel_cpu = sel_idx.cpu().numpy()
                ori_cpu = orient_idx.cpu().numpy()
                
                for j, i in enumerate(active_envs):
                    env = envs[i]
                    x, y = int(xy[j, 0]), int(xy[j, 1])
                    
                    # 최적화 2: NumPy 인덱싱 (item() 대신)
                    obs, reward, done, info = env.step((x, y,
                                                         int(sel_cpu[j]),
                                                         int(ori_cpu[j])))
                    
                    # --- 모든 스텝 reason 집계 ---
                    step_key = (info or {}).get("reason", "unknown")
                    step_reason_counts[step_key] += 1
                    
                    # 최적화 3: GPU에 저장 (CPU 전송 안 함!)
                    gpu_buffers[i]['boxes'].append(boxes[j:j+1])
                    gpu_buffers[i]['cont_flat'].append(cont_flat[j:j+1])
                    gpu_buffers[i]['raw_flat'].append(raw_flat[j:j+1])
                    gpu_buffers[i]['used_mask'].append(used_mask[j:j+1])
                    gpu_buffers[i]['pos_idx'].append(pos_idx[j:j+1])
                    gpu_buffers[i]['sel_idx'].append(sel_idx[j:j+1])
                    gpu_buffers[i]['orient_idx'].append(orient_idx[j:j+1])
                    gpu_buffers[i]['logp_old'].append(logp[j:j+1])
                    gpu_buffers[i]['values'].append(V[j:j+1])
                    
                    # rewards, dones는 작으니 바로 CPU 저장
                    trajs[i]['rewards'].append(torch.tensor([reward], dtype=torch.float32))
                    trajs[i]['dones'].append(torch.tensor([done], dtype=torch.float32))
                    
                    last_done[i] = bool(done)
                    
                    if done:
                        reason = (info or {}).get("reason", "unknown")
                        detail = (info or {}).get("detail", None)
                        done_reason_counts[reason] += 1
                        if len(done_reason_samples) < cfg.print_first_k_done:
                            done_reason_samples.append((t, i, reason, detail))
                        # --- UR(best), UR(last) 추적: 완료 시에만 ---
                        if info.get("reason") == "complete":
                            # reset 전에 저장
                            current_ur = env.utilization_rate()
                            
                            # best 업데이트
                            if current_ur > best_ur_per_env[i]:
                                best_ur_per_env[i] = current_ur
                                best_snap_per_env[i] = (
                                    list(env.placed_boxes),
                                    env.current_max_height()
                                )
                            
                            # last 저장
                            last_ur_per_env[i] = current_ur
                            last_complete_state[i] = (
                                list(env.placed_boxes),
                                env.current_max_height()
                            )
                        env.reset()
                        # (Rollout 루프 끝난 후)
            
            # 최적화 4: GPU → CPU 전송 (한번에!)
            for i in range(num_envs):
                # GPU 텐서들 concat → CPU 전송
                for key in ['boxes', 'cont_flat', 'raw_flat', 'used_mask',
                           'pos_idx', 'sel_idx', 'orient_idx', 'logp_old', 'values']:
                    if len(gpu_buffers[i][key]) > 0:
                        trajs[i][key] = torch.cat(gpu_buffers[i][key], dim=0).cpu()
                    else:
                        trajs[i][key] = torch.empty(0)
                
                # rewards, dones concat (이미 CPU)
                if len(trajs[i]['rewards']) > 0:
                    trajs[i]['rewards'] = torch.cat(trajs[i]['rewards'], dim=0)
                    trajs[i]['dones'] = torch.cat(trajs[i]['dones'], dim=0)
                else:
                    trajs[i]['rewards'] = torch.empty(0)
                    trajs[i]['dones'] = torch.empty(0)
            
            # GPU 버퍼 메모리 해제
            del gpu_buffers

            # Bootstrap value calculation - GPU batch processing (same as rollout)
            heightmaps_boot = np.stack([envs[i].height for i in range(num_envs)])  # (B, H, W)
            boxes_list_boot = [envs[i].boxes for i in range(num_envs)]
            
            # GPU preprocessing
            boot_raw = preprocess_batch_gpu(heightmaps_boot, patch=10, device=device)  # (B, 100, 7)
            boot_cont = boot_raw
            
            # Boxes to GPU
            boxes_arr_boot = np.array([list(b) for b in boxes_list_boot], dtype=np.float32)  # (B, N, 3)
            boot_box = torch.from_numpy(boxes_arr_boot).float().to(device)  # (B, N, 3)

            with torch.inference_mode():
                boot_V = value(boot_box, boot_cont).squeeze(-1).detach().cpu()

            # --- 시각화 저장 ---
            if update % cfg.log_interval == 0:
                # (1) 마지막 완주 스냅샷
                if last_complete_state[0] is not None:
                    last_boxes0, last_H0 = last_complete_state[0]
                    out_base_last = f"results/plots/packing_u{update:05d}"
                    save_packing_3d(last_boxes0,
                                    container=(env_cfg.L, env_cfg.W, last_H0),
                                    out_path=out_base_last+".png")
                    save_packing_3d_interactive(last_boxes0,
                                    container=(env_cfg.L, env_cfg.W, last_H0),
                                    out_path=out_base_last+".html")
                # (2) 최고 UR 스냅샷: 모든 env 중 best_ur 최대인 env 선택
                cand_best = [(i, u) for i, u in enumerate(best_ur_per_env) if u >= 0.0]
                if cand_best:
                    env_best_idx = max(cand_best, key=lambda x: x[1])[0]
                    if best_snap_per_env[env_best_idx] is not None:
                        best_boxes, best_H = best_snap_per_env[env_best_idx]
                        out_base_best = f"results/plots/packing_best_u{update:05d}"
                        save_packing_3d(best_boxes,
                                        container=(env_cfg.L, env_cfg.W, best_H),
                                        out_path=out_base_best+".png")
                        save_packing_3d_interactive(best_boxes,
                                        container=(env_cfg.L, env_cfg.W, best_H),
                                        out_path=out_base_best+".html")

            # --- 모든 스텝 reason 요약 ---
            if step_reason_counts:
                total_steps = sum(step_reason_counts.values())
                dprint("[step-reasons] total:", total_steps,
                       "| " + " | ".join(f"{k}:{v}" for k,v in step_reason_counts.most_common()))
            else:
                dprint("[step-reasons] (none)")

            # --- 종료(reason) 요약 ---
            if update % cfg.log_done_reasons_every == 0:
                if done_reason_counts:
                    total_done = sum(done_reason_counts.values())
                    dprint("[done-reasons] total:", total_done,
                           "| " + " | ".join(f"{k}:{v}" for k,v in done_reason_counts.most_common()))
                    if done_reason_samples:
                        dprint("[done-reasons] samples (t, env, key, detail) ↓")
                        for s in done_reason_samples:
                            dprint("   ", s)
                else:
                    dprint("[done-reasons] no termination observed in rollout window")

            # --- 에피소드 통계 ---
            return_sum = float(np.mean([trajs[i]['rewards'].sum().item() for i in range(num_envs)]))
            ur_last = float(np.mean([u for u in last_ur_per_env if u >= 0.0])) if any(u >= 0.0 for u in last_ur_per_env) else 0.0
            valid = [u for u in best_ur_per_env if u >= 0.0]
            ur_best = max(valid) if valid else 0.0
            mean_step = float(np.mean([getattr(env, "step_count", 0) for env in envs]))
            mean_invalid = float(np.mean([getattr(env, "invalid_count", 0) for env in envs]))
            dprint(f"[collect] trajectories ready | T={n_steps * num_envs} | "
                   f"return={return_sum:.2f} | UR(last)={ur_last:.4f} | UR(best)={ur_best:.4f} | "
                   f"step(avg)={mean_step:.1f} | invalid(avg)={mean_invalid:.1f}")

            # --- rollout 병합 ---
            traj = {k: torch.cat([trajs[i][k] for i in range(num_envs)], dim=0)
                    for k in trajs[0].keys()}
            
            # === 여기서 '항상' 텐서로 준비 ===
            device_opt = dict(device=device, non_blocking=True)
            rewards_T      = traj['rewards'   ].squeeze(-1).to(**device_opt)  # CPU→GPU
            dones_T        = traj['dones'     ].squeeze(-1).to(**device_opt)  # CPU→GPU
            values_T       = traj['values'    ].squeeze(-1).to(**device_opt)  # 불필요!
            logp_old_T     = traj['logp_old'  ].squeeze(-1).to(**device_opt)  # 불필요!
            pos_idx_T      = traj['pos_idx'   ].squeeze(-1).to(**device_opt).long()  # 불필요!
            sel_idx_T      = traj['sel_idx'   ].squeeze(-1).to(**device_opt).long()  # 불필요!
            orient_idx_T   = traj['orient_idx'].squeeze(-1).to(**device_opt).long()  # 불필요!

            T = n_steps * num_envs

            # env별로 구간을 쪼개어 GAE 계산 (부트스트랩 포함)
            offset = 0
            rets_list, adv_list = [], []
            for i in range(num_envs):
                Ti = len(trajs[i]['rewards'])  # 해당 env의 길이
                r_i = rewards_T[offset:offset+Ti]
                d_i = dones_T[offset:offset+Ti]
                v_i = values_T[offset:offset+Ti]

                # 마지막이 terminal이면 0, truncated(미종료)이면 V(next)
                bootstrap = 0.0 if (Ti == 0 or last_done[i]) else float(boot_V[i].item())

                ret_i, adv_i = compute_gae(r_i, v_i, d_i, cfg.gamma, cfg.gae_lambda, cfg.device, bootstrap_value=bootstrap)
                rets_list.append(ret_i)
                adv_list.append(adv_i)
                offset += Ti

            returns_T = torch.cat(rets_list, dim=0)
            adv_T     = torch.cat(adv_list, dim=0)
            adv_T = (adv_T - adv_T.mean()) / (adv_T.std() + 1e-8)

            # ---- 최적화: 이미 GPU에 있으므로 그대로 사용 ----
            boxes_B = traj['boxes'].squeeze(1).contiguous().pin_memory()
            cont_B  = traj['cont_flat'].squeeze(1).contiguous().pin_memory()
            raw_B   = traj['raw_flat'].squeeze(1).contiguous().pin_memory()
            used_B  = traj['used_mask'].squeeze(1).contiguous().pin_memory()

            pos_B = pos_idx_T
            sel_B = sel_idx_T
            or_B  = orient_idx_T

            adv_T     = adv_T.to(device)
            returns_T = returns_T.to(device)
            logp_old_T= logp_old_T.to(device)

            accum = getattr(cfg, "grad_accum_steps", 1)

            policy.train(); value.train()

            for epoch in range(cfg.epochs_per_update):
                approx_kls, clip_fracs = [], []
                idx = torch.randperm(T)

                # 누적을 위해 zero_grad를 에폭 시작에서 1회
                optim_actor.zero_grad(set_to_none=True)
                optim_critic.zero_grad(set_to_none=True)

                mb = 0
                for start in range(0, T, cfg.batch_size):
                    b_cpu = idx[start:start+cfg.batch_size]
                    b_gpu = b_cpu.to(device, non_blocking=True)

                    boxes_b = boxes_B.index_select(0, b_cpu).to(device, non_blocking=True)
                    cont_b  = cont_B .index_select(0, b_cpu).to(device, non_blocking=True)
                    raw_b   = raw_B  .index_select(0, b_cpu).to(device, non_blocking=True)
                    used_b  = used_B .index_select(0, b_cpu).to(device, non_blocking=True).bool()

                    # --- GPU 소스 → GPU 인덱싱 ---
                    pos_b       = pos_B      .index_select(0, b_gpu)
                    sel_b       = sel_B      .index_select(0, b_gpu)
                    or_b        = or_B       .index_select(0, b_gpu)
                    logp_old_b  = logp_old_T .index_select(0, b_gpu)
                    adv_b       = adv_T      .index_select(0, b_gpu)
                    rets_b      = returns_T  .index_select(0, b_gpu)

                    with autocast(device_type=device.type, enabled=(device.type=="cuda")):
                        logp_new, ent = policy.evaluate_actions(
                            boxes_b, cont_b, raw_b, used_b, pos_b, sel_b, or_b
                        )
                        V_pred = value(boxes_b, cont_b).squeeze(-1)

                        ratio = (logp_new - logp_old_b.to(logp_new.dtype)).exp()
                        surr1 = ratio * adv_b
                        surr2 = torch.clamp(ratio, 1 - cfg.ppo_clip, 1 + cfg.ppo_clip) * adv_b
                        actor_loss   = -torch.min(surr1, surr2).mean()

                        approx_kl_batch = (logp_old_b - logp_new).mean()                   # 스칼라
                        clip_frac_batch = ((ratio - 1.0).abs() > cfg.ppo_clip).float().mean()

                        approx_kls.append(approx_kl_batch.detach())
                        clip_fracs.append(clip_frac_batch.detach())

                        # --- PPO-style value clipping ---
                        V_old_b = values_T.index_select(0, b_gpu).detach().to(V_pred.dtype)
                        value_clip_eps = getattr(cfg, "value_clip_eps", 0.2)

                        V_pred_clipped = V_old_b + (V_pred - V_old_b).clamp(-value_clip_eps, value_clip_eps)
                        
                        rets_b = rets_b.to(V_pred.dtype)  # 손실 계산 dtype 통일
                        critic_loss_1 = F.mse_loss(V_pred, rets_b)
                        critic_loss_2 = F.mse_loss(V_pred_clipped, rets_b)
                        critic_loss = torch.max(critic_loss_1, critic_loss_2)
                        
                        entropy_mean = ent.mean()

                        # 0..until 구간에서만 선형, 그 이후엔 1.0로 고정
                        ent_start = cfg.entropy_coef
                        ent_end   = 0.01
                        total     = int(cfg.num_updates)
                        until     = 0.6   # 60%

                        prog_raw = update / float(total - 1) if total > 1 else 1.0       # 0..1
                        prog = min(1.0, prog_raw / until)                                # 0..1로 압축
                        entropy_coef_t = ent_start * (1 - prog) + ent_end * prog
                        total_actor_loss = actor_loss - entropy_coef_t * entropy_mean
                    
                    # --- NaN guard: backward 전에! ---
                    if (not torch.isfinite(actor_loss)) or (not torch.isfinite(critic_loss)) or (not torch.isfinite(entropy_mean)):
                        dprint("[nan] detected — skipping step",
                            "| total_actor:", total_actor_loss.item() if torch.isfinite(total_actor_loss) else "NaN",
                            "| critic:", critic_loss.item() if torch.isfinite(critic_loss) else "NaN")
                        optim_actor.zero_grad(set_to_none=True)
                        optim_critic.zero_grad(set_to_none=True)
                        # 누적 카운트 증가/step 판단 전에 건너뛰기
                        continue

                    # ---- 그래디언트 누적 (loss/accum) ----
                    scaler_actor.scale(total_actor_loss / accum).backward()
                    scaler_critic.scale(critic_loss / accum).backward()

                    # ---- step 타이밍: 매 accum 스텝마다 ----
                    mb += 1
                    should_step = ((mb % accum) == 0) or (start + cfg.batch_size >= T)

                    if should_step:
                        # Actor step
                        scaler_actor.unscale_(optim_actor)
                        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                        scaler_actor.step(optim_actor); scaler_actor.update()
                        optim_actor.zero_grad(set_to_none=True)

                        # Critic step
                        scaler_critic.unscale_(optim_critic)
                        torch.nn.utils.clip_grad_norm_(value.parameters(), 1.0)
                        scaler_critic.step(optim_critic); scaler_critic.update()
                        optim_critic.zero_grad(set_to_none=True)
                
                # <<< 여기! 미니배치 루프 종료 직후 에폭 평균 계산 >>>
                approx_kl_epoch  = torch.stack(approx_kls).mean().item() if approx_kls else 0.0
                clip_frac_epoch  = torch.stack(clip_fracs).mean().item() if clip_fracs else 0.0

            # --- 로그 ---
            if update % cfg.log_interval == 0:
                # 그리디 평가 (완주) 스냅샷
                try:
                    eval_ret, eval_ur = render_eval(policy, envs[0], device, f"results/plots/eval_u{update:05d}")
                except Exception as e:
                    eval_ret, eval_ur = 0.0, 0.0
                    dprint("[plot] eval render failed:", e)

                dprint(f"[update {update}] T={T} | "
                    f"Stage={stage_info['stage_name']} ({stage_info['progress']:.0%}) | "
                    f"Return={returns_T.sum().item():.1f} | ActorLoss={actor_loss.item():.4f} | CriticLoss={critic_loss.item():.4f} | "
                    f"Entropy={entropy_mean.item():.4f} | "
                    f"EntropyCoef(t)={entropy_coef_t:.6f} | "
                    f"approx_KL={approx_kl_epoch:.4f} | clip_frac={clip_frac_epoch:.3f} | "
                    f"eval_return={eval_ret:.2f} | eval_UR={eval_ur:.4f}")
                
            
            # CSV 로그 기록 (누적)
            try:
                csv_w.writerow([
                    run_name, session_tag, update, T,
                    f"{return_sum:.4f}",
                    f"{ur_last:.6f}", f"{ur_best:.6f}",
                    f"{actor_loss.item():.6f}", f"{critic_loss.item():.6f}", f"{entropy_mean.item():.6f}",
                    f"{mean_step:.2f}", f"{mean_invalid:.2f}",
                    f"{eval_ret:.4f}", f"{eval_ur:.6f}",
                    f"{approx_kl_epoch:.6f}", f"{clip_frac_epoch:.6f}"
                ])
                csv_f.flush()
            except Exception as e:
                dprint("[warn] CSV write failed:", e)

            # --- 체크포인트 저장 ---
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
                # Train 데이터 (모든 update)
                updates, rets, urs_last, urs_best = [], [], [], []
                # Eval 데이터 (0이 아닌 값만, 즉 log_interval에 해당하는 update만)
                eval_updates, eval_rets, eval_urs = [], [], []
                
                with open(log_path, "r", newline="") as f:
                    rd = csv.DictReader(f)
                    for row in rd:
                        try:
                            upd = int(row["update"])
                            updates.append(upd)
                            rets.append(float(row["return_sum"]))
                            urs_last.append(float(row.get("mean_UR_last", row.get("mean_UR", 0.0))))
                            urs_best.append(float(row.get("mean_UR_best", 0.0)))
                            
                            # eval 값이 0이 아닌 경우만 별도 리스트에 추가
                            eval_ret = float(row.get("return_eval", 0.0))
                            eval_ur = float(row.get("UR_eval", 0.0))
                            if eval_ret != 0.0 or eval_ur != 0.0:  # 둘 중 하나라도 0이 아니면
                                eval_updates.append(upd)
                                eval_rets.append(eval_ret)
                                eval_urs.append(eval_ur)
                        except Exception:
                            continue

                out_dir = os.path.join("results", "plots")
                os.makedirs(out_dir, exist_ok=True)

                # --- Return Plot ---
                plt.figure()
                plt.plot(updates, rets, label="Train Return", linewidth=1.8)
                if eval_updates:  # eval 데이터가 있으면
                    plt.plot(eval_updates, eval_rets, 'o-', label="Eval Return", 
                            linewidth=1.8, markersize=4, alpha=0.8)
                plt.title(f"Return (train vs eval) — {run_name}")
                plt.xlabel("Update")
                plt.ylabel("Return")
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(os.path.join(out_dir, f"{run_name}_return.png"), dpi=150)
                plt.close()

                # --- UR Plot ---
                plt.figure()
                plt.plot(updates, urs_last, label="UR(last)", linewidth=1.8)
                plt.plot(updates, urs_best, label="UR(best)", linewidth=1.8)
                if eval_updates:  # eval 데이터가 있으면
                    plt.plot(eval_updates, eval_urs, 'o-', label="UR(eval)", 
                            linewidth=1.8, markersize=4, alpha=0.8)
                plt.title(f"Utilization Rate (train vs eval) — {run_name}")
                plt.xlabel("Update")
                plt.ylabel("Utilization Rate")
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(os.path.join(out_dir, f"{run_name}_ur.png"), dpi=150)
                plt.close()

                dprint(f"[plot] saved return/UR plots with eval curves")
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
    ecfg = EnvConfig(L=100, W=100, N=20, seed=None, num_envs=4, max_steps=200)

    # ---------- YAML 설정 연결 ----------
    # configs/env.yaml 로드
    try:
        env_y = load_yaml(os.path.join("configs", "env.yaml"))
        L = int(env_y["container"]["length"])
        W = int(env_y["container"]["width"])
        seed = int(env_y["container"]["seed"])
        N = int(env_y["boxes"]["count"])
        invalid_penalty = float(env_y["reward"]["invalid_penalty"])
        num_envs = int(env_y["env"]["num_envs"])
        max_steps = int(env_y["env"]["max_steps"])
        ecfg = EnvConfig(L=L, W=W, N=N, seed=seed, num_envs=num_envs, max_steps=max_steps)
        dprint(f"[cfg] env.yaml loaded → L={L}, W={W}, N={N}, seed={seed}, num_envs={num_envs}, max_steps={max_steps}, invalid_penalty={invalid_penalty}")
    except Exception as e:
        dprint("[cfg] env.yaml not loaded, using defaults:", e)

    # configs/train.yaml 로드
    try:
        train_y = load_yaml(os.path.join("configs", "train.yaml"))
        ppo = train_y.get("ppo", {})
        opt = train_y.get("optimizer", {})
        sch = train_y.get("schedule", {})
        amp = train_y.get("amp", {})

        tcfg.gamma        = float(ppo.get("gamma", tcfg.gamma))
        tcfg.gae_lambda   = float(ppo.get("gae_lambda", tcfg.gae_lambda))
        tcfg.ppo_clip     = float(ppo.get("clip_epsilon", tcfg.ppo_clip))
        tcfg.entropy_coef = float(ppo.get("entropy_coef", tcfg.entropy_coef))

        tcfg.lr_actor     = float(opt.get("policy_lr", tcfg.lr_actor))
        tcfg.lr_critic    = float(opt.get("value_lr", tcfg.lr_critic))

        tcfg.n_steps           = int(sch.get("n_steps", tcfg.n_steps))
        tcfg.num_updates       = int(sch.get("num_updates", tcfg.num_updates))
        tcfg.epochs_per_update = int(sch.get("epochs_per_update", tcfg.epochs_per_update))
        tcfg.batch_size        = int(sch.get("batch_size", tcfg.batch_size))
        tcfg.log_interval      = int(sch.get("log_interval", tcfg.log_interval))
        tcfg.save_interval     = int(sch.get("save_interval", tcfg.save_interval))

        tcfg.grad_accum_steps = int(amp.get("grad_accum_steps", tcfg.grad_accum_steps))

        dprint(f"[cfg] train.yaml loaded → n_step={tcfg.n_steps}, updates={tcfg.num_updates}, epochs={tcfg.epochs_per_update}, "
               f"batch={tcfg.batch_size}, log_interval={tcfg.log_interval}, save_interval={tcfg.save_interval}, grad_accum_steps={tcfg.grad_accum_steps} | "
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
