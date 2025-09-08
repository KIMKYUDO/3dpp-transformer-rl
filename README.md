# Transformer-based 3D Container Packing with PPO (ReturnAll 협업)

강화학습(PPO)과 트랜스포머 정책망으로 3D 컨테이너 적재(가변 높이)를 최적화하는 프로젝트입니다. 목표 지표는 **공간 활용률(UR)** 극대화이며, 논문 *"Solving 3D packing problem using Transformer network and reinforcement learning" (Que et al., ESWA 2023)*의 핵심 아이디어(상태 표현, **Plane Features**, **Action Order**, **Downsampling**)를 구현했습니다.

---

## ✅ Key Features
- **상태 분리 표현**: Box state + Container state(Heightmap + Plane Features)
- **트랜스포머 백본**: Box/Container 인코더 + Position/Selection/Orientation 디코더
- **체인룰 정책**: π(a|s)=π^p·π^s·π^o (**Position → Selection → Orientation**)
- **PPO + GAE**: 안정적 정책 학습, Entropy 보너스로 탐색 조절
- **실험 편의성**: `configs/*.yaml`, `results/{logs,plots,ckpt}/` 구조, 테스트 스위트 제공

---

## 📂 Folder Structure
```
projects/
  agents/{backbone.py, heads.py, value_head.py}
  configs/{env.yaml, model.yaml, train.yaml}
  envs/{container_sim.py}
  results/{ckpt/, logs/, plots/}
  tests/{test_backbone.py, test_env.py, test_heads.py, test_preprocess.py, test_value_head.py}
  train/{train_ppo.py}
  utils/{preprocess.py}
```
---

## 🧪 Environment & Setup
```bash
# 1) Create & activate venv
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

# 2) Install requirements
pip install -r requirements-gpu.txt    # (CUDA 사용 시)
# or
pip install -r requirements-cpu.txt    # (CPU 전용)
```
> 권장 Python: 3.9+ / PyTorch 최신 안정 버전

---

## ⚙️ Configs
세 개의 YAML로 분리되어 있습니다. (기본값은 논문 세팅을 따름)

### `configs/env.yaml`
```yaml
container: {length: 100, width: 100, seed: 42}
boxes:
  min_length: 10
  max_length: 50
  min_width: 10
  max_width: 50
  min_height: 10
  max_height: 50
  count: 30
reward:
  invalid_penalty: -1.0
```

### `configs/model.yaml`
```yaml
transformer:
  d_model: 128
  num_encoder_layers: 2
  num_decoder_layers: 2
  num_heads_encoder: 4
  num_heads_decoder: 8
  dropout: 0.1
embedding:
  use_positional_encoding: true
  orientation_classes: 6

# 추가: Downsampling/Plane Features 설정(필요 시)
preprocess:
  plane_features: true          # heightmap에서 7채널 plane features 생성
  downsample:
    enabled: true
    patch: 10                   # 100x100 → (100/10)x(100/10)=10x10
    select_metric: "el_x_ew"    # 각 패치에서 argmax(e_l × e_w) 위치의 7채널 유지
```

### `configs/train.yaml`
```yaml
ppo:
  gamma: 0.99
  gae_lambda: 0.96
  clip_epsilon: 0.12
  entropy_coef: 0.01
  value_coef: 0.5
  max_grad_norm: 0.5
optimizer:
  policy_lr: 1e-5
  value_lr: 1e-4
  weight_decay: 0.0
schedule:
  num_updates: 10000
  epochs_per_update: 4
  batch_size: 32
  log_interval: 50
  save_interval: 500
```

---

## 🧠 Method Brief
- **Container State 7채널 (100×100×7)**: 높이 `h`, 경계거리 `e_l/e_w/e_-l/e_-w`, 이웃 고지점 거리 `f_l/f_w`
- **Downsampling**: 10×10 패치로 나눈 뒤, 각 패치에서 `argmax(e_l×e_w)` 위치의 피처만 보존 → 10×10×7 → flatten(100×7)
- **Encoders**: Box/Container 각각 Transformer Encoder
- **Decoders**:
  - **Position** (Q=ContainerEnc, KV=BoxEnc) → (x,y)
  - **Position Embedding Builder** (컨테이너 enc (x,y) + 원시 state(x,y) 결합) → 위치-조건 맥락
  - **Selection** (Q=BoxEnc, KV=PosEmb) → box index
  - **OrientationEmbedder** (선택 박스 6회전 임베딩) → **Orientation** (Q=OrientEmb, KV=PosEmb)
- **Policy Factorization**: π(a|s)=π^p(a_p|s)·π^s(a_s|a_p,s)·π^o(a_o|a_p,a_s,s)
- **Reward**: r_i = g_{i-1} - g_i,  g_i = L·W·Ĥ_i − Σ_j l_j w_j h_j
- **Objective**: PPO(clip) + value loss + β·entropy

---

## 🧩 Plane Features — 정의와 계산
컨테이너 바닥을 H×W 격자의 heightmap으로 나타낼 때, 각 셀 (i,j)에 대해 다음 **7채널**을 구성합니다.

| 채널 | 의미(직관) | 요약 |
|---|---|---|
| `h_{ij}` | 현재 칸의 높이 | 이미 적재된 높이 |
| `e^l_{ij}` | **왼쪽 경계까지의 여유** | i열 고정, j에서 **왼쪽 벽**까지 빈 칸 거리 |
| `e^w_{ij}` | **위쪽 경계까지의 여유** | j행 고정, i에서 **위(상단) 벽**까지 빈 칸 거리 |
| `e^{-l}_{ij}` | **오른쪽 경계까지의 여유** | 오른쪽 벽까지 빈 칸 거리 |
| `e^{-w}_{ij}` | **아래쪽 경계까지의 여유** | 아래(하단) 벽까지 빈 칸 거리 |
| `f^l_{ij}` | **왼쪽 방향 가장 가까운 더 높은 지점까지 거리** | 지형의 ‘턱/절벽’까지의 거리(수평) |
| `f^w_{ij}` | **위쪽 방향 가장 가까운 더 높은 지점까지 거리** | 지형의 ‘턱/절벽’까지의 거리(수직) |

> 구현: `utils/preprocess.py: compute_plane_features(heightmap)` — 넘파이로 선형 스캔하여 경계거리 e·와 고지점거리 f·를 채웁니다. 테스트: `tests/test_preprocess.py`.

**왜 필요한가?**
- `h`만으로는 **경계/턱 정보**가 부족합니다. e·/f·는 “어디가 넓게 비었는지”, “어디가 막혀 있는지”를 인코딩해 **좌표 선택(Position)**과 **회전(Orientation)**의 불확실성을 크게 줄여줍니다.

---

## 🔻 Downsampling — 100×100 → 10×10×7 → 100×7
컨테이너 상태는 100×100×7로 크고 희소성이 큽니다. 연산량과 잡음을 줄이기 위해 **패치 기반 다운샘플링**을 적용합니다.

1. **패치 분할**: (H,W)=(100,100)을 `patch=10`으로 균등 분할 → 10×10 패치(각 패치 10×10 크기)
2. **대표 위치 선택**: 각 패치에서 `argmax(e_l×e_w)`인 좌표 (넓은 경계 여유를 가진 지점)을 고릅니다.
3. **특징 추출**: 선택된 (x,y)의 **7채널**만 유지 → 결과는 10×10×7
4. **Flatten**: 10×10×7 → **100×7** 시퀀스로 변환 후 컨테이너 인코더 입력

> 구현: `utils/preprocess.py: downsample_patches(features, patch=10, select="el_x_ew")`  
> 장점: (i) **연산량↓**, (ii) **노이즈↓**, (iii) **탐색 효율↑** — 특히 Position 디코더의 수렴을 가속합니다.

---

## 🔁 Action Order — Position → Selection → Orientation
이 프로젝트는 기존 연구들과 달리 **좌표를 먼저 고르고(Position)**, **그 좌표에 가장 잘 맞는 박스를 고른 뒤(Selection)**, **마지막으로 회전(Orientation)**을 결정합니다.

- **정당성(체인룰)**:  
  π(a|s)=π^p(a_p|s)·π^s(a_s|a_p,s)·π^o(a_o|a_p,a_s,s)  
  좌표 조건의 **맥락(PosEmb)**을 Selection/Orientation에 공유하여 **조건부 의사결정**을 강화합니다.
- **효과**: 좌표를 먼저 고르면 **경계/지형 제약**을 초기에 반영할 수 있어, 박스 선택·회전의 탐색 공간이 줄고 **수렴 속도**와 **UR**이 개선됩니다.
- **코드 연결**:
  - `agents/heads.py`: PositionDecoder → PositionEmbeddingBuilder → SelectionDecoder → OrientationDecoder
  - `agents/value_head.py`: 동일 입력 기반의 ValueNet

---

## 🔌 API (전처리) — 빠른 예시
```python
import numpy as np
from utils.preprocess import compute_plane_features, downsample_patches, flatten_for_encoder

H, W = 100, 100
heightmap = np.zeros((H, W), dtype=np.int32)

# 1) Plane Features (H×W×7)
pf = compute_plane_features(heightmap)

# 2) Downsampling (10×10 패치 → 10×10×7)
pf_ds = downsample_patches(pf, patch=10, select="el_x_ew")

# 3) Flatten for Transformer Encoder (100×7)
enc_in = flatten_for_encoder(pf_ds)
```

---

## 🚀 Quick Start
```bash
# 0) (선택) results 디렉터리 준비
mkdir -p results/{logs,plots,ckpt}

# 1) 테스트 스위트
pytest -q

# 2) 학습 실행 (train/train_ppo.py는 Step 6 원본 유지)
python train/train_ppo.py
```
학습 후 다음이 생성되는지 확인:
- `results/logs/` : 요약 로그/텐서보드 로그
- `results/ckpt/` : 체크포인트(`.pt`)
- `results/plots/` : (선택) 리워드/UR 그래프

> 런 네이밍 권장: `3dpp_d128_N30_lr1e-5_YYYYMMDD-HHMM`

---

## 📊 Metrics
- **UR (Utilization Rate)**: Σ_i l_i w_i h_i / (L·W·Ĥ)
- **Wasted Space(gap)**: terminal 시 g_n
- **학습 곡선**: mean reward, mean UR, losses(actor/critic/entropy), clip fraction, explained var

---

## 🗂 Results Conventions
- Logs: `results/logs/{run_name}.csv` 또는 `results/logs/{run_name}/`
- Plots: `results/plots/{run_name}_reward.png`, `{run_name}_ur.png`
- Ckpt: `results/ckpt/{run_name}_u{update:05d}.pt`

.gitignore 권장 설정:
```
results/logs/*
results/plots/*
results/ckpt/*
!results/**/.gitkeep
```

---

## 🗺 Roadmap
- [ ] Behavioral Cloning 사전학습(휴리스틱 궤적 모사)
- [ ] 대형 컨테이너(200×200, 400×200) 확장 실험
- [ ] 안정성/지지면 제약 추가
- [ ] 3D 시각화 도구 연동 (mesh/voxel)

---

## 📚 Reference
- **Que, Yang, Zhang (2023)**. *Solving 3D packing problem using Transformer network and reinforcement learning*. Expert Systems With Applications, 214:119153.

---

## 🙌 Acknowledgements
- ReturnAll(리터놀) 산학협력 배경의 문제 정의와 KPI 제공
- Open-source 커뮤니티 및 관련 연구에 감사드립니다.

---

## 📄 License
MIT 또는 내부 정책에 맞게 선택 후 표기하세요.

---

## 🔗 Project Resources

- 📄 **Notion 문서**: [팀 노션 페이지](https://www.notion.so/268beaebaf7c80329846c1c1f46c79b2?source=copy_link)
- 📕 **프로젝트 제안서 (PDF)**: [2025-09-08_3DPP_RL_Proposal_v1.0_ko.pdf](./2025-09-08_3DPP_RL_Proposal_v1.0_ko.pdf)
