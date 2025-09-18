# 2025-2학기 경북대학교

## 프로젝트 개요
- 강화학습 기반 3D 컨테이너 적재 최적화 시스템 개발
  - 3D 포장 문제 정의 및 환경 구성
  - 트랜스포머 기반 강화학습 에이전트 학습
  - 시뮬레이션 및 시각화 기반 실험 수행

## 프로젝트 문서
- 프로젝트 문서는 아래 Notion 페이지에서 정리합니다.
- Notion : https://www.notion.so/returnall/2025-2-26adad4051a680a7bedfd65b457deed5?source=copy_link

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

## 📁 Folder Structure (핵심 역할 주석 포함)

```text
projects/
├─ agents/
│  ├─ backbone.py            # Box/Container 인코더 포함 공용 Transformer 백본(포지셔널 인코딩·멀티헤드·층수 등 모델 골격)
│  ├─ heads.py               # 세 디코더(위치/선택/방향) + PositionEmbeddingBuilder + OrientationEmbedder; log_softmax 출력
│  └─ value_head.py          # 정책과 입력 동일 구조로 V(s) 스칼라 추정하는 ValueNet(critic 헤드)
│
├─ configs/
│  ├─ env.yaml               # 환경 파라미터: 컨테이너 L/W/H, seed, 박스크기분포·개수, invalid_penalty 등
│  ├─ model.yaml             # 모델 파라미터: d_model, 인코더/디코더 레이어 수, heads(enc/dec), dropout, orient_classes 등
│  └─ train.yaml             # 학습 파라미터: PPO(γ, λ, clip, entropy/value coef), lr(policy/value), 배치·에폭·저장주기 등
│
├─ envs/
│  └─ container_sim.py       # heightmap 기반 3D 적재 Gym 환경: action=(x,y,box_idx,orient), 충돌/경계/지지면 체크, 보상/UR 계산
│
├─ results/
│  ├─ ckpt/                  # 체크포인트 저장(자동 재개용 *_latest_{resume,post}.pt + 마일스톤 *_u{global}.pt)
│  ├─ logs/                  # CSV/TensorBoard 로그
│  └─ plots/                 # 학습 곡선/지표 시각화 이미지
│
├─ tests/
│  ├─ test_backbone.py       # backbone 입출력 shape·마스킹·attention 동작 단위테스트
│  ├─ test_env.py            # 환경 step/reset/보상/terminal·invalid penalty·gap 계산 검증
│  ├─ test_heads.py          # 위치/선택/방향 디코더 확률분포 합=1·shape 검증, 위치임베딩 빌더 확인
│  ├─ test_preprocess.py     # 전처리 다운샘플/플래튼/경계거리 채널 생성 검증
│  └─ test_value_head.py     # ValueNet 전파·출력 스칼라·loss 역전파 동작 확인
│
├─ train/
│  └─ train_ppo.py           # PPO+GAE 학습 스크립트: YAML 로드, 루프/로깅/플롯, 안전 저장·자동 재개·KeyboardInterrupt 대응
│
├─ utils/
│  └─ preprocess.py          # 컨테이너 7채널 plane features 생성, 100×100→10×10 패치 다운샘플, encoder용 flatten 유틸
│
├─ .gitignore                # venv/ckpt/logs/plots/pycache 등 제외 규칙
├─ README.md                 # 프로젝트 개요/설치/실행법/구조/지표 설명(본 섹션 붙여넣기 위치)
├─ requirements-cpu.txt      # CPU 환경 의존성(pytorch CPU 빌드 등)
└─ requirements-gpu.txt      # GPU 환경 의존성(cuda/cudnn 맞춤 pytorch 버전 등)
```

### 파일별 핵심 개념 요약
- **agents/backbone.py**: Box/Container 두 인코더를 통해 상태를 임베딩하고, 디코더·Value 헤드가 재사용할 공용 표현을 만듭니다.  
- **agents/heads.py**: 체인룰 정책(**위치→박스→방향**)을 구현하는 세 디코더와 보조 임베딩 모듈을 제공합니다.  
- **agents/value_head.py**: 동일 입력으로 상태가치 V(s)를 추정해 PPO의 critic 손실을 계산합니다.  
- **configs/\*.yaml**: 실험을 코드 수정 없이 바꾸도록 분리(환경/모델/훈련 하이퍼).  
- **envs/container_sim.py**: 높이맵 기반 쌓기·충돌·경계·지지면(안정성) 규칙을 적용하고, 보상 r = g_{i-1} − g_i 및 활용률(UR)을 계산합니다.  
- **results/**: 학습 산출물 표준 경로(재현성·중단복구).  
- **tests/**: 각 컴포넌트별 최소 보증(스모크+shape+확률합+수치 검증).  
- **train/train_ppo.py**: PPO 루프(수집→업데이트→로그), 체크포인트 네이밍 규칙과 자동 재개 우선순위(pre→post→milestone) 포함.  
- **utils/preprocess.py**: 경계/에지/높이 등 7채널 plane features와 패치 다운샘플(100×100×7→10×10×7→flatten 100×7).  
- **requirements-\*.txt**: 환경 재현용 의존성 핀 고정.  

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
- 📕 **프로젝트 계획서 (PDF)**: [열기](./docs/2025-09-08_3DPP_RL_Proposal_v1.0_ko.pdf)
- 📝 **수행계획서 양식 (HWP)**: [다운로드](./docs/[양식]산학협력프로젝트_수행계획서_프로젝트명.hwp)