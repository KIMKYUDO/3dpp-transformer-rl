# Transformer-based 3D Container Packing with PPO (ReturnAll 협업)

강화학습(PPO)과 트랜스포머 정책망으로 3D 컨테이너 적재(가변 높이)를 최적화하는 프로젝트입니다. 목표 지표는 **공간 활용률(UR)** 극대화이며, 논문 *"Solving 3D packing problem using Transformer network and reinforcement learning" (Que et al., ESWA 2023)*의 핵심 아이디어(상태 표현, Plane Features, 액션 순서, 다운샘플링)를 구현했습니다.

---

## ✅ Key Features
- **상태 분리 표현**: Box state + Container state(Heightmap + Plane Features)
- **트랜스포머 백본**: Box/Container 인코더 + Position/Selection/Orientation 디코더
- **체인룰 정책**: π(a|s)=π^p·π^s·π^o (Position→Selection→Orientation)
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

## 🧠 Method Brief
- **Container State 7채널 (100×100×7)**: 높이 ℎ, 경계거리 e_l/e_w/e_-l/e_-w, 이웃 고지점 거리 f_l/f_w
- **Downsampling**: 10×10 패치로 나눈 뒤, 각 패치에서 `argmax(e_l×e_w)` 위치의 피처만 보존 → 10×10×7 → flatten(100×7)
- **Encoders**: Box/Container 각각 Transformer Encoder
- **Decoders**:
  - Position(Q=ContainerEnc, KV=BoxEnc) → (x,y)
  - Position Embedding Builder(컨테이너 enc (x,y) + 원시 state(x,y) 결합)
  - Selection(Q=BoxEnc, KV=PosEmb) → box index
  - OrientationEmbedder(선택 박스 6회전 임베딩) → Orientation(Q=OrientEmb, KV=PosEmb)
- **Policy Factorization**: π(a|s)=π^p(a_p|s)·π^s(a_s|a_p,s)·π^o(a_o|a_p,a_s,s)
- **Reward**: r_i = g_{i-1} - g_i,  g_i = L·W·Ĥ_i − Σ_j l_j w_j h_j
- **Objective**: PPO(clip) + value loss + β·entropy

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

## 🧰 Troubleshooting
- **CUDA가 인식되지 않아요**: GPU 환경이면 `requirements-gpu.txt` 사용, PyTorch CUDA 빌드 확인, 드라이버/툴킷 버전 호환성 점검.
- **학습이 매우 느려요**: 처음엔 스모크런(예: `num_updates=2`, `batch_size=1`)으로 파이프라인만 검증 후 본 학습으로 전환.
- **메모리 이슈**: `d_model` 축소(64/96), batch size 축소, 시퀀스 길이(박스 수) 줄이기.

---

## 🗺 Roadmap
- [ ] Behavioral Cloning 사전학습(휴리스틱 궤적模仿)
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

