# train/hpo_stage1_optuna.py

from __future__ import annotations
import os, sys, csv
from pathlib import Path

import optuna
from optuna.trial import TrialState, create_trial

# 프로젝트 루트 잡기 (train_ppo.py랑 같은 방식)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from train.train_ppo import TrainConfig, train, load_yaml
from envs.container_sim import EnvConfig


def make_base_cfgs() -> tuple[TrainConfig, EnvConfig]:
    """env.yaml / train.yaml을 읽어서 기본 cfg를 만든다."""
    tcfg = TrainConfig()
    # env 기본값
    ecfg = EnvConfig(L=100, W=100, N=20, seed=None, num_envs=4, max_steps=200)

    # ----- env.yaml -----
    env_path = PROJECT_ROOT / "configs" / "env.yaml"
    if env_path.exists():
        env_y = load_yaml(str(env_path))
        L = int(env_y["container"]["length"])
        W = int(env_y["container"]["width"])
        seed = int(env_y["container"]["seed"])
        N = int(env_y["boxes"]["count"])
        invalid_penalty = float(env_y["reward"]["invalid_penalty"])
        num_envs = int(env_y["env"]["num_envs"])
        max_steps = int(env_y["env"]["max_steps"])
        ecfg = EnvConfig(
            L=L, W=W, N=N,
            seed=seed,
            invalid_penalty=invalid_penalty,
            num_envs=num_envs,
            max_steps=max_steps,
        )

    # ----- train.yaml -----
    train_path = PROJECT_ROOT / "configs" / "train.yaml"
    if train_path.exists():
        train_y = load_yaml(str(train_path))
        ppo = train_y.get("ppo", {})
        opt = train_y.get("optimizer", {})
        sch = train_y.get("schedule", {})
        amp = train_y.get("amp", {})
        noise = train_y.get("parameter_noise", {})

        # PPO
        tcfg.gamma         = float(ppo.get("gamma", tcfg.gamma))
        tcfg.gae_lambda    = float(ppo.get("gae_lambda", tcfg.gae_lambda))
        tcfg.ppo_clip      = float(ppo.get("clip_epsilon", tcfg.ppo_clip))
        tcfg.entropy_coef  = float(ppo.get("entropy_coef", tcfg.entropy_coef))
        tcfg.kl_stop       = float(ppo.get("kl_stop", getattr(tcfg, "kl_stop", 0.03)))
        tcfg.clipfrac_stop = float(ppo.get("clipfrac_stop", getattr(tcfg, "clipfrac_stop", 0.35)))

        # LR
        tcfg.lr_actor  = float(opt.get("policy_lr", tcfg.lr_actor))
        tcfg.lr_critic = float(opt.get("value_lr", getattr(tcfg, "lr_critic", 1e-4)))

        # Schedule
        tcfg.n_steps           = int(sch.get("n_steps", tcfg.n_steps))
        tcfg.num_updates       = int(sch.get("num_updates", tcfg.num_updates))
        tcfg.epochs_per_update = int(sch.get("epochs_per_update", tcfg.epochs_per_update))
        tcfg.batch_size        = int(sch.get("batch_size", tcfg.batch_size))
        tcfg.log_interval      = int(sch.get("log_interval", tcfg.log_interval))
        tcfg.save_interval     = int(sch.get("save_interval", tcfg.save_interval))

        # AMP
        tcfg.grad_accum_steps = int(amp.get("grad_accum_steps", tcfg.grad_accum_steps))

        # Parameter Noise
        tcfg.param_noise_std  = float(noise.get("std", tcfg.param_noise_std))

        tcfg.pretrain_ckpt = None  # HPO 시에는 기본적으로 없음

    return tcfg, ecfg


def read_best_ur_fixed(log_path: str) -> float:
    if not os.path.exists(log_path):
        return 0.0

    best = 0.0
    with open(log_path, "r", newline="") as f:
        rd = csv.DictReader(f)
        for row in rd:
            try:
                val = row.get("UR_eval_fixed", "")
                if val in ("", "nan", "NaN", None):
                    continue
                ur = float(val)
            except Exception:
                continue

            if ur > best:
                best = ur

    return best


def resume_interrupted_trial_if_any(study: optuna.Study):
    trials = study.get_trials(deepcopy=False)

    # 1) RUNNING trial 우선 (전원 OFF 같은 케이스)
    running_trials = [t for t in trials if t.state == TrialState.RUNNING]
    if running_trials:
        trial = running_trials[-1]  # 가장 마지막 RUNNING 하나만 재개
        print(f"[resume] Found RUNNING trial #{trial.number}, params={trial.params}")

        # cfg 복원
        tcfg, ecfg = make_base_cfgs()
        tcfg.lr_actor = float(trial.params["lr_actor"])
        tcfg.lr_critic = float(trial.params["lr_critic"])
        tcfg.entropy_coef = float(trial.params["entropy_coef"])
        tcfg.param_noise_std = float(trial.params["param_noise_std"])

        tcfg.num_updates = 300          # 추가 구간 길이
        tcfg.pretrain_ckpt = "results/ckpt/hpo1_t010_u00288.pt"

        run_name = f"hpo1_t{trial.number:03d}"

        # train 재개 (ckpt/CSV 기준으로 알아서 resume)
        train(tcfg, ecfg, run_name, propagate_interrupt=True)

        # 완료 후, best UR_eval_fixed 읽어서 이 RUNNING trial을 COMPLETE로 마무리
        log_path = os.path.join("results", "logs", f"{run_name}.csv")
        best_ur = read_best_ur_fixed(log_path)
        print(f"[resume] Finished RUNNING trial #{trial.number}, best_ur={best_ur:.4f}")

        # RUNNING → COMPLETE 업데이트
        study.tell(trial.number, best_ur)
        return  # 여기서 끝

    # 2) RUNNING 없으면, FAIL 중에서 "아직 복구 안 된" 것만 찾기 (Ctrl+C 케이스)
    #    이미 같은 파라미터로 COMPLETE trial이 있으면 스킵
    complete_param_keys = {
        (t.params.get("lr_actor"),
         t.params.get("lr_critic"),
         t.params.get("entropy_coef"),
         t.params.get("param_noise_std"))
        for t in trials
        if t.state == TrialState.COMPLETE
    }

    fail_candidates = []
    for t in trials:
        if t.state != TrialState.FAIL:
            continue
        key = (
            t.params.get("lr_actor"),
            t.params.get("lr_critic"),
            t.params.get("entropy_coef"),
            t.params.get("param_noise_std"))
        if key in complete_param_keys:
            # 이미 동일 파라미터 조합으로 COMPLETE가 있으니 이 FAIL은 무시
            continue
        fail_candidates.append(t)

    if not fail_candidates:
        return

    # 가장 최근 FAIL 하나만 복구
    trial = fail_candidates[-1]
    print(f"[resume] Found pure FAILED trial #{trial.number}, params={trial.params}")

    # cfg 복원
    tcfg, ecfg = make_base_cfgs()
    tcfg.lr_actor = float(trial.params["lr_actor"])
    tcfg.lr_critic = float(trial.params["lr_critic"])
    tcfg.entropy_coef = float(trial.params["entropy_coef"])
    tcfg.param_noise_std = float(trial.params["param_noise_std"])
    tcfg.num_updates = 900

    run_name = f"hpo1_t{trial.number:03d}"

    # train 재개
    train(tcfg, ecfg, run_name, propagate_interrupt=True)

    # 로그에서 best UR 읽기
    log_path = os.path.join("results", "logs", f"{run_name}.csv")
    best_ur = read_best_ur_fixed(log_path)
    print(f"[resume] Finished resumed FAILED trial #{trial.number}, best_ur={best_ur:.4f}")

    # FAIL trial은 직접 상태를 바꾸지 않고,
    # 동일 파라미터에 대해 NEW COMPLETE trial 하나를 만들고 추가
    resumed_trial = create_trial(
        params=trial.params,
        distributions=trial.distributions,
        value=best_ur,
        state=TrialState.COMPLETE,
    )
    study.add_trial(resumed_trial)
    print(
        f"[resume] Recorded resumed trial as NEW trial #{resumed_trial.number} "
        f"(from failed trial #{trial.number}), value={best_ur:.4f}"
    )


def objective(trial: optuna.trial.Trial) -> float:
    tcfg, ecfg = make_base_cfgs()

    # 1) 튜닝할 하이퍼파라미터
    tcfg.lr_actor = trial.suggest_float("lr_actor", 5e-5, 5e-5, log=True)
    tcfg.lr_critic = trial.suggest_float("lr_critic", 3.0e-4, 3.0e-4, log=True)
    tcfg.entropy_coef = trial.suggest_float("entropy_coef", 2e-2, 2e-2, log=True)
    tcfg.param_noise_std = trial.suggest_float("param_noise_std", 0.015, 0.015, step=0.001)

    # 2) "추가" 업데이트 300번만 돌리기 (의미상 300→600 구간)
    tcfg.num_updates = 900

    # 3) update 300에서 만든 베이스 체크포인트를 출발점으로 사용
    tcfg.pretrain_ckpt = "results/ckpt/hpo1_t010_u00288.pt"   # 네가 만든 파일 경로로 수정

    run_name = f"hpo1_t{trial.number:03d}"

    train(tcfg, ecfg, run_name, propagate_interrupt=True)

    log_path = os.path.join("results", "logs", f"{run_name}.csv")
    best_ur = read_best_ur_fixed(log_path)
    return best_ur


if __name__ == "__main__":
    storage = "sqlite:///results/hpo_stage1.db"

    study = optuna.create_study(
        study_name="3dpp_stage1_lr_entropy",
        storage=storage,
        load_if_exists=True,
        direction="maximize",
    )

    # 1) 먼저, 이전에 실패한 trial이 있으면 그 trial부터 재개
    resume_interrupted_trial_if_any(study)

    # 2) 그 다음, 새 trial들에 대해 HPO 이어서 실행
    try:
        study.optimize(objective, n_trials=20)
    except KeyboardInterrupt:
        print("\n[HPO] Ctrl+C 감지 → 현재 trial에서 즉시 중단하고 파일 종료")

    # 3) best_trial 출력은 "완료된 trial이 있을 때만" 안전하게
    from optuna.trial import TrialState
    completed = [
        t for t in study.get_trials(deepcopy=False)
        if t.state == TrialState.COMPLETE
    ]
    if not completed:
        print("아직 완료된 trial이 없어서 best_trial을 계산할 수 없습니다.")
    else:
        best = max(completed, key=lambda t: t.value)
        print("=== [Stage1] Best Trial So Far ===")
        print("best value (max UR_eval_fixed):", best.value)
        print("best params:", best.params)
