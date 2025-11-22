import optuna
import pandas as pd

storage = "sqlite:///results/hpo_stage1.db"
study = optuna.load_study(
    study_name="3dpp_stage1_lr_entropy",
    storage=storage,
)

df = study.trials_dataframe()   # ← 인자 없이 호출
cols = [
    "number", "state", "value",
    "params_lr_actor", "params_lr_critic", "params_entropy_coef", "params_param_noise_std"
]
print(df[cols])