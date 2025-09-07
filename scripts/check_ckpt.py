import torch

path = r"results/ckpt/ppo_step_0001.pt"
ckpt = torch.load(path, map_location="cpu")

print("== keys ==")
print(list(ckpt.keys()))

for k, v in ckpt.items():
    if hasattr(v, "keys"):
        print(f"[{k}] -> {list(v.keys())[:10]}")
