# scripts/plot_logs.py
import csv, sys, os
from pathlib import Path
import matplotlib.pyplot as plt

path = Path(sys.argv[1])  # results/logs/xxx.csv
out_dir = Path("results/plots")
out_dir.mkdir(parents=True, exist_ok=True)

updates, ret, ur = [], [], []
with path.open("r", newline="") as f:
    rd = csv.DictReader(f)
    for row in rd:
        updates.append(int(row["update"]))
        ret.append(float(row["return_sum"]))
        ur.append(float(row["mean_UR"]))

stem = path.stem  # 예: 3dpp_20250908-123456

# Return plot -> PNG 저장
plt.figure()
plt.plot(updates, ret)
plt.title("Return (sum per update)")
plt.xlabel("update"); plt.ylabel("return")
plt.tight_layout()
plt.savefig(out_dir / f"{stem}_return.png", dpi=150)
plt.close()

# UR plot -> PNG 저장
plt.figure()
plt.plot(updates, ur)
plt.title("Utilization Rate")
plt.xlabel("update"); plt.ylabel("UR")
plt.tight_layout()
plt.savefig(out_dir / f"{stem}_ur.png", dpi=150)
plt.close()

print(f"saved: {out_dir / f'{stem}_return.png'}")
print(f"saved: {out_dir / f'{stem}_ur.png'}")
