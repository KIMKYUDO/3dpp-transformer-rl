import os, csv, torch

def read_last_update_from_csv(log_path: str) -> int:
    """CSV 파일에서 마지막 update 값을 읽음"""
    if not os.path.exists(log_path):
        return 0
    last_update = 0
    with open(log_path, "r", newline="") as f:
        rd = csv.DictReader(f)
        for row in rd:
            try:
                gu = int(row.get("update", 0))
                last_update = gu
            except Exception:
                continue
    return last_update


def rollback_csv(log_path: str, rollback_to: int):
    """update > rollback_to 인 모든 줄 삭제"""
    if not os.path.exists(log_path):
        return
    with open(log_path, "r") as f:
        lines = f.readlines()
    if not lines:
        return

    header = lines[0].strip().split(",")
    keep = []
    for row in csv.DictReader(lines):
        try:
            gu = int(row["update"])
            if gu <= rollback_to:
                keep.append(row)
        except Exception:
            continue

    with open(log_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        w.writerows(keep)


def resolve_resume(cfg, run_name: str, log_path: str):
    """
    Resume-safe 로직:
    - save_interval 배수에서 멈추면 그 ckpt는 버리고 이전 milestone으로 롤백
    - 그 외에는 milestone까지만 보존
    """
    last_update = read_last_update_from_csv(log_path)
    if last_update == 0:
        return 0, None

    milestone = (last_update // cfg.save_interval) * cfg.save_interval

    if last_update == milestone:
        # === 정확히 milestone에서 멈춤 ===
        bad_ckpt = os.path.join(cfg.ckpt_dir, f"{run_name}_u{milestone:05d}.pt")
        if os.path.exists(bad_ckpt):
            os.remove(bad_ckpt)

        rollback_to = milestone - cfg.save_interval
        rollback_csv(log_path, rollback_to)

        ckpt_path = os.path.join(cfg.ckpt_dir, f"{run_name}_u{rollback_to:05d}.pt")
        return rollback_to, ckpt_path if os.path.exists(ckpt_path) else None
    else:
        # === 배수가 아닌 곳에서 멈춤 ===
        rollback_to = milestone
        rollback_csv(log_path, rollback_to)

        ckpt_path = os.path.join(cfg.ckpt_dir, f"{run_name}_u{rollback_to:05d}.pt")
        return rollback_to, ckpt_path if os.path.exists(ckpt_path) else None
