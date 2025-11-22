import os, csv, torch, tempfile, shutil

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
    """update > rollback_to 인 모든 줄 삭제 (안전하게 재기록)"""
    if not os.path.exists(log_path):
        return

    with open(log_path, "r", newline="", encoding="utf-8") as f:
        lines = f.readlines()

    if not lines:
        # 빈 파일이면 건드리지 않고 종료
        return

    # 1) 헤더 정규화(셀 단위 strip + BOM 제거)
    raw_header = lines[0].rstrip("\n")
    header = [h.lstrip("\ufeff").strip() for h in raw_header.split(",")]

    keep = []
    # 2) 강제 헤더로 파싱(초과키 무시, 누락키 채움)
    reader = csv.DictReader(
        lines[1:], fieldnames=header, restval="", skipinitialspace=True
    )

    for row in reader:
        try:
            row.pop(None, None)  # 초과 열 제거
            row = {k: row.get(k, "") for k in header}  # 누락 키 채움

            # update 필드가 없거나 숫자 변환 실패 시 스킵
            if "update" not in row:
                continue
            gu = int(row["update"])
            if gu <= rollback_to:
                keep.append(row)
        except Exception:
            continue

    # 3) 원자적 쓰기: 임시 파일에 기록 후 교체
    dirpath = os.path.dirname(log_path) or "."
    fd, tmp_path = tempfile.mkstemp(prefix=".rollback_", dir=dirpath)
    os.close(fd)
    try:
        with open(tmp_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
            w.writeheader()
            w.writerows(keep)
        shutil.move(tmp_path, log_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


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
