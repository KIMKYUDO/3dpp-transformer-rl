#from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import numpy as np

try:
    # 전처리와 연동이 필요할 때 사용 (테스트에서는 옵션)
    from utils.preprocess import compute_plane_features
except Exception:
    compute_plane_features = None


# Box 타입
Box = Tuple[int, int, int]  # (l, w, h)


@dataclass
class EnvConfig:
    L: int = 100
    W: int = 100
    N: int = 20
    l_range: Tuple[int, int] = (10, 50)
    w_range: Tuple[int, int] = (10, 50)
    h_range: Tuple[int, int] = (10, 50)
    seed: Optional[int] = None


class PackingEnv:
    """Heightmap 기반 3D-PP 환경.

    - 상태: heightmap (H=W=컨테이너 평면), 내부적으로 gap/UR 계산 지원
    - step(action): action = (x, y, box_idx, orient_idx)
      * x, y: 좌상단 좌표 (정수 그리드)
      * box_idx: 미배치 박스 인덱스
      * orient_idx: 0..5 (6가지 직교 회전)
    """

    def __init__(self, cfg: EnvConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.L = int(cfg.L)
        self.W = int(cfg.W)
        self.height = np.zeros((self.W, self.L), dtype=np.int32)  # (H, W) = (y, x)
        self.boxes: List[Box] = []
        self.used = np.zeros(0, dtype=bool)
        self.t = 0
        self._sum_vol = 0  # \sum lwh for placed boxes
        self._g_prev = 0   # g_{i-1}

    # ---------- 유틸 ----------
    @staticmethod
    def _orientations(l: int, w: int, h: int) -> List[Box]:
        # 6개의 축정렬 회전 (중복 제거)
        perms = {
            (l, w, h), (l, h, w), (w, l, h), (w, h, l), (h, l, w), (h, w, l)
        }
        return list(perms)

    def _max_height_in_region(self, x0: int, y0: int, l: int, w: int) -> int:
        region = self.height[y0:y0+w, x0:x0+l]
        if region.size == 0:
            return 0
        return int(region.max(initial=0))

    def _raise_region(self, x0: int, y0: int, l: int, w: int, top: int, h: int) -> None:
        # region을 top+h까지 채움 (바닥은 top)
        self.height[y0:y0+w, x0:x0+l] = top + h

    def _H_tilde(self) -> int:
        return int(self.height.max(initial=0))

    def _gap(self) -> int:
        # g_i = L*W*H_tilde_i - sum_volumes
        return int(self.L * self.W * self._H_tilde() - self._sum_vol)

    # ---------- API ----------
    def reset(self, boxes: Optional[List[Box]] = None) -> Dict:
        self.height.fill(0)
        self.t = 0
        self._sum_vol = 0
        self._g_prev = 0
        if boxes is None:
            self.boxes = self._sample_boxes()
        else:
            self.boxes = [tuple(map(int, b)) for b in boxes]
        self.used = np.zeros(len(self.boxes), dtype=bool)
        obs = {
            "height": self.height.copy(),
            "remaining": (~self.used).sum(),
            "gap": self._gap(),
        }
        return obs

    def _sample_boxes(self) -> List[Box]:
        Lr = self.cfg.l_range
        Wr = self.cfg.w_range
        Hr = self.cfg.h_range
        boxes = []
        for _ in range(self.cfg.N):
            l = int(self.rng.integers(Lr[0], Lr[1] + 1))
            w = int(self.rng.integers(Wr[0], Wr[1] + 1))
            h = int(self.rng.integers(Hr[0], Hr[1] + 1))
            boxes.append((l, w, h))
        return boxes

    def step(self, action: Tuple[int, int, int, int]):
        """Place a box with action (x, y, box_idx, orient_idx).
        Returns: obs, reward, done, info
        """
        x, y, bidx, oidx = map(int, action)
        assert 0 <= bidx < len(self.boxes), "invalid box index"
        assert not self.used[bidx], "box already used"

        # 회전 적용
        l, w, h = self.boxes[bidx]
        orients = self._orientations(l, w, h)
        assert 0 <= oidx < len(orients), "invalid orientation index"
        l2, w2, h2 = orients[oidx]

        # 경계 체크 (좌상단 기준, 우/하 경계 포함 X)
        if x < 0 or y < 0 or x + l2 > self.L or y + w2 > self.W:
            # 간단히 무효 처리: 큰 페널티 주고 종료하지는 않음
            reward = -1.0
            info = {"invalid": True}
            obs = {"height": self.height.copy(), "remaining": (~self.used).sum(), "gap": self._gap()}
            return obs, reward, False, info

        # 지지면: 해당 영역의 현재 최대 높이 위에 올린다 (스텝함수 높이맵).
        top = self._max_height_in_region(x, y, l2, w2)
        self._raise_region(x, y, l2, w2, top, h2)

        # 누적 부피/단계 증가
        self.used[bidx] = True
        self.t += 1
        self._sum_vol += int(l2 * w2 * h2)

        g_now = self._gap()
        reward = float(self._g_prev - g_now)  # r_i = g_{i-1} - g_i
        self._g_prev = g_now

        done = bool(self.used.all())
        info = {
            "H_tilde": self._H_tilde(),
            "gap": g_now,
            "sum_vol": self._sum_vol,
            "placed_box": (l2, w2, h2),
        }
        obs = {
            "height": self.height.copy(),
            "remaining": (~self.used).sum(),
            "gap": g_now,
        }
        return obs, reward, done, info

    # 선택: 정책 입력용 컨테이너 상태 7채널
    def container_state7(self) -> Optional[np.ndarray]:
        if compute_plane_features is None:
            return None
        return compute_plane_features(self.height.astype(np.float32))

    def utilization_rate(self) -> float:
        Ht = self._H_tilde()
        denom = float(self.L * self.W * max(Ht, 1))
        if denom == 0:
            return 0.0
        return float(self._sum_vol) / denom
