import numpy as np
import pytest
from envs.container_sim import PackingEnv, EnvConfig


def test_reset_initial_gap_zero():
    env = PackingEnv(EnvConfig(L=10, W=10, N=0, seed=42))
    obs = env.reset()
    assert obs["gap"] == 0
    assert env._H_tilde() == 0


def test_single_box_reward_and_terminal_gap():
    env = PackingEnv(EnvConfig(L=10, W=10, N=1, seed=0))
    # 고정 박스 1개: (l,w,h)=(5,5,5)
    env.reset(boxes=[(5,5,5)])

    # (0,0)에 배치, orientation=0
    obs, r1, done, info = env.step((0, 0, 0, 0))
    # g1 = L*W*Ht - sum_vol = 10*10*5 - 5*5*5 = 500 - 125 = 375
    assert info["H_tilde"] == 5
    assert info["gap"] == 500 - 125
    assert pytest.approx(r1, rel=0, abs=1e-6) == -375.0  # r1 = g0 - g1 = -g1
    assert done is True


def test_two_boxes_terminal_gap_equals_wasted_space():
    env = PackingEnv(EnvConfig(L=10, W=10, N=2, seed=0))
    env.reset(boxes=[(5,5,5), (5,5,4)])

    # 첫 박스 (0,0)
    obs, r1, done, info1 = env.step((0, 0, 0, 0))
    # 두 번째 박스 (5,0) 옆에
    obs, r2, done, info2 = env.step((5, 0, 1, 0))

    # 최종 H~ = 5, 총부피 = 125 + 100 = 225
    # g_N = 10*10*5 - 225 = 500 - 225 = 275
    assert done is True
    assert info2["H_tilde"] == 5
    assert info2["gap"] == 500 - 225


def test_out_of_bounds_penalty():
    env = PackingEnv(EnvConfig(L=10, W=10, N=1, seed=0))
    env.reset(boxes=[(8,8,2)])
    # (5,5)에 두면 5+8>10 → 경계 밖 → invalid
    obs, r, done, info = env.step((5, 5, 0, 0))
    assert info.get("invalid", False) is True
    assert r == -1.0
    assert done is False
