"""
Curriculum Learning Scheduler for 3D Packing Problem
점진적 난이도 증가로 학습 안정성 향상
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, List
import numpy as np


@dataclass
class CurriculumStage:
    """각 학습 단계의 난이도 설정"""
    name: str
    box_count_range: Tuple[int, int]  # (min, max) 박스 개수
    size_range: Tuple[int, int]        # (min, max) 박스 크기
    duration_updates: int              # 이 단계에서 머무를 업데이트 횟수
    success_threshold: float = 0.0     # 다음 단계로 넘어가는 UR 기준 (선택)


class CurriculumScheduler:
    """
    학습 진행에 따라 환경 난이도를 자동 조정
    
    Usage:
        scheduler = CurriculumScheduler()
        for update in range(num_updates):
            stage = scheduler.get_stage(update)
            env_cfg.N = stage.sample_box_count()
            env_cfg.l_range = stage.size_range
            # ... train
    """
    
    def __init__(self, stages: List[CurriculumStage] | None = None):
        if stages is None:
            # 기본 4단계 커리큘럼 (논문 Table 2 기준)
            self.stages = [
                CurriculumStage(
                    name="Stage1_Easy",
                    box_count_range=(10, 10),
                    size_range=(20, 40),
                    duration_updates=500,  # 초기 500 업데이트
                    success_threshold=0.70
                ),
                CurriculumStage(
                    name="Stage2_Medium",
                    box_count_range=(15, 15),
                    size_range=(15, 45),
                    duration_updates=1000,
                    success_threshold=0.75
                ),
                CurriculumStage(
                    name="Stage3_Hard",
                    box_count_range=(20, 20),
                    size_range=(12, 50),
                    duration_updates=2000,
                    success_threshold=0.80
                ),
                CurriculumStage(
                    name="Stage4_Expert",
                    box_count_range=(20, 20),
                    size_range=(10, 50),
                    duration_updates=999999,  # 끝까지
                    success_threshold=0.82
                )
            ]
        else:
            self.stages = stages
        
        # 누적 업데이트 경계값 계산
        self._boundaries = []
        cumsum = 0
        for stage in self.stages[:-1]:  # 마지막 단계는 무한
            cumsum += stage.duration_updates
            self._boundaries.append(cumsum)
        
        self.rng = np.random.default_rng()
    
    def get_stage(self, current_update: int) -> CurriculumStage:
        """현재 업데이트에 맞는 단계 반환"""
        for i, boundary in enumerate(self._boundaries):
            if current_update < boundary:
                return self.stages[i]
        return self.stages[-1]  # 최종 단계
    
    def get_stage_info(self, current_update: int) -> dict:
        """현재 단계 정보 (로깅용)"""
        stage = self.get_stage(current_update)
        stage_idx = self.stages.index(stage)
        
        # 진행률 계산
        if stage_idx == 0:
            prev_boundary = 0
        else:
            prev_boundary = self._boundaries[stage_idx - 1]
        
        if stage_idx < len(self._boundaries):
            next_boundary = self._boundaries[stage_idx]
            progress = (current_update - prev_boundary) / stage.duration_updates
        else:
            progress = 1.0  # 최종 단계
        
        return {
            "stage_name": stage.name,
            "stage_index": stage_idx + 1,
            "total_stages": len(self.stages),
            "progress": min(progress, 1.0),
            "box_count_range": stage.box_count_range,
            "size_range": stage.size_range
        }
    
    def sample_box_count(self, current_update: int) -> int:
        """현재 단계에서 박스 개수 샘플링"""
        stage = self.get_stage(current_update)
        min_n, max_n = stage.box_count_range
        return int(self.rng.integers(min_n, max_n + 1))
    
    def should_advance(self, current_update: int, current_ur: float) -> bool:
        """
        성공률 기반 조기 단계 전환 판정 (선택 기능)
        
        Returns:
            True if ready to advance to next stage early
        """
        stage = self.get_stage(current_update)
        stage_idx = self.stages.index(stage)
        
        # 마지막 단계면 전환 불가
        if stage_idx >= len(self.stages) - 1:
            return False
        
        # UR이 threshold를 초과하면 조기 전환 허용
        return current_ur >= stage.success_threshold