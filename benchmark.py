#!/usr/bin/env python3
"""
TF32 vs FP32 속도 비교 벤치마크
TF32가 FP32보다 빠르다는 것을 명확하게 증명합니다.
"""

import torch
import time
import sys
import numpy as np
from typing import List, Tuple

print("=" * 80)
print("TF32 vs FP32 속도 벤치마크")
print("=" * 80)

# ============================================================================
# 1. 환경 확인
# ============================================================================

print("\n[1] 환경 확인")
print("-" * 80)

# PyTorch 버전
print(f"PyTorch 버전: {torch.__version__}")
pytorch_version = tuple(map(int, torch.__version__.split('.')[:2]))

# CUDA 사용 가능 여부
if not torch.cuda.is_available():
    print("❌ ERROR: CUDA를 사용할 수 없습니다!")
    sys.exit(1)

print(f"✅ CUDA 사용 가능")

# GPU 정보
device_name = torch.cuda.get_device_name(0)
compute_capability = torch.cuda.get_device_capability(0)
cc = float(f"{compute_capability[0]}.{compute_capability[1]}")

print(f"GPU: {device_name}")
print(f"Compute Capability: {cc}")

# TF32 지원 확인
if cc >= 8.0:
    print(f"✅ Ampere 이상 GPU - TF32 지원!")
    tf32_supported = True
elif cc >= 7.0:
    print(f"⚠️  Volta/Turing GPU - TF32 미지원 (테스트 진행하지만 차이 없을 것)")
    tf32_supported = False
else:
    print(f"❌ TF32 미지원 GPU - 테스트 의미 없음")
    tf32_supported = False

# ============================================================================
# 2. 벤치마크 함수들
# ============================================================================

def benchmark_matmul(size: int, dtype: torch.dtype, use_tf32: bool, 
                     iterations: int = 100, warmup: int = 10) -> float:
    """행렬 곱셈 벤치마크"""
    
    device = torch.device('cuda')
    A = torch.randn(size, size, dtype=dtype, device=device)
    B = torch.randn(size, size, dtype=dtype, device=device)
    
    # TF32 설정
    if dtype == torch.float32:
        torch.backends.cuda.matmul.allow_tf32 = use_tf32
        torch.backends.cudnn.allow_tf32 = use_tf32
        if pytorch_version >= (2, 0):
            torch.set_float32_matmul_precision("high" if use_tf32 else "highest")
    
    # Warmup
    for _ in range(warmup):
        C = A @ B
    torch.cuda.synchronize()
    
    # 측정
    torch.cuda.synchronize()
    start = time.perf_counter()
    
    for _ in range(iterations):
        C = A @ B
    
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    
    return elapsed / iterations


def benchmark_linear_layers(batch: int, seq_len: int, hidden: int, 
                            use_tf32: bool, iterations: int = 100) -> float:
    """Linear layer 벤치마크 (Transformer 시뮬레이션)"""
    
    device = torch.device('cuda')
    
    # TF32 설정
    torch.backends.cuda.matmul.allow_tf32 = use_tf32
    torch.backends.cudnn.allow_tf32 = use_tf32
    if pytorch_version >= (2, 0):
        torch.set_float32_matmul_precision("high" if use_tf32 else "highest")
    
    # 간단한 네트워크 (Transformer FFN 유사)
    model = torch.nn.Sequential(
        torch.nn.Linear(hidden, hidden * 4),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden * 4, hidden)
    ).cuda()
    
    x = torch.randn(batch, seq_len, hidden, device=device)
    
    # Warmup
    for _ in range(10):
        _ = model(x)
    torch.cuda.synchronize()
    
    # 측정
    torch.cuda.synchronize()
    start = time.perf_counter()
    
    for _ in range(iterations):
        _ = model(x)
    
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    
    return elapsed / iterations


def benchmark_attention(batch: int, seq_len: int, hidden: int, 
                       use_tf32: bool, iterations: int = 50) -> float:
    """Self-Attention 벤치마크"""
    
    device = torch.device('cuda')
    
    # TF32 설정
    torch.backends.cuda.matmul.allow_tf32 = use_tf32
    torch.backends.cudnn.allow_tf32 = use_tf32
    if pytorch_version >= (2, 0):
        torch.set_float32_matmul_precision("high" if use_tf32 else "highest")
    
    Q = torch.randn(batch, seq_len, hidden, device=device)
    K = torch.randn(batch, seq_len, hidden, device=device)
    V = torch.randn(batch, seq_len, hidden, device=device)
    
    # Warmup
    for _ in range(10):
        scores = Q @ K.transpose(-2, -1) / (hidden ** 0.5)
        attn = torch.softmax(scores, dim=-1)
        out = attn @ V
    torch.cuda.synchronize()
    
    # 측정
    torch.cuda.synchronize()
    start = time.perf_counter()
    
    for _ in range(iterations):
        scores = Q @ K.transpose(-2, -1) / (hidden ** 0.5)
        attn = torch.softmax(scores, dim=-1)
        out = attn @ V
    
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    
    return elapsed / iterations


def benchmark_backward(size: int, use_tf32: bool, iterations: int = 50) -> float:
    """Forward + Backward 벤치마크"""
    
    device = torch.device('cuda')
    
    # TF32 설정
    torch.backends.cuda.matmul.allow_tf32 = use_tf32
    torch.backends.cudnn.allow_tf32 = use_tf32
    if pytorch_version >= (2, 0):
        torch.set_float32_matmul_precision("high" if use_tf32 else "highest")
    
    model = torch.nn.Sequential(
        torch.nn.Linear(size, size * 2),
        torch.nn.ReLU(),
        torch.nn.Linear(size * 2, size),
    ).cuda()
    
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    
    x = torch.randn(64, size, device=device)
    target = torch.randn(64, size, device=device)
    
    # Warmup
    for _ in range(10):
        optimizer.zero_grad()
        output = model(x)
        loss = torch.nn.functional.mse_loss(output, target)
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize()
    
    # 측정
    torch.cuda.synchronize()
    start = time.perf_counter()
    
    for _ in range(iterations):
        optimizer.zero_grad()
        output = model(x)
        loss = torch.nn.functional.mse_loss(output, target)
        loss.backward()
        optimizer.step()
    
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    
    return elapsed / iterations


# ============================================================================
# 3. 벤치마크 실행
# ============================================================================

print("\n[2] 벤치마크 시작")
print("-" * 80)

results = []

# Test 1: 순수 행렬 곱셈 (다양한 크기)
print("\n📊 Test 1: 순수 행렬 곱셈 (A @ B)")
print("-" * 80)

matrix_sizes = [512, 1024, 2048, 4096]
for size in matrix_sizes:
    print(f"\n크기: {size}x{size}")
    
    # FP32
    time_fp32 = benchmark_matmul(size, torch.float32, use_tf32=False, iterations=100)
    print(f"  FP32 (highest): {time_fp32*1000:.3f}ms")
    
    # TF32
    time_tf32 = benchmark_matmul(size, torch.float32, use_tf32=True, iterations=100)
    print(f"  TF32 (high):    {time_tf32*1000:.3f}ms")
    
    speedup = time_fp32 / time_tf32
    print(f"  ⚡ 속도 향상:     {speedup:.2f}x")
    
    results.append({
        'test': f'MatMul {size}x{size}',
        'fp32': time_fp32 * 1000,
        'tf32': time_tf32 * 1000,
        'speedup': speedup
    })


# Test 2: Linear Layers (Transformer FFN)
print("\n\n📊 Test 2: Linear Layers (Transformer FFN)")
print("-" * 80)

configs = [
    (32, 128, 512),   # (batch, seq_len, hidden)
    (16, 256, 768),
    (8, 512, 1024),
]

for batch, seq_len, hidden in configs:
    print(f"\n설정: batch={batch}, seq={seq_len}, hidden={hidden}")
    
    # FP32
    time_fp32 = benchmark_linear_layers(batch, seq_len, hidden, use_tf32=False, iterations=100)
    print(f"  FP32: {time_fp32*1000:.3f}ms")
    
    # TF32
    time_tf32 = benchmark_linear_layers(batch, seq_len, hidden, use_tf32=True, iterations=100)
    print(f"  TF32: {time_tf32*1000:.3f}ms")
    
    speedup = time_fp32 / time_tf32
    print(f"  ⚡ 속도 향상: {speedup:.2f}x")
    
    results.append({
        'test': f'Linear B{batch}S{seq_len}H{hidden}',
        'fp32': time_fp32 * 1000,
        'tf32': time_tf32 * 1000,
        'speedup': speedup
    })


# Test 3: Self-Attention
print("\n\n📊 Test 3: Self-Attention (Q @ K^T, Attn @ V)")
print("-" * 80)

attn_configs = [
    (16, 128, 512),
    (8, 256, 768),
    (4, 512, 1024),
]

for batch, seq_len, hidden in attn_configs:
    print(f"\n설정: batch={batch}, seq={seq_len}, hidden={hidden}")
    
    # FP32
    time_fp32 = benchmark_attention(batch, seq_len, hidden, use_tf32=False, iterations=50)
    print(f"  FP32: {time_fp32*1000:.3f}ms")
    
    # TF32
    time_tf32 = benchmark_attention(batch, seq_len, hidden, use_tf32=True, iterations=50)
    print(f"  TF32: {time_tf32*1000:.3f}ms")
    
    speedup = time_fp32 / time_tf32
    print(f"  ⚡ 속도 향상: {speedup:.2f}x")
    
    results.append({
        'test': f'Attention B{batch}S{seq_len}H{hidden}',
        'fp32': time_fp32 * 1000,
        'tf32': time_tf32 * 1000,
        'speedup': speedup
    })


# Test 4: Forward + Backward
print("\n\n📊 Test 4: Forward + Backward (학습 시뮬레이션)")
print("-" * 80)

train_sizes = [256, 512, 1024]
for size in train_sizes:
    print(f"\n크기: {size}")
    
    # FP32
    time_fp32 = benchmark_backward(size, use_tf32=False, iterations=50)
    print(f"  FP32: {time_fp32*1000:.3f}ms")
    
    # TF32
    time_tf32 = benchmark_backward(size, use_tf32=True, iterations=50)
    print(f"  TF32: {time_tf32*1000:.3f}ms")
    
    speedup = time_fp32 / time_tf32
    print(f"  ⚡ 속도 향상: {speedup:.2f}x")
    
    results.append({
        'test': f'Training {size}',
        'fp32': time_fp32 * 1000,
        'tf32': time_tf32 * 1000,
        'speedup': speedup
    })


# ============================================================================
# 4. 결과 요약
# ============================================================================

print("\n\n" + "=" * 80)
print("📊 최종 결과 요약")
print("=" * 80)

print(f"\n{'테스트':<30} {'FP32 (ms)':<12} {'TF32 (ms)':<12} {'속도 향상':<12}")
print("-" * 80)

speedups = []
for r in results:
    print(f"{r['test']:<30} {r['fp32']:>10.3f}ms {r['tf32']:>10.3f}ms {r['speedup']:>10.2f}x")
    speedups.append(r['speedup'])

avg_speedup = np.mean(speedups)
min_speedup = np.min(speedups)
max_speedup = np.max(speedups)

print("-" * 80)
print(f"\n📈 통계:")
print(f"  평균 속도 향상: {avg_speedup:.2f}x")
print(f"  최소 속도 향상: {min_speedup:.2f}x")
print(f"  최대 속도 향상: {max_speedup:.2f}x")


# ============================================================================
# 5. 결론
# ============================================================================

print("\n\n" + "=" * 80)
print("🎯 결론")
print("=" * 80)

if tf32_supported:
    if avg_speedup >= 1.3:
        print(f"\n✅ TF32가 FP32보다 평균 {avg_speedup:.2f}배 빠릅니다!")
        print(f"   → 행렬 곱셈이 많은 딥러닝 학습에서 {((avg_speedup-1)*100):.0f}% 속도 향상!")
        
        if avg_speedup >= 2.0:
            print(f"   → 🚀 매우 큰 속도 향상! 반드시 TF32를 사용하세요!")
        elif avg_speedup >= 1.5:
            print(f"   → ⚡ 상당한 속도 향상! TF32 사용 권장!")
        else:
            print(f"   → 💡 중간 정도 속도 향상. TF32 사용 추천!")
    else:
        print(f"\n⚠️ TF32의 속도 향상이 기대보다 작습니다 ({avg_speedup:.2f}x)")
        print("   가능한 원인:")
        print("   - GPU 사용률이 낮음 (다른 병목)")
        print("   - 행렬 크기가 작음")
        print("   - 메모리 대역폭 제한")
else:
    print(f"\n⚠️  현재 GPU는 TF32를 지원하지 않습니다.")
    print(f"   Compute Capability: {cc} (8.0 이상 필요)")
    print(f"   TF32 지원 GPU: RTX 3000/4000 시리즈, A100, H100 등")

print("\n" + "=" * 80)

# ============================================================================
# 6. 추가 정보
# ============================================================================

print("\n[참고] TF32 활성화 방법:")
print("-" * 80)
print("""
# train_ppo.py 맨 위에 추가:
import torch
torch.set_float32_matmul_precision("high")
torch.backends.cudnn.allow_tf32 = True

# 확인:
print(torch.get_float32_matmul_precision())  # "high"
print(torch.backends.cuda.matmul.allow_tf32)  # True
""")

print("\n벤치마크 완료!")