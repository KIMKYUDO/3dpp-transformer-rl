# check_param_noise.py
import torch
from torch import nn

# ---- 1. 아주 작은 네트워크 하나 정의 ----
class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 8)
        self.fc2 = nn.Linear(8, 2)

    def forward(self, x):
        x = self.fc1(x).relu()
        x = self.fc2(x)
        return x


# ---- 2. parameter noise 간단 버전 ----
def apply_parameter_noise(module: nn.Module, std: float):
    """모든 trainable 파라미터에 N(0, std^2) 노이즈 추가하고,
    나중에 되돌릴 수 있도록 noise dict 반환."""
    if std <= 0.0:
        return None

    noise = {}
    with torch.no_grad():
        for name, p in module.named_parameters():
            if not p.requires_grad:
                continue
            eps = torch.randn_like(p) * std
            p.add_(eps)
            noise[name] = eps
    return noise


def remove_parameter_noise(module: nn.Module, noise):
    """apply_parameter_noise에서 반환한 noise dict로 파라미터 되돌리기."""
    if noise is None:
        return
    with torch.no_grad():
        for name, p in module.named_parameters():
            eps = noise.get(name)
            if eps is None:
                continue
            p.sub_(eps)


# ---- 3. 진짜로 잘 동작하는지 확인 ----
def main():
    torch.manual_seed(0)

    net = TinyNet()

    # (1) 원래 파라미터/출력 저장
    params_before = [p.detach().clone() for p in net.parameters()]
    x = torch.randn(3, 4)
    out_before = net(x).detach().clone()

    # (2) 노이즈 적용
    std = 0.01
    noise = apply_parameter_noise(net, std=std)

    params_noisy = [p.detach().clone() for p in net.parameters()]
    out_noisy = net(x).detach().clone()

    # (3) 노이즈 제거
    remove_parameter_noise(net, noise)
    params_after = [p.detach().clone() for p in net.parameters()]
    out_after = net(x).detach().clone()

    # ---- 결과 출력 ----
    # 파라미터가 실제로 바뀌었는지
    max_diff_noise = max(
        (pb - pn).abs().max().item()
        for pb, pn in zip(params_before, params_noisy)
    )
    # 제거 후 원래로 돌아왔는지
    max_diff_restore = max(
        (pb - pa).abs().max().item()
        for pb, pa in zip(params_before, params_after)
    )

    print(f"[1] 노이즈 적용 후 파라미터 변화량 max |Δθ| = {max_diff_noise:.6e}")
    print(f"[2] 노이즈 제거 후 복원 오차 max |θ_before - θ_after| = {max_diff_restore:.6e}")

    # 출력도 확인
    print(f"[3] 원래 출력과 노이즈 적용 출력이 같은가?      {torch.allclose(out_before, out_noisy)}")
    print(f"[4] 원래 출력과 노이즈 제거 후 출력이 같은가? {torch.allclose(out_before, out_after)}")

    # 직관적인 OK / FAIL 표시
    if max_diff_noise > 0 and max_diff_restore < 1e-7 and not torch.allclose(out_before, out_noisy) and torch.allclose(out_before, out_after):
        print("\n✅ parameter noise가 의도대로 동작합니다.")
    else:
        print("\n⚠️  뭔가 이상합니다. 구현을 다시 확인하세요.")


if __name__ == "__main__":
    main()