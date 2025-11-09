# utils/plotting.py

# --- backend 선택 & GUI import 가드 ---
USE_GUI = False

import os

if USE_GUI:
    os.environ.pop("MPLBACKEND", None)
    import matplotlib
    matplotlib.use("TkAgg", force=True)
    import tkinter as tk
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
else:
    os.environ.setdefault("MPLBACKEND", "Agg")
    import matplotlib
    matplotlib.use("Agg", force=True)

# === plotting (자동 저장) ===
try:
    import matplotlib.pyplot as plt
    if not USE_GUI:
        plt.ioff()  # 인터랙티브 off
    import atexit
    atexit.register(lambda: plt.close('all'))
    MATPLOTLIB_OK = True
except Exception:
    MATPLOTLIB_OK = False
    plt = None

from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.cm as cm
import numpy as np
try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except Exception:
    HAS_PLOTLY = False


# =====================
# Matplotlib 정적 3D 플롯
# =====================
def _draw_box(ax, x, y, z, dx, dy, dz, color="skyblue", alpha=0.6):
    """3D 박스 하나를 그리기 (반투명 + 깊이순서)"""
    v = [
        [x, y, z],
        [x+dx, y, z],
        [x+dx, y+dy, z],
        [x, y+dy, z],
        [x, y, z+dz],
        [x+dx, y, z+dz],
        [x+dx, y+dy, z+dz],
        [x, y+dy, z+dz]
    ]
    faces = [
        [v[i] for i in [0,1,2,3]],  # 아래
        [v[i] for i in [4,5,6,7]],  # 위
        [v[i] for i in [0,1,5,4]],
        [v[i] for i in [2,3,7,6]],
        [v[i] for i in [1,2,6,5]],
        [v[i] for i in [0,3,7,4]],
    ]
    poly3d = Poly3DCollection(
        faces,
        facecolors=color,
        edgecolors="k",
        linewidths=0.5,
        alpha=alpha
    )
    poly3d.set_zsort("min")  # 깊이순서 반영
    ax.add_collection3d(poly3d)


def save_packing_3d(boxes, container=(100,100,100), out_path="results/plots/packing_ep.png"):
    """적재 결과를 3D 시각화하여 저장 (Matplotlib, 바닥 격자 포함)"""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection="3d")

    L, W, H = container
    H_safe = max(1, int(H))
    ax.set_xlim(0, L)
    ax.set_ylim(0, W)
    ax.set_zlim(0, H_safe)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    # 바닥 격자
    xx, yy = np.meshgrid(np.linspace(0, L, 11), np.linspace(0, W, 11))
    zz = np.zeros_like(xx)
    ax.plot_wireframe(xx, yy, zz, color="gray", linewidth=0.5, alpha=0.7)

    # 단일톤 블루 계열 색상
    n = len(boxes)
    colors = [cm.Blues(0.3 + 0.7 * (i % 10) / 10.0) for i in range(n)]

    for (box, color) in zip(boxes, colors):
        x, y, z, l, w, h = box
        _draw_box(ax, x, y, z, l, w, h, color=color, alpha=0.6)

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)


# =====================
# Plotly 인터랙티브 3D 플롯
# =====================
def save_packing_3d_interactive(boxes, container=(100,100,100), out_path="results/plots/packing_ep.html"):
    """
    적재 결과를 Plotly 기반 인터랙티브 3D로 저장.
    HTML 파일을 저장하고, 브라우저에서 마우스로 회전/확대/이동 가능.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    L, W, H = container
    H_safe = max(1, int(H))
    fig = go.Figure()

    # 색상 팔레트 (tab20)
    colors = [cm.tab20(i % 20) for i in range(len(boxes))]

    for idx, (x, y, z, dx, dy, dz) in enumerate(boxes):
        verts = [
            (x,     y,     z),
            (x+dx,  y,     z),
            (x+dx,  y+dy,  z),
            (x,     y+dy,  z),
            (x,     y,     z+dz),
            (x+dx,  y,     z+dz),
            (x+dx,  y+dy,  z+dz),
            (x,     y+dy,  z+dz),
        ]
        faces = [
            (0,1,2), (0,2,3),  # 아래
            (4,5,6), (4,6,7),  # 위
            (0,1,5), (0,5,4),  # 앞
            (2,3,7), (2,7,6),  # 뒤
            (1,2,6), (1,6,5),  # 오른쪽
            (0,3,7), (0,7,4),  # 왼쪽
        ]
        rgba = "rgba({},{},{},{})".format(
            int(colors[idx][0]*255),
            int(colors[idx][1]*255),
            int(colors[idx][2]*255),
            0.8
        )
        fig.add_trace(go.Mesh3d(
            x=[v[0] for v in verts],
            y=[v[1] for v in verts],
            z=[v[2] for v in verts],
            i=[f[0] for f in faces],
            j=[f[1] for f in faces],
            k=[f[2] for f in faces],
            color=rgba,
            opacity=0.9,
            name=f"Box {idx}"
        ))

    # 컨테이너 범위
    fig.update_layout(
        scene=dict(
            xaxis=dict(range=[0,L]),
            yaxis=dict(range=[0,W]),
            zaxis=dict(range=[0,H_safe])
        ),
        width=900,
        height=700
    )

    fig.write_html(out_path)
    print(f"[saved] interactive plot → {out_path}")
