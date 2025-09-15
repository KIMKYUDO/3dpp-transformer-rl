import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def _draw_box(ax, x, y, z, dx, dy, dz, color="skyblue"):
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
        [v[i] for i in [0,1,2,3]],
        [v[i] for i in [4,5,6,7]],
        [v[i] for i in [0,1,5,4]],
        [v[i] for i in [2,3,7,6]],
        [v[i] for i in [1,2,6,5]],
        [v[i] for i in [0,3,7,4]],
    ]
    ax.add_collection3d(Poly3DCollection(faces, alpha=0.5, edgecolor="k"))

def save_packing_3d(boxes, container=(100,100,100), out_path="results/plots/packing_ep.png"):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection="3d")
    L, W, H = container
    ax.set_xlim(0, L)
    ax.set_ylim(0, W)
    ax.set_zlim(0, H)
    for (x,y,z,l,w,h) in boxes:
        _draw_box(ax, x, y, z, l, w, h)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)
