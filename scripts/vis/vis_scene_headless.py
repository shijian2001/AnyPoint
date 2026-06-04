#!/usr/bin/env python3
"""Headless point-cloud scene visualizer for this GPU server (no X display).

The machine has no X11/EGL display, so Open3D's interactive window and its
EGL/Filament OffscreenRenderer both fail. This script renders the (N,6) xyzrgb
scene .npy files with backends that need no GL context:

    plotly  -> interactive .html  (rotate/zoom in a browser; default, best)
    mpl     -> static .png        (matplotlib Agg; zero GL deps, smallest)

Usage:
    python vis_scene_headless.py /path/to/scene.npy
    python vis_scene_headless.py /tmp/qa_smoke_test/pcd/000000.npy -o out.html
    python vis_scene_headless.py scene.npy --backend mpl --max-points 40000

Then copy the .html/.png to your laptop (scp / VSCode "Download") and open it.
"""
import argparse
import os
import numpy as np


def load_xyzrgb(path, max_points, seed=0):
    a = np.load(path)
    if a.ndim != 2 or a.shape[1] < 3:
        raise ValueError(f"Expected (N,>=3) array, got {a.shape}")
    n = a.shape[0]
    if max_points and n > max_points:
        idx = np.random.RandomState(seed).choice(n, size=max_points, replace=False)
        a = a[idx]
    xyz = a[:, :3].astype(np.float32)
    if a.shape[1] >= 6:
        rgb = a[:, 3:6].astype(np.float32)
        if rgb.max() > 1.5:  # stored as 0-255
            rgb = rgb / 255.0
        rgb = np.clip(rgb, 0.0, 1.0)
    else:
        rgb = None
    return xyz, rgb, n


def render_plotly(xyz, rgb, out, point_size):
    import plotly.graph_objects as go
    if rgb is not None:
        color = ['rgb(%d,%d,%d)' % (int(r * 255), int(g * 255), int(b * 255))
                 for r, g, b in rgb]
    else:
        color = xyz[:, 1]  # height-colored fallback
    fig = go.Figure(go.Scatter3d(
        x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2],
        mode='markers',
        marker=dict(size=point_size, color=color, opacity=0.9),
    ))
    fig.update_layout(scene=dict(aspectmode='data'),
                      margin=dict(l=0, r=0, t=0, b=0))
    fig.write_html(out)
    return out


def render_mpl(xyz, rgb, out, point_size):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    c = rgb if rgb is not None else xyz[:, 1]
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=c, s=point_size, linewidths=0)
    ax.set_box_aspect((np.ptp(xyz[:, 0]) or 1, np.ptp(xyz[:, 1]) or 1, np.ptp(xyz[:, 2]) or 1))
    ax.view_init(elev=20, azim=45)
    plt.savefig(out, dpi=120, bbox_inches='tight')
    plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("npy", help="scene .npy file (N,6 xyzrgb) or (N,3)")
    ap.add_argument("-o", "--output", help="output file (default: <input>.html/.png)")
    ap.add_argument("--backend", choices=["plotly", "mpl"], default="plotly")
    ap.add_argument("--max-points", type=int, default=0,
                    help="downsample cap for speed/filesize (0 = all points, the default)")
    ap.add_argument("--point-size", type=float, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    xyz, rgb, n_total = load_xyzrgb(args.npy, args.max_points, args.seed)
    shown = xyz.shape[0]

    if args.backend == "plotly":
        out = args.output or (os.path.splitext(args.npy)[0] + ".html")
        ps = args.point_size if args.point_size is not None else 1.5
        out = render_plotly(xyz, rgb, out, ps)
    else:
        out = args.output or (os.path.splitext(args.npy)[0] + ".png")
        ps = args.point_size if args.point_size is not None else 0.5
        out = render_mpl(xyz, rgb, out, ps)

    size_mb = os.path.getsize(out) / 1e6
    print(f"OK -> {out}  ({shown:,}/{n_total:,} pts, {size_mb:.2f} MB)")
    print("Copy it to your laptop and open it (html=interactive, png=static).")


if __name__ == "__main__":
    main()
