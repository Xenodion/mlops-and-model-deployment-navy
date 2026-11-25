"""Export simple editable architecture diagrams without external diagram software."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = Path(__file__).resolve().parent


def diagram(name, title, boxes, arrows, note):
    fig, ax = plt.subplots(figsize=(12, 5), layout="constrained")
    ax.set(xlim=(0, 12), ylim=(0, 5))
    ax.axis("off")
    for text, x, y, w, h in boxes:
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.08,rounding_size=0.12",
                facecolor="#eef4fb",
                edgecolor="#2563a8",
                linewidth=1.5,
            )
        )
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=11, color="#172334")
    for x1, y1, x2, y2 in arrows:
        ax.add_patch(
            FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="->", mutation_scale=14, color="#475569", linewidth=1.5)
        )
    ax.set_title(title, loc="left", fontsize=17, weight="bold", color="#172334")
    fig.text(0.03, 0.02, note, fontsize=10, color="#475569")
    for ext in ("png", "svg"):
        target = HERE / f"{name}.{ext}"
        fig.savefig(target, dpi=200, bbox_inches="tight", facecolor="white")
        if ext == "svg":
            target.write_text(
                "\n".join(line.rstrip() for line in target.read_text(encoding="utf-8").splitlines()) + "\n",
                encoding="utf-8",
            )
    plt.close(fig)


diagram(
    "measured-replay",
    "Measured: joint API replay on one Vertex worker",
    [
        ("Saved ride\nreadings", 0.2, 2, 1.7, 0.8),
        ("Replay client\n100 held-out rides", 2.4, 2, 2, 0.8),
        ("FastAPI\nboth outputs", 5, 2, 1.9, 0.8),
        ("Ride classifier\nCPU or CUDA", 8, 3.2, 2.5, 0.8),
        ("Fall-risk model\nCPU", 8, 0.8, 2.5, 0.8),
    ],
    [(1.9, 2.4, 2.4, 2.4), (4.4, 2.4, 5, 2.4), (6.9, 2.5, 8, 3.5), (6.9, 2.3, 8, 1.3)],
    "The client and API share the worker; HTTP is local. Raw timings and logs are saved separately.",
)

diagram(
    "current-system",
    "Current project: training and joint serving",
    [
        ("GBG500 + SRV3490\nDVC data", 0.2, 3.2, 2.3, 0.8),
        ("Training and\nevaluation", 3.1, 3.2, 2, 0.8),
        ("MLflow runs\nand registry", 5.8, 3.2, 2, 0.8),
        ("Versioned models\nand API image", 8.5, 3.2, 2.5, 0.8),
        ("Live / recorded\nride readings", 0.2, 0.8, 2.3, 0.8),
        ("FastAPI\ntwo models", 4.8, 0.8, 2.4, 0.8),
        ("Ride category +\nfall-risk output", 8.5, 0.8, 2.5, 0.8),
    ],
    [
        (2.5, 3.6, 3.1, 3.6),
        (5.1, 3.6, 5.8, 3.6),
        (7.8, 3.6, 8.5, 3.6),
        (2.5, 1.2, 4.8, 1.2),
        (7.2, 1.2, 8.5, 1.2),
        (9.5, 3.2, 7, 1.6),
    ],
    "Components exist in the project; this diagram does not claim a production deployment.",
)

diagram(
    "proposed-system",
    "Proposed: scalable joint serving",
    [
        ("Live or post-ride\nclient", 0.2, 2, 1.8, 0.8),
        ("Ingress", 2.5, 2, 1.3, 0.8),
        ("Joint API replicas\nride classifier + fall risk", 4.6, 2, 3.0, 0.8),
        ("Metrics and logs\nlatency / errors / memory", 8.4, 3.3, 2.9, 0.8),
        ("Pinned image and\ntwo model versions", 8.4, 0.7, 2.9, 0.8),
    ],
    [(2, 2.4, 2.5, 2.4), (3.8, 2.4, 4.6, 2.4), (7.6, 2.7, 8.4, 3.5), (8.4, 1.3, 7.6, 2.1)],
    "Scaling and recovery are proposed; replica capacity and rollback time remain unmeasured.",
)
