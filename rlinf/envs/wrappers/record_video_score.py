# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Right-hand score panel for the side-by-side SAFE overlay videos.

Generates a matplotlib figure showing, up to the *current* decision step:
  * the accumulated SAFE failure ``score`` trajectory, and
  * the precomputed functional-``cp_band`` threshold trajectory.

The panel height matches the left-hand video frame; it is meant to be stacked
side-by-side (``np.hstack``) with the robot frame to form one output video frame.

To keep runtime low, callers should build one panel *per decision step* and reuse
it for all ``replan_steps`` video frames that belong to that step.
"""

from typing import Optional, Sequence

import matplotlib
import numpy as np

matplotlib.use("Agg")

import matplotlib.pyplot as plt

# Default panel width in pixels (scaled to match left frame height if needed).
_PANEL_WIDTH = 480
# Use a light but not fully transparent background so score/band lines pop.
_BG = "#fcfcfc"
_TEXT_COLOR = "#222222"
_SCORE_COLOR = "#1f77b4"
_BAND_COLOR = "#d62728"
_FLAG_COLOR = "#ff0000"


def compute_panel_width(frame_height: int, panel_width: int = _PANEL_WIDTH) -> int:
    """Pick a panel width that keeps a sensible aspect ratio with the frame."""
    return max(int(panel_width), frame_height // 2)


def build_score_panel(
    frame_height: int,
    scores: Sequence[float],
    band: np.ndarray,
    step_idx: int,
    replan_steps: int,
    flagged: bool = False,
    success: Optional[int] = None,
    detector_predicted_failure: Optional[bool] = None,
    title_lines: Optional[Sequence[str]] = None,
    panel_width: int = _PANEL_WIDTH,
) -> np.ndarray:
    """Render the right-hand panel up to decision step ``step_idx``.

    Args:
        frame_height: pixel height of the left-hand (robot) frame; the panel is
            sized to match.
        scores: SAFE score per decision step for this episode (full trajectory).
        band: precomputed functional-CP band (full trajectory, may be longer
            than ``scores``).
        step_idx: current (exclusive) number of decision steps to display.
        replan_steps: number of video frames per decision step (used only to map
            the x axis label to actual rollout time).
        flagged: whether a detection (score >= band) has fired so far.
        success: ground-truth episode success (0/1) if known, else None.
        detector_predicted_failure: whether the detector predicted a failure.
        title_lines: extra text lines rendered in the panel title area.
        panel_width: desired panel width in pixels.

    Returns:
        numpy ``(frame_height, panel_width, 3)`` uint8 RGB image.
    """
    width = compute_panel_width(frame_height, panel_width)
    height = frame_height
    fig = plt.figure(figsize=(width / 100, height / 100), dpi=100)
    ax = fig.add_subplot(111)
    ax.set_facecolor(_BG)
    fig.patch.set_facecolor(_BG)

    x = list(range(1, step_idx + 1))  # decision steps, 1-indexed
    shown_scores = list(scores[:step_idx])

    # Edge-pad the (fixed-length) CP band to the full displayed horizon so the
    # threshold line reaches the end of the trajectory, matching the offline
    # ``np.pad(..., mode='edge')`` alignment used in SAFE evaluation.
    band_full = None
    if band is not None and band.size > 0:
        band_flat = np.asarray(band, dtype=np.float64).reshape(-1)
        if len(band_flat) < step_idx:
            band_full = np.pad(band_flat, (0, step_idx - len(band_flat)), mode="edge")
        else:
            band_full = band_flat[:step_idx]

    # y limits: adaptive over the shown scores and the corresponding bands.
    band_shown = band_full if band_full is not None else np.array([])
    ymax = 1.0
    if shown_scores:
        ymax = max(float(np.max(shown_scores)), ymax)
    if band_shown.size:
        ymax = max(float(np.max(band_shown)), ymax)
    ymax = float(ymax)
    if ymax <= 0:
        ymax = 1.0
    ymax *= 1.1  # headroom

    # Band line (up to current step).
    if band_full is not None:
        bx = list(range(1, step_idx + 1))
        by = list(band_full[: len(bx)])
        ax.plot(bx, by, color=_BAND_COLOR, linewidth=2.0,
                label="threshold (CP band)")
        ax.fill_between(bx, 0, by, color=_BAND_COLOR, alpha=0.12)

    # Score line.
    if shown_scores:
        ax.plot(x, shown_scores, color=_SCORE_COLOR, linewidth=2.4,
                marker="o", markersize=3.5, label="safe score")
        if flagged and shown_scores:
            ax.scatter([x[-1]], [shown_scores[-1]], color=_FLAG_COLOR,
                       s=60, zorder=5)
            ax.annotate("DETECTED", (x[-1], shown_scores[-1]),
                        textcoords="offset points", xytext=(6, -2),
                        color=_FLAG_COLOR, fontsize=9, fontweight="bold")

    ax.set_xlim(0.0, max(step_idx + 1, 1))
    ax.set_ylim(0.0, ymax)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_xlabel("decision step", fontsize=9)
    ax.set_ylabel("failure score", fontsize=9)
    ax.tick_params(labelsize=8)

    # Compose title block (episode outcome + detector verdict if known).
    title_lines = list(title_lines or [])
    if success is not None:
        title_lines.append(f"outcome: {'success' if success == 1 else 'fail'}")
    if detector_predicted_failure is not None:
        title_lines.append(
            "detector: fail" if detector_predicted_failure else "detector: ok"
        )
    if title_lines:
        ax.set_title(" | ".join(title_lines), fontsize=9, color=_TEXT_COLOR,
                     loc="left")

    ax.legend(loc="upper left", fontsize=8, framealpha=0.6)
    fig.tight_layout(pad=0.4)

    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    img = np.asarray(buf)[..., :3]  # (H, W, 3) RGB, drop alpha
    plt.close(fig)

    # Resize to exact target (height-frame, width-panel).
    if img.shape[0] != height or img.shape[1] != width:
        from PIL import Image

        img = np.array(Image.fromarray(img).resize((width, height)))
    return img


def add_red_border(image: np.ndarray, thickness: int = 8) -> np.ndarray:
    """Draw a red border around a video frame."""
    out = np.ascontiguousarray(image.copy())
    out[:thickness, :, 0] = 255
    out[:thickness, :, 1] = 0
    out[:thickness, :, 2] = 0
    out[-thickness:, :, 0] = 255
    out[-thickness:, :, 1] = 0
    out[-thickness:, :, 2] = 0
    out[:, :thickness, 0] = 255
    out[:, :thickness, 1] = 0
    out[:, :thickness, 2] = 0
    out[:, -thickness:, 0] = 255
    out[:, -thickness:, 1] = 0
    out[:, -thickness:, 2] = 0
    return out
