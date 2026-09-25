#!/usr/bin/env python3
"""Dynamic Power BI screenshot cropper.

Detects the likely report canvas structurally instead of using fixed pixel
coordinates, so it can tolerate changes in resolution, DPI, zoom and pane size.

Usage:
    python cropper.py -i raw.png -o cropped.png
    python cropper.py -i raw.png -o cropped.png --debug debug.png

Dependencies:
    pip install opencv-python numpy
"""

from __future__ import annotations
import argparse
from pathlib import Path

import cv2
import numpy as np


def _resize_for_analysis(img: np.ndarray, target_width: int):
    h, w = img.shape[:2]
    if w <= target_width:
        return img.copy(), 1.0
    scale = target_width / float(w)
    resized = cv2.resize(
        img, (target_width, max(1, round(h * scale))),
        interpolation=cv2.INTER_AREA
    )
    return resized, scale


def _smooth(values: np.ndarray, window: int):
    window = max(3, int(window) | 1)
    kernel = np.ones(window, dtype=np.float32) / window
    return np.convolve(values.astype(np.float32), kernel, mode="same")


def _peaks(values: np.ndarray, percentile: float = 72):
    threshold = np.percentile(values, percentile)
    ids = np.flatnonzero(values >= threshold)
    if not len(ids):
        return []
    groups = np.split(ids, np.where(np.diff(ids) > 1)[0] + 1)
    return [int(g[np.argmax(values[g])]) for g in groups if len(g)]


def _boundaries(gray: np.ndarray):
    h, w = gray.shape
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

    vx = _smooth(np.mean(np.abs(gx), axis=0), max(5, w // 250))
    hy = _smooth(np.mean(np.abs(gy), axis=1), max(5, h // 180))
    return _peaks(vx), _peaks(hy), vx, hy


def _score(gray, x1, y1, x2, y2, vx, hy):
    h, w = gray.shape
    rw, rh = x2 - x1, y2 - y1

    if rw <= 0 or rh <= 0:
        return -1e9

    area_ratio = (rw * rh) / float(w * h)
    aspect = rw / float(rh)

    if area_ratio < 0.18 or not 0.65 <= aspect <= 4.5:
        return -1e9

    cx = (x1 + x2) / (2.0 * w)
    cy = (y1 + y2) / (2.0 * h)
    center_penalty = abs(cx - 0.55) + 0.55 * abs(cy - 0.55)

    boundary_strength = (
        vx[min(x1, len(vx) - 1)]
        + vx[min(x2 - 1, len(vx) - 1)]
        + hy[min(y1, len(hy) - 1)]
        + hy[min(y2 - 1, len(hy) - 1)]
    )

    roi = gray[y1:y2, x1:x2]
    variance_bonus = min(float(np.std(roi)) / 60.0, 1.0)

    # Slightly discourage using the screenshot edges unless structural evidence
    # suggests the report genuinely fills the image.
    edge_count = sum((x1 == 0, y1 == 0, x2 == w, y2 == h))
    edge_penalty = 0.20 * edge_count

    return (
        7.0 * area_ratio
        + 0.025 * boundary_strength
        + 0.30 * variance_bonus
        - 1.70 * center_penalty
        - edge_penalty
    )


def detect_powerbi_canvas(img: np.ndarray, analysis_width: int = 1600):
    small, scale = _resize_for_analysis(img, analysis_width)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    h, w = gray.shape
    vertical, horizontal, vx, hy = _boundaries(gray)

    lefts = [x for x in vertical if 0.04 * w <= x <= 0.45 * w] + [0]
    rights = [x for x in vertical if 0.55 * w <= x <= 0.99 * w] + [w]
    tops = [y for y in horizontal if 0.04 * h <= y <= 0.42 * h] + [0]
    bottoms = [y for y in horizontal if 0.58 * h <= y <= 0.99 * h] + [h]

    best = None
    best_score = -1e9

    for x1 in lefts:
        for x2 in rights:
            if x2 - x1 < 0.38 * w:
                continue
            for y1 in tops:
                for y2 in bottoms:
                    if y2 - y1 < 0.38 * h:
                        continue
                    s = _score(gray, x1, y1, x2, y2, vx, hy)
                    if s > best_score:
                        best = (x1, y1, x2, y2)
                        best_score = s

    if best is None:
        raise RuntimeError("Could not identify a plausible Power BI canvas.")

    inv = 1.0 / scale
    oh, ow = img.shape[:2]
    x1, y1, x2, y2 = best

    x1 = max(0, min(ow - 1, round(x1 * inv)))
    x2 = max(x1 + 1, min(ow, round(x2 * inv)))
    y1 = max(0, min(oh - 1, round(y1 * inv)))
    y2 = max(y1 + 1, min(oh, round(y2 * inv)))

    return (x1, y1, x2, y2), float(best_score)


def crop_powerbi(
    input_path: str,
    output_path: str,
    debug_path: str | None = None,
    analysis_width: int = 1600,
    padding: int = 0,
):
    src = Path(input_path)
    dst = Path(output_path)

    img = cv2.imread(str(src))
    if img is None:
        raise FileNotFoundError(f"Cannot read input image: {src}")

    (x1, y1, x2, y2), score = detect_powerbi_canvas(img, analysis_width)

    h, w = img.shape[:2]
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(w, x2 + padding)
    y2 = min(h, y2 + padding)

    crop = img[y1:y2, x1:x2]
    dst.parent.mkdir(parents=True, exist_ok=True)

    if not cv2.imwrite(str(dst), crop):
        raise RuntimeError(f"Could not write output image: {dst}")

    if debug_path:
        debug = img.copy()
        cv2.rectangle(debug, (x1, y1), (x2 - 1, y2 - 1), (0, 0, 255), 3)
        cv2.putText(
            debug,
            f"score={score:.2f} crop={x1},{y1},{x2},{y2}",
            (max(5, x1), max(25, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        dbg = Path(debug_path)
        dbg.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(dbg), debug)

    print(f"Input : {src}")
    print(f"Output: {dst}")
    print(f"Crop  : x={x1}, y={y1}, width={x2-x1}, height={y2-y1}")
    print(f"Score : {score:.2f}")


def main():
    parser = argparse.ArgumentParser(
        description="Dynamically crop a Power BI report canvas from a screenshot."
    )
    parser.add_argument("-i", "--input", required=True, help="Full screenshot path")
    parser.add_argument("-o", "--output", required=True, help="Cropped image path")
    parser.add_argument("--debug", help="Optional debug image with detected rectangle")
    parser.add_argument("--analysis-width", type=int, default=1600)
    parser.add_argument("--padding", type=int, default=0)
    args = parser.parse_args()

    crop_powerbi(
        args.input,
        args.output,
        args.debug,
        args.analysis_width,
        args.padding,
    )


if __name__ == "__main__":
    main()
