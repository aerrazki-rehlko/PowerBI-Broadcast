#!/usr/bin/env python3
"""Crop the report surface from a Power BI screenshot.

Install: python -m pip install opencv-python numpy
Example (Windows cmd):
    python cropper.py -i "C:\\Reports\\raw.png" -o "C:\\Reports\\canvas.png" --debug "C:\\Reports\\debug.png"

The detector looks for long, rectangular *surface* boundaries, never the bounds
of individual visuals. A genuinely borderless white toolbar cannot always be
distinguished from a white canvas using pixels alone; such ambiguous captures
raise an error instead of publishing an arbitrary crop.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import cv2
import numpy as np


def _read(path: Path) -> np.ndarray:
    # imdecode/imencode accept Unicode Windows paths (cv2.imread may not).
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"Cannot read image {path}: {exc}") from exc
    if image is None:
        raise RuntimeError(f"Cannot decode image: {path}")
    return image


def _write(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix not in ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'):
        raise ValueError(f"Unsupported output image extension: {path.suffix}")
    ok, encoded = cv2.imencode(suffix, image)
    if not ok:
        raise RuntimeError(f"Could not encode image: {path}")
    try:
        encoded.tofile(str(path))
    except OSError as exc:
        raise RuntimeError(f"Cannot save image {path}: {exc}") from exc


def _smooth(a: np.ndarray, length: int) -> np.ndarray:
    length = max(3, int(length) | 1)
    return cv2.GaussianBlur(a.astype(np.float32).reshape(1, -1),
                            (length, 1), 0).ravel()


def _candidates(signal: np.ndarray, lo: int, hi: int, count: int = 12) -> list[int]:
    """Separated local maxima; a thick border contributes one candidate."""
    lo, hi = max(2, lo), min(len(signal) - 2, hi)
    if hi <= lo:
        return []
    ids = np.arange(lo, hi)
    peaks = ids[(signal[ids] >= signal[ids-1]) &
                (signal[ids] >= signal[ids+1])]
    peaks = sorted(peaks, key=lambda i: (-float(signal[i]), int(i)))
    picked = []
    separation = max(4, len(signal) // 130)
    for i in peaks:
        if all(abs(int(i) - j) >= separation for j in picked):
            picked.append(int(i))
        if len(picked) == count:
            break
    return picked


def _boundary(mask: np.ndarray, gray: np.ndarray, axis: int,
              span: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Score transitions on long scanlines, suppressing small chart edges.

    In the scanline, gradient support counts only pixels with a real change;
    rows/columns containing a few toolbar icons therefore score weakly.
    """
    a, b = span
    if axis == 0:  # vertical lines, aggregate over rows
        m, g = mask[a:b, :], gray[a:b, :]
        diff = np.abs(np.diff(g, axis=1, prepend=g[:, :1]))
        density = m.mean(axis=0)
        support = (diff > 7).mean(axis=0)
        magnitude = np.minimum(diff, 45).mean(axis=0) / 45
    else:
        m, g = mask[:, a:b], gray[:, a:b]
        diff = np.abs(np.diff(g, axis=0, prepend=g[:1, :]))
        density = m.mean(axis=1)
        support = (diff > 7).mean(axis=1)
        magnitude = np.minimum(diff, 45).mean(axis=1) / 45
    n = len(density)
    k = max(3, n // 300)
    density = _smooth(density, k)
    support = _smooth(support, k)
    magnitude = _smooth(magnitude, k)
    step = max(4, n // 110)
    # A sustained change of surface colour is useful even when no dark line exists.
    before = density[np.maximum(0, np.arange(n) - step)]
    after = density[np.minimum(n - 1, np.arange(n) + step)]
    transition = np.abs(after - before)
    strength = .48 * support + .22 * magnitude + .30 * transition
    return strength, density


def _detect(image: np.ndarray, analysis_width: int, brightness: int,
            channel_spread: int):
    oh, ow = image.shape[:2]
    scale = min(1., analysis_width / ow)
    w, h = max(1, round(ow * scale)), max(1, round(oh * scale))
    small = (cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)
             if scale < 1 else image)
    low = small.min(axis=2).astype(np.int16)
    high = small.max(axis=2).astype(np.int16)
    mask = ((low >= brightness) & (high - low <= channel_spread)).astype(np.uint8)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # Search patches near the centre, allowing a visual to cover the exact centre.
    window = max(9, min(w, h) // 38 | 1)
    whiteness = cv2.boxFilter(mask.astype(np.float32), -1, (window, window))
    xa, xb = int(.32*w), int(.74*w)
    ya, yb = int(.31*h), int(.78*h)
    central = whiteness[ya:yb, xa:xb]
    if central.size == 0:
        raise RuntimeError("Screenshot is too small for canvas detection")
    iy, ix = np.unravel_index(int(np.argmax(central)), central.shape)
    sx, sy = xa + int(ix), ya + int(iy)
    seed_quality = float(central[iy, ix])
    if seed_quality < .65:
        raise RuntimeError("No near-white central report surface found; try --brightness 205")

    # Initial whole-image profiles propose boundaries. Each candidate is then
    # measured again over its partner span, ensuring a separator extends over
    # the report rather than just a table or a toolbar icon.
    vx, _ = _boundary(mask, gray, 0, (int(.26*h), int(.88*h)))
    hy, _ = _boundary(mask, gray, 1, (int(.24*w), int(.85*w)))
    left = _candidates(vx, int(.015*w), min(sx-2, int(.58*w)), 13)
    right = _candidates(vx, max(sx+2, int(.48*w)), int(.985*w), 13)
    top = _candidates(hy, int(.025*h), min(sy-2, int(.60*h)), 18)
    bottom = _candidates(hy, max(sy+2, int(.48*h)), int(.99*h), 13)
    if not all((left, right, top, bottom)):
        raise RuntimeError("Could not find four canvas boundaries; inspect screenshot or adjust --brightness")

    # Integral mask makes evaluating thousands of rectangles inexpensive.
    integral = cv2.integral(mask)
    def white(x1, y1, x2, y2):
        x1, x2 = max(0, x1), min(w, x2)
        y1, y2 = max(0, y1), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return 0.
        return float(integral[y2, x2] - integral[y1, x2] -
                     integral[y2, x1] + integral[y1, x1]) / ((x2-x1)*(y2-y1))

    best = None
    vertical_cache = {}
    # Cache boundary profiles by proposed opposite coordinate ranges.
    for x1 in left:
        for x2 in right:
            if x2-x1 < .37*w or not x1 < sx < x2:
                continue
            # Ignore edge pixels within the vertical pane separators.
            tx1, tx2 = x1 + max(3, (x2-x1)//60), x2 - max(3, (x2-x1)//60)
            hs, _ = _boundary(mask, gray, 1, (tx1, tx2))
            for y1 in top:
                for y2 in bottom:
                    rw, rh = x2-x1, y2-y1
                    if rh < .37*h or not y1 < sy < y2:
                        continue
                    aspect = rw / rh
                    if not .62 <= aspect <= 5.2:
                        continue
                    ly1, ly2 = y1 + max(3, rh//55), y2 - max(3, rh//55)
                    if (ly1, ly2) not in vertical_cache:
                        vertical_cache[(ly1, ly2)] = _boundary(mask, gray, 0, (ly1, ly2))[0]
                    vs = vertical_cache[(ly1, ly2)]
                    inset = max(3, min(rw, rh)//100)
                    inside = white(x1+inset, y1+inset, x2-inset, y2-inset)
                    if inside < .63:
                        continue
                    # Compare full-length strips on both sides, especially for
                    # white toolbars: colour alone cannot validate the top.
                    band = max(5, min(rw, rh)//60)
                    t_in = white(tx1, y1+inset, tx2, y1+band+inset)
                    t_out = white(tx1, y1-band, tx2, y1-inset)
                    b_in = white(tx1, y2-band-inset, tx2, y2-inset)
                    b_out = white(tx1, y2+inset, tx2, y2+band)
                    l_in = white(x1+inset, ly1, x1+band+inset, ly2)
                    l_out = white(x1-band, ly1, x1-inset, ly2)
                    r_in = white(x2-band-inset, ly1, x2-inset, ly2)
                    r_out = white(x2+inset, ly1, x2+band, ly2)
                    edges = [float(vs[x1]), float(vs[x2]),
                             float(hs[y1]), float(hs[y2])]
                    contrasts = [l_in-l_out, r_in-r_out,
                                 t_in-t_out, b_in-b_out]
                    # A white header with no sustained divider is ambiguous.
                    # Top edge must have either sustained contrast or a strong
                    # horizontal structural signal across the actual width.
                    if edges[2] < .065 and contrasts[2] < .11:
                        continue
                    if edges[3] < .045 and contrasts[3] < .09:
                        continue
                    if edges[0] < .045 and contrasts[0] < .09:
                        continue
                    if edges[1] < .045 and contrasts[1] < .09:
                        continue
                    area = rw*rh/(w*h)
                    if area > .92:
                        continue
                    border = sum(min(.30, e*2.0) + max(-.10, min(.30, c))
                                 for e, c in zip(edges, contrasts))
                    # Area is a weak tie-breaker; the toolbar cannot win merely
                    # by expanding the rectangle upward.
                    # A quiet part of the lower canvas should have a broadly
                    # uniform light surface; a included grey Pages pane lowers
                    # this even if it passes the near-white threshold.
                    quiet = gray[y1+rh//2:y2-max(2,rh//18), x1+inset:x2-inset]
                    light = float(np.mean(quiet)) / 255. if quiet.size else 0.
                    inside_brightness = float(np.mean(gray[ly1:ly2, x1+inset:x1+band+inset])) / 255.
                    score = (border + .30*inside + .12*seed_quality + .10*area
                             + .65*light + 2.0*inside_brightness
                             - .13*abs((x1+x2)/(2*w)-.53)
                             - .10*abs((y1+y2)/(2*h)-.56))
                    item = (score, (x1,y1,x2,y2), edges, contrasts, inside)
                    if best is None or score > best[0]:
                        best = item
    if best is None:
        raise RuntimeError("Canvas boundaries are ambiguous; try --brightness 205 and capture a screenshot with visible pane separators")
    score, box, edges, contrasts, inside = best
    if inside < .68 or min(max(e*2., c) for e,c in zip(edges, contrasts)) < .075:
        raise RuntimeError("Low-confidence canvas detection; adjust --brightness or capture with visible separators")
    x1,y1,x2,y2 = box
    # Smoothed profiles locate a boundary neighbourhood. Snap back to the
    # actual sustained pixel transition so smoothing does not trim a margin.
    def snap(position, axis, a, b, radius):
        low = max(1, position-radius)
        high = min((w if axis == 0 else h)-1, position+radius+1)
        if high <= low:
            return position
        if axis == 0:
            changes = np.abs(gray[a:b, low:high] - gray[a:b, low-1:high-1])
            values = np.mean(np.minimum(changes, 80), axis=0)
        else:
            changes = np.abs(gray[low:high, a:b] - gray[low-1:high-1, a:b])
            values = np.mean(np.minimum(changes, 80), axis=1)
        return low + int(np.argmax(values))
    r = max(3, min(w,h)//65)
    x1 = snap(x1, 0, y1+10, y2-10, r)
    x2 = snap(x2, 0, y1+10, y2-10, r)
    y1 = snap(y1, 1, x1+10, x2-10, r)
    y2 = snap(y2, 1, x1+10, x2-10, r)
    rect = (max(0, math.floor(x1*ow/w)), max(0, math.floor(y1*oh/h)),
            min(ow, math.ceil(x2*ow/w)), min(oh, math.ceil(y2*oh/h)))
    seed = (round(sx*ow/w), round(sy*oh/h))
    return rect, score, seed, mask


def crop_powerbi(input_path, output_path, debug_path=None, analysis_width=1600,
                 padding=0, brightness=218, channel_spread=35):
    """Save an original-resolution canvas PNG and return (x1,y1,x2,y2)."""
    if analysis_width < 300 or not 0 <= brightness <= 255 or not 0 <= channel_spread <= 255:
        raise ValueError("analysis_width must be >=300; brightness and channel_spread must be 0..255")
    if padding < 0:
        raise ValueError("padding must be nonnegative")
    src, dst = Path(input_path), Path(output_path)
    if dst.suffix.lower() != '.png':
        raise ValueError("Output must be a PNG filename")
    image = _read(src)
    rect, score, seed, mask = _detect(image, analysis_width, brightness, channel_spread)
    h,w = image.shape[:2]
    x1,y1,x2,y2 = rect
    rect = (max(0,x1-padding), max(0,y1-padding),
            min(w,x2+padding), min(h,y2+padding))
    x1,y1,x2,y2 = rect
    # Reject aliasing the input before any write.
    if src.resolve() == dst.resolve():
        raise ValueError("Input and output paths must differ")
    _write(dst, image[y1:y2, x1:x2])
    if debug_path is not None:
        debug = Path(debug_path)
        overlay = image.copy()
        thick = max(2, round(w/700))
        cv2.rectangle(overlay, (x1,y1), (x2-1,y2-1), (0,0,255), thick)
        cv2.drawMarker(overlay, seed, (255,0,255), cv2.MARKER_CROSS,
                       max(12, thick*6), thick)
        label = f"Score {score:.3f}  Crop ({x1},{y1})-({x2},{y2})"
        font_scale = max(.5, min(1.1, w/1500))
        (tw,th),_ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
        cv2.rectangle(overlay, (0,0), (min(w,tw+16), min(h,th+18)), (0,0,0), -1)
        cv2.putText(overlay, label, (8,th+8), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (255,255,255), 2, cv2.LINE_AA)
        _write(debug, overlay)
        # The mask is intentionally at analysis resolution, as used by detection.
        _write(debug.with_name(debug.stem+'_white_mask.png'), mask*255)
    print(f"Input : {src}")
    print(f"Output: {dst}")
    print(f"Crop  : x={x1}, y={y1}, width={x2-x1}, height={y2-y1}")
    print(f"Score : {score:.3f}")
    return rect


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-i','--input', required=True)
    parser.add_argument('-o','--output', required=True)
    parser.add_argument('--debug')
    parser.add_argument('--analysis-width', type=int, default=1600)
    parser.add_argument('--padding', type=int, default=0)
    parser.add_argument('--brightness', type=int, default=218)
    parser.add_argument('--channel-spread', type=int, default=35)
    args = parser.parse_args()
    try:
        crop_powerbi(args.input, args.output, args.debug, args.analysis_width,
                     args.padding, args.brightness, args.channel_spread)
    except (RuntimeError, ValueError, OSError, cv2.error) as exc:
        print(f"Crop failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
