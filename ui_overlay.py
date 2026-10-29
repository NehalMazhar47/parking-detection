"""
ui_overlay.py
-------------
OpenCV drawing utilities for the parking detection system.

Handles:
  - Bounding boxes (green = allowed, red = unauthorized)
  - Confidence + label text on each box
  - Alert banner when unauthorized vehicle is detected
  - Dashboard panel (top-left): FPS, allowed type, per-class counts
  - Timestamp overlay
"""

import cv2
import numpy as np
import time
from collections import defaultdict

# ── Colour palette (BGR) ──────────────────────────────────────────────────────
COLOR_ALLOWED       = (34,  197,  94)   # green
COLOR_UNAUTHORIZED  = (59,   30, 240)   # red  (BGR: red channel last)
COLOR_ALERT_BG      = (0,    0,  180)   # dark red for alert banner
COLOR_DASHBOARD_BG  = (20,  20,   20)   # near-black panel
COLOR_WHITE         = (255, 255, 255)
COLOR_YELLOW        = (0,  220,  255)
COLOR_TEXT_SHADOW   = (0,    0,    0)

# ── Font settings ─────────────────────────────────────────────────────────────
FONT            = cv2.FONT_HERSHEY_DUPLEX
FONT_SMALL      = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE_LG   = 0.75
FONT_SCALE_SM   = 0.55
FONT_SCALE_XS   = 0.45
FONT_THICK      = 2
FONT_THICK_SM   = 1


def _put_text_with_shadow(img, text, pos, font=FONT, scale=FONT_SCALE_SM,
                          color=COLOR_WHITE, thickness=FONT_THICK_SM):
    """Draw text with a drop-shadow for readability on any background."""
    x, y = pos
    cv2.putText(img, text, (x + 1, y + 1), font, scale,
                COLOR_TEXT_SHADOW, thickness + 1, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)


def draw_detections(frame: np.ndarray, evaluated: list[dict],
                    show_confidence: bool = True) -> np.ndarray:
    """
    Draw bounding boxes and labels for all evaluated detections.

    Args:
        frame:           BGR frame from OpenCV.
        evaluated:       List of dicts from ParkingRuleEngine.evaluate_frame().
        show_confidence: Whether to render confidence score on each box.

    Returns:
        Frame with overlays drawn (in-place modification).
    """
    for det in evaluated:
        x1, y1, x2, y2 = det["bbox"]
        label    = det["label"].capitalize()
        conf     = det["confidence"]
        allowed  = det["allowed"]

        color = COLOR_ALLOWED if allowed else COLOR_UNAUTHORIZED
        status_icon = "✓" if allowed else "✗"

        # ── Bounding box ──────────────────────────────────────────────────────
        # Outer glow effect (thicker, semi-transparent feel)
        cv2.rectangle(frame, (x1 - 1, y1 - 1), (x2 + 1, y2 + 1), color, 4)
        cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_WHITE, 1)

        # ── Label background pill ─────────────────────────────────────────────
        conf_str = f" {conf:.0%}" if show_confidence else ""
        tag_text = f" {status_icon} {label}{conf_str} "
        (tw, th), _ = cv2.getTextSize(tag_text, FONT, FONT_SCALE_SM, FONT_THICK_SM)

        tag_y1 = max(y1 - th - 10, 0)
        tag_y2 = tag_y1 + th + 10
        tag_x2 = x1 + tw + 4

        cv2.rectangle(frame, (x1, tag_y1), (tag_x2, tag_y2), color, -1)
        cv2.putText(frame, tag_text, (x1 + 2, tag_y2 - 5),
                    FONT, FONT_SCALE_SM, COLOR_WHITE, FONT_THICK_SM, cv2.LINE_AA)

    return frame


def draw_alert_banner(frame: np.ndarray, alert_counter: int,
                      alert_duration: int, unauthorized_labels: list[str]) -> np.ndarray:
    """
    Draw a flashing alert banner at the top of the frame when unauthorized
    vehicles are detected.

    Args:
        frame:               BGR frame.
        alert_counter:       Remaining frames to show alert (counts down).
        alert_duration:      Total alert duration in frames (for flash timing).
        unauthorized_labels: List of unauthorized vehicle label strings.

    Returns:
        Frame with alert banner drawn.
    """
    if alert_counter <= 0:
        return frame

    h, w = frame.shape[:2]
    banner_h = 60

    # Flash every 10 frames
    flash_on = (alert_counter % 20) < 10

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, banner_h), COLOR_ALERT_BG, -1)
    alpha = 0.85
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    # Warning icon + text
    labels_str = ", ".join(set(l.capitalize() for l in unauthorized_labels))
    main_text   = f"  ⚠  UNAUTHORIZED VEHICLE DETECTED: {labels_str}"
    sub_text    = "  This parking zone is RESTRICTED. Please move your vehicle."

    text_color = COLOR_YELLOW if flash_on else COLOR_WHITE
    cv2.putText(frame, main_text, (10, 24),
                FONT, FONT_SCALE_SM, text_color, FONT_THICK_SM, cv2.LINE_AA)
    cv2.putText(frame, sub_text, (10, 48),
                FONT_SMALL, FONT_SCALE_XS, COLOR_WHITE, FONT_THICK_SM, cv2.LINE_AA)

    # Red border flash
    if flash_on:
        border = 4
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1),
                      (0, 0, 255), border)

    return frame


def draw_dashboard(frame: np.ndarray, fps: float, allowed_vehicle: str,
                   evaluated: list[dict], show_dashboard: bool) -> np.ndarray:
    """
    Draw a semi-transparent dashboard panel in the top-left corner.

    Shows: FPS, allowed vehicle type, count per detected class.
    """
    if not show_dashboard:
        return frame

    # Count vehicles per class
    counts: dict[str, dict] = defaultdict(lambda: {"total": 0, "allowed": 0, "blocked": 0})
    for det in evaluated:
        lbl = det["label"]
        counts[lbl]["total"] += 1
        if det["allowed"]:
            counts[lbl]["allowed"] += 1
        else:
            counts[lbl]["blocked"] += 1

    panel_w   = 240
    row_h     = 22
    n_rows    = 4 + len(counts)   # header rows + vehicle rows
    panel_h   = n_rows * row_h + 16
    margin    = 10
    top_offset = 70              # sit below alert banner area

    # Semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(overlay,
                  (margin, top_offset),
                  (margin + panel_w, top_offset + panel_h),
                  COLOR_DASHBOARD_BG, -1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)

    # Accent bar
    cv2.rectangle(frame,
                  (margin, top_offset),
                  (margin + 4, top_offset + panel_h),
                  COLOR_ALLOWED, -1)

    y = top_offset + row_h
    x = margin + 12

    # Title
    cv2.putText(frame, "PARKING MONITOR", (x, y),
                FONT, FONT_SCALE_XS, COLOR_YELLOW, FONT_THICK_SM, cv2.LINE_AA)

    y += row_h
    cv2.line(frame, (x, y - 6), (margin + panel_w - 8, y - 6), (60, 60, 60), 1)

    # FPS
    fps_color = (0, 200, 0) if fps >= 20 else (0, 165, 255) if fps >= 10 else (0, 0, 220)
    _put_text_with_shadow(frame, f"FPS:      {fps:.1f}", (x, y),
                          color=fps_color, scale=FONT_SCALE_XS)
    y += row_h

    # Allowed type
    _put_text_with_shadow(frame, f"Allowed:  {allowed_vehicle.upper()}", (x, y),
                          color=COLOR_ALLOWED, scale=FONT_SCALE_XS)
    y += row_h

    # Separator
    cv2.line(frame, (x, y - 6), (margin + panel_w - 8, y - 6), (60, 60, 60), 1)
    _put_text_with_shadow(frame, "Detections:", (x, y),
                          color=(180, 180, 180), scale=FONT_SCALE_XS)
    y += row_h

    if not counts:
        _put_text_with_shadow(frame, "  None", (x, y),
                              color=(120, 120, 120), scale=FONT_SCALE_XS)
    else:
        for lbl, cnt in sorted(counts.items()):
            icon  = "✓" if lbl == allowed_vehicle else "✗"
            color = COLOR_ALLOWED if lbl == allowed_vehicle else COLOR_UNAUTHORIZED
            text  = f"  {icon} {lbl.capitalize()}: {cnt['total']}"
            _put_text_with_shadow(frame, text, (x, y), color=color, scale=FONT_SCALE_XS)
            y += row_h

    return frame


def draw_timestamp(frame: np.ndarray) -> np.ndarray:
    """Draw current date-time in the bottom-right corner."""
    h, w = frame.shape[:2]
    ts = time.strftime("%Y-%m-%d  %H:%M:%S")
    (tw, th), _ = cv2.getTextSize(ts, FONT_SMALL, FONT_SCALE_XS, 1)
    x = w - tw - 10
    y = h - 10
    _put_text_with_shadow(frame, ts, (x, y), font=FONT_SMALL,
                          scale=FONT_SCALE_XS, color=(200, 200, 200), thickness=1)
    return frame
