# motion_analysis.py
import numpy as np
from scipy.signal import savgol_filter, find_peaks


def smooth_series(series, window=7, poly=3):
    """Savitzky-Golay 平滑"""
    arr = np.array([x if x is not None else np.nan for x in series], dtype=float)
    # 简单线性插值补 NaN
    mask = np.isnan(arr)
    if mask.all():
        return arr
    arr[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), arr[~mask])
    if len(arr) < window:
        return arr
    # window 必须是奇数且 > poly
    if window % 2 == 0:
        window += 1
    if window <= poly:
        return arr
    return savgol_filter(arr, window, poly)


def extract_peak_frames(angle_series, fps, min_range_deg=15.0):
    """
    返回: [(peak_frame_idx, peak_angle), ...]，按帧索引排序
    - 同时检测正向峰（如屈曲）与负向峰（如后伸，为负值）
    - 只有整体幅度超过 min_range_deg 才算有效动作
    """
    if len(angle_series) < 10:
        return []
    window = min(int(fps // 2) * 2 + 1, 11)
    smoothed = smooth_series(angle_series, window=window)
    if np.ptp(smoothed) < min_range_deg:
        return []

    distance = max(int(fps), 1)
    prominence = min_range_deg / 2

    # 正向峰（屈曲等正值极值）
    pos_peaks, _ = find_peaks(smoothed, distance=distance, prominence=prominence)
    # 负向峰（后伸等负值极值）
    neg_peaks, _ = find_peaks(-smoothed, distance=distance, prominence=prominence)

    all_peaks = [(int(p), float(smoothed[p])) for p in pos_peaks] + \
                [(int(p), float(smoothed[p])) for p in neg_peaks]
    all_peaks.sort(key=lambda x: x[0])
    return all_peaks


def segment_phases(angle_series, fps):
    """
    返回每帧的相位标签: 'rest' / 'ascending' / 'peak' / 'descending'
    """
    window = min(int(fps // 2) * 2 + 1, 11)
    smoothed = smooth_series(angle_series, window=window)
    velocity = np.gradient(smoothed)
    v_thresh = max(0.5, np.std(velocity) * 0.3)
    phases = []
    max_val = np.max(smoothed)
    # 避免 max_val 为 0 或负值时的判断异常
    peak_threshold = max_val * 0.92 if max_val > 0 else max_val - abs(max_val) * 0.08
    for v, a in zip(velocity, smoothed):
        if a > peak_threshold:
            phases.append('peak')
        elif v > v_thresh:
            phases.append('ascending')
        elif v < -v_thresh:
            phases.append('descending')
        else:
            phases.append('rest')
    return phases


def recognize_primary_action(rom_statistics: dict, min_range=20.0) -> dict:
    """
    根据各关节 ROM 的变化幅度识别主动作。
    返回 {
        'action':      str,   # key（如 'shoulder_sagittal_left'）
        'action_name': str,   # 中文可读名（如 '肩前屈'）
        'confidence':  float, # 0~1
        'range':       float, # 主动作幅度(度)
    }
    """
    ranges = {}
    for key, stats in rom_statistics.items():
        angles = stats.get("angles", [])
        if len(angles) < 5:
            continue
        ranges[key] = float(np.ptp(angles))

    if not ranges:
        return {
            "action": "unknown",
            "action_name": "未知",
            "confidence": 0.0,
            "range": 0.0,
        }

    dominant = max(ranges, key=ranges.get)
    dom_range = ranges[dominant]
    dom_name = rom_statistics[dominant].get("name", dominant)

    if dom_range < min_range:
        return {
            "action": "static_pose",
            "action_name": "静态姿势",
            "confidence": 0.3,
            "range": round(dom_range, 2),
        }

    # 置信度：主动作幅度 / 第二大幅度，归一化到 [0, 1]
    sorted_vals = sorted(ranges.values(), reverse=True)
    if len(sorted_vals) < 2:
        conf = 1.0
    else:
        conf = min(1.0, sorted_vals[0] / (sorted_vals[1] + 1e-6) / 3)

    return {
        "action": dominant,
        "action_name": dom_name,
        "confidence": round(conf, 2),
        "range": round(dom_range, 2),
    }