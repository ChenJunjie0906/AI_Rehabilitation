'''
motion_analysis.py —— 动作识别与相位分析
'''
import numpy as np
from scipy.signal import savgol_filter, find_peaks


# ──────────────────────────────────────────────────────────────=
# 配置
# ──────────────────────────────────────────────────────────────

# 一个通道内正负号代表两个互为反向的动作（前屈+/后伸-）
# key 前缀 -> (正向名, 负向名)
BIDIRECTIONAL_CHANNELS = {
    "shoulder_sagittal": ("肩前屈", "肩后伸"),
    "shoulder_extension": ("肩前屈", "肩后伸")
    # 以后有其他带符号通道可以加，例如：
    # "trunk_sagittal":  ("躯干前屈", "躯干后伸"),
}

# 判"真的出现了某方向"的最小绝对角度，用来过滤抖动
DIRECTION_THRESHOLD_DEG = 5.0

# 识别主动作所需的最小幅度
DEFAULT_MIN_RANGE = 20.0

# 判定"多个动作并存"时，次要动作需达到主动作幅度的比例
SECONDARY_ACTION_RATIO = 0.6


def smooth_series(series, window=7, poly=3):
    """Savitzky-Golay 平滑"""
    arr = np.array([x if x is not None else np.nan for x in series], dtype=float)
    mask = np.isnan(arr)
    if mask.all():
        return arr
    arr[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), arr[~mask])
    if len(arr) < window:
        return arr
    if window % 2 == 0:
        window += 1
    if window <= poly:
        return arr
    return savgol_filter(arr, window, poly)


def extract_peak_frames(angle_series, fps, min_range_deg=15.0):
    """
    返回: [(peak_frame_idx, peak_angle), ...]，按帧索引排序
    - 同时检测正向峰与负向峰
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

    pos_peaks, _ = find_peaks(smoothed, distance=distance, prominence=prominence)
    neg_peaks, _ = find_peaks(-smoothed, distance=distance, prominence=prominence)

    all_peaks = [(int(p), float(smoothed[p])) for p in pos_peaks] + \
                [(int(p), float(smoothed[p])) for p in neg_peaks]
    all_peaks.sort(key=lambda x: x[0])
    return all_peaks


def segment_phases(angle_series, fps):
    """返回每帧的相位标签: 'rest' / 'ascending' / 'peak' / 'descending'"""
    window = min(int(fps // 2) * 2 + 1, 11)
    smoothed = smooth_series(angle_series, window=window)
    velocity = np.gradient(smoothed)
    v_thresh = max(0.5, np.std(velocity) * 0.3)
    phases = []
    max_val = np.max(smoothed)
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


# ──────────────────────────────────────────────────────────────
# 方向细化工具
# ──────────────────────────────────────────────────────────────
def _match_bidirectional(action_key: str):
    """若 action_key 前缀匹配双向通道，返回 (正向名, 负向名)，否则 None"""
    for prefix, names in BIDIRECTIONAL_CHANNELS.items():
        if action_key.startswith(prefix):
            return names
    return None


def _analyze_signed_channel(action_key: str, angles: np.ndarray) -> dict:
    """
    对带符号的通道做方向分析。
    返回:
        {
          'action_name': str,     # 根据实际数据推断的动作名
          'direction_type': str,  # 'flexion_only' / 'extension_only' / 'bidirectional' / 'unidirectional'
          'flexion_max': float,   # 正向峰值的绝对值 (仅双向通道有意义)
          'extension_max': float, # 负向峰值的绝对值 (仅双向通道有意义)
          'range': float,         # peak-to-peak
        }
    """
    a_max = float(np.max(angles))
    a_min = float(np.min(angles))
    rng   = a_max - a_min

    names = _match_bidirectional(action_key)
    if names is None:
        # 普通单向通道，保持原语义
        return {
            "direction_type": "unidirectional",
            "range": rng,
        }

    pos_name, neg_name = names
    has_pos = a_max >  DIRECTION_THRESHOLD_DEG
    has_neg = a_min < -DIRECTION_THRESHOLD_DEG

    # 修改逻辑：如果同时有正负值，选择幅度最大的方向作为主要方向
    if has_pos and has_neg:
        # 选择绝对值最大的方向
        if abs(a_max) >= abs(a_min):
            # 正值幅度更大或相等，选择正向（前屈）
            action_name = pos_name
            direction_type = "flexion_only"
        else:
            # 负值绝对值更大，选择负向（后伸）
            action_name = neg_name
            direction_type = "extension_only"
    elif has_neg and not has_pos:
        action_name = neg_name
        direction_type = "extension_only"
    elif has_pos and not has_neg:
        action_name = pos_name
        direction_type = "flexion_only"
    else:
        # 如果没有超过阈值的值，则根据绝对值大小判断主导方向
        if abs(a_min) > abs(a_max):
            action_name = neg_name  # 负值绝对值更大，显示负向名称
            direction_type = "extension_only"
        else:
            action_name = pos_name  # 正值绝对值更大或相等，显示正向名称
            direction_type = "flexion_only"

    return {
        "action_name":    action_name,
        "direction_type": direction_type,
        "flexion_max":    round(max(a_max,  0.0), 2),
        "extension_max":  round(max(-a_min, 0.0), 2),
        "range":          rng,
    }

def _effective_range(action_key: str, angles: np.ndarray) -> float:
    """
    用于主动作排序的"有效幅度"。
    - 单向通道：就是 peak-to-peak
    - 双向通道：改为选择两个方向中绝对值最大的那个
    """
    if _match_bidirectional(action_key) is None:
        return float(np.ptp(angles))

    a_max = float(np.max(angles))
    a_min = float(np.min(angles))
    # 修改：不再使用双向运动的特殊逻辑，而是取最大绝对值
    return max(abs(a_max), abs(a_min))


# ──────────────────────────────────────────────────────────────
# 主动作识别
# ──────────────────────────────────────────────────────────────
def recognize_primary_action(rom_statistics: dict,
                             min_range: float = DEFAULT_MIN_RANGE) -> dict:
    """
    根据各关节 ROM 的变化幅度识别主动作。
    返回:
        {
          'action':         str,      # 通道 key
          'action_name':    str,      # 中文可读名
          'confidence':     float,    # 0~1
          'range':          float,    # peak-to-peak
          'direction_type': str,      # 'unidirectional' / 'flexion_only' / 'extension_only' / 'bidirectional'
          'flexion_max':    float,    # 仅双向通道
          'extension_max':  float,    # 仅双向通道
          'secondary_actions': [      # 同时存在的其他显著动作
              {'action': ..., 'action_name': ..., 'range': ..., 'direction_type': ...},
              ...
          ],
        }
    """
    # 1) 收集每个通道的幅度
    channel_info = {}
    for key, stats in rom_statistics.items():
        angles = np.asarray(
            [a for a in stats.get("angles", []) if a is not None],
            dtype=float,
        )
        if angles.size < 5:
            continue
        channel_info[key] = {
            "angles":     angles,
            "raw_range":  float(np.ptp(angles)),
            "eff_range":  _effective_range(key, angles),
            "stats_name": stats.get("name", key),
        }

    if not channel_info:
        return {
            "action": "unknown",
            "action_name": "未知",
            "confidence": 0.0,
            "range": 0.0,
            "direction_type": "unidirectional",
            "secondary_actions": [],
        }

    # 2) 按 eff_range 选主动作
    dominant = max(channel_info, key=lambda k: channel_info[k]["eff_range"])
    dom_info = channel_info[dominant]
    dom_eff  = dom_info["eff_range"]

    # 3) 静态姿势兜底
    if dom_eff < min_range:
        return {
            "action": "static_pose",
            "action_name": "静态姿势",
            "confidence": 0.3,
            "range": round(dom_info["raw_range"], 2),
            "direction_type": "unidirectional",
            "secondary_actions": [],
        }

    # 4) 方向细化
    signed = _analyze_signed_channel(dominant, dom_info["angles"])
    direction_type = signed["direction_type"]

    if direction_type == "unidirectional":
        action_name = dom_info["stats_name"]
    else:
        action_name = signed["action_name"]

        # 5) 置信度：基于 eff_range 的主 vs 次比
    sorted_eff = sorted((v["eff_range"] for v in channel_info.values()),
                        reverse=True)
    if len(sorted_eff) < 2:
        conf = 1.0
    else:
        # 原逻辑: min(1, first / second / 3)
        conf = min(1.0, sorted_eff[0] / (sorted_eff[1] + 1e-6) / 3)

    # 双向动作略微加权（前屈+后伸本就是一次完整 ROM 测试，值得更高置信）
    if direction_type == "bidirectional":
        conf = min(1.0, conf * 1.25)

    # 6) 次要动作：所有幅度达到主动作 SECONDARY_ACTION_RATIO 的通道
    threshold = dom_eff * SECONDARY_ACTION_RATIO
    secondary = []
    for key, info in channel_info.items():
        if key == dominant:
            continue
        if info["eff_range"] < threshold:
            continue
        sig = _analyze_signed_channel(key, info["angles"])
        sec_name = sig.get("action_name", info["stats_name"])
        secondary.append({
            "action":         key,
            "action_name":    sec_name,
            "range":          round(info["raw_range"], 2),
            "direction_type": sig["direction_type"],
        })
    secondary.sort(key=lambda x: x["range"], reverse=True)

    # 7) 组装结果
    result = {
        "action":            dominant,
        "action_name":       action_name,
        "confidence":        round(conf, 2),
        "range":             round(dom_info["raw_range"], 2),
        "direction_type":    direction_type,
        "secondary_actions": secondary,
    }
    if direction_type in ("bidirectional", "flexion_only", "extension_only"):
        result["flexion_max"]   = signed["flexion_max"]
        result["extension_max"] = signed["extension_max"]

    return result