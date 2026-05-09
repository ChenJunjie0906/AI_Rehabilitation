"""
pose_analyzer_base.py —— 姿态分析器共用基类
"""

import cv2
import numpy as np
from PIL import ImageFont
from rom_rules import (
    POSE_LANDMARKS, SKELETON_CONNECTIONS, ROM_DEFINITIONS,
    VIEW_DETECTION, MEASURE_VIEW_CONFIG, UNRELIABLE_WARNINGS, NORMAL_ROM,
    COMPENSATION_RULES,
)


class PoseAnalyzerBase:
    """姿态分析基类"""

    @staticmethod
    def _load_chinese_font(font_size: int = 40):
        """加载中文字体，支持多平台"""
        candidates = [
            "C:/Windows/Fonts/msyh.ttc",
            "C:/Windows/Fonts/simhei.ttf",
            "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
            "/System/Library/Fonts/PingFang.ttc",
        ]
        for path in candidates:
            try:
                return ImageFont.truetype(path, font_size)
            except OSError:
                continue
        return ImageFont.load_default()


    _POINT_META = {
        ("shoulder_flexion", "left"): (("左肩峰", 11), ("左肘", 13), ("垂直向下参考点(虚拟)", "↓")),
        ("shoulder_flexion", "right"): (("右肩峰", 12), ("右肘", 14), ("垂直向下参考点(虚拟)", "↓")),
        ("shoulder_abduction", "left"): (("左肩峰", 11), ("左肘", 13), ("垂直向下参考点(虚拟)", "↓")),
        ("shoulder_abduction", "right"): (("右肩峰", 12), ("右肘", 14), ("垂直向下参考点(虚拟)", "↓")),
        ("shoulder_extension", "left"): (("左肩峰", 11), ("左肘", 13), ("垂直向下参考点(虚拟)", "↓")),
        ("shoulder_extension", "right"): (("右肩峰", 12), ("右肘", 14), ("垂直向下参考点(虚拟)", "↓")),
        ("elbow_flexion",        "left"):   (("左肘",          13), ("左腕",           15), ("左肩",            11)),
        ("elbow_flexion",        "right"):  (("右肘",          14), ("右腕",           16), ("右肩",            12)),
        ("hip_flexion", "left"): (("左髋", 23), ("左膝", 25), ("垂直向下参考点(虚拟)", "↓")),
        ("hip_flexion", "right"): (("右髋", 24), ("右膝", 26), ("垂直向下参考点(虚拟)", "↓")),
        ("hip_abduction", "left"): (("左髋", 23), ("左膝", 25), ("垂直向下参考点(虚拟)", "↓")),
        ("hip_abduction", "right"): (("右髋", 24), ("右膝", 26), ("垂直向下参考点(虚拟)", "↓")),
        ("knee_flexion",         "left"):   (("左膝",          25), ("左踝",           27), ("左髋",            23)),
        ("knee_flexion",         "right"):  (("右膝",          26), ("右踝",           28), ("右髋",            24)),
        ("ankle_dorsiflexion",   "left"):   (("左踝",          27), ("左足中点(虚拟)", "29+31"), ("左膝",        25)),
        ("ankle_dorsiflexion",   "right"):  (("右踝",          28), ("右足中点(虚拟)", "30+32"), ("右膝",        26)),
        ("ankle_plantarflexion", "left"):   (("左踝",          27), ("左足中点(虚拟)", "29+31"), ("左膝",        25)),
        ("ankle_plantarflexion", "right"):  (("右踝",          28), ("右足中点(虚拟)", "30+32"), ("右膝",        26)),
        ("trunk_flexion",        None):     (("双髋中点(虚拟)", "23+24"), ("双肩中点(虚拟)", "11+12"), ("垂直向上参考点(虚拟)", "↑")),
        ("neck_flexion", "center"): (("双肩中点(虚拟)", "11+12"), ("鼻", 0), ("垂直向上参考点(虚拟)", "↑")),

    }

    # 头/颈中线点：无论侧视图左近还是右近都应绘制
    _CENTER_INDICES = {0, 7, 8}

    # 判断关键点是否在画面内时的归一化容差
    _FRAME_MARGIN = 0.01

    def __init__(self):
        self._lateral_calc_methods = {
            "shoulder_flexion":     self._calculate_shoulder_flexion,
            "shoulder_abduction":   self._calculate_shoulder_abduction,
            "shoulder_extension":   self._calculate_shoulder_extension,
            "elbow_flexion":        self._calculate_elbow_flexion,
            "hip_flexion":          self._calculate_hip_flexion,
            "hip_abduction":        self._calculate_hip_abduction,
            "knee_flexion":         self._calculate_knee_flexion,
            "ankle_dorsiflexion":   self._calculate_ankle_flexion,
            "ankle_plantarflexion": self._calculate_ankle_flexion,
        }

    # ------------------------------------------------------------------
    # 视角检测
    # ------------------------------------------------------------------

    def detect_view_angle(self, landmarks) -> str:
        shoulder_diff = abs(landmarks[11].x - landmarks[12].x)
        hip_diff      = abs(landmarks[23].x - landmarks[24].x)
        avg_diff      = (shoulder_diff + hip_diff) / 2
        if avg_diff < VIEW_DETECTION["side_threshold"]:
            return 'side'
        elif avg_diff > VIEW_DETECTION["front_threshold"]:
            return 'front'
        else:
            return 'oblique'

    def get_near_side(self, landmarks) -> str:
        l_z = (landmarks[11].z + landmarks[23].z) / 2
        r_z = (landmarks[12].z + landmarks[24].z) / 2
        return 'left' if l_z < r_z else 'right'

    # ------------------------------------------------------------------
    # 朝向推断（人体面朝 +X 还是 -X）
    # ------------------------------------------------------------------
    @staticmethod
    def _get_facing_direction(landmarks) -> int:
        """
        返回 +1 (面向 X 正方向) 或 -1 (面向 X 负方向)
        以鼻子相对双肩中点的 X 偏移为依据；若偏移太小则退回用双耳 z 判断。
        """
        mid_shoulder_x = (landmarks[11].x + landmarks[12].x) / 2
        nose_offset = landmarks[0].x - mid_shoulder_x

        if abs(nose_offset) > 0.015:
            return 1 if nose_offset > 0 else -1

        # 退化：用耳朵 z（更深的一侧在后方）
        try:
            l_ear_z = landmarks[7].z
            r_ear_z = landmarks[8].z
            if l_ear_z < r_ear_z:
                return 1 if nose_offset >= 0 else -1
            else:
                return -1 if nose_offset <= 0 else 1
        except Exception:
            return 1

    # ------------------------------------------------------------------
    # 工具方法
    # ------------------------------------------------------------------

    @staticmethod
    def calculate_angle(point1, point2, point3):
        p1 = np.array(point1, dtype=float)
        p2 = np.array(point2, dtype=float)
        p3 = np.array(point3, dtype=float)
        v1, v2 = p1 - p2, p3 - p2
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 < 1e-6 or n2 < 1e-6:
            return 0.0
        cos_a = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
        return round(float(np.degrees(np.arccos(cos_a))), 2)

    @staticmethod
    def _lm_xyz(lm):
        return np.array([lm.x, lm.y, lm.z])

    def _check_visibility(self, landmarks, indices, threshold=0.3):
        return all(getattr(landmarks[i], 'visibility', 1.0) >= threshold for i in indices)

    def _is_in_frame(self, landmarks, indices, margin=None):
        """
        检查给定关键点是否全部位于画面之内（归一化坐标 0~1）。
        只要任一点越出画面（或越出容差），即视为画面不全。
        """
        if margin is None:
            margin = self._FRAME_MARGIN
        lo, hi = -margin, 1.0 + margin
        for i in indices:
            lm = landmarks[i]
            if not (lo <= lm.x <= hi) or not (lo <= lm.y <= hi):
                return False
        return True

    def _check_foot_visibility(self, landmarks, side="left", vis_threshold=0.5):
        """
        足部可见性更严格的校验：
          1) 脚跟与脚趾 visibility 必须达到阈值；
          2) 脚跟与脚趾必须位于画面内；
          3) 脚跟—脚趾水平间距必须足够大（否则是 MediaPipe
             在足部被裁/被裤管遮挡时常见的"塌陷"误估，
             会把足中点甩到踝关节正下方，造成 ankle 角度
             从 90° 翻到 170°，误报跖屈/背屈）。
        """
        h_idx, t_idx = (29, 31) if side == "left" else (30, 32)
        heel, toe = landmarks[h_idx], landmarks[t_idx]

        if heel.visibility < vis_threshold or toe.visibility < vis_threshold:
            return False
        if not self._is_in_frame(landmarks, [h_idx, t_idx]):
            return False
        # 脚跟-脚趾水平分离度过小：视为未检测到足朝向
        if abs(heel.x - toe.x) < 0.01:
            return False
        return True

    def _check_joint_visibility(self, landmarks, measure_type, side, threshold=0.3):
        """
        统一的关节可见性判断：
          - visibility 必须达阈值
          - 所有必需点都必须位于画面之内（否则判为画面不全，不显示）
          - 足部另加严格校验（见 _check_foot_visibility）
        """
        if measure_type in ("shoulder_flexion", "shoulder_abduction", "shoulder_extension"):
            s, e = (11, 13) if side == "left" else (12, 14)
            h = 23 if side == "left" else 24
            req = [s, e, h]
        elif measure_type == "elbow_flexion":
            req = [11, 13, 15] if side == "left" else [12, 14, 16]
        elif measure_type in ("hip_flexion", "hip_abduction"):
            req = [23, 25] if side == "left" else [24, 26]
        elif measure_type == "knee_flexion":
            req = [23, 25, 27] if side == "left" else [24, 26, 28]
        elif measure_type in ("ankle_dorsiflexion", "ankle_plantarflexion"):
            k, a = (25, 27) if side == "left" else (26, 28)
            req = [k, a]
            if not all(landmarks[i].visibility >= threshold for i in req):
                return False
            if not self._is_in_frame(landmarks, req):
                return False
            return self._check_foot_visibility(landmarks, side)
        else:
            return True

        if not all(landmarks[i].visibility >= threshold for i in req):
            return False
        if not self._is_in_frame(landmarks, req):
            return False
        return True

    # ------------------------------------------------------------------
    # 中间点 / 参考点计算
    # ------------------------------------------------------------------

    def _get_mid_shoulder(self, landmarks):
        return np.array([
            (landmarks[11].x + landmarks[12].x) / 2,
            (landmarks[11].y + landmarks[12].y) / 2,
            (landmarks[11].z + landmarks[12].z) / 2,
        ])

    def _get_mid_hip(self, landmarks):
        return np.array([
            (landmarks[23].x + landmarks[24].x) / 2,
            (landmarks[23].y + landmarks[24].y) / 2,
            (landmarks[23].z + landmarks[24].z) / 2,
        ])

    def _calculate_foot_center(self, landmarks, side="left"):
        h, t = (landmarks[29], landmarks[31]) if side == "left" else (landmarks[30], landmarks[32])
        return np.array([(h.x + t.x) / 2, (h.y + t.y) / 2, (h.z + t.z) / 2])

    # ------------------------------------------------------------------
    # 轴心-移动臂-固定臂 计算方法
    # ------------------------------------------------------------------

    def _calculate_shoulder_flexion(self, landmarks, side="left"):
        s, e = (11, 13) if side == "left" else (12, 14)
        shoulder_point = self._lm_xyz(landmarks[s])
        vertical_ref = shoulder_point + np.array([0.0, 1.0, 0.0])
        return shoulder_point, self._lm_xyz(landmarks[e]), vertical_ref

    def _calculate_shoulder_abduction(self, landmarks, side="left"):
        s, e = (11, 13) if side == "left" else (12, 14)
        shoulder_point = self._lm_xyz(landmarks[s])
        vertical_ref = shoulder_point + np.array([0.0, 1.0, 0.0])
        return shoulder_point, self._lm_xyz(landmarks[e]), vertical_ref

    def _calculate_shoulder_extension(self, landmarks, side="left"):
        """轴=肩 | moving→肘（上臂）| fixed→肩关节垂直向下 | 临床=几何角"""
        s, e = (11, 13) if side == "left" else (12, 14)
        shoulder_point = self._lm_xyz(landmarks[s])
        vertical_ref = shoulder_point + np.array([0.0, 1.0, 0.0])
        return shoulder_point, self._lm_xyz(landmarks[e]), vertical_ref

    # ──────────────────────────────────────────────────────────────
    # 根据真实朝向判断肩前屈/后伸
    # ──────────────────────────────────────────────────────────────
    @staticmethod
    def _determine_shoulder_direction(landmarks, side="left"):
        """
        判断肩关节运动方向(前屈/后伸)
        返回: 'flexion' (前屈) 或 'extension' (后伸)
        """
        s_idx = 11 if side == "left" else 12
        e_idx = 13 if side == "left" else 14

        x_diff = landmarks[e_idx].x - landmarks[s_idx].x
        facing = PoseAnalyzerBase._get_facing_direction(landmarks)

        projected = x_diff * facing

        if abs(projected) < 0.02:
            return 'flexion'
        return 'flexion' if projected > 0 else 'extension'

    @staticmethod
    def _apply_shoulder_sign(measure_type, angle, landmarks, side):
        """
        为肩关节角度应用正负号：前屈正值、后伸负值。其他关节保持不变。
        """
        if angle is None:
            return None
        if measure_type not in ("shoulder_flexion", "shoulder_extension"):
            return angle
        direction = PoseAnalyzerBase._determine_shoulder_direction(landmarks, side)
        return angle if direction == 'flexion' else -angle

    def _calculate_elbow_flexion(self, landmarks, side="left"):
        """轴=肘 | moving→腕（前臂）| fixed→肩（上臂）| 临床=180°−几何角"""
        s, e, w = (11, 13, 15) if side == "left" else (12, 14, 16)
        return (self._lm_xyz(landmarks[e]),
                self._lm_xyz(landmarks[w]),
                self._lm_xyz(landmarks[s]))

    def _calculate_hip_flexion(self, landmarks, side="left"):
        h, k = (23, 25) if side == "left" else (24, 26)
        hip_point = self._lm_xyz(landmarks[h])
        vertical_ref = hip_point + np.array([0.0, 1.0, 0.0])
        return hip_point, self._lm_xyz(landmarks[k]), vertical_ref

    def _calculate_hip_abduction(self, landmarks, side="left"):
        h, k = (23, 25) if side == "left" else (24, 26)
        hip_point = self._lm_xyz(landmarks[h])
        vertical_ref = hip_point + np.array([0.0, 1.0, 0.0])
        return hip_point, self._lm_xyz(landmarks[k]), vertical_ref

    def _calculate_knee_flexion(self, landmarks, side="left"):
        h, k, a = (23, 25, 27) if side == "left" else (24, 26, 28)
        return self._lm_xyz(landmarks[k]), self._lm_xyz(landmarks[a]), self._lm_xyz(landmarks[h])

    def _calculate_ankle_flexion(self, landmarks, side="left"):
        k, a = (25, 27) if side == "left" else (26, 28)
        return (self._lm_xyz(landmarks[a]),
                self._calculate_foot_center(landmarks, side),
                self._lm_xyz(landmarks[k]))

    def _calculate_trunk_flexion(self, landmarks):
        mh = self._get_mid_hip(landmarks)
        ms = self._get_mid_shoulder(landmarks)
        return mh, ms, mh + np.array([0.0, -1.0, 0.0])

    def _calculate_neck_flexion(self, landmarks):
        shoulder_mid = self._get_mid_shoulder(landmarks)
        vertical_ref = shoulder_mid + np.array([0.0, -1.0, 0.0])
        return shoulder_mid, self._lm_xyz(landmarks[0]), vertical_ref

    # ------------------------------------------------------------------
    # 角度计算与临床修正
    # ------------------------------------------------------------------

    @staticmethod
    def _get_joint_angle(axis, moving, fixed):
        """计算轴心-移动点-固定点之间的 3D 角度（兼容保留）"""
        mv = moving - axis
        fv = fixed  - axis
        n1, n2 = np.linalg.norm(mv), np.linalg.norm(fv)
        if n1 < 1e-6 or n2 < 1e-6:
            return None
        cos_a = np.clip(np.dot(mv, fv) / (n1 * n2), -1.0, 1.0)
        return round(float(np.degrees(np.arccos(cos_a))), 2)

    @staticmethod
    def _get_joint_angle_2d(axis, moving, fixed):
        """
        只取 x, y 计算角度 —— 避免 MediaPipe z 轴噪声污染。
        ROM 测量只在"运动平面与相机平面对齐"的视角下被标 reliable，
        因此 2D 投影就是我们真正需要的临床角。
        """
        mv = np.array([moving[0] - axis[0], moving[1] - axis[1]])
        fv = np.array([fixed[0]  - axis[0], fixed[1]  - axis[1]])
        n1, n2 = np.linalg.norm(mv), np.linalg.norm(fv)
        if n1 < 1e-6 or n2 < 1e-6:
            return None
        cos_a = np.clip(np.dot(mv, fv) / (n1 * n2), -1.0, 1.0)
        return round(float(np.degrees(np.arccos(cos_a))), 2)

    @staticmethod
    def _apply_clinical_correction(measure_type, angle):
        if angle is None:
            return None
        if measure_type in ("knee_flexion", "elbow_flexion"):
            return round(max(0.0, 180.0 - angle), 2)
        if measure_type == "ankle_dorsiflexion":
            # 解剖上几何角应在 60°~120° 之间，超出说明足部关键点被误估
            if not (60.0 <= angle <= 120.0):
                return None
            return round(max(0.0, 90.0 - angle), 2)
        if measure_type == "ankle_plantarflexion":
            if not (60.0 <= angle <= 120.0):
                return None
            return round(max(0.0, angle - 90.0), 2)
        return angle

    # ------------------------------------------------------------------
    # 躯干补偿角（前倾正、后仰负，考虑朝向）
    # ------------------------------------------------------------------

    def _get_trunk_compensation(self, landmarks):
        ms = self._get_mid_shoulder(landmarks)
        mh = self._get_mid_hip(landmarks)
        tv = np.array([ms[0] - mh[0], ms[1] - mh[1], 0.0])
        n  = np.linalg.norm(tv)
        if n < 1e-6:
            return 0.0
        cos_a = np.clip(np.dot(tv, np.array([0.0, -1.0, 0.0])) / n, -1.0, 1.0)
        tilt  = float(np.degrees(np.arccos(cos_a)))

        facing = self._get_facing_direction(landmarks)
        lateral = (ms[0] - mh[0]) * facing
        return round(tilt if lateral >= 0 else -tilt, 2)

    # ------------------------------------------------------------------
    # 代偿信号计算（修正：躯干侧屈用 midspine，不用肩连线）
    # ------------------------------------------------------------------

    def compute_compensation_signals(self, landmarks) -> dict:
        """提取代偿分析所需的所有原始信号"""
        ms = self._get_mid_shoulder(landmarks)
        mh = self._get_mid_hip(landmarks)

        # ── 躯干前/后倾（矢状面，带符号）──
        trunk_vec = np.array([ms[0] - mh[0], ms[1] - mh[1]])
        ref = np.array([0.0, -1.0])
        n = np.linalg.norm(trunk_vec)
        if n < 1e-6:
            trunk_signed = 0.0
        else:
            cos_a = np.clip(np.dot(trunk_vec, ref) / n, -1.0, 1.0)
            tilt = float(np.degrees(np.arccos(cos_a)))
            facing = self._get_facing_direction(landmarks)
            lateral = (ms[0] - mh[0]) * facing
            trunk_signed = tilt if lateral >= 0 else -tilt

        view = self.detect_view_angle(landmarks)
        coronal_reliable = (view == 'front')

        # ── ★ 躯干冠状面侧屈：用 midspine（肩中→髋中），而不是肩连线 ──
        # 肩连线会因肩胛上回旋而倾斜，不代表躯干真正侧屈
        if coronal_reliable:
            dx = ms[0] - mh[0]          # 水平分量（归一化坐标）
            dy = mh[1] - ms[1]          # 竖直分量（图像 y 向下为正，髋在下）
            if dy > 1e-6:
                trunk_lateral = float(np.degrees(np.arctan2(abs(dx), dy)))
            else:
                trunk_lateral = 0.0

            # 骨盆侧倾：仍可用髋连线（下肢不受肩胛影响）
            hx = landmarks[24].x - landmarks[23].x
            hy = landmarks[24].y - landmarks[23].y
            seg = float(np.hypot(hx, hy))
            if seg > 1e-6:
                pelvis_tilt = float(np.degrees(np.arcsin(np.clip(hy / seg, -1.0, 1.0))))
            else:
                pelvis_tilt = 0.0
        else:
            trunk_lateral = 0.0
            pelvis_tilt = 0.0

        # 耸肩比
        shoulder_width = abs(landmarks[11].x - landmarks[12].x) + 1e-6
        l_elev = (landmarks[7].y - landmarks[11].y) / shoulder_width
        r_elev = (landmarks[8].y - landmarks[12].y) / shoulder_width
        shoulder_elev_ratio = abs((l_elev + r_elev) / 2) * 100

        return {
            "trunk_tilt_signed":        round(trunk_signed, 2),
            "trunk_lateral_tilt":       round(abs(trunk_lateral), 2),
            "pelvis_lateral_tilt":      round(abs(pelvis_tilt), 2),
            "shoulder_elevation_ratio": round(shoulder_elev_ratio, 2),
            "coronal_reliable":         coronal_reliable,
        }

    def analyze_compensation(self, measure_type: str, signals: dict) -> list:
        """根据主动作类型和当前信号，返回触发的代偿项"""
        rules = COMPENSATION_RULES.get(measure_type, [])
        triggered = []
        for rule in rules:
            val = signals.get(rule["signal"])
            if val is None:
                continue
            if rule["direction"] == "negative":
                v = -val if val < 0 else 0
            else:
                v = val if val > 0 else 0
            if v >= rule["threshold_severe"]:
                level = "severe"
            elif v >= rule["threshold_mild"]:
                level = "mild"
            else:
                continue
            triggered.append({
                "code":       rule["code"],
                "name":       rule["name"],
                "level":      level,
                "value":      round(v, 2),
                "suggestion": rule["suggestion"],
            })
        return triggered

    def _compute_joint_vectors(self, landmarks, measure_type, side):
        eff_side = side if side in ("left", "right") else "left"
        if measure_type in self._lateral_calc_methods:
            axis, moving, fixed = self._lateral_calc_methods[measure_type](landmarks, eff_side)
            visible = self._check_joint_visibility(landmarks, measure_type, eff_side)
        elif measure_type == "trunk_flexion":
            axis, moving, fixed = self._calculate_trunk_flexion(landmarks)
            visible = self._check_visibility(landmarks, [11, 12, 23, 24])
        elif measure_type == "neck_flexion":
            axis, moving, fixed = self._calculate_neck_flexion(landmarks)
            visible = self._check_visibility(landmarks, [0, 11, 12, 23, 24])
        else:
            return None, None, None, False
        return axis, moving, fixed, visible

    @staticmethod
    def _build_entry(defn, measure_type, key, name, side, angle, visible,
                     reliable, warning, view, trunk_tilt):
        """
        构建标准 ROM 报告条目字典。
        out_of_range: reliable=True 且角度超出 NORMAL_ROM 定义的正常范围时为 True。
        """
        lo, hi       = NORMAL_ROM.get(measure_type, (0, float('inf')))
        out_of_range = bool(reliable and angle is not None and not (lo <= angle <= hi))

        return {
            "key":          key,
            "name":         name,
            "measure_type": measure_type,
            "side":         side,
            "angle":        angle,
            "unit":         "°",
            "visible":      visible,
            "reliable":     reliable,
            "out_of_range": out_of_range,
            "warning":      warning,
            "view":         view,
            "rule":         defn["rule"],
            "normal_range": defn["normal_range"],
            "trunk_tilt":   trunk_tilt,
        }

    def _attach_verbose(self, entry, measure_type, side, axis, moving, fixed):
        side_key = side if side in ("left", "right", "center") else None
        meta = (self._POINT_META.get((measure_type, side_key)) or
                self._POINT_META.get((measure_type, "left")))
        if meta:
            (an, ai), (mn, mi), (fn, fi) = meta
            entry["axis"]   = {"name": an, "landmark_indices": ai,
                               "normalized": [round(float(v), 6) for v in axis]}
            entry["moving"] = {"name": mn, "landmark_indices": mi,
                               "normalized": [round(float(v), 6) for v in moving]}
            entry["fixed"]  = {"name": fn, "landmark_indices": fi,
                               "normalized": [round(float(v), 6) for v in fixed]}

    # ------------------------------------------------------------------
    # ROM 报告生成（核心）—— 全部使用 2D 角度计算
    # ------------------------------------------------------------------

    def calculate_rom_report(self, landmarks, frame_width, frame_height, verbose=False):
        report = []
        trunk_tilt = round(self._get_trunk_compensation(landmarks), 2)

        view = self.detect_view_angle(landmarks)
        near_side = self.get_near_side(landmarks)
        processed = set()

        for defn in ROM_DEFINITIONS:
            measure_type = defn["measure_type"]
            side = defn.get("side")
            eff_side = side if side in ("left", "right") else "left"
            view_cfg = MEASURE_VIEW_CONFIG.get(measure_type, {})
            preferred_view = view_cfg.get("preferred_view", "both")

            # ══ 侧视图 ════════════════════════════════════════════════
            if view == 'side':
                if measure_type in ("shoulder_flexion", "shoulder_extension"):
                    shoulder_key = "shoulder_sagittal"
                    if shoulder_key in processed:
                        continue
                    processed.add(shoulder_key)
                    if measure_type != "shoulder_flexion":
                        continue
                elif measure_type in processed:
                    continue
                else:
                    processed.add(measure_type)

                is_bilateral = side in ("left", "right")
                calc_side = near_side if is_bilateral else eff_side
                merged_name = view_cfg.get("side_merged_name", defn["name"])
                merged_key = f"{measure_type}_side" if is_bilateral else defn["key"]

                if preferred_view == 'side':
                    axis, moving, fixed, visible = self._compute_joint_vectors(
                        landmarks, measure_type, calc_side)
                    geo = self._get_joint_angle_2d(axis, moving, fixed) if axis is not None else None

                    actual_measure_type = measure_type
                    if measure_type in ("shoulder_flexion", "shoulder_extension"):
                        direction = self._determine_shoulder_direction(landmarks, calc_side)
                        if direction == 'flexion':
                            actual_measure_type = "shoulder_flexion"
                            for flex_defn in ROM_DEFINITIONS:
                                if (flex_defn["measure_type"] == "shoulder_flexion"
                                        and flex_defn["side"] == calc_side):
                                    defn = flex_defn
                                    break
                            merged_name = "肩前屈"
                        else:
                            actual_measure_type = "shoulder_extension"
                            for ext_defn in ROM_DEFINITIONS:
                                if (ext_defn["measure_type"] == "shoulder_extension"
                                        and ext_defn["side"] == calc_side):
                                    defn = ext_defn
                                    break
                            merged_name = "肩后伸"
                        merged_key = f"shoulder_sagittal_{calc_side}"

                    angle = self._apply_clinical_correction(actual_measure_type, geo)
                    angle = self._apply_shoulder_sign(actual_measure_type, angle, landmarks, calc_side)
                    if not visible:
                        angle = None
                    entry = self._build_entry(
                        defn, actual_measure_type, merged_key, merged_name,
                        calc_side, angle, visible,
                        reliable=True, warning="", view=view, trunk_tilt=trunk_tilt,
                    )
                    if verbose and axis is not None:
                        self._attach_verbose(entry, actual_measure_type,
                                             calc_side, axis, moving, fixed)
                else:
                    warning = UNRELIABLE_WARNINGS.get((measure_type, 'side'), "侧视图不可靠")
                    entry = self._build_entry(
                        defn, measure_type, merged_key, merged_name,
                        None, None, False,
                        reliable=False, warning=warning, view=view, trunk_tilt=trunk_tilt,
                    )
                report.append(entry)

            # ══ 正视图 ════════════════════════════════════════════════
            elif view == 'front':
                if preferred_view == 'front':
                    axis, moving, fixed, visible = self._compute_joint_vectors(
                        landmarks, measure_type, eff_side)
                    geo = self._get_joint_angle_2d(axis, moving, fixed) if axis is not None else None
                    angle = self._apply_clinical_correction(measure_type, geo)
                    angle = self._apply_shoulder_sign(measure_type, angle, landmarks, eff_side)
                    if not visible:
                        angle = None
                    entry = self._build_entry(
                        defn, measure_type, defn["key"], defn["name"],
                        side, angle, visible,
                        reliable=True, warning="", view=view, trunk_tilt=trunk_tilt,
                    )
                    if verbose and axis is not None:
                        self._attach_verbose(entry, measure_type, side, axis, moving, fixed)
                else:
                    warning = UNRELIABLE_WARNINGS.get((measure_type, 'front'), "正视图不可靠")
                    entry = self._build_entry(
                        defn, measure_type, defn["key"], defn["name"],
                        side, None, False,
                        reliable=False, warning=warning, view=view, trunk_tilt=trunk_tilt,
                    )
                report.append(entry)

            # ══ 斜视图 ════════════════════════════════════════════════
            else:
                axis, moving, fixed, visible = self._compute_joint_vectors(
                    landmarks, measure_type, eff_side)
                geo = self._get_joint_angle_2d(axis, moving, fixed) if axis is not None else None
                angle = self._apply_clinical_correction(measure_type, geo)
                angle = self._apply_shoulder_sign(measure_type, angle, landmarks, eff_side)
                if not visible:
                    angle = None
                entry = self._build_entry(
                    defn, measure_type, defn["key"], defn["name"],
                    side, angle, visible,
                    reliable=False, warning="斜视图，建议调整为侧视图或正视图",
                    view=view, trunk_tilt=trunk_tilt,
                )
                if verbose and axis is not None:
                    self._attach_verbose(entry, measure_type, side, axis, moving, fixed)
                report.append(entry)

        return report

    # ------------------------------------------------------------------
    # 骨架 / 关键点绘制
    # ------------------------------------------------------------------

    def _draw_skeleton(self, image, landmarks, w, h, near_side=None):
        """
        绘制骨架连线。
        - 侧视图 (near_side 指定) 时只画近侧肢体，但头颈中线点始终绘制。
        - 额外绘制"肩中点 → 鼻"作为颈部前屈可视化。
        """
        pts2d = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]

        if near_side:
            left_indices  = {11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31}
            right_indices = {12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32}
            side_set = left_indices if near_side == 'left' else right_indices
            visible_indices = side_set | self._CENTER_INDICES
            midline_pairs = {(11, 12), (23, 24), (11, 23), (12, 24)}

            for a, b in SKELETON_CONNECTIONS:
                if (a in visible_indices and b in visible_indices) or \
                   ((a, b) in midline_pairs):
                    cv2.line(image, pts2d[a], pts2d[b], (0, 200, 255), 2)
        else:
            for a, b in SKELETON_CONNECTIONS:
                cv2.line(image, pts2d[a], pts2d[b], (0, 200, 255), 2)

        # 颈部参考线：肩中点 → 鼻
        ms_x = int((landmarks[11].x + landmarks[12].x) / 2 * w)
        ms_y = int((landmarks[11].y + landmarks[12].y) / 2 * h)
        nose = pts2d[0]
        cv2.line(image, (ms_x, ms_y), nose, (0, 200, 255), 2)
        cv2.circle(image, (ms_x, ms_y), 4, (0, 255, 0), -1)

    def _draw_keypoints(self, image, landmarks, w, h, show_index=False, near_side=None):
        """绘制关键点；侧视图下头颈中线点也要画。"""
        left_indices  = {11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31}
        right_indices = {12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32}

        for idx, lm in enumerate(landmarks):
            if near_side:
                allowed = (left_indices if near_side == 'left' else right_indices) \
                          | self._CENTER_INDICES
                if idx not in allowed:
                    continue

            x, y = int(lm.x * w), int(lm.y * h)
            vis = getattr(lm, 'visibility', 1.0)
            color = (0, 255, 0) if vis >= 0.5 else (128, 128, 128)
            cv2.circle(image, (x, y), 5 if show_index else 4, color, -1)
            if show_index:
                cv2.putText(image, str(idx), (x + 5, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 0), 1)