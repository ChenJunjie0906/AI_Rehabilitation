"""
rom_rules.py —— 关节活动度规则定义

包含：
  - 关键点名称映射
  - 骨架连线
  - 正常关节活动范围
  - ROM 测量项目定义列表
  - 各测量项底层向量计算逻辑 CALC_LOGIC
  - 视角检测配置 VIEW_DETECTION
  - 测量类型视角映射 MEASURE_VIEW_CONFIG
  - 不可靠测量警告文本 UNRELIABLE_WARNINGS

向量约定（与代码完全对应）：
  moving_vec = moving_point - axis   （移动臂向量）
  fixed_vec  = fixed_point  - axis   （固定臂向量）
  angle = arccos( dot(moving_vec, fixed_vec) / (|moving_vec| × |fixed_vec|) )
"""

# ══════════════════════════════════════════════════════════════════════
# 常量
# ══════════════════════════════════════════════════════════════════════

VIS_THRESHOLD = 0.3  # 关键点最低可见度阈值

# ── MediaPipe Pose 33 关键点索引 → 中文名称 ──────────────────────────
POSE_LANDMARKS = {
    0:  "鼻",
    1:  "左眼内角",  2:  "左眼中心",  3:  "左眼外角",
    4:  "右眼内角",  5:  "右眼中心",  6:  "右眼外角",
    7:  "左耳",      8:  "右耳",
    9:  "嘴左角",    10: "嘴右角",
    11: "左肩",      12: "右肩",
    13: "左肘",      14: "右肘",
    15: "左腕",      16: "右腕",
    17: "左小指",    18: "右小指",
    19: "左食指",    20: "右食指",
    21: "左拇指",    22: "右拇指",
    23: "左髋",      24: "右髋",
    25: "左膝",      26: "右膝",
    27: "左踝",      28: "右踝",
    29: "左脚跟",    30: "右脚跟",
    31: "左脚趾",    32: "右脚趾",
}

# ── 骨架连线（绘制用）────────────────────────────────────────────────
SKELETON_CONNECTIONS = [
    # 头颈
    (0, 7), (0, 8),
    # 躯干
    (11, 12), (11, 23), (12, 24), (23, 24),
    # 左臂
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
    # 右臂
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22),
    # 左腿
    (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),
    # 右腿
    (24, 26), (26, 28), (28, 30), (28, 32), (30, 32),
]

# ══════════════════════════════════════════════════════════════════════
# 视角检测配置
# ══════════════════════════════════════════════════════════════════════
#
# 判据：双肩 + 双髋 x 坐标差均值
#   < side_threshold  → 侧视图（人体侧面对相机）
#   > front_threshold → 正视图（人体正面对相机）
#   介于两者之间      → 斜视图（oblique）

VIEW_DETECTION = {
    "side_threshold":  0.04,   # 归一化坐标差；小于此值视为侧视图
    "front_threshold": 0.05,   # 归一化坐标差；大于此值视为正视图
}

# ══════════════════════════════════════════════════════════════════════
# 测量类型 → 视角配置
# ══════════════════════════════════════════════════════════════════════
#
# preferred_view   : 该测量类型的最佳视角 ("side" | "front")
# side_merged_name : 侧视图下合并左右后的显示名称（不区分左右）
# motion_plane     : 运动平面 ("sagittal" | "coronal")

MEASURE_VIEW_CONFIG = {
    "shoulder_flexion":     {"preferred_view": "side",  "side_merged_name": "肩前屈",  "motion_plane": "sagittal"},
    "shoulder_extension":   {"preferred_view": "side",  "side_merged_name": "肩后伸",  "motion_plane": "sagittal"},  # ← 新增
    "shoulder_abduction":   {"preferred_view": "front", "side_merged_name": "肩外展",  "motion_plane": "coronal"},
    "elbow_flexion":        {"preferred_view": "side",  "side_merged_name": "肘屈曲",   "motion_plane": "sagittal"},
    "hip_flexion":          {"preferred_view": "side",  "side_merged_name": "髋前屈",   "motion_plane": "sagittal"},
    "hip_abduction":        {"preferred_view": "front", "side_merged_name": "髋外展",   "motion_plane": "coronal"},
    "knee_flexion":         {"preferred_view": "side",  "side_merged_name": "膝屈曲",   "motion_plane": "sagittal"},
    "ankle_dorsiflexion":   {"preferred_view": "side",  "side_merged_name": "踝背屈",   "motion_plane": "sagittal"},
    "ankle_plantarflexion": {"preferred_view": "side",  "side_merged_name": "踝跖屈",   "motion_plane": "sagittal"},
    "trunk_flexion":        {"preferred_view": "side",  "side_merged_name": "躯干前屈", "motion_plane": "sagittal"},
    "neck_flexion":         {"preferred_view": "side",  "side_merged_name": "颈部前屈", "motion_plane": "sagittal"},
}

# ══════════════════════════════════════════════════════════════════════
# 不可靠测量警告文本
# ══════════════════════════════════════════════════════════════════════
#
# 键：(measure_type, view)
# 值：显示在界面和 JSON 中的警告文字

UNRELIABLE_WARNINGS = {
    # 侧视图下的冠状面运动
    ("shoulder_abduction",   "side"):    "冠状面，侧视Z轴不可测",
    ("hip_abduction",        "side"):    "冠状面，侧视Z轴不可测",
    # 正视图下的矢状面运动
    ("shoulder_flexion",     "front"):   "矢状面，正视Z轴不可测",
    ("shoulder_extension",  "front"):    "矢状面，正视Z轴不可测",
    ("elbow_flexion",        "front"):   "矢状面，正视Z轴不可测",
    ("hip_flexion",          "front"):   "矢状面，正视Z轴不可测",
    ("knee_flexion",         "front"):   "矢状面，正视Z轴不可测",
    ("ankle_dorsiflexion",   "front"):   "矢状面，正视Z轴不可测",
    ("ankle_plantarflexion", "front"):   "矢状面，正视Z轴不可测",
    ("trunk_flexion",        "front"):   "矢状面，正视Z轴不可测",
    ("neck_flexion",         "front"):   "矢状面，正视Z轴不可测",
}

# ══════════════════════════════════════════════════════════════════════
# 正常关节活动范围（中立位 0° 法，参考临床规范）
# ══════════════════════════════════════════════════════════════════════
NORMAL_ROM = {
    "shoulder_flexion":      (0, 180),
    "shoulder_extension":    (-50, 0),
    "shoulder_abduction":    (0, 180),
    "elbow_flexion":         (0, 145),
    "wrist_dorsiflexion":    (0,  70),
    "wrist_palmarflexion":   (0,  90),
    "hip_flexion":           (0,  90),
    "hip_extension":         (0,  15),
    "hip_abduction":         (0,  45),
    "hip_adduction":         (0,  20),
    "hip_ext_rotation":      (0,  45),
    "hip_int_rotation":      (0,  45),
    "knee_flexion":          (0, 130),
    "ankle_dorsiflexion":    (0,  20),
    "ankle_plantarflexion":  (0,  45),
    "neck_flexion":          (0,  45),
    "neck_extension":        (0,  50),
    "trunk_flexion":         (0,  45),
    "trunk_extension":       (0,  30),
}

# ══════════════════════════════════════════════════════════════════════
# 底层向量计算逻辑
# ══════════════════════════════════════════════════════════════════════
#
# 键：(measure_type, side)，side 取 "left" / "right" / "center"
#
# 字段含义：
#   axis        顶点（关节轴心）
#   moving_arm  移动臂向量 = moving_point - axis
#   fixed_arm   固定臂向量 = fixed_point  - axis
#   formula     角度计算公式（展开形式）
#   description 临床语义说明

CALC_LOGIC = {

    # ── 肩关节前屈 ───────────────────────────────────────────────────
    ("shoulder_flexion", "left"): {
        "axis": "左肩关节 (lm11)",
        "moving_arm": "向量 lm11 → lm13，左肩指向左肘",
        "fixed_arm": "向量 lm11 → 垂直向下参考点，即 lm11.xyz + [0, 1, 0]，沿左肩垂直向下",
        "formula": "arccos( dot(lm13-lm11, 垂直向下-lm11) / (|lm13-lm11| × |垂直向下-lm11|) )",
        "description": "以左肩为顶点，上臂（肩→肘）为移动臂，肩关节垂直向下参考线为固定臂",
    },
    ("shoulder_flexion", "right"): {
        "axis": "右肩关节 (lm12)",
        "moving_arm": "向量 lm12 → lm14，右肩指向右肘",
        "fixed_arm": "向量 lm12 → 垂直向下参考点，即 lm12.xyz + [0, 1, 0]，沿右肩垂直向下",
        "formula": "arccos( dot(lm14-lm12, 垂直向下-lm12) / (|lm14-lm12| × |垂直向下-lm12|) )",
        "description": "以右肩为顶点，上臂（肩→肘）为移动臂，肩关节垂直向下参考线为固定臂",
    },

    # ── 肩关节外展 ───────────────────────────────────────────────────
    ("shoulder_abduction", "left"): {
        "axis": "左肩关节 (lm11)",
        "moving_arm": "向量 lm11 → lm13，左肩指向左肘",
        "fixed_arm": "向量 lm11 → 垂直向下参考点，即 lm11.xyz + [0, 1, 0]，沿左肩垂直向下",
        "formula": "arccos( dot(lm13-lm11, 垂直向下-lm11) / (|lm13-lm11| × |垂直向下-lm11|) )",
        "description": "以左肩为顶点，上臂（肩→肘）为移动臂，肩关节垂直向下参考线为固定臂",
    },
    ("shoulder_abduction", "right"): {
        "axis": "右肩关节 (lm12)",
        "moving_arm": "向量 lm12 → lm14，右肩指向右肘",
        "fixed_arm": "向量 lm12 → 垂直向下参考点，即 lm12.xyz + [0, 1, 0]，沿右肩垂直向下",
        "formula": "arccos( dot(lm14-lm12, 垂直向下-lm12) / (|lm14-lm12| × |垂直向下-lm12|) )",
        "description": "以右肩为顶点，上臂（肩→肘）为移动臂，肩关节垂直向下参考线为固定臂",
    },

    # ── 肩关节后伸 ───────────────────────────────────────────────────
    ("shoulder_extension", "left"): {
        "axis": "左肩关节 (lm11)",
        "moving_arm": "向量 lm11 → lm13，左肩指向左肘",
        "fixed_arm": "向量 lm11 → 垂直向下参考点，即 lm11.xyz + [0, 1, 0]，沿左肩垂直向下",
        "formula": "arccos( dot(lm13-lm11, 垂直向下-lm11) / (|lm13-lm11| × |垂直向下-lm11|) )",
        "description": "以左肩为顶点，上臂（肩→肘）为移动臂，肩关节垂直向下参考线为固定臂，手臂向后时的几何夹角即后伸幅度",
    },
    ("shoulder_extension", "right"): {
        "axis": "右肩关节 (lm12)",
        "moving_arm": "向量 lm12 → lm14，右肩指向右肘",
        "fixed_arm": "向量 lm12 → 垂直向下参考点，即 lm12.xyz + [0, 1, 0]，沿右肩垂直向下",
        "formula": "arccos( dot(lm14-lm12, 垂直向下-lm12) / (|lm14-lm12| × |垂直向下-lm12|) )",
        "description": "以右肩为顶点，上臂（肩→肘）为移动臂，肩关节垂直向下参考线为固定臂，手臂向后时的几何夹角即后伸幅度",
    },

    # ── 肘关节屈曲 ───────────────────────────────────────────────────
    # 轴=肘；移动臂=肘→腕（前臂）；固定臂=肘→肩（上臂）
    # 完全伸直时几何角≈180°；临床屈曲角 = 180° − 几何角
    ("elbow_flexion", "left"): {
        "axis":        "左肘关节 (lm13)",
        "moving_arm":  "向量 lm13 → lm15，即 lm15.xyz - lm13.xyz，左肘指向左腕（前臂方向）",
        "fixed_arm":   "向量 lm13 → lm11，即 lm11.xyz - lm13.xyz，左肘指向左肩（上臂方向）",
        "formula":     "arccos( dot(lm15-lm13, lm11-lm13) / (|lm15-lm13| × |lm11-lm13|) )",
        "description": "以左肘为顶点，前臂向量（肘→腕）为移动臂，上臂向量（肘→肩）为固定臂；完全伸直时几何角≈180°，临床屈曲角 = 180° − 几何角",
    },
    ("elbow_flexion", "right"): {
        "axis":        "右肘关节 (lm14)",
        "moving_arm":  "向量 lm14 → lm16，即 lm16.xyz - lm14.xyz，右肘指向右腕（前臂方向）",
        "fixed_arm":   "向量 lm14 → lm12，即 lm12.xyz - lm14.xyz，右肘指向右肩（上臂方向）",
        "formula":     "arccos( dot(lm16-lm14, lm12-lm14) / (|lm16-lm14| × |lm12-lm14|) )",
        "description": "以右肘为顶点，前臂向量（肘→腕）为移动臂，上臂向量（肘→肩）为固定臂；完全伸直时几何角≈180°，临床屈曲角 = 180° − 几何角",
    },

    # ── 髋关节前屈 ───────────────────────────────────────────────────

    ("hip_flexion", "left"): {
        "axis": "左髋关节 (lm23)",
        "moving_arm": "向量 lm23 → lm25，即 lm25.xyz - lm23.xyz，左髋指向左膝",
        "fixed_arm": "向量 lm23 → 垂直向下参考点，即 lm23.xyz + [0, 1, 0]，沿髋关节垂直向下",
        "formula": "arccos( dot(lm25-lm23, 垂直向下-lm23) / (|lm25-lm23| × |垂直向下-lm23|) )",
        "description": "以左髋为顶点，大腿向量（髋→膝）为移动臂，髋关节垂直向下参考线为固定臂，两向量夹角即髋关节前屈幅度",
    },
    ("hip_flexion", "right"): {
        "axis": "右髋关节 (lm24)",
        "moving_arm": "向量 lm24 → lm26，即 lm26.xyz - lm24.xyz，右髋指向右膝",
        "fixed_arm": "向量 lm24 → 垂直向下参考点，即 lm24.xyz + [0, 1, 0]，沿髋关节垂直向下",
        "formula": "arccos( dot(lm26-lm24, 垂直向下-lm24) / (|lm26-lm24| × |垂直向下-lm24|) )",
        "description": "以右髋为顶点，大腿向量（髋→膝）为移动臂，髋关节垂直向下参考线为固定臂，两向量夹角即髋关节前屈幅度",
    },

    # ── 髋关节外展 ──────────────────────────────────────────────────
    ("hip_abduction", "left"): {
        "axis": "左髋关节 (lm23)",
        "moving_arm": "向量 lm23 → lm25，即 lm25.xyz - lm23.xyz，左髋指向左膝",
        "fixed_arm": "向量 lm23 → 垂直向下参考点，即 lm23.xyz + [0, 1, 0]，沿髋关节垂直向下",
        "formula": "arccos( dot(lm25-lm23, 垂直向下-lm23) / (|lm25-lm23| × |垂直向下-lm23|) )",
        "description": "以左髋为顶点，大腿向量（髋→膝）为移动臂，髋关节垂直向下参考线为固定臂，两向量夹角即髋关节外展幅度",
    },
    ("hip_abduction", "right"): {
        "axis": "右髋关节 (lm24)",
        "moving_arm": "向量 lm24 → lm26，即 lm26.xyz - lm24.xyz，右指向右膝",
        "fixed_arm": "向量 lm24 → 垂直向下参考点，即 lm24.xyz + [0, 1, 0]，沿髋关节垂直向下",
        "formula": "arccos( dot(lm26-lm24, 垂直向下-lm24) / (|lm26-lm24| × |垂直向下-lm24|) )",
        "description": "以右为顶点，大腿向量（髋→膝）为移动臂，髋关节垂直向下参考线为固定臂，两向量夹角即髋关节外展幅度",
    },

    # ── 膝关节屈曲 ───────────────────────────────────────────────────
    ("knee_flexion", "left"): {
        "axis":        "左膝关节 (lm25)",
        "moving_arm":  "向量 lm25 → lm27，即 lm27.xyz - lm25.xyz，左膝指向左踝（小腿方向）",
        "fixed_arm":   "向量 lm25 → lm23，即 lm23.xyz - lm25.xyz，左膝指向左髋（大腿方向）",
        "formula":     "arccos( dot(lm27-lm25, lm23-lm25) / (|lm27-lm25| × |lm23-lm25|) )",
        "description": "以左膝为顶点，小腿向量（膝→踝）为移动臂，大腿向量（膝→髋）为固定臂；完全伸直时几何角≈180°，临床屈曲角 = 180° − 几何角",
    },
    ("knee_flexion", "right"): {
        "axis":        "右膝关节 (lm26)",
        "moving_arm":  "向量 lm26 → lm28，即 lm28.xyz - lm26.xyz，右膝指向右踝（小腿方向）",
        "fixed_arm":   "向量 lm26 → lm24，即 lm24.xyz - lm26.xyz，右膝指向右髋（大腿方向）",
        "formula":     "arccos( dot(lm28-lm26, lm24-lm26) / (|lm28-lm26| × |lm24-lm26|) )",
        "description": "以右膝为顶点，小腿向量（膝→踝）为移动臂，大腿向量（膝→髋）为固定臂；完全伸直时几何角≈180°，临床屈曲角 = 180° − 几何角",
    },

    # ── 踝关节背屈 ───────────────────────────────────────────────────
    ("ankle_dorsiflexion", "left"): {
        "axis":        "左踝关节 (lm27)",
        "moving_arm":  "向量 lm27 → 左足中点，即 (lm29+lm31)/2 - lm27.xyz，左踝指向足部中心",
        "fixed_arm":   "向量 lm27 → lm25，即 lm25.xyz - lm27.xyz，左踝指向左膝（小腿方向）",
        "formula":     "arccos( dot(足中点-lm27, lm25-lm27) / (|足中点-lm27| × |lm25-lm27|) )",
        "description": "以左踝为顶点，足部向量（踝→足中点）为移动臂，小腿向量（踝→膝）为固定臂；临床背屈角 = max(0, 90° − 几何角)",
    },
    ("ankle_dorsiflexion", "right"): {
        "axis":        "右踝关节 (lm28)",
        "moving_arm":  "向量 lm28 → 右足中点，即 (lm30+lm32)/2 - lm28.xyz，右踝指向足部中心",
        "fixed_arm":   "向量 lm28 → lm26，即 lm26.xyz - lm28.xyz，右踝指向右膝（小腿方向）",
        "formula":     "arccos( dot(足中点-lm28, lm26-lm28) / (|足中点-lm28| × |lm26-lm28|) )",
        "description": "以右踝为顶点，足部向量（踝→足中点）为移动臂，小腿向量（踝→膝）为固定臂；临床背屈角 = max(0, 90° − 几何角)",
    },

    # ── 踝关节跖屈（与背屈共用同一向量，仅临床转换公式不同）───────────
    ("ankle_plantarflexion", "left"): {
        "axis":        "左踝关节 (lm27)",
        "moving_arm":  "向量 lm27 → 左足中点，即 (lm29+lm31)/2 - lm27.xyz，左踝指向足部中心",
        "fixed_arm":   "向量 lm27 → lm25，即 lm25.xyz - lm27.xyz，左踝指向左膝（小腿方向）",
        "formula":     "arccos( dot(足中点-lm27, lm25-lm27) / (|足中点-lm27| × |lm25-lm27|) )",
        "description": "与背屈共用同一向量夹角；临床跖屈角 = max(0, 几何角 − 90°)，以 90° 中立位为分界",
    },
    ("ankle_plantarflexion", "right"): {
        "axis":        "右踝关节 (lm28)",
        "moving_arm":  "向量 lm28 → 右足中点，即 (lm30+lm32)/2 - lm28.xyz，右踝指向足部中心",
        "fixed_arm":   "向量 lm28 → lm26，即 lm26.xyz - lm28.xyz，右踝指向右膝（小腿方向）",
        "formula":     "arccos( dot(足中点-lm28, lm26-lm28) / (|足中点-lm28| × |lm26-lm28|) )",
        "description": "与背屈共用同一向量夹角；临床跖屈角 = max(0, 几何角 − 90°)，以 90° 中立位为分界",
    },

    # ── 颈部前屈 ─────────────────────────────────────────────────────
    ("neck_flexion", "center"): {
        "axis": "双肩中点 = (lm11.xyz + lm12.xyz) / 2",
        "moving_arm": "向量 肩中点 → lm0，即 lm0.xyz - 肩中点，肩中点指向鼻尖",
        "fixed_arm": "向量 肩中点 → 垂直向上参考点，即 肩中点 + [0, -1, 0]，沿肩中点垂直向上",
        "formula": "arccos( dot(lm0-肩中点, 垂直向上-肩中点) / (|lm0-肩中点| × |垂直向上-肩中点|) )",
        "description": "以双肩中点为顶点，颈部向量（肩中点→鼻）为移动臂，肩中点垂直向上参考线为固定臂，两向量夹角即颈部前屈幅度",
    },

}

# ══════════════════════════════════════════════════════════════════════
# ROM 测量项目定义列表
# ══════════════════════════════════════════════════════════════════════
#
# rule 字段格式（与 CALC_LOGIC 及代码三者对应）：
#   "轴=<axis> | moving_vec=<moving>-<axis>（语义） | fixed_vec=<fixed>-<axis>（语义） | 临床角=<公式>"
#
ROM_DEFINITIONS = [

    # ────────────────── 肩关节前屈 ──────────────────────────────────
    {
        "key": "shoulder_flexion_left",
        "name": "左肩前屈",
        "measure_type": "shoulder_flexion",
        "side": "left",
        "rule": "轴=lm11(左肩) | moving_vec=lm13-lm11(肩→肘) | fixed_vec=左肩垂直向下参考线 | 临床=几何角",
        "normal_range": "0°~180°",
    },
    {
        "key": "shoulder_flexion_right",
        "name": "右肩前屈",
        "measure_type": "shoulder_flexion",
        "side": "right",
        "rule": "轴=lm12(右肩) | moving_vec=lm14-lm12(肩→肘) | fixed_vec=右肩垂直向下参考线 | 临床=几何角",
        "normal_range": "0°~180°",
    },

    # ────────────────── 肩关节外展 ──────────────────────────────────
    {
        "key": "shoulder_abduction_left",
        "name": "左肩外展",
        "measure_type": "shoulder_abduction",
        "side": "left",
        "rule": "轴=lm11(左肩) | moving_vec=lm13-lm11(肩→肘) | fixed_vec=左肩垂直向下参考线 | 临床=几何角",
        "normal_range": "0°~180°",
    },
    {
        "key": "shoulder_abduction_right",
        "name": "右肩外展",
        "measure_type": "shoulder_abduction",
        "side": "right",
        "rule": "轴=lm12(右肩) | moving_vec=lm14-lm12(肩→肘) | fixed_vec=右肩垂直向下参考线 | 临床=几何角",
        "normal_range": "0°~180°",
    },
    {
        "key": "shoulder_extension_left",
        "name": "左肩后伸",
        "measure_type": "shoulder_extension",
        "side": "left",
        "rule": "轴=lm11(左肩) | moving_vec=lm13-lm11(肩→肘) | fixed_vec=左肩垂直向下参考线 | 临床=几何角",
        "normal_range": "0°~50°",
    },
    {
        "key": "shoulder_extension_right",
        "name": "右肩后伸",
        "measure_type": "shoulder_extension",
        "side": "right",
        "rule": "轴=lm12(右肩) | moving_vec=lm14-lm12(肩→肘) | fixed_vec=右肩垂直向下参考线 | 临床=几何角",
        "normal_range": "0°~50°",
    },
    # ────────────────── 肘关节屈曲 ──────────────────────────────────
    {
        "key":          "elbow_flexion_left",
        "name":         "左肘屈曲",
        "measure_type": "elbow_flexion",
        "side":         "left",
        "rule":         "轴=lm13(左肘) | moving_vec=lm15-lm13(肘→腕，前臂) | fixed_vec=lm11-lm13(肘→肩，上臂) | 临床=180°−几何角",
        "normal_range": "0°~145°",
    },
    {
        "key":          "elbow_flexion_right",
        "name":         "右肘屈曲",
        "measure_type": "elbow_flexion",
        "side":         "right",
        "rule":         "轴=lm14(右肘) | moving_vec=lm16-lm14(肘→腕，前臂) | fixed_vec=lm12-lm14(肘→肩，上臂) | 临床=180°−几何角",
        "normal_range": "0°~145°",
    },

    # ────────────────── 髋关节前屈 ──────────────────────────────────
    {
        "key": "hip_flexion_left",
        "name": "左髋前屈",
        "measure_type": "hip_flexion",
        "side": "left",
        "rule": "轴=lm23(左髋) | moving_vec=lm25-lm23(髋→膝) | fixed_vec=髋关节垂直向下参考线 | 临床=几何角",
        "normal_range": "0°~90°",
    },
    {
        "key": "hip_flexion_right",
        "name": "右髋前屈",
        "measure_type": "hip_flexion",
        "side": "right",
        "rule": "轴=lm24(右髋) | moving_vec=lm26-lm24(髋→膝) | fixed_vec=髋关节垂直向下参考线 | 临床=几何角",
        "normal_range": "0°~90°",
    },

    # ────────────────── 髋关节外展 ──────────────────────────────────
    {
        "key": "hip_abduction_left",
        "name": "左髋外展",
        "measure_type": "hip_abduction",
        "side": "left",
        "rule": "轴=lm23(左髋) | moving_vec=lm25-lm23(髋→膝) | fixed_vec=髋关节垂直向下参考线 | 临床=几何角",
        "normal_range": "0°~45°",
    },
    {
        "key": "hip_abduction_right",
        "name": "右髋外展",
        "measure_type": "hip_abduction",
        "side": "right",
        "rule": "轴=lm24(右髋) | moving_vec=lm26-lm24(髋→膝) | fixed_vec=髋关节垂直向下参考线 | 临床=几何角",
        "normal_range": "0°~45°",
    },

    # ────────────────── 膝关节屈曲 ──────────────────────────────────
    {
        "key":          "knee_flexion_left",
        "name":         "左膝屈曲",
        "measure_type": "knee_flexion",
        "side":         "left",
        "rule":         "轴=lm25(左膝) | moving_vec=lm27-lm25(膝→踝) | fixed_vec=lm23-lm25(膝→髋) | 临床=180°−几何角",
        "normal_range": "0°~130°",
    },
    {
        "key":          "knee_flexion_right",
        "name":         "右膝屈曲",
        "measure_type": "knee_flexion",
        "side":         "right",
        "rule":         "轴=lm26(右膝) | moving_vec=lm28-lm26(膝→踝) | fixed_vec=lm24-lm26(膝→髋) | 临床=180°−几何角",
        "normal_range": "0°~130°",
    },

    # ────────────────── 踝关节背屈 ──────────────────────────────────
    {
        "key":          "ankle_dorsiflexion_left",
        "name":         "左踝背屈",
        "measure_type": "ankle_dorsiflexion",
        "side":         "left",
        "rule":         "轴=lm27(左踝) | moving_vec=足中点-lm27(踝→足中点) | fixed_vec=lm25-lm27(踝→膝) | 临床=max(0, 90°−几何角)",
        "normal_range": "0°~20°",
    },
    {
        "key":          "ankle_dorsiflexion_right",
        "name":         "右踝背屈",
        "measure_type": "ankle_dorsiflexion",
        "side":         "right",
        "rule":         "轴=lm28(右踝) | moving_vec=足中点-lm28(踝→足中点) | fixed_vec=lm26-lm28(踝→膝) | 临床=max(0, 90°−几何角)",
        "normal_range": "0°~20°",
    },

    # ────────────────── 踝关节跖屈 ──────────────────────────────────
    {
        "key":          "ankle_plantarflexion_left",
        "name":         "左踝跖屈",
        "measure_type": "ankle_plantarflexion",
        "side":         "left",
        "rule":         "轴=lm27(左踝) | moving_vec=足中点-lm27(踝→足中点) | fixed_vec=lm25-lm27(踝→膝) | 临床=max(0, 几何角−90°)",
        "normal_range": "0°~45°",
    },
    {
        "key":          "ankle_plantarflexion_right",
        "name":         "右踝跖屈",
        "measure_type": "ankle_plantarflexion",
        "side":         "right",
        "rule":         "轴=lm28(右踝) | moving_vec=足中点-lm28(踝→足中点) | fixed_vec=lm26-lm28(踝→膝) | 临床=max(0, 几何角−90°)",
        "normal_range": "0°~45°",
    },

    # ────────────────── 躯干前屈 ────────────────────────────────────
    {
        "key":          "trunk_flexion",
        "name":         "躯干前屈",
        "measure_type": "trunk_flexion",
        "side":         None,
        "rule":         "脊柱轴向量=肩中点-髋中点 与 垂直向上 [0,-1,0] 夹角 | 临床=几何角",
        "normal_range": "0°~45°",
    },




    {
        "key":          "neck_flexion",
        "name":         "颈部前屈",
        "measure_type": "neck_flexion",
        "side":         "center",
        "rule":         "轴=肩中点 | moving_vec=lm0-肩中点(肩中点→鼻) | fixed_vec=肩中点垂直向上参考线 | 临床=几何角",
        "normal_range": "0°~45°",
    },


]

# ══════════════════════════════════════════════════════════════════════
# 代偿规则定义
# ══════════════════════════════════════════════════════════════════════
# 键：主动作 measure_type
# 值：代偿检测项列表

COMPENSATION_RULES = {
    "shoulder_flexion": [
        {
            "code": "trunk_backward_lean",
            "name": "躯干后仰代偿",
            "threshold_mild":   10.0,   # 度
            "threshold_severe": 20.0,
            "signal": "trunk_tilt_signed",   # 负值=后仰
            "direction": "negative",
            "suggestion": "躯干后仰>10°，可能通过脊柱伸展代偿肩屈受限",
        },
        {
            "code": "shoulder_elevation",
            "name": "耸肩代偿",
            "threshold_mild": 15.0,
            "threshold_severe": 25.0,
            "signal": "shoulder_elevation_ratio",
            "direction": "positive",
            "suggestion": "肩胛上提，提示盂肱活动受限",
        },
    ],
    "shoulder_abduction": [
        {
            "code": "trunk_lateral_opposite",
            "name": "对侧躯干侧屈代偿",
            "threshold_mild": 8.0,
            "threshold_severe": 15.0,
            "signal": "trunk_lateral_tilt",
            "direction": "positive",
            "suggestion": "对侧躯干侧屈>8°，可能代偿肩外展不足",
        },
    ],
    "hip_abduction": [
        {
            "code": "pelvis_tilt",
            "name": "骨盆倾斜代偿",
            "threshold_mild": 6.0,
            "threshold_severe": 12.0,
            "signal": "pelvis_lateral_tilt",
            "direction": "positive",
            "suggestion": "骨盆倾斜>6°，髋外展受限",
        },
    ],
    "hip_flexion": [
        {
            "code": "trunk_flexion_comp",
            "name": "躯干前屈代偿",
            "threshold_mild": 10.0,
            "threshold_severe": 20.0,
            "signal": "trunk_tilt_signed",
            "direction": "positive",
            "suggestion": "躯干前倾代偿髋屈受限",
        },
    ],
    "knee_flexion": [
        {
            "code": "hip_flex_comp",
            "name": "过度髋屈曲代偿",
            "threshold_mild": 30.0,
            "threshold_severe": 60.0,
            "signal": "hip_flexion_angle",
            "direction": "positive",
            "suggestion": "膝屈时髋过度屈曲，可能腘绳肌或膝关节囊受限",
        },
    ],
}