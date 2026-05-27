"""
动态动作分析（视频）.py —— 基于视频的逐帧关节活动度分析（性能优化版）
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import json
import os
import time
import numpy as np
from PIL import ImageDraw, Image
from pose_analyzer_base import PoseAnalyzerBase
from signal_filters import LandmarkSmoother
from motion_analysis import extract_peak_frames, recognize_primary_action
from rom_rules import COMPENSATION_RULES




_VIEW_ZH = {'side': '侧视图', 'front': '正视图', 'oblique': '斜视图'}
_NEAR_ZH = {'left': '左近',   'right': '右近'}

_COLOR_OK     = (0,  255, 128)
_COLOR_EXCEED = (0,   60, 255)
_COLOR_HEADER = (0,  220, 255)


def _resolve_measure_type(rom_key: str) -> str | None:
    if "shoulder_sagittal" in rom_key:
        return "shoulder_flexion"
    for mt in COMPENSATION_RULES.keys():
        if rom_key.startswith(mt):
            return mt
    return None


class VideoPoseAnalyzer(PoseAnalyzerBase):
    """视频姿态分析器（支持性能档位）"""

    WINDOW_NAME = "Video ROM Analysis"

    # 性能档位
    # process_max_side: 整帧长边上限（像素）；None 表示不缩放。
    # 4K/2K 视频强烈建议开启整帧降采样，绘制/编码/I-O 都会成倍加速。
    PRESETS = {
        "fast": {
            "model_suffix":     "full",   # lite > full > heavy
            "analysis_stride":  2,        # 每 N 帧做一次检测
            "detect_scale":     1.0,      # 检测前再次降采样（现整帧已缩，设 1.0）
            "panel_refresh":    3,        # 面板每 N 帧重绘一次
            "display_stride":   2,        # display 窗口每 N 帧刷新
            "process_max_side": 1280,     # ★ 整帧长边上限
        },
        "balanced": {
            "model_suffix":     "full",
            "analysis_stride":  1,
            "detect_scale":     1.0,
            "panel_refresh":    2,
            "display_stride":   1,
            "process_max_side": 1920,
        },
        "accurate": {
            "model_suffix":     "heavy",
            "analysis_stride":  1,
            "detect_scale":     1.0,
            "panel_refresh":    1,
            "display_stride":   1,
            "process_max_side": None,
        },
    }

    def __init__(self, model_dir='models', output_root='output', preset='fast'):
        super().__init__()
        self.preset_name = preset
        self.preset = self.PRESETS[preset].copy()

        model_path = os.path.join(
            model_dir, f"pose_landmarker_{self.preset['model_suffix']}.task"
        )
        if not os.path.exists(model_path):
            # 兜底：找任意一个可用模型
            for s in ("full", "lite", "heavy"):
                alt = os.path.join(model_dir, f"pose_landmarker_{s}.task")
                if os.path.exists(alt):
                    print(f"[warn] {model_path} 不存在，改用 {alt}")
                    model_path = alt
                    self.preset["model_suffix"] = s
                    break

        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.detector  = vision.PoseLandmarker.create_from_options(options)
        self._pil_font = self._load_chinese_font(font_size=36)
        self.output_root = output_root
        self.smoother = None

        # 缓存：面板图像（透明）
        self._cached_panel = None
        self._cached_panel_key = None

    # ──────────────────────────────────────────
    # 高效中文文本绘制（只在需要时重建 PIL 图）
    # ──────────────────────────────────────────
    def _render_panel(self, lines, colors_bgr, width, height, line_height=44):
        """渲染一个带中文的半透明面板 RGBA 图，供 cv2 叠加使用。"""
        panel = np.zeros((height, width, 4), dtype=np.uint8)
        # 半透明黑底
        panel[:, :, 0:3] = 20
        panel[:, :, 3]   = 150

        img_pil = Image.fromarray(panel, mode='RGBA')
        draw    = ImageDraw.Draw(img_pil)
        y = 8
        for i, line in enumerate(lines):
            c   = colors_bgr[i] if i < len(colors_bgr) else (0, 255, 128)
            rgb = (c[2], c[1], c[0], 255)
            draw.text((8, y), line, font=self._pil_font, fill=rgb)
            y += line_height
        return np.array(img_pil)

    @staticmethod
    def _overlay_rgba(frame_bgr, rgba_panel, x=0, y=0):
        """把 RGBA 面板叠到 BGR 帧上（原地修改）。"""
        ph, pw = rgba_panel.shape[:2]
        fh, fw = frame_bgr.shape[:2]
        x2, y2 = min(x + pw, fw), min(y + ph, fh)
        if x2 <= x or y2 <= y:
            return
        panel = rgba_panel[: y2 - y, : x2 - x]
        alpha = panel[:, :, 3:4].astype(np.float32) / 255.0
        roi   = frame_bgr[y:y2, x:x2].astype(np.float32)
        roi   = roi * (1 - alpha) + panel[:, :, :3].astype(np.float32) * alpha
        frame_bgr[y:y2, x:x2] = roi.astype(np.uint8)

    def _show_preview(self, frame, frame_count, total_frames,
                      tag_text="") -> bool:
        fh, fw = frame.shape[:2]
        max_dw, max_dh = 1024, 576
        if fw > max_dw or fh > max_dh:
            scale = min(max_dw / fw, max_dh / fh)
            disp = cv2.resize(frame, (int(fw * scale), int(fh * scale)))
        else:
            disp = frame

        cv2.putText(disp, f"Frame {frame_count}/{total_frames}",
                    (10, disp.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        if tag_text:
            cv2.putText(disp, tag_text, (10, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 255), 2)

        cv2.imshow(self.WINDOW_NAME, disp)
        return (cv2.waitKey(1) & 0xFF) != ord('q')

    # ──────────────────────────────────────────
    # 对帧做降采样检测，但 landmark 归一化坐标与尺寸无关，所以直接用
    # ──────────────────────────────────────────
    def _detect_pose(self, frame_bgr, timestamp_ms):
        scale = self.preset["detect_scale"]
        if scale < 0.999:
            h, w = frame_bgr.shape[:2]
            small = cv2.resize(frame_bgr, (int(w * scale), int(h * scale)),
                               interpolation=cv2.INTER_LINEAR)
            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        else:
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        return self.detector.detect_for_video(mp_img, timestamp_ms)

    def analyze_video(self, video_path, save_json=True, save_video=True,
                      display=False, max_frames=None, output_subdir=None):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"无法打开视频: {video_path}")

        fps          = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        src_w        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        src_h        = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # ── 计算整帧缩放 ──────────────────────────
        max_side = self.preset.get("process_max_side")
        if max_side is not None and max(src_w, src_h) > max_side:
            scale_ratio = max_side / max(src_w, src_h)
            fw = int(round(src_w * scale_ratio))
            fh = int(round(src_h * scale_ratio))
            do_resize = True
        else:
            fw, fh = src_w, src_h
            do_resize = False

        print(f"视频: {src_w}×{src_h}, {fps:.2f} FPS, {total_frames} 帧")
        if do_resize:
            print(f"整帧缩放 → {fw}×{fh} (长边≤{max_side})")
        print(f"性能档位: {self.preset_name} → {self.preset}")

        video_name = os.path.splitext(os.path.basename(video_path))[0]
        if output_subdir is None:
            output_subdir = video_name
        output_dir = os.path.join(self.output_root, output_subdir)
        os.makedirs(output_dir, exist_ok=True)

        self.smoother = LandmarkSmoother(fps=fps, mincutoff=1.0, beta=0.01)

        out = None
        if save_video:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(
                os.path.join(output_dir, f"{video_name}_rom_result.mp4"),
                fourcc, fps, (fw, fh))

        if display:
            cv2.namedWindow(self.WINDOW_NAME, cv2.WINDOW_NORMAL)

        all_frames_data = []
        rom_statistics  = {}
        frame_count     = 0
        user_abort      = False

        analysis_stride = self.preset["analysis_stride"]
        panel_refresh   = self.preset["panel_refresh"]
        display_stride  = self.preset["display_stride"]

        # 上一次的检测结果，用于跳过帧时沿用
        last_landmarks = None
        last_view = None
        last_near = None
        last_rom  = []
        last_panel_rgba = None

        t_start = time.time()

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if max_frames is not None and frame_count >= max_frames:
                    break

                # ★ 整帧降采样：之后所有处理/绘制/输出都在低分辨率做
                if do_resize:
                    frame = cv2.resize(frame, (fw, fh),
                                       interpolation=cv2.INTER_AREA)

                need_detect = (frame_count % analysis_stride == 0)
                ts_ms = int(frame_count * 1000 / fps)

                if need_detect:
                    res = self._detect_pose(frame, ts_ms)
                    if res.pose_landmarks:
                        lms = self.smoother.smooth(res.pose_landmarks[0])
                        last_landmarks = lms
                        last_view = self.detect_view_angle(lms)
                        last_near = self.get_near_side(lms)
                        last_rom  = self.calculate_rom_report(lms, fw, fh, verbose=False)

                        # 统计
                        for rom in last_rom:
                            key = rom["key"]
                            if key not in rom_statistics:
                                rom_statistics[key] = {
                                    "name": rom["name"],
                                    "measure_type": rom["measure_type"],
                                    "angles": [],
                                    "normal_range": rom["normal_range"],
                                }
                            if rom["angle"] is not None:
                                rom_statistics[key]["angles"].append(rom["angle"])

                        # 代偿信号
                        comp_signals = self.compute_compensation_signals(lms)
                    else:
                        last_landmarks = None
                        comp_signals = {}
                else:
                    comp_signals = {}

                frame_data = {
                    "frame_index":  frame_count,
                    "timestamp_ms": ts_ms,
                    "view":         last_view,
                    "near_side":    last_near,
                    "joint_rom_report": last_rom if need_detect else [],
                    "compensation_signals": comp_signals,
                }
                all_frames_data.append(frame_data)

                # ── 绘制（用最近一次的 landmarks，即使是跳过帧也能看起来连贯）──
                if last_landmarks is not None:
                    draw_near = last_near if last_view == 'side' else None
                    self._draw_skeleton(frame, last_landmarks, fw, fh, near_side=draw_near)
                    self._draw_keypoints(frame, last_landmarks, fw, fh,
                                         show_index=False, near_side=draw_near)

                    # 面板：只在 panel_refresh 周期重建
                    if frame_count % panel_refresh == 0 or last_panel_rgba is None:
                        reliable = [r for r in last_rom
                                    if r["reliable"] and r["angle"] is not None]
                        rows = reliable[:8]
                        header = (f"[{_VIEW_ZH.get(last_view, last_view)}]"
                                  + (f" {_NEAR_ZH.get(last_near, '')}"
                                     if last_view == 'side' else ""))
                        lines, colors = [header], [_COLOR_HEADER]
                        for r in rows:
                            if r["out_of_range"]:
                                lines.append(f"⚠ {r['name']}: {r['angle']:.1f}°")
                                colors.append(_COLOR_EXCEED)
                            else:
                                lines.append(f"{r['name']}: {r['angle']:.1f}°")
                                colors.append(_COLOR_OK)
                        panel_h = len(lines) * 44 + 16
                        last_panel_rgba = self._render_panel(
                            lines, colors, width=360, height=panel_h)

                    if last_panel_rgba is not None:
                        self._overlay_rgba(frame, last_panel_rgba, x=0, y=0)

                if out:
                    out.write(frame)

                if display and (frame_count % display_stride == 0):
                    if not self._show_preview(frame, frame_count, total_frames,
                                              tag_text=f"[{self.preset_name}]"):
                        print("用户中断")
                        user_abort = True
                        break

                frame_count += 1
                if frame_count % 60 == 0:
                    elapsed = time.time() - t_start
                    cur_fps = frame_count / elapsed if elapsed > 0 else 0
                    eta     = (total_frames - frame_count) / cur_fps if cur_fps > 0 else -1
                    print(f"  {frame_count}/{total_frames}  "
                          f"处理 {cur_fps:.1f} FPS  ETA {eta:.1f}s")

        finally:
            cap.release()
            if out:
                out.release()
            cv2.destroyAllWindows()

        elapsed = time.time() - t_start
        avg_fps = frame_count / elapsed if elapsed > 0 else 0
        print(f"\n✓ 共处理 {frame_count} 帧，用时 {elapsed:.1f}s，平均 {avg_fps:.1f} FPS")

        # ═════════════════════════════════════════════════
        # 代偿分析 & 汇总（保持原逻辑）
        # ═════════════════════════════════════════════════
        compensation_summary = {}
        for key, stats in rom_statistics.items():
            angles = stats["angles"]
            if not angles:
                continue
            peaks = extract_peak_frames(angles, fps, min_range_deg=15.0)
            if not peaks:
                continue
            mt = _resolve_measure_type(key)
            if mt is None:
                continue
            comp_list_agg = {}
            for peak_idx, _ in peaks:
                if peak_idx >= len(all_frames_data):
                    continue
                signals = all_frames_data[peak_idx].get("compensation_signals", {})
                if not signals:
                    continue
                for c in self.analyze_compensation(mt, signals):
                    k = c["code"]
                    if k not in comp_list_agg or c["value"] > comp_list_agg[k]["value"]:
                        comp_list_agg[k] = c
            if comp_list_agg:
                compensation_summary[key] = {
                    "name": stats["name"],
                    "measure_type": mt,
                    "compensations": list(comp_list_agg.values()),
                }

        primary_action = recognize_primary_action(rom_statistics)
        print(f"识别到主动作: "
              f"{primary_action.get('action_name', primary_action['action'])} "
              f"(置信度 {primary_action['confidence']})")

        rom_summary = {}
        for key, stats in rom_statistics.items():
            angles = stats["angles"]
            if not angles:
                continue
            peaks = extract_peak_frames(angles, fps, min_range_deg=15.0)
            min_val = round(float(np.min(angles)), 2)
            max_val = round(float(np.max(angles)), 2)
            if peaks:
                peak_rom = round(max(peaks, key=lambda p: abs(p[1]))[1], 2)
            else:
                peak_rom = round(max_val if abs(max_val) >= abs(min_val) else min_val, 2)
            rom_summary[key] = {
                "name": stats["name"],
                "peak_rom": peak_rom,
                "peak_frames": [p[0] for p in peaks],
                "max_angle": max_val,
                "min_angle": min_val,
                "range": f"{min_val}°~{max_val}°",
                "normal_range": stats["normal_range"],
                "num_repetitions": len(peaks),
            }

        result_data = {
            "video": video_path,
            "video_info": {
                "src_width":  src_w, "src_height": src_h,
                "width": fw, "height": fh, "fps": fps,
                "total_frames":    total_frames,
                "analyzed_frames": frame_count,
                "processing_fps":  round(avg_fps, 2),
                "preset":          self.preset_name,
                "resized":         do_resize,
                "user_abort":      user_abort,
            },
            "rom_summary": rom_summary,
            "compensation_summary": compensation_summary,
            "primary_action": primary_action,
        }

        if save_json:
            json_path = os.path.join(output_dir, f"{video_name}_rom_analysis.json")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(result_data, f, indent=2, ensure_ascii=False)
            print(f"已导出: {json_path}")

        return result_data


if __name__ == "__main__":
    # ─────────────────────────────────────────────────────
    # 档位选择:
    #   "fast"     —— full 模型 + 隔帧检测 + 整帧缩到 1280，最快（推荐日常用）
    #   "balanced" —— full 模型 + 逐帧 + 整帧缩到 1920，质量与速度折中
    #   "accurate" —— heavy 模型 + 逐帧 + 原分辨率，最准（最慢）
    # ─────────────────────────────────────────────────────

    analyzer = VideoPoseAnalyzer(
        model_dir='models',
        output_root='output',
        preset='fast',
    )
    result = analyzer.analyze_video(
        video_path="素材库/视频/肩关节/前屈.mp4",
        save_json=True,
        save_video=True,
        display=True,
        max_frames=None,
    )

    print(f"\n处理速度: {result['video_info']['processing_fps']} FPS "
          f"(档位: {result['video_info']['preset']})")