"""
ROM Assessment API - 小程序后端服务
基于 MediaPipe 进行姿态检测和关节活动度分析
"""

import os
import uuid
import json
import tempfile
import shutil
from datetime import datetime
from typing import Optional, List
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse

# 导入视频分析器
try:
    from video_analyzer import VideoPoseAnalyzer
    VIDEO_ANALYZER_AVAILABLE = True
except ImportError as e:
    print(f"[warn] 视频分析器导入失败: {e}")
    VIDEO_ANALYZER_AVAILABLE = False

# ============== 应用配置 ==============
app = FastAPI(
    title="ROM Assessment API",
    description="关节活动度评估后端服务",
    version="1.0.0"
)

# CORS 配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 数据存储路径
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
REPORTS_DIR = DATA_DIR / "reports"
OUTPUT_DIR = BASE_DIR / "output"
MODELS_DIR = BASE_DIR / "models"

# 确保目录存在
DATA_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# ============== 全局分析器实例 ==============
video_analyzer_cache = {}

def get_video_analyzer(preset: str = "fast"):
    """获取或创建视频分析器实例（按档位缓存）"""
    if not VIDEO_ANALYZER_AVAILABLE:
        raise HTTPException(status_code=503, detail="视频分析器不可用")

    if preset not in video_analyzer_cache:
        video_analyzer_cache[preset] = VideoPoseAnalyzer(
            model_dir=str(MODELS_DIR),
            output_root=str(OUTPUT_DIR),
            preset=preset
        )
    return video_analyzer_cache[preset]


# ============== API 路由 ==============
@app.get("/")
async def root():
    """健康检查"""
    return {
        "message": "您的康复评估助手即将上线，敬请期待！"
    }


@app.post("/api/analyze/video")
async def analyze_video_upload(
    file: UploadFile = File(...),
    assessment_type: str = Form("前屈"),
    preset: str = Form("fast"),
    user_id: Optional[str] = Form(None)
):
    """
    上传视频文件进行关节活动度分析
    """
    temp_video_path = None
    try:
        # 验证文件类型
        allowed_extensions = ('.mp4', '.mov', '.avi', '.mkv')
        if not file.filename.lower().endswith(allowed_extensions):
            raise HTTPException(status_code=400, detail="不支持的视频格式")

        # 创建临时文件
        temp_dir = tempfile.mkdtemp()
        file_extension = os.path.splitext(file.filename)[1] or '.mp4'
        temp_video_path = os.path.join(temp_dir, f"upload_{uuid.uuid4().hex}{file_extension}")

        # 保存上传的视频
        with open(temp_video_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)

        print(f"[info] 收到视频分析请求: {file.filename}, 类型: {assessment_type}, 档位: {preset}")

        # 获取视频分析器
        analyzer = get_video_analyzer(preset)

        # 执行分析
        result = analyzer.analyze_video(
            video_path=temp_video_path,
            save_json=True,
            save_video=True,
            display=False,
            max_frames=None,
            output_subdir=f"user_{user_id or 'anonymous'}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )

        # 保存报告到用户目录
        if user_id:
            user_report_dir = REPORTS_DIR / user_id
            user_report_dir.mkdir(exist_ok=True)

            report_file = user_report_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.json"
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)

        # 查找生成的结果视频
        result_video = None
        video_files = list(OUTPUT_DIR.rglob("*_rom_result.mp4"))
        if video_files:
            latest_video = max(video_files, key=lambda x: x.stat().st_mtime)
            result_video = f"/api/download/{latest_video.name}"

        return JSONResponse(content={
            "status": "success",
            "message": "视频分析完成",
            "result": result,
            "result_video": result_video
        })

    except HTTPException:
        raise
    except Exception as e:
        print(f"[error] 视频分析失败: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"视频分析失败: {str(e)}")

    finally:
        # 清理临时文件
        if temp_video_path and os.path.exists(temp_video_path):
            try:
                os.remove(temp_video_path)
                temp_dir = os.path.dirname(temp_video_path)
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir, ignore_errors=True)
            except:
                pass


@app.get("/api/download/{filename}")
async def download_result_video(filename: str):
    """下载分析结果视频"""
    video_path = OUTPUT_DIR / filename

    if not video_path.exists():
        raise HTTPException(status_code=404, detail="视频文件不存在")

    return FileResponse(
        path=str(video_path),
        media_type='video/mp4',
        filename=filename
    )


@app.get("/api/reports/{user_id}")
async def get_user_reports(user_id: str):
    """获取用户的所有报告"""
    user_dir = REPORTS_DIR / user_id
    reports = []

    if user_dir.exists():
        for f in sorted(user_dir.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
            with open(f, 'r', encoding='utf-8') as fp:
                reports.append(json.load(fp))

    return {"reports": reports, "count": len(reports)}


# ============== 启动信息 ==============
if __name__ == "__main__":
    import uvicorn
    print("=" * 60)
    print("🚀 ROM Assessment API 启动")
    print(f"📂 工作目录: {BASE_DIR}")
    print(f"🎥 视频分析器: {'✅ 可用' if VIDEO_ANALYZER_AVAILABLE else '❌ 不可用'}")
    print("📖 API 文档: http://localhost:80/docs")
    print("=" * 60)
    uvicorn.run(app, host="0.0.0.0", port=80)
