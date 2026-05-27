"""
llm_api.py - 大模型API调用模块
用于生成康复评估报告的专业总结
"""
import requests
import os


def call_llm_for_summary(report_data: dict, api_key: str = None) -> str:
    """
    调用火山引擎大模型对ROM分析报告进行总结

    Args:
        report_data: 视频分析生成的JSON报告数据
        api_key: 火山引擎API密钥，如果为None则从环境变量ARK_API_KEY获取

    Returns:
        大模型生成的总结文本
    """
    # 获取API密钥
    if api_key is None:
        api_key = os.getenv("ARK_API_KEY")

    if not api_key:
        raise ValueError("未设置API密钥，请通过参数传入或设置环境变量 ARK_API_KEY")

    # 提取关键信息构建提示词
    primary_action = report_data.get("primary_action", {})
    rom_summary = report_data.get("rom_summary", {})
    compensation_summary = report_data.get("compensation_summary", {})
    video_info = report_data.get("video_info", {})

    # 构建动作信息
    action_name = primary_action.get('action_name', '未知')
    direction_type = primary_action.get('direction_type', '')

    # 方向类型说明
    direction_explain = {
        "bidirectional": "双向运动（包含前屈和后伸）",
        "flexion_only": "仅前屈方向",
        "extension_only": "仅后伸方向",
        "unidirectional": "单向运动"
    }.get(direction_type, "未知类型")

    # 构建 ROM 详情
    rom_details = []
    for key, data in rom_summary.items():
        min_angle = data.get('min_angle', 0)
        max_angle = data.get('max_angle', 0)
        rom_range = data.get('range', 'N/A')
        peak_rom = data.get('peak_rom', 0)
        num_rep = data.get('num_repetitions', 0)

        rom_details.append(
            f"- {data['name']}: "
            f"范围 {rom_range} (最小{min_angle}°~最大{max_angle}°), "
            f"峰值 {peak_rom}°, 重复 {num_rep} 次"
        )

    # 构建代偿问题
    comp_issues = []
    for key, data in compensation_summary.items():
        for comp in data.get("compensations", []):
            comp_issues.append(
                f"- {data['name']}存在代偿: {comp['description']} "
                f"(严重程度: {comp['severity']})"
            )

    # 构建提示词
    prompt = f"""你是一位专业的康复治疗师，请根据以下关节活动度(ROM)视频分析报告给出专业总结和建议：

【评估动作】
{action_name} ({direction_explain})
置信度: {primary_action.get('confidence', 0)}

【重要说明】
- 这是视频分析，记录了完整的动作过程
- 重复次数表示完成的动作周期数
- visible=false 表示该关节在当前视角下不可测量，不是活动度为0

【关节活动度数据】
{chr(10).join(rom_details) if rom_details else '无数据'}

【代偿问题】
{chr(10).join(comp_issues) if comp_issues else '未检测到明显代偿'}

【视频信息】
处理速度: {video_info.get('processing_fps', 0)} FPS
分析帧数: {video_info.get('analyzed_frames', 0)} / {video_info.get('total_frames', 0)}

请以专业但易懂的语言提供：
1. 整体评估结论（关注活动范围是否充分，注意负值表示反向运动）
2. 主要问题分析（注意区分"未测量"和"测量值为 0"）
3. 康复训练建议
4. 注意事项

【输出格式要求】
- 使用纯文本，不要使用 Markdown 格式
- 不要使用 ** 或 __ 等加粗符号
- 使用数字编号 1. 2. 3. 4. 列出要点
- 简洁明了，重点突出，控制在 300 字以内
- 正确理解双向运动的含义（负值不等于异常起始位）
- 不要将"未检测到数据"误判为"活动度为 0"
"""

    # 调用火山引擎API
    url = "https://ark.cn-beijing.volces.com/api/v3/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    payload = {
        "model": "deepseek-v4-pro-260425",
        "messages": [
            {"role": "system", "content": "你是专业的康复治疗师，擅长分析关节活动度报告并给出专业建议。"},
            {"role": "user", "content": prompt}
        ]
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()

        result = response.json()
        summary = result["choices"][0]["message"]["content"]
        return summary

    except requests.exceptions.RequestException as e:
        raise Exception(f"调用大模型API失败: {str(e)}")
    except (KeyError, IndexError) as e:
        raise Exception(f"解析API响应失败: {str(e)}")
