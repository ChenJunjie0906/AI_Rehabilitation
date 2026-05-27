
"""
pdf_report_generator.py - 生成PDF格式的康复评估报告
集成AI总结功能
"""
import os
import json
from datetime import datetime
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# 导入大模型API
from llm_api import call_llm_for_summary


class PDFReportGenerator:
    """PDF报告生成器（集成AI总结）"""

    def __init__(self, output_dir="output"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # 注册中文字体
        self._register_fonts()

        # 页面设置
        self.page_width = A4[0]
        self.page_height = A4[1]
        self.margin = 2 * cm

    def _register_fonts(self):
        """注册中文字体"""
        try:
            # Windows系统字体路径
            font_paths = {
                'SimHei': r'C:\Windows\Fonts\simhei.ttf',
                'MicrosoftYaHei': r'C:\Windows\Fonts\msyh.ttc',
            }

            for font_name, font_path in font_paths.items():
                if os.path.exists(font_path):
                    pdfmetrics.registerFont(TTFont(font_name, font_path))
                    print(f"[info] 已加载字体: {font_name}")
                    return

            # 如果找不到中文字体，使用默认字体
            print("[warn] 未找到中文字体，将使用默认字体")

        except Exception as e:
            print(f"[warn] 字体加载失败: {e}，使用默认字体")

    def generate_pdf(self, report_data: dict, api_key: str = None,
                     output_filename: str = None, enable_ai_summary: bool = True) -> str:
        """
        生成PDF报告（可选AI总结）

        Args:
            report_data: 视频分析生成的JSON报告数据
            api_key: 火山引擎API密钥（如果需要AI总结）
            output_filename: 输出文件名，如果为None则自动生成
            enable_ai_summary: 是否启用AI总结

        Returns:
            PDF文件路径
        """
        # 如果需要AI总结且报告中还没有
        if enable_ai_summary and "ai_summary" not in report_data:
            if api_key is None:
                api_key = os.getenv("ARK_API_KEY")

            if api_key:
                try:
                    print("正在调用AI生成报告总结...")
                    summary_text = call_llm_for_summary(report_data, api_key)
                    report_data["ai_summary"] = {
                        "summary": summary_text,
                        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "model": "deepseek-v4-pro-260425"
                    }
                    print("✓ AI总结生成完成")
                except Exception as e:
                    print(f"[warn] AI总结生成失败: {e}，将继续生成不含AI总结的报告")
            else:
                print("[warn] 未提供API密钥，跳过AI总结")

        # 生成文件名
        if output_filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_filename = f"ROM_Report_{timestamp}.pdf"

        pdf_path = os.path.join(self.output_dir, output_filename)

        # 创建PDF文档
        doc = SimpleDocTemplate(
            pdf_path,
            pagesize=A4,
            leftMargin=self.margin,
            rightMargin=self.margin,
            topMargin=self.margin,
            bottomMargin=self.margin
        )

        # 构建内容
        story = []

        # 1. 标题页
        story.extend(self._create_title_page(report_data))
        story.append(PageBreak())

        # 2. 执行摘要（如果有AI总结）
        if "ai_summary" in report_data:
            story.extend(self._create_summary_section(report_data["ai_summary"]))
            story.append(Spacer(1, 0.5 * cm))

        # 3. 主动作识别
        story.extend(self._create_primary_action_section(report_data.get("primary_action", {})))
        story.append(Spacer(1, 0.5 * cm))

        # 4. ROM数据分析
        story.extend(self._create_rom_section(report_data.get("rom_summary", {})))
        story.append(Spacer(1, 0.5 * cm))

        # 5. 代偿问题分析
        compensation = report_data.get("compensation_summary", {})
        if compensation:
            story.extend(self._create_compensation_section(compensation))
            story.append(Spacer(1, 0.5 * cm))

        # 6. 技术信息
        story.extend(self._create_technical_info_section(report_data.get("video_info", {})))

        # 生成PDF
        doc.build(story)

        print(f"✓ PDF报告已生成: {pdf_path}")
        return pdf_path

    def _create_title_page(self, report_data: dict):
        """创建标题页"""
        elements = []
        styles = getSampleStyleSheet()

        # 检查是否有中文字体
        font_name = 'SimHei' if 'SimHei' in pdfmetrics.getRegisteredFontNames() else 'Helvetica'

        # 标题样式
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=28,
            textColor=colors.HexColor('#1E3A8A'),
            spaceAfter=1 * cm,
            alignment=1,  # 居中
            fontName=font_name
        )

        subtitle_style = ParagraphStyle(
            'CustomSubtitle',
            parent=styles['Normal'],
            fontSize=16,
            textColor=colors.HexColor('#4B5563'),
            spaceAfter=0.5 * cm,
            alignment=1,
            fontName=font_name
        )

        # 标题
        elements.append(Paragraph("关节活动度(ROM)评估报告", title_style))
        elements.append(Spacer(1, 0.5 * cm))

        # 副标题 - 主动作
        primary_action = report_data.get("primary_action", {})
        action_name = primary_action.get("action_name", "未知动作")
        elements.append(Paragraph(f"评估动作: {action_name}", subtitle_style))

        # 生成时间
        gen_time = datetime.now().strftime("%Y年%m月%d日 %H:%M:%S")
        time_style = ParagraphStyle(
            'TimeInfo',
            parent=styles['Normal'],
            fontSize=12,
            textColor=colors.grey,
            alignment=1,
            fontName=font_name
        )
        elements.append(Spacer(1, 1 * cm))
        elements.append(Paragraph(f"报告生成时间: {gen_time}", time_style))

        # 分隔线
        elements.append(Spacer(1, 1 * cm))
        line = Paragraph("_" * 50, ParagraphStyle(
            'Line',
            parent=styles['Normal'],
            fontSize=8,
            textColor=colors.lightgrey,
            alignment=1,
            fontName=font_name
        ))
        elements.append(line)

        return elements

    def _create_summary_section(self, ai_summary: dict):
        """创建AI总结部分"""
        elements = []
        styles = getSampleStyleSheet()
        font_name = 'SimHei' if 'SimHei' in pdfmetrics.getRegisteredFontNames() else 'Helvetica'

        section_title = ParagraphStyle(
            'SectionTitle',
            parent=styles['Heading2'],
            fontSize=18,
            textColor=colors.HexColor('#1E3A8A'),
            spaceBefore=0.5 * cm,
            spaceAfter=0.3 * cm,
            fontName=font_name
        )

        elements.append(Paragraph("一、专业评估总结", section_title))

        # 总结内容
        summary_text = ai_summary.get("summary", "无总结内容")
        content_style = ParagraphStyle(
            'SummaryContent',
            parent=styles['Normal'],
            fontSize=11,
            leading=18,  # 行间距
            spaceAfter=0.3 * cm,
            textColor=colors.HexColor('#374151'),
            fontName=font_name
        )

        elements.append(Paragraph(summary_text, content_style))

        return elements

    def _create_primary_action_section(self, primary_action: dict):
        """创建主动作识别部分"""
        elements = []
        styles = getSampleStyleSheet()
        font_name = 'SimHei' if 'SimHei' in pdfmetrics.getRegisteredFontNames() else 'Helvetica'

        section_title = ParagraphStyle(
            'SectionTitle',
            parent=styles['Heading2'],
            fontSize=18,
            textColor=colors.HexColor('#1E3A8A'),
            spaceBefore=0.5 * cm,
            spaceAfter=0.3 * cm,
            fontName=font_name
        )

        elements.append(Paragraph("二、主动作识别", section_title))

        # 主动作信息表格
        action_name = primary_action.get("action_name", "未知")
        confidence = primary_action.get("confidence", 0)
        direction_type = primary_action.get("direction_type", "")
        rom_range = primary_action.get("range", 0)

        data = [
            ["识别动作", action_name],
            ["置信度", f"{confidence:.0%}"],
            ["运动类型", self._get_direction_type_cn(direction_type)],
            ["活动范围", f"{rom_range:.1f}°"]
        ]

        # 添加双向运动的详细信息
        if direction_type in ["bidirectional", "flexion_only", "extension_only"]:
            flex_max = primary_action.get("flexion_max", 0)
            ext_max = primary_action.get("extension_max", 0)
            if flex_max > 0:
                data.append(["前屈峰值", f"{flex_max:.1f}°"])
            if ext_max > 0:
                data.append(["后伸峰值", f"{ext_max:.1f}°"])

        table = Table(data, colWidths=[4 * cm, 10 * cm])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#F3F4F6')),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor('#1F2937')),
            ('FONTNAME', (0, 0), (-1, -1), font_name),
            ('FONTSIZE', (0, 0), (-1, -1), 11),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#D1D5DB')),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ]))

        elements.append(table)

        # 次要动作
        secondary_actions = primary_action.get("secondary_actions", [])
        if secondary_actions:
            elements.append(Spacer(1, 0.3 * cm))
            elements.append(Paragraph("次要动作:", ParagraphStyle(
                'SubTitle',
                parent=styles['Normal'],
                fontSize=12,
                textColor=colors.HexColor('#4B5563'),
                spaceAfter=0.2 * cm,
                fontName=font_name
            )))

            for sec in secondary_actions:
                sec_text = f"• {sec['action_name']} (范围: {sec['range']:.1f}°)"
                elements.append(Paragraph(sec_text, ParagraphStyle(
                    'ListItem',
                    parent=styles['Normal'],
                    fontSize=10,
                    textColor=colors.HexColor('#6B7280'),
                    spaceAfter=0.1 * cm,
                    fontName=font_name
                )))

        return elements

    def _create_rom_section(self, rom_summary: dict):
        """创建ROM数据分析部分"""
        elements = []
        styles = getSampleStyleSheet()
        font_name = 'SimHei' if 'SimHei' in pdfmetrics.getRegisteredFontNames() else 'Helvetica'

        section_title = ParagraphStyle(
            'SectionTitle',
            parent=styles['Heading2'],
            fontSize=18,
            textColor=colors.HexColor('#1E3A8A'),
            spaceBefore=0.5 * cm,
            spaceAfter=0.3 * cm,
            fontName=font_name
        )

        elements.append(Paragraph("三、关节活动度分析", section_title))

        if not rom_summary:
            elements.append(Paragraph("无ROM数据", styles['Normal']))
            return elements

        # ROM数据表格
        table_data = [["关节/动作", "活动范围", "峰值角度", "重复次数"]]

        for key, data in rom_summary.items():
            name = data.get("name", key)
            rom_range = data.get("range", "N/A")
            peak_rom = data.get("peak_rom", 0)
            num_rep = data.get("num_repetitions", 0)

            table_data.append([
                name,
                rom_range,
                f"{peak_rom:.1f}°",
                str(num_rep)
            ])

        table = Table(table_data, colWidths=[5 * cm, 4 * cm, 3 * cm, 2.5 * cm])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1E3A8A')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
            ('FONTNAME', (0, 0), (-1, -1), font_name),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#D1D5DB')),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ]))

        elements.append(table)

        return elements

    def _create_compensation_section(self, compensation_summary: dict):
        """创建代偿问题分析部分"""
        elements = []
        styles = getSampleStyleSheet()
        font_name = 'SimHei' if 'SimHei' in pdfmetrics.getRegisteredFontNames() else 'Helvetica'

        section_title = ParagraphStyle(
            'SectionTitle',
            parent=styles['Heading2'],
            fontSize=18,
            textColor=colors.HexColor('#1E3A8A'),
            spaceBefore=0.5 * cm,
            spaceAfter=0.3 * cm,
            fontName=font_name
        )

        elements.append(Paragraph("四、代偿问题分析", section_title))

        for key, data in compensation_summary.items():
            name = data.get("name", key)
            compensations = data.get("compensations", [])

            # 关节名称
            elements.append(Paragraph(f"【{name}】", ParagraphStyle(
                'JointTitle',
                parent=styles['Normal'],
                fontSize=13,
                textColor=colors.HexColor('#DC2626'),
                spaceBefore=0.3 * cm,
                spaceAfter=0.2 * cm,
                fontName=font_name
            )))

            for comp in compensations:
                description = comp.get("description", "未知代偿")
                severity = comp.get("severity", "unknown")

                # 根据严重程度设置颜色
                severity_colors = {
                    "high": "#DC2626",
                    "medium": "#F59E0B",
                    "low": "#10B981"
                }
                color = severity_colors.get(severity, "#6B7280")

                severity_cn = {
                    "high": "严重",
                    "medium": "中等",
                    "low": "轻微"
                }
                severity_text = severity_cn.get(severity, severity)

                comp_text = f"• {description} (严重程度: {severity_text})"
                elements.append(Paragraph(comp_text, ParagraphStyle(
                    'CompItem',
                    parent=styles['Normal'],
                    fontSize=10,
                    leading=16,
                    spaceAfter=0.15 * cm,
                    textColor=colors.HexColor('#374151'),
                    fontName=font_name
                )))

        return elements

    def _create_technical_info_section(self, video_info: dict):
        """创建技术信息部分"""
        elements = []
        styles = getSampleStyleSheet()
        font_name = 'SimHei' if 'SimHei' in pdfmetrics.getRegisteredFontNames() else 'Helvetica'

        section_title = ParagraphStyle(
            'SectionTitle',
            parent=styles['Heading2'],
            fontSize=18,
            textColor=colors.HexColor('#1E3A8A'),
            spaceBefore=0.5 * cm,
            spaceAfter=0.3 * cm,
            fontName=font_name
        )

        elements.append(Paragraph("五、技术信息", section_title))

        preset = video_info.get("preset", "unknown")
        processing_fps = video_info.get("processing_fps", 0)
        analyzed_frames = video_info.get("analyzed_frames", 0)
        total_frames = video_info.get("total_frames", 0)
        resized = video_info.get("resized", False)

        info_text = f"""
        <para>
        • 处理档位: {preset}<br/>
        • 处理速度: {processing_fps:.1f} FPS<br/>
        • 分析帧数: {analyzed_frames} / {total_frames}<br/>
        • 分辨率优化: {"是" if resized else "否"}<br/>
        </para>
        """

        elements.append(Paragraph(info_text, ParagraphStyle(
            'TechInfo',
            parent=styles['Normal'],
            fontSize=10,
            leading=16,
            textColor=colors.HexColor('#6B7280'),
            fontName=font_name
        )))

        # 免责声明
        elements.append(Spacer(1, 1 * cm))
        disclaimer = Paragraph("""
        <para>
        <b>免责声明:</b><br/>
        本报告由AI辅助生成，仅供参考。具体康复方案请咨询专业康复治疗师。<br/>
        如有不适，请立即停止训练并就医。
        </para>
        """, ParagraphStyle(
            'Disclaimer',
            parent=styles['Normal'],
            fontSize=9,
            leading=14,
            textColor=colors.grey,
            spaceBefore=0.5 * cm,
            fontName=font_name
        ))
        elements.append(disclaimer)

        return elements

    @staticmethod
    def _get_direction_type_cn(direction_type: str) -> str:
        """将方向类型转换为中文"""
        mapping = {
            "unidirectional": "单向运动",
            "flexion_only": "仅前屈",
            "extension_only": "仅后伸",
            "bidirectional": "双向运动"
        }
        return mapping.get(direction_type, direction_type)


def generate_pdf_with_ai_summary(report_data: dict, api_key: str = None,
                                 output_dir: str = "output",
                                 output_filename: str = None) -> str:
    """
    便捷函数：生成带AI总结的PDF报告

    Args:
        report_data: 视频分析生成的JSON报告数据
        api_key: 火山引擎API密钥
        output_dir: 输出目录
        output_filename: 输出文件名

    Returns:
        PDF文件路径
    """
    generator = PDFReportGenerator(output_dir=output_dir)
    pdf_path = generator.generate_pdf(
        report_data=report_data,
        api_key=api_key,
        output_filename=output_filename,
        enable_ai_summary=True
    )
    return pdf_path


if __name__ == "__main__":
    # 测试示例
    import sys

    # PyCharm 运行时使用默认参数
    if len(sys.argv) < 2:
        # 默认测试路径（修改为你的实际 JSON 报告路径）
        report_path = r"output/前屈/前屈_rom_analysis.json"
        api_key = None  # 或者填写你的 API 密钥

        if not os.path.exists(report_path):
            print(f"错误: 找不到报告文件 {report_path}")
            print("\n请修改代码中的 report_path 为实际的 JSON 报告路径")
            print("或者在运行配置中添加参数: python pdf_report_generator.py <report_path> [api_key]")
            sys.exit(1)
    else:
        report_path = sys.argv[1]
        api_key = sys.argv[2] if len(sys.argv) > 2 else None

    # 读取报告
    with open(report_path, 'r', encoding='utf-8') as f:
        report_data = json.load(f)

    # 生成PDF（自动调用AI总结）
    print("正在生成PDF报告（包含AI总结）...")
    pdf_path = generate_pdf_with_ai_summary(report_data, api_key)

    print(f"\n✓ PDF报告已生成: {pdf_path}")
