"""
配置中心界面
"""
import gradio as gr
import pandas as pd
from typing import List, Tuple, Optional
from services.config_service import ConfigService
from utils.logger import setup_logger
from ui.constants import DataFrameHeaders, create_empty_dataframe
from ui.ui_utils import UIHelper

logger = setup_logger(__name__, "config_center_ui.log")


class ConfigCenterUI:
    """配置中心界面"""

    def __init__(self):
        self.config_service = ConfigService()

    # ========== LLM 配置管理 ==========

    @UIHelper.create_error_handler("加载LLM配置")
    def load_llm_configs(self) -> pd.DataFrame:
        """加载 LLM 配置列表"""
        configs = self.config_service.get_all_llm_configs(active_only=False)

        if not configs:
            return create_empty_dataframe(DataFrameHeaders.LLM_CONFIG)

        data = []
        for config in configs:
            data.append({
                "ID": config.id,
                "名称": config.name,
                "提供商": config.provider,
                "模型": config.model_name or "-",
                "状态": "启用" if config.is_active else "禁用",
                "默认": "✓" if config.is_default else ""
            })

        return pd.DataFrame(data)

    def create_llm_config(
        self,
        name: str,
        provider: str,
        api_key: str,
        api_base_url: str,
        model_name: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        is_default: bool
    ) -> Tuple[str, pd.DataFrame]:
        """创建 LLM 配置"""
        try:
            if not name or not provider:
                return "❌ 请填写配置名称和提供商", self.load_llm_configs()

            config_id = self.config_service.create_llm_config(
                name=name,
                provider=provider,
                api_key=api_key,
                api_base_url=api_base_url,
                model_name=model_name,
                max_tokens=int(max_tokens),
                temperature=float(temperature),
                top_p=float(top_p),
                is_default=is_default
            )

            if config_id:
                return f"✅ 创建成功！配置 ID: {config_id}", self.load_llm_configs()
            else:
                return "❌ 创建失败", self.load_llm_configs()

        except Exception as e:
            logger.error(f"创建 LLM 配置失败: {e}")
            return f"❌ 创建失败: {str(e)}", self.load_llm_configs()

    def delete_llm_config(self, config_id: int) -> Tuple[str, pd.DataFrame]:
        """删除 LLM 配置"""
        try:
            if not config_id or config_id <= 0:
                return "❌ 请选择要删除的配置", self.load_llm_configs()

            success = self.config_service.delete_llm_config(int(config_id))

            if success:
                return f"✅ 删除成功！配置 ID: {config_id}", self.load_llm_configs()
            else:
                return "❌ 删除失败", self.load_llm_configs()

        except Exception as e:
            logger.error(f"删除 LLM 配置失败: {e}")
            return f"❌ 删除失败: {str(e)}", self.load_llm_configs()

    def get_llm_config_choices(self) -> List[Tuple[str, int]]:
        """获取 LLM 配置选项（用于下拉框）"""
        try:
            configs = self.config_service.get_all_llm_configs(active_only=True)
            return [(f"{cfg.name} ({cfg.provider})", cfg.id) for cfg in configs]
        except Exception as e:
            logger.error(f"获取 LLM 配置选项失败: {e}")
            return []

    # ========== Agent 提示词模版管理 ==========

    @UIHelper.create_error_handler("加载提示词模版")
    def load_prompt_templates(self) -> pd.DataFrame:
        """加载 Agent 提示词模版列表"""
        templates = self.config_service.get_all_prompt_templates(active_only=False)

        if not templates:
            return create_empty_dataframe(DataFrameHeaders.PROMPT_TEMPLATE)

        data = []
        for template in templates:
            desc = template.description or ""
            if len(desc) > 50:
                desc = desc[:50] + "..."

            data.append({
                "ID": template.id,
                "名称": template.name,
                "分类": template.category or "-",
                "描述": desc,
                "状态": "启用" if template.is_active else "禁用",
                "默认": "✓" if template.is_default else ""
            })

        return pd.DataFrame(data)

    def create_prompt_template(
        self,
        name: str,
        content: str,
        description: str,
        category: str,
        is_default: bool
    ) -> Tuple[str, pd.DataFrame]:
        """创建 Agent 提示词模版"""
        try:
            if not name or not content:
                return "❌ 请填写模版名称和内容", self.load_prompt_templates()

            template_id = self.config_service.create_prompt_template(
                name=name,
                content=content,
                description=description,
                category=category,
                is_default=is_default
            )

            if template_id:
                return f"✅ 创建成功！模版 ID: {template_id}", self.load_prompt_templates()
            else:
                return "❌ 创建失败", self.load_prompt_templates()

        except Exception as e:
            logger.error(f"创建 Agent 提示词模版失败: {e}")
            return f"❌ 创建失败: {str(e)}", self.load_prompt_templates()

    def delete_prompt_template(self, template_id: int) -> Tuple[str, pd.DataFrame]:
        """删除 Agent 提示词模版"""
        try:
            if not template_id or template_id <= 0:
                return "❌ 请选择要删除的模版", self.load_prompt_templates()

            success = self.config_service.delete_prompt_template(int(template_id))

            if success:
                return f"✅ 删除成功！模版 ID: {template_id}", self.load_prompt_templates()
            else:
                return "❌ 删除失败", self.load_prompt_templates()

        except Exception as e:
            logger.error(f"删除 Agent 提示词模版失败: {e}")
            return f"❌ 删除失败: {str(e)}", self.load_prompt_templates()

    def get_prompt_template_choices(self) -> List[Tuple[str, str]]:
        """获取 Agent 提示词模版选项（用于下拉框）"""
        try:
            templates = self.config_service.get_all_prompt_templates(active_only=True)
            return [(tpl.name, tpl.content) for tpl in templates]
        except Exception as e:
            logger.error(f"获取 Agent 提示词模版选项失败: {e}")
            return []

    def load_template_content(self, template_name: str) -> str:
        """加载模版内容"""
        try:
            templates = self.config_service.get_all_prompt_templates(active_only=True)
            for template in templates:
                if template.name == template_name:
                    return template.content
            return ""
        except Exception as e:
            logger.error(f"加载模版内容失败: {e}")
            return ""

    # ========== UI 构建 ==========

    def build_ui(self):
        """构建界面"""
        with gr.Column():
            gr.Markdown("## 配置中心")

            with gr.Tabs():
                # Tab 1: LLM 配置管理
                with gr.Tab("LLM 配置"):
                    gr.Markdown("### LLM 配置管理")
                    gr.Markdown("管理 AI Agent 使用的 LLM 配置（Claude、Qwen、Ollama、OpenAI）")

                    with gr.Row():
                        # 左侧：配置列表
                        with gr.Column(scale=2):
                            llm_configs_table = gr.DataFrame(
                                value=self.load_llm_configs(),
                                label="LLM 配置列表",
                                interactive=False
                            )

                            with gr.Row():
                                llm_refresh_btn = gr.Button("🔄 刷新列表", size="sm")
                                llm_delete_id = gr.Number(
                                    label="配置 ID",
                                    value=0,
                                    minimum=0,
                                    scale=1
                                )
                                llm_delete_btn = gr.Button("🗑️ 删除", variant="stop", size="sm")

                        # 右侧：创建配置
                        with gr.Column(scale=3):
                            gr.Markdown("#### 新建 LLM 配置")

                            llm_name = gr.Textbox(
                                label="配置名称",
                                placeholder="例如：Claude Sonnet 3.5"
                            )

                            llm_provider = gr.Dropdown(
                                label="LLM 提供商",
                                choices=["claude", "qwen", "ollama", "openai"],
                                value="claude"
                            )

                            with gr.Row():
                                llm_api_key = gr.Textbox(
                                    label="API Key",
                                    type="password",
                                    placeholder="sk-xxx..."
                                )

                                llm_api_base_url = gr.Textbox(
                                    label="API Base URL",
                                    placeholder="https://api.anthropic.com (可选)"
                                )

                            llm_model_name = gr.Textbox(
                                label="模型名称",
                                placeholder="claude-3-5-sonnet-20241022"
                            )

                            with gr.Row():
                                llm_max_tokens = gr.Number(
                                    label="最大 Token 数",
                                    value=4096,
                                    minimum=1,
                                    maximum=200000
                                )

                                llm_temperature = gr.Slider(
                                    label="温度 (Temperature)",
                                    minimum=0.0,
                                    maximum=2.0,
                                    value=0.7,
                                    step=0.1
                                )

                                llm_top_p = gr.Slider(
                                    label="Top P",
                                    minimum=0.0,
                                    maximum=1.0,
                                    value=1.0,
                                    step=0.05
                                )

                            llm_is_default = gr.Checkbox(
                                label="设为默认配置",
                                value=False
                            )

                            llm_create_btn = gr.Button("➕ 创建配置", variant="primary")

                            llm_result = gr.Textbox(
                                label="操作结果",
                                interactive=False
                            )

                    # 事件绑定
                    llm_refresh_btn.click(
                        fn=lambda: self.load_llm_configs(),
                        inputs=[],
                        outputs=[llm_configs_table]
                    )

                    llm_create_btn.click(
                        fn=lambda name, provider, api_key, api_base_url, model_name, max_tokens, temperature, top_p, is_default: self.create_llm_config(
                            name, provider, api_key, api_base_url, model_name, max_tokens, temperature, top_p, is_default
                        ),
                        inputs=[
                            llm_name, llm_provider, llm_api_key, llm_api_base_url,
                            llm_model_name, llm_max_tokens, llm_temperature, llm_top_p,
                            llm_is_default
                        ],
                        outputs=[llm_result, llm_configs_table]
                    )

                    llm_delete_btn.click(
                        fn=lambda config_id: self.delete_llm_config(config_id),
                        inputs=[llm_delete_id],
                        outputs=[llm_result, llm_configs_table]
                    )

                # Tab 2: Agent 提示词模版管理
                with gr.Tab("Agent 提示词模版"):
                    gr.Markdown("### Agent 提示词模版管理")
                    gr.Markdown("管理可复用的 Agent 提示词模版")

                    with gr.Row():
                        # 左侧：模版列表
                        with gr.Column(scale=2):
                            prompt_templates_table = gr.DataFrame(
                                value=self.load_prompt_templates(),
                                label="提示词模版列表",
                                interactive=False
                            )

                            with gr.Row():
                                prompt_refresh_btn = gr.Button("🔄 刷新列表", size="sm")
                                prompt_delete_id = gr.Number(
                                    label="模版 ID",
                                    value=0,
                                    minimum=0,
                                    scale=1
                                )
                                prompt_delete_btn = gr.Button("🗑️ 删除", variant="stop", size="sm")

                        # 右侧：创建模版
                        with gr.Column(scale=3):
                            gr.Markdown("#### 新建提示词模版")

                            prompt_name = gr.Textbox(
                                label="模版名称",
                                placeholder="例如：保守型策略"
                            )

                            prompt_category = gr.Dropdown(
                                label="分类",
                                choices=["conservative", "aggressive", "balanced", "custom"],
                                value="balanced"
                            )

                            prompt_description = gr.Textbox(
                                label="模版描述",
                                placeholder="简要描述此模版的用途和特点",
                                lines=2
                            )

                            prompt_content = gr.Textbox(
                                label="提示词内容",
                                placeholder="输入详细的 Agent 提示词...",
                                lines=10
                            )

                            prompt_is_default = gr.Checkbox(
                                label="设为默认模版",
                                value=False
                            )

                            prompt_create_btn = gr.Button("➕ 创建模版", variant="primary")

                            prompt_result = gr.Textbox(
                                label="操作结果",
                                interactive=False
                            )

                    # 事件绑定
                    prompt_refresh_btn.click(
                        fn=lambda: self.load_prompt_templates(),
                        inputs=[],
                        outputs=[prompt_templates_table]
                    )

                    prompt_create_btn.click(
                        fn=lambda name, content, description, category, is_default: self.create_prompt_template(
                            name, content, description, category, is_default
                        ),
                        inputs=[
                            prompt_name, prompt_content, prompt_description,
                            prompt_category, prompt_is_default
                        ],
                        outputs=[prompt_result, prompt_templates_table]
                    )

                    prompt_delete_btn.click(
                        fn=lambda template_id: self.delete_prompt_template(template_id),
                        inputs=[prompt_delete_id],
                        outputs=[prompt_result, prompt_templates_table]
                    )


def create_config_center_ui():
    """创建配置中心界面"""
    ui = ConfigCenterUI()
    return ui.build_ui()
