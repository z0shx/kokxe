"""
配置管理服务
管理 LLM 配置和 Agent 提示词模版
"""
from typing import Optional, List, Dict
from database.db import get_db
from database.models import LLMConfig, AgentPromptTemplate
from utils.logger import setup_logger

logger = setup_logger(__name__, "config_service.log")


class ConfigService:
    """配置管理服务"""

    # ========== LLM 配置管理 ==========

    @classmethod
    def create_llm_config(
        cls,
        name: str,
        provider: str,
        api_key: Optional[str] = None,
        api_base_url: Optional[str] = None,
        model_name: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        top_p: float = 1.0,
        extra_params: Optional[dict] = None,
        is_default: bool = False
    ) -> Optional[int]:
        """
        创建 LLM 配置

        Args:
            name: 配置名称
            provider: LLM 提供商（claude/qwen/ollama/openai）
            api_key: API Key
            api_base_url: API 基础 URL
            model_name: 模型名称
            max_tokens: 最大 token 数
            temperature: 温度参数
            top_p: Top P 参数
            extra_params: 其他参数
            is_default: 是否默认配置

        Returns:
            配置 ID，失败返回 None
        """
        try:
            with get_db() as db:
                # 如果设置为默认，先清除其他默认配置
                if is_default:
                    db.query(LLMConfig).filter(
                        LLMConfig.provider == provider,
                        LLMConfig.is_default == True
                    ).update({"is_default": False})

                config = LLMConfig(
                    name=name,
                    provider=provider,
                    api_key=api_key,
                    api_base_url=api_base_url,
                    model_name=model_name,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    extra_params=extra_params,
                    is_default=is_default
                )

                db.add(config)
                db.commit()
                db.refresh(config)

                logger.info(f"创建 LLM 配置成功: ID={config.id}, Name={name}")
                return config.id

        except Exception as e:
            logger.error(f"创建 LLM 配置失败: {e}")
            return None

    @classmethod
    def get_llm_config(cls, config_id: int) -> Optional[LLMConfig]:
        """获取 LLM 配置"""
        with get_db() as db:
            return db.query(LLMConfig).filter(LLMConfig.id == config_id).first()

    @classmethod
    def get_all_llm_configs(cls, active_only: bool = True) -> List[LLMConfig]:
        """
        获取所有 LLM 配置

        Args:
            active_only: 是否只返回启用的配置

        Returns:
            LLM 配置列表
        """
        with get_db() as db:
            query = db.query(LLMConfig)
            if active_only:
                query = query.filter(LLMConfig.is_active == True)
            return query.order_by(LLMConfig.created_at.desc()).all()

    @classmethod
    def get_llm_configs_by_provider(cls, provider: str, active_only: bool = True) -> List[LLMConfig]:
        """
        按提供商获取 LLM 配置

        Args:
            provider: LLM 提供商
            active_only: 是否只返回启用的配置

        Returns:
            LLM 配置列表
        """
        with get_db() as db:
            query = db.query(LLMConfig).filter(LLMConfig.provider == provider)
            if active_only:
                query = query.filter(LLMConfig.is_active == True)
            return query.order_by(LLMConfig.created_at.desc()).all()

    @classmethod
    def get_default_llm_config(cls, provider: Optional[str] = None) -> Optional[LLMConfig]:
        """
        获取默认 LLM 配置

        Args:
            provider: LLM 提供商（可选）

        Returns:
            默认配置，如果没有则返回 None
        """
        with get_db() as db:
            query = db.query(LLMConfig).filter(
                LLMConfig.is_default == True,
                LLMConfig.is_active == True
            )
            if provider:
                query = query.filter(LLMConfig.provider == provider)

            return query.first()

    @classmethod
    def update_llm_config(
        cls,
        config_id: int,
        name: Optional[str] = None,
        api_key: Optional[str] = None,
        api_base_url: Optional[str] = None,
        model_name: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        extra_params: Optional[dict] = None,
        is_active: Optional[bool] = None,
        is_default: Optional[bool] = None
    ) -> bool:
        """
        更新 LLM 配置

        Returns:
            是否成功
        """
        try:
            with get_db() as db:
                config = db.query(LLMConfig).filter(LLMConfig.id == config_id).first()
                if not config:
                    logger.error(f"LLM 配置不存在: ID={config_id}")
                    return False

                # 如果设置为默认，先清除其他默认配置
                if is_default:
                    db.query(LLMConfig).filter(
                        LLMConfig.provider == config.provider,
                        LLMConfig.is_default == True,
                        LLMConfig.id != config_id
                    ).update({"is_default": False})

                # 更新字段
                if name is not None:
                    config.name = name
                if api_key is not None:
                    config.api_key = api_key
                if api_base_url is not None:
                    config.api_base_url = api_base_url
                if model_name is not None:
                    config.model_name = model_name
                if max_tokens is not None:
                    config.max_tokens = max_tokens
                if temperature is not None:
                    config.temperature = temperature
                if top_p is not None:
                    config.top_p = top_p
                if extra_params is not None:
                    config.extra_params = extra_params
                if is_active is not None:
                    config.is_active = is_active
                if is_default is not None:
                    config.is_default = is_default

                db.commit()
                logger.info(f"更新 LLM 配置成功: ID={config_id}")
                return True

        except Exception as e:
            logger.error(f"更新 LLM 配置失败: {e}")
            return False

    @classmethod
    def delete_llm_config(cls, config_id: int) -> bool:
        """删除 LLM 配置"""
        try:
            with get_db() as db:
                config = db.query(LLMConfig).filter(LLMConfig.id == config_id).first()
                if config:
                    db.delete(config)
                    db.commit()
                    logger.info(f"删除 LLM 配置成功: ID={config_id}")
                    return True
                return False
        except Exception as e:
            logger.error(f"删除 LLM 配置失败: {e}")
            return False

    # ========== Agent 提示词模版管理 ==========

    @classmethod
    def create_prompt_template(
        cls,
        name: str,
        content: str,
        description: Optional[str] = None,
        category: Optional[str] = None,
        tags: Optional[list] = None,
        is_default: bool = False
    ) -> Optional[int]:
        """
        创建 Agent 提示词模版

        Args:
            name: 模版名称
            content: 提示词内容
            description: 模版描述
            category: 分类（conservative/aggressive/balanced/custom）
            tags: 标签列表
            is_default: 是否默认模版

        Returns:
            模版 ID，失败返回 None
        """
        try:
            with get_db() as db:
                # 如果设置为默认，先清除其他默认模版
                if is_default:
                    db.query(AgentPromptTemplate).filter(
                        AgentPromptTemplate.is_default == True
                    ).update({"is_default": False})

                template = AgentPromptTemplate(
                    name=name,
                    content=content,
                    description=description,
                    category=category,
                    tags=tags,
                    is_default=is_default
                )

                db.add(template)
                db.commit()
                db.refresh(template)

                logger.info(f"创建 Agent 提示词模版成功: ID={template.id}, Name={name}")
                return template.id

        except Exception as e:
            logger.error(f"创建 Agent 提示词模版失败: {e}")
            return None

    @classmethod
    def get_prompt_template(cls, template_id: int) -> Optional[AgentPromptTemplate]:
        """获取 Agent 提示词模版"""
        with get_db() as db:
            return db.query(AgentPromptTemplate).filter(
                AgentPromptTemplate.id == template_id
            ).first()

    @classmethod
    def get_all_prompt_templates(cls, active_only: bool = True) -> List[AgentPromptTemplate]:
        """
        获取所有 Agent 提示词模版

        Args:
            active_only: 是否只返回启用的模版

        Returns:
            模版列表
        """
        with get_db() as db:
            query = db.query(AgentPromptTemplate)
            if active_only:
                query = query.filter(AgentPromptTemplate.is_active == True)
            return query.order_by(AgentPromptTemplate.created_at.desc()).all()

    @classmethod
    def get_prompt_templates_by_category(
        cls,
        category: str,
        active_only: bool = True
    ) -> List[AgentPromptTemplate]:
        """
        按分类获取 Agent 提示词模版

        Args:
            category: 分类
            active_only: 是否只返回启用的模版

        Returns:
            模版列表
        """
        with get_db() as db:
            query = db.query(AgentPromptTemplate).filter(
                AgentPromptTemplate.category == category
            )
            if active_only:
                query = query.filter(AgentPromptTemplate.is_active == True)
            return query.order_by(AgentPromptTemplate.created_at.desc()).all()

    @classmethod
    def get_default_prompt_template(cls) -> Optional[AgentPromptTemplate]:
        """
        获取默认 Agent 提示词模版

        Returns:
            默认模版，如果没有则返回 None
        """
        with get_db() as db:
            return db.query(AgentPromptTemplate).filter(
                AgentPromptTemplate.is_default == True,
                AgentPromptTemplate.is_active == True
            ).first()

    @classmethod
    def update_prompt_template(
        cls,
        template_id: int,
        name: Optional[str] = None,
        content: Optional[str] = None,
        description: Optional[str] = None,
        category: Optional[str] = None,
        tags: Optional[list] = None,
        is_active: Optional[bool] = None,
        is_default: Optional[bool] = None
    ) -> bool:
        """
        更新 Agent 提示词模版

        Returns:
            是否成功
        """
        try:
            with get_db() as db:
                template = db.query(AgentPromptTemplate).filter(
                    AgentPromptTemplate.id == template_id
                ).first()
                if not template:
                    logger.error(f"Agent 提示词模版不存在: ID={template_id}")
                    return False

                # 如果设置为默认，先清除其他默认模版
                if is_default:
                    db.query(AgentPromptTemplate).filter(
                        AgentPromptTemplate.is_default == True,
                        AgentPromptTemplate.id != template_id
                    ).update({"is_default": False})

                # 更新字段
                if name is not None:
                    template.name = name
                if content is not None:
                    template.content = content
                if description is not None:
                    template.description = description
                if category is not None:
                    template.category = category
                if tags is not None:
                    template.tags = tags
                if is_active is not None:
                    template.is_active = is_active
                if is_default is not None:
                    template.is_default = is_default

                db.commit()
                logger.info(f"更新 Agent 提示词模版成功: ID={template_id}")
                return True

        except Exception as e:
            logger.error(f"更新 Agent 提示词模版失败: {e}")
            return False

    @classmethod
    def delete_prompt_template(cls, template_id: int) -> bool:
        """删除 Agent 提示词模版"""
        try:
            with get_db() as db:
                template = db.query(AgentPromptTemplate).filter(
                    AgentPromptTemplate.id == template_id
                ).first()
                if template:
                    db.delete(template)
                    db.commit()
                    logger.info(f"删除 Agent 提示词模版成功: ID={template_id}")
                    return True
                return False
        except Exception as e:
            logger.error(f"删除 Agent 提示词模版失败: {e}")
            return False
