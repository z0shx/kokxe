"""
增强的AI Agent对话管理服务
重构版：支持完整上下文管理、配置化系统提示词、持久化对话记录
"""
import json
import pandas as pd
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, AsyncGenerator, Any
from enum import Enum
from sqlalchemy import and_, desc, or_, text as sa_text
from database.db import get_db
from database.models import (
    AgentConversation, AgentMessage, TradingPlan, TrainingRecord,
    PredictionData, AgentPromptTemplate, LLMConfig, now_beijing
)
from utils.logger import setup_logger

logger = setup_logger(__name__, "enhanced_conversation.log")


class ConversationType(Enum):
    """对话类型"""
    MANUAL_CHAT = "manual_chat"          # 手动对话
    AUTO_INFERENCE = "auto_inference"    # 自动推理
    KLINE_EVENT = "kline_event"          # K线事件触发
    SYSTEM_INIT = "system_init"          # 系统初始化


class MessageRole(Enum):
    """消息角色"""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class MessageSubType(Enum):
    """消息子类型"""
    TEXT = "text"
    THINKING = "thinking"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    SYSTEM_PROMPT = "system_prompt"
    KLINE_DATA = "kline_data"
    EVENT_NOTIFICATION = "event_notification"


class EnhancedConversationService:
    """增强的对话服务类"""

    @staticmethod
    def create_or_get_conversation(
        plan_id: int,
        conversation_type: ConversationType,
        reset_context: bool = False,
        session_name: Optional[str] = None
    ) -> AgentConversation:
        """
        创建或获取对话会话

        Args:
            plan_id: 计划ID
            conversation_type: 对话类型
            reset_context: 是否重置上下文（创建新会话）
            session_name: 会话名称

        Returns:
            对话会话对象
        """
        try:
            with get_db() as db:
                if not reset_context:
                    # 尝试获取现有对话
                    existing_conversation = db.query(AgentConversation).filter(
                        and_(
                            AgentConversation.plan_id == plan_id,
                            AgentConversation.conversation_type == conversation_type.value,
                            AgentConversation.status == 'active'
                        )
                    ).order_by(desc(AgentConversation.last_message_at)).first()

                    if existing_conversation:
                        logger.info(f"复用现有对话: conversation_id={existing_conversation.id}, type={conversation_type.value}")
                        return existing_conversation

                # 创建新对话
                conversation = AgentConversation(
                    plan_id=plan_id,
                    conversation_type=conversation_type.value,
                    session_name=session_name or f"{conversation_type.value}_{now_beijing().strftime('%Y%m%d_%H%M%S')}",
                    status='active',
                    started_at=now_beijing(),
                    last_message_at=now_beijing()
                )

                db.add(conversation)
                db.commit()
                db.refresh(conversation)

                logger.info(f"创建新对话: conversation_id={conversation.id}, type={conversation_type.value}")
                return conversation

        except Exception as e:
            logger.error(f"创建对话会话失败: {e}")
            raise

    @staticmethod
    def validate_tools_configuration() -> bool:
        """
        验证工具配置是否正确

        Returns:
            bool: 工具配置是否有效
        """
        try:
            from services.agent_tools import get_all_tools

            tools = get_all_tools()
            if not tools:
                logger.warning("没有找到任何可用工具")
                return False

            # 检查关键工具是否存在
            critical_tools = [
                "get_account_balance",
                "get_current_price",
                "place_order",
                "get_order_info"
            ]

            missing_tools = []
            for tool_name in critical_tools:
                if tool_name not in tools:
                    missing_tools.append(tool_name)

            if missing_tools:
                logger.error(f"缺少关键工具: {missing_tools}")
                return False

            logger.info(f"工具配置验证通过，共有 {len(tools)} 个工具可用")
            return True

        except Exception as e:
            logger.error(f"工具配置验证失败: {e}")
            return False

    @staticmethod
    def generate_tools_description() -> str:
        """
        生成可调用工具的说明描述

        Returns:
            工具说明字符串
        """
        try:
            from services.agent_tools import get_all_tools

            tools = get_all_tools()
            if not tools:
                logger.error("没有找到任何可用工具，无法生成工具说明")
                return ""

            # 验证工具配置
            if not EnhancedConversationService.validate_tools_configuration():
                logger.warning("工具配置验证失败，但仍尝试生成说明")

            # 按分类组织工具
            tools_by_category = {
                "query": [],
                "trade": [],
                "monitor": []
            }

            for tool_name, tool in tools.items():
                category = tools_by_category.get(tool.category.value, [])
                category.append({
                    "name": tool_name,
                    "description": tool.description,
                    "required_params": tool.required_params,
                    "risk_level": tool.risk_level
                })

            description = "\n\n## 🛠️ 可用工具说明\n\n"

            # 查询类工具
            if tools_by_category["query"]:
                description += "### 🔍 查询类工具 (低风险)\n\n"
                for tool in tools_by_category["query"]:
                    desc = f"**{tool['name']}**: {tool['description']}\n"
                    if tool['required_params']:
                        desc += f"- 必填参数: {', '.join(tool['required_params'])}\n"
                    desc += f"- 风险级别: {tool['risk_level']}\n\n"
                    description += desc

            # 交易类工具
            if tools_by_category["trade"]:
                description += "### 💰 交易类工具 (中高风险)\n\n"
                description += "**重要提醒**: 下单前必须先调用 `get_account_balance` 确认资金充足\n\n"
                for tool in tools_by_category["trade"]:
                    desc = f"**{tool['name']}**: {tool['description']}\n"
                    if tool['required_params']:
                        desc += f"- 必填参数: {', '.join(tool['required_params'])}\n"
                    desc += f"- 风险级别: {tool['risk_level']}\n\n"
                    description += desc

            # 监控类工具
            if tools_by_category["monitor"]:
                description += "### 📊 数据管理工具 (低风险)\n\n"
                for tool in tools_by_category["monitor"]:
                    desc = f"**{tool['name']}**: {tool['description']}\n"
                    if tool['required_params']:
                        desc += f"- 必填参数: {', '.join(tool['required_params'])}\n"
                    desc += f"- 风险级别: {tool['risk_level']}\n\n"
                    description += desc

            description += """### 📋 工具使用规范

1. **查询优先**: 做交易决策前先查询账户余额、持仓和当前价格
2. **资金确认**: 下单前必须调用 `get_account_balance` 确保有足够资金
3. **风险评估**: 高风险工具使用时要谨慎，确保符合风险管理要求
4. **参数验证**: 确保所有必填参数都已提供且格式正确
5. **订单确认**: 下单后调用相应查询工具确认订单状态

当需要执行交易操作时，请使用 `place_order` 工具进行下单。"""

            return description

        except Exception as e:
            logger.error(f"生成工具说明失败: {e}")
            return ""

    @staticmethod
    def get_system_prompt_content(plan: TradingPlan) -> str:
        """
        获取系统提示词内容（优先使用agent_prompt字段，并自动追加工具说明）

        Args:
            plan: 交易计划对象

        Returns:
            系统提示词内容
        """
        try:
            base_prompt = ""
            prompt_source = ""

            # 优先使用计划的agent_prompt字段
            agent_prompt = getattr(plan, 'agent_prompt', None)
            if agent_prompt and agent_prompt.strip():
                base_prompt = agent_prompt.strip()
                prompt_source = "计划配置的agent_prompt"
                logger.info(f"使用计划配置的agent_prompt")
            else:
                # 如果没有agent_prompt，检查数据库中是否有prompt_template_id字段
                with get_db() as db:
                    # 检查trading_plans表是否有prompt_template_id字段
                    result = db.execute(sa_text("""
                        SELECT column_name
                        FROM information_schema.columns
                        WHERE table_name = 'trading_plans' AND column_name = 'prompt_template_id'
                    """))
                    has_field = result.fetchone() is not None

                    if has_field:
                        prompt_template_id = getattr(plan, 'prompt_template_id', None)
                        if prompt_template_id:
                            template = db.query(AgentPromptTemplate).filter(
                                AgentPromptTemplate.id == prompt_template_id,
                                AgentPromptTemplate.is_active == True
                            ).first()

                            if template:
                                base_prompt = template.content
                                prompt_source = f"计划配置的提示词模板: {template.name}"
                                logger.info(f"使用计划配置的提示词模板: {template.name}")

                    # 如果仍然没有找到，使用默认模板
                    if not base_prompt:
                        default_template = db.query(AgentPromptTemplate).filter(
                            AgentPromptTemplate.is_default == True,
                            AgentPromptTemplate.is_active == True
                        ).first()

                        if default_template:
                            base_prompt = default_template.content
                            prompt_source = f"默认提示词模板: {default_template.name}"
                            logger.info(f"使用默认提示词模板: {default_template.name}")

            # 最后使用硬编码的基础提示词
            if not base_prompt:
                base_prompt = EnhancedConversationService._get_default_system_prompt(plan)
                prompt_source = "硬编码的基础提示词"
                logger.warning("未找到配置的提示词，使用基础提示词")

            # 自动追加工具说明
            tools_description = EnhancedConversationService.generate_tools_description()

            if tools_description:
                final_prompt = base_prompt + tools_description
                logger.info(f"系统提示词已生成，来源: {prompt_source}，已自动追加工具说明")
                return final_prompt
            else:
                logger.warning("工具说明生成失败，仅使用基础提示词")
                return base_prompt

        except Exception as e:
            logger.error(f"获取系统提示词失败: {e}")
            # 即使出错也要尝试返回基础提示词
            return EnhancedConversationService._get_default_system_prompt(plan)

    @staticmethod
    def _get_default_system_prompt(plan: TradingPlan) -> str:
        """获取默认系统提示词（兜底方案）"""
        return f"""你是一个专业的加密货币交易AI助手，负责分析市场数据并做出交易决策。

**交易计划信息**:
- 交易对: {plan.inst_id}
- 时间周期: {plan.interval}
- 环境: {'🧪 模拟盘' if plan.is_demo else '💰 实盘'}
- 计划状态: {plan.status}

**工作原则**:
1. 基于数据驱动的决策
2. 严格执行风险管理
3. 保持客观和理性
4. 使用ReAct模式进行思考
5. 详细记录决策过程

请根据提供的预测数据和市场信息，进行专业的分析和推理。"""

    @staticmethod
    def add_system_prompt_message(
        conversation_id: int,
        content: str,
        template_id: Optional[int] = None
    ) -> AgentMessage:
        """
        添加系统提示词消息

        Args:
            conversation_id: 对话ID
            content: 提示词内容
            template_id: 模板ID

        Returns:
            消息对象
        """
        try:
            with get_db() as db:
                message = AgentMessage(
                    conversation_id=conversation_id,
                    role=MessageRole.SYSTEM.value,
                    message_type=MessageSubType.SYSTEM_PROMPT.value,
                    content=content,
                    metadata=json.dumps({
                        "template_id": template_id,
                        "timestamp": now_beijing().isoformat()
                    }),
                    created_at=now_beijing()
                )

                db.add(message)

                # 更新对话的最后消息时间
                db.query(AgentConversation).filter(
                    AgentConversation.id == conversation_id
                ).update({
                    'last_message_at': now_beijing()
                })

                db.commit()
                db.refresh(message)

                logger.debug(f"添加系统提示词消息: conversation_id={conversation_id}")
                return message

        except Exception as e:
            logger.error(f"添加系统提示词消息失败: {e}")
            raise

    @staticmethod
    def add_kline_data_message(
        conversation_id: int,
        prediction_data: List[PredictionData],
        trigger_event: str = "manual_inference"
    ) -> AgentMessage:
        """
        添加K线数据消息（CSV格式）

        Args:
            conversation_id: 对话ID
            prediction_data: 预测数据列表
            trigger_event: 触发事件类型

        Returns:
            消息对象
        """
        try:
            # 转换为CSV格式
            if not prediction_data:
                csv_content = "暂无预测数据"
            else:
                # 构建DataFrame并转换为CSV
                df_data = []
                for data in prediction_data:
                    df_data.append({
                        'timestamp': data.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC'),
                        'close': data.close,
                        'close_min': data.close_min,
                        'close_max': data.close_max,
                        'upward_probability': data.upward_probability,
                        'volatility_amplification_probability': data.volatility_amplification_probability
                    })

                df = pd.DataFrame(df_data)
                csv_content = df.to_csv(index=False)

            content = f"""**最新预测数据** (CSV格式):

{csv_content}

**数据来源**: {trigger_event}
**记录数量**: {len(prediction_data)}
**生成时间**: {now_beijing().strftime('%Y-%m-%d %H:%M:%S UTC+8')}"""

            with get_db() as db:
                message = AgentMessage(
                    conversation_id=conversation_id,
                    role=MessageRole.USER.value,
                    message_type=MessageSubType.KLINE_DATA.value,
                    content=content,
                    metadata=json.dumps({
                        "trigger_event": trigger_event,
                        "record_count": len(prediction_data),
                        "data_timestamps": [d.timestamp.isoformat() for d in prediction_data[:5]]  # 只保留前5个时间戳
                    }),
                    created_at=now_beijing()
                )

                db.add(message)

                # 更新对话的最后消息时间
                db.query(AgentConversation).filter(
                    AgentConversation.id == conversation_id
                ).update({
                    'last_message_at': now_beijing()
                })

                db.commit()
                db.refresh(message)

                logger.info(f"添加K线数据消息: conversation_id={conversation_id}, records={len(prediction_data)}")
                return message

        except Exception as e:
            logger.error(f"添加K线数据消息失败: {e}")
            raise

    @staticmethod
    async def add_assistant_message_stream(
        conversation_id: int,
        content_stream: AsyncGenerator[Dict[str, Any], None]
    ) -> AsyncGenerator[Tuple[AgentMessage, Dict[str, Any]], None]:
        """
        流式添加助手消息

        Args:
            conversation_id: 对话ID
            content_stream: 内容流生成器

        Yields:
            (消息对象, 流式数据) 元组
        """
        message_id = None
        accumulated_content = ""

        try:
            with get_db() as db:
                # 创建初始消息
                message = AgentMessage(
                    conversation_id=conversation_id,
                    role=MessageRole.ASSISTANT.value,
                    message_type=MessageSubType.TEXT.value,
                    content="",
                    metadata=json.dumps({
                        "streaming": True,
                        "started_at": now_beijing().isoformat()
                    }),
                    created_at=now_beijing()
                )

                db.add(message)
                db.commit()
                db.refresh(message)
                message_id = message.id

                logger.info(f"开始流式助手消息: conversation_id={conversation_id}, message_id={message_id}")

            # 处理流式内容
            async for chunk_data in content_stream:
                chunk_content = chunk_data.get('content', '')
                chunk_type = chunk_data.get('type', 'content')

                if chunk_type in ['content', 'thinking', 'tool_call', 'tool_result']:
                    accumulated_content += chunk_content

                    # 更新消息内容
                    with get_db() as db:
                        db.query(AgentMessage).filter(
                            AgentMessage.id == message_id
                        ).update({
                            'content': accumulated_content,
                            'metadata': json.dumps({
                                "streaming": True,
                                "last_update": now_beijing().isoformat(),
                                "chunk_count": chunk_data.get('chunk_count', 0)
                            })
                        })
                        db.commit()

                yield (message, chunk_data)

            # 标记流式完成
            with get_db() as db:
                db.query(AgentMessage).filter(
                    AgentMessage.id == message_id
                ).update({
                    'metadata': json.dumps({
                        "streaming": False,
                        "completed_at": now_beijing().isoformat(),
                        "final_length": len(accumulated_content)
                    })
                })
                db.commit()

            logger.info(f"完成流式助手消息: conversation_id={conversation_id}, message_id={message_id}")

        except Exception as e:
            logger.error(f"流式助手消息失败: {e}")

            # 标记错误状态
            if message_id:
                try:
                    with get_db() as db:
                        db.query(AgentMessage).filter(
                            AgentMessage.id == message_id
                        ).update({
                            'metadata': json.dumps({
                                "streaming": False,
                                "error": str(e),
                                "failed_at": now_beijing().isoformat()
                            })
                        })
                        db.commit()
                except Exception as update_error:
                    logger.error(f"更新错误状态失败: {update_error}")

            raise

    @staticmethod
    def get_conversation_messages(
        conversation_id: int,
        include_metadata: bool = False
    ) -> List[Dict[str, Any]]:
        """
        获取对话消息（转换为Chatbot格式）

        Args:
            conversation_id: 对话ID
            include_metadata: 是否包含元数据

        Returns:
            消息列表
        """
        try:
            with get_db() as db:
                messages = db.query(AgentMessage).filter(
                    AgentMessage.conversation_id == conversation_id
                ).order_by(AgentMessage.created_at.asc()).all()

                chatbot_messages = []

                for msg in messages:
                    # 系统提示词转换为assistant角色以便在chatbot中正确显示
                    if msg.message_type == MessageSubType.SYSTEM_PROMPT.value:
                        chatbot_msg = {
                            "role": "assistant",  # 系统提示词使用assistant角色显示
                            "content": f"🤖 **系统提示词**\n\n{msg.content}",
                            "timestamp": msg.created_at.isoformat(),
                            "metadata": {"collapsible": True, "default_collapsed": False, "type": "system"}
                        }
                    elif msg.message_type == MessageSubType.KLINE_DATA.value:
                        chatbot_msg = {
                            "role": "user",  # 预测数据作为用户输入显示
                            "content": f"📊 **预测数据**\n\n{msg.content}",
                            "timestamp": msg.created_at.isoformat(),
                            "metadata": {"collapsible": True, "default_collapsed": False, "type": "data"}
                        }
                    else:
                        chatbot_msg = {
                            "role": msg.role,
                            "content": msg.content,
                            "timestamp": msg.created_at.isoformat()
                        }

                        # 添加特殊格式化和元数据
                        if msg.message_type == MessageSubType.THINKING.value:
                            chatbot_msg["content"] = f"🧠 **AI思考过程**\n\n{msg.content}"
                            chatbot_msg["metadata"] = {"collapsible": True, "default_collapsed": True, "type": "thinking"}
                        elif msg.message_type == MessageSubType.TOOL_CALL.value:
                            chatbot_msg["metadata"] = {"collapsible": True, "default_collapsed": False, "type": "tool_call"}
                        elif msg.message_type == MessageSubType.TOOL_RESULT.value:
                            chatbot_msg["metadata"] = {"collapsible": True, "default_collapsed": False, "type": "tool_result"}

                    # 包含原始元数据（如果有的话）
                    if include_metadata and msg.metadata:
                        try:
                            original_metadata = json.loads(msg.metadata)
                            if "metadata" in chatbot_msg:
                                chatbot_msg["metadata"].update(original_metadata)
                            else:
                                chatbot_msg["metadata"] = original_metadata
                        except json.JSONDecodeError:
                            pass

                    chatbot_messages.append(chatbot_msg)

                return chatbot_messages

        except Exception as e:
            logger.error(f"获取对话消息失败: {e}")
            return []

    @staticmethod
    def get_latest_conversation_by_type(
        plan_id: int,
        conversation_type: ConversationType
    ) -> Optional[AgentConversation]:
        """获取指定类型的最新对话"""
        try:
            with get_db() as db:
                return db.query(AgentConversation).filter(
                    and_(
                        AgentConversation.plan_id == plan_id,
                        AgentConversation.conversation_type == conversation_type.value,
                        AgentConversation.status == 'active'
                    )
                ).order_by(desc(AgentConversation.last_message_at)).first()

        except Exception as e:
            logger.error(f"获取最新对话失败: {e}")
            return None

    @staticmethod
    def format_for_chatbot(messages: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """
        格式化消息为Chatbot格式

        Args:
            messages: 消息列表

        Returns:
            格式化后的消息列表
        """
        formatted_messages = []

        for msg in messages:
            formatted_msg = {
                "role": msg.get("role", "user"),
                "content": msg.get("content", "")
            }

            # 添加特殊样式
            if msg.get("metadata", {}).get("collapsible"):
                formatted_msg["metadata"] = msg["metadata"]

            formatted_messages.append(formatted_msg)

        return formatted_messages


# 全局实例
enhanced_conversation_service = EnhancedConversationService()