"""
增强的AI Agent流式对话服务
重构版：支持thinking模式、ReAct累积展示、配置化系统提示词
"""
import json
import asyncio
from datetime import datetime
from typing import Dict, List, Optional, AsyncGenerator, Any
from database.models import TradingPlan, TrainingRecord, PredictionData, LLMConfig, AgentConversation, now_beijing
from database.db import get_db
from utils.logger import setup_logger
from services.agent_tools import get_all_tools
from services.enhanced_conversation_service import (
    enhanced_conversation_service, ConversationType, MessageSubType
)

logger = setup_logger(__name__, "enhanced_agent_stream.log")


class EnhancedAgentStreamService:
    """增强的AI Agent流式对话服务"""

    @classmethod
    async def initialize_conversation(
        cls,
        plan_id: int,
        conversation_type: ConversationType,
        reset_context: bool = True
    ) -> int:
        """
        初始化对话（添加系统提示词）

        Args:
            plan_id: 计划ID
            conversation_type: 对话类型
            reset_context: 是否重置上下文

        Returns:
            对话ID
        """
        try:
            # 获取计划信息
            with get_db() as db:
                plan = db.query(TradingPlan).filter(TradingPlan.id == plan_id).first()
                if not plan:
                    raise ValueError(f"计划不存在: {plan_id}")

                # 获取最新的训练记录
                latest_training = db.query(TrainingRecord).filter(
                    TrainingRecord.plan_id == plan_id,
                    TrainingRecord.status == 'completed',
                    TrainingRecord.is_active == True
                ).order_by(TrainingRecord.created_at.desc()).first()

            # 创建或获取对话
            conversation = enhanced_conversation_service.create_or_get_conversation(
                plan_id=plan_id,
                conversation_type=conversation_type,
                reset_context=reset_context,
                session_name=f"{conversation_type.value}_{now_beijing().strftime('%Y%m%d_%H%M%S')}"
            )

            # 获取系统提示词
            system_prompt = enhanced_conversation_service.get_system_prompt_content(plan)

            # 添加系统提示词消息
            template_id = getattr(plan, 'prompt_template_id', None)
            enhanced_conversation_service.add_system_prompt_message(
                conversation_id=conversation.id,
                content=system_prompt,
                template_id=template_id
            )

            logger.info(f"初始化对话完成: conversation_id={conversation.id}, type={conversation_type.value}")
            return conversation.id

        except Exception as e:
            logger.error(f"初始化对话失败: {e}")
            raise

    @classmethod
    async def add_prediction_data_message(
        cls,
        conversation_id: int,
        plan_id: int,
        trigger_event: str = "manual_inference"
    ) -> bool:
        """
        添加预测数据消息

        Args:
            conversation_id: 对话ID
            plan_id: 计划ID
            trigger_event: 触发事件

        Returns:
            是否成功
        """
        try:
            # 获取最新预测数据
            with get_db() as db:
                latest_training = db.query(TrainingRecord).filter(
                    TrainingRecord.plan_id == plan_id,
                    TrainingRecord.status == 'completed',
                    TrainingRecord.is_active == True
                ).order_by(TrainingRecord.created_at.desc()).first()

                if not latest_training:
                    logger.warning(f"计划 {plan_id} 没有可用的训练记录")
                    return False

                prediction_data = db.query(PredictionData).filter(
                    PredictionData.training_record_id == latest_training.id
                ).order_by(PredictionData.timestamp.desc()).limit(10).all()

                if not prediction_data:
                    logger.warning(f"计划 {plan_id} 没有预测数据")
                    return False

            # 添加K线数据消息
            enhanced_conversation_service.add_kline_data_message(
                conversation_id=conversation_id,
                prediction_data=prediction_data,
                trigger_event=trigger_event
            )

            logger.info(f"添加预测数据消息完成: conversation_id={conversation_id}, records={len(prediction_data)}")
            return True

        except Exception as e:
            logger.error(f"添加预测数据消息失败: {e}")
            return False

    @classmethod
    async def chat_with_tools_stream(
        cls,
        conversation_id: int,
        user_message: str = "",
        use_thinking_mode: bool = True
    ) -> AsyncGenerator[str, None]:
        """
        与工具进行流式对话

        Args:
            conversation_id: 对话ID
            user_message: 用户消息（可选，如果为空则使用上下文中的数据）
            use_thinking_mode: 是否使用thinking模式

        Yields:
            流式响应字符串（JSON格式）
        """
        try:
            # 获取对话信息
            with get_db() as db:
                conversation = db.query(AgentConversation).filter(
                    AgentConversation.id == conversation_id
                ).first()

                if not conversation:
                    yield json.dumps({"type": "error", "content": "对话不存在"})
                    return

                plan = db.query(TradingPlan).filter(TradingPlan.id == conversation.plan_id).first()
                if not plan:
                    yield json.dumps({"type": "error", "content": "计划不存在"})
                    return

                # 获取LLM配置
                llm_config = None
                if plan.llm_config_id:
                    llm_config = db.query(LLMConfig).filter(LLMConfig.id == plan.llm_config_id).first()

                if not llm_config:
                    yield json.dumps({"type": "error", "content": "未配置LLM"})
                    return

            # 获取对话历史
            conversation_history = enhanced_conversation_service.get_conversation_messages(conversation_id)

            # 构建LLM消息格式
            messages = []
            for msg in conversation_history:
                if msg["role"] in ["system", "user", "assistant"]:
                    messages.append({
                        "role": msg["role"],
                        "content": msg["content"]
                    })

            # 添加当前用户消息（如果有）
            if user_message:
                messages.append({"role": "user", "content": user_message})

            # 获取可用工具
            tools_config = plan.agent_tools_config or {}
            available_tools = {}
            enabled_tools = []

            for tool_name, tool_obj in get_all_tools().items():
                if tools_config.get(tool_name, False):
                    available_tools[tool_name] = tool_obj
                    enabled_tools.append({
                        "type": "function",
                        "function": {
                            "name": tool_name,
                            "description": tool_obj.description,
                            "parameters": tool_obj.parameters
                        }
                    })

            # 初始化LLM客户端
            llm_client = cls._get_llm_client(llm_config)

            if use_thinking_mode and cls._supports_thinking_mode(llm_config):
                # 使用thinking模式
                async for response_chunk in cls._stream_with_thinking_mode(
                    llm_client, messages, enabled_tools, plan
                ):
                    yield response_chunk
            else:
                # 使用普通模式
                async for response_chunk in cls._stream_normal_mode(
                    llm_client, messages, enabled_tools, plan
                ):
                    yield response_chunk

        except Exception as e:
            logger.error(f"流式对话失败: {e}")
            yield json.dumps({"type": "error", "content": f"对话失败: {str(e)}"})

    @classmethod
    def _get_llm_client(cls, llm_config):
        """获取LLM客户端"""
        try:
            if llm_config.provider == 'qwen':
                import openai
                return openai.AsyncOpenAI(
                    api_key=llm_config.api_key,
                    base_url=llm_config.api_base_url
                )
            elif llm_config.provider == 'openai':
                import openai
                return openai.AsyncOpenAI(
                    api_key=llm_config.api_key
                )
            elif llm_config.provider == 'anthropic':
                # 这里可以添加Claude客户端实现
                logger.warning(f"Anthropic客户端适配尚未完成")
                return None
            else:
                logger.error(f"不支持的LLM提供商: {llm_config.provider}")
                return None

        except Exception as e:
            logger.error(f"获取LLM客户端失败: {e}")
            raise

    @classmethod
    def _safe_parse_arguments(cls, arguments_str: str) -> Dict[str, Any]:
        """安全解析工具参数"""
        if not arguments_str or not arguments_str.strip():
            return {}

        try:
            return json.loads(arguments_str)
        except json.JSONDecodeError as e:
            logger.warning(f"工具参数JSON解析失败: {e}, 原始参数: {arguments_str}")
            return {}

    @staticmethod
    def _is_json_complete(json_str: str) -> bool:
        """检查JSON字符串是否完整"""
        if not json_str:
            return False

        json_str = json_str.strip()

        # 基本检查：开始和结束符号
        if not (json_str.startswith('{') and json_str.endswith('}')):
            return False

        # 大括号计数检查
        brace_count = 0
        in_string = False
        escape_next = False

        for char in json_str:
            if escape_next:
                escape_next = False
                continue

            if char == '\\':
                escape_next = True
                continue

            if char == '"' and not escape_next:
                in_string = not in_string
                continue

            if not in_string:
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        return True

        return False

    @classmethod
    def _supports_thinking_mode(cls, llm_config) -> bool:
        """检查是否支持thinking模式"""
        # Qwen 默认支持thinking模式
        # Claude 也支持thinking模式
        return llm_config.provider in ["qwen", "anthropic"]

    @classmethod
    async def _stream_with_thinking_mode(
        cls,
        llm_client,
        messages: List[Dict],
        tools: List[Dict],
        plan: TradingPlan
    ) -> AsyncGenerator[str, None]:
        """使用thinking模式的流式响应"""
        chunk_count = 0

        try:
            # 发送thinking开始事件
            yield json.dumps({
                "type": "thinking_start",
                "content": "开始思考分析...",
                "chunk_count": chunk_count
            })
            chunk_count += 1

            # 获取LLM配置
            with get_db() as db:
                llm_config = db.query(LLMConfig).filter(LLMConfig.id == plan.llm_config_id).first()

            # 调用LLM进行流式推理
            response = await llm_client.chat.completions.create(
                model=llm_config.model_name,
                messages=messages,
                tools=tools if tools else None,
                tool_choice="auto" if tools else None,
                temperature=llm_config.temperature,
                max_tokens=llm_config.max_tokens,
                stream=True
            )

            thinking_content = ""
            content = ""
            current_tool_call = None

            async for chunk in response:
                delta = chunk.choices[0].delta

                # 处理thinking（如果有）
                if hasattr(delta, 'reasoning_content') and delta.reasoning_content:
                    thinking_content += delta.reasoning_content
                    yield json.dumps({
                        "type": "thinking",
                        "content": thinking_content,
                        "chunk_count": chunk_count
                    })
                    chunk_count += 1

                # 处理正常内容
                if delta.content:
                    content += delta.content
                    yield json.dumps({
                        "type": "content",
                        "content": content,
                        "chunk_count": chunk_count
                    })
                    chunk_count += 1

                # 处理工具调用
                if delta.tool_calls:
                    for tool_call in delta.tool_calls:
                        if not current_tool_call or current_tool_call.get("id") != tool_call.id:
                            if current_tool_call:
                                # 等待前一个工具调用的参数完整
                                if cls._is_json_complete(current_tool_call["arguments"]):
                                    logger.info(f"工具调用参数完整，开始执行: {current_tool_call['name']}")
                                    try:
                                        async for result_chunk in cls._execute_tool_call_stream(
                                            current_tool_call["name"],
                                            cls._safe_parse_arguments(current_tool_call["arguments"]),
                                            plan.id
                                        ):
                                            yield json.dumps({
                                                "type": "tool_result",
                                                "tool_name": current_tool_call["name"],
                                                "result": result_chunk,
                                                "chunk_count": chunk_count
                                            })
                                            chunk_count += 1
                                    except Exception as e:
                                        logger.error(f"工具调用失败，继续对话: {current_tool_call['name']}, error: {e}")
                                        yield json.dumps({
                                            "type": "tool_result",
                                            "tool_name": current_tool_call["name"],
                                            "result": {
                                                "success": False,
                                                "error": f"工具调用失败: {str(e)}"
                                            },
                                            "chunk_count": chunk_count
                                        })
                                        chunk_count += 1
                                        # 工具调用失败，继续对话而不是抛出异常
                                else:
                                    logger.warning(f"工具调用参数不完整，跳过执行: {current_tool_call['name']}")

                            current_tool_call = {
                                "id": tool_call.id,
                                "name": tool_call.function.name,
                                "arguments": tool_call.function.arguments or ""
                            }

                            yield json.dumps({
                                "type": "tool_call_start",
                                "tool_name": tool_call.function.name,
                                "arguments": "",  # 不显示参数片段，避免混淆
                                "chunk_count": chunk_count
                            })
                            chunk_count += 1
                        else:
                            # 累积参数片段
                            current_tool_call["arguments"] += tool_call.function.arguments or ""
                            logger.debug(f"累积参数片段: {repr(tool_call.function.arguments)}")

            # 执行最后的工具调用
            if current_tool_call:
                # 等待参数完整
                if cls._is_json_complete(current_tool_call["arguments"]):
                    logger.info(f"最后工具调用参数完整，开始执行: {current_tool_call['name']}")
                    try:
                        async for result_chunk in cls._execute_tool_call_stream(
                            current_tool_call["name"],
                            cls._safe_parse_arguments(current_tool_call["arguments"]),
                            plan.id
                        ):
                            yield json.dumps({
                                "type": "tool_result",
                                "tool_name": current_tool_call["name"],
                                "result": result_chunk,
                                "chunk_count": chunk_count
                            })
                            chunk_count += 1
                    except Exception as e:
                        logger.error(f"最后工具调用失败，继续对话: {current_tool_call['name']}, error: {e}")
                        yield json.dumps({
                            "type": "tool_result",
                            "tool_name": current_tool_call["name"],
                            "result": {
                                "success": False,
                                "error": f"工具调用失败: {str(e)}"
                            },
                            "chunk_count": chunk_count
                        })
                        chunk_count += 1
                        # 工具调用失败，继续对话而不是抛出异常
                else:
                    logger.warning(f"最后工具调用参数不完整，跳过执行: {current_tool_call['name']}")

        except Exception as e:
            logger.error(f"thinking模式流式响应失败: {e}")
            yield json.dumps({
                "type": "error",
                "content": f"thinking模式失败: {str(e)}",
                "chunk_count": chunk_count
            })

    @classmethod
    async def _stream_normal_mode(
        cls,
        llm_client,
        messages: List[Dict],
        tools: List[Dict],
        plan: TradingPlan
    ) -> AsyncGenerator[str, None]:
        """普通模式的流式响应"""
        chunk_count = 0

        try:
            # 获取LLM配置
            with get_db() as db:
                llm_config = db.query(LLMConfig).filter(LLMConfig.id == plan.llm_config_id).first()

            # 调用LLM进行流式推理
            response = await llm_client.chat.completions.create(
                model=llm_config.model_name,
                messages=messages,
                tools=tools if tools else None,
                tool_choice="auto" if tools else None,
                temperature=llm_config.temperature,
                max_tokens=llm_config.max_tokens,
                stream=True
            )

            content = ""
            current_tool_call = None

            async for chunk in response:
                delta = chunk.choices[0].delta

                # 处理正常内容
                if delta.content:
                    content += delta.content
                    yield json.dumps({
                        "type": "content",
                        "content": content,
                        "chunk_count": chunk_count
                    })
                    chunk_count += 1

                # 处理工具调用
                if delta.tool_calls:
                    for tool_call in delta.tool_calls:
                        if not current_tool_call or current_tool_call.get("id") != tool_call.id:
                            if current_tool_call:
                                # 等待前一个工具调用的参数完整
                                if cls._is_json_complete(current_tool_call["arguments"]):
                                    logger.info(f"工具调用参数完整，开始执行: {current_tool_call['name']}")
                                    try:
                                        async for result_chunk in cls._execute_tool_call_stream(
                                            current_tool_call["name"],
                                            cls._safe_parse_arguments(current_tool_call["arguments"]),
                                            plan.id
                                        ):
                                            yield json.dumps({
                                                "type": "tool_result",
                                                "tool_name": current_tool_call["name"],
                                                "result": result_chunk,
                                                "chunk_count": chunk_count
                                            })
                                            chunk_count += 1
                                    except Exception as e:
                                        logger.error(f"工具调用失败，继续对话: {current_tool_call['name']}, error: {e}")
                                        yield json.dumps({
                                            "type": "tool_result",
                                            "tool_name": current_tool_call["name"],
                                            "result": {
                                                "success": False,
                                                "error": f"工具调用失败: {str(e)}"
                                            },
                                            "chunk_count": chunk_count
                                        })
                                        chunk_count += 1
                                        # 工具调用失败，继续对话而不是抛出异常
                                else:
                                    logger.warning(f"工具调用参数不完整，跳过执行: {current_tool_call['name']}")

                            current_tool_call = {
                                "id": tool_call.id,
                                "name": tool_call.function.name,
                                "arguments": tool_call.function.arguments or ""
                            }

                            yield json.dumps({
                                "type": "tool_call_start",
                                "tool_name": tool_call.function.name,
                                "arguments": "",  # 不显示参数片段，避免混淆
                                "chunk_count": chunk_count
                            })
                            chunk_count += 1
                        else:
                            # 累积参数片段
                            current_tool_call["arguments"] += tool_call.function.arguments or ""
                            logger.debug(f"累积参数片段: {repr(tool_call.function.arguments)}")

            # 执行最后的工具调用
            if current_tool_call:
                # 等待参数完整
                if cls._is_json_complete(current_tool_call["arguments"]):
                    logger.info(f"最后工具调用参数完整，开始执行: {current_tool_call['name']}")
                    try:
                        async for result_chunk in cls._execute_tool_call_stream(
                            current_tool_call["name"],
                            cls._safe_parse_arguments(current_tool_call["arguments"]),
                            plan.id
                        ):
                            yield json.dumps({
                                "type": "tool_result",
                                "tool_name": current_tool_call["name"],
                                "result": result_chunk,
                                "chunk_count": chunk_count
                            })
                            chunk_count += 1
                    except Exception as e:
                        logger.error(f"最后工具调用失败，继续对话: {current_tool_call['name']}, error: {e}")
                        yield json.dumps({
                            "type": "tool_result",
                            "tool_name": current_tool_call["name"],
                            "result": {
                                "success": False,
                                "error": f"工具调用失败: {str(e)}"
                            },
                            "chunk_count": chunk_count
                        })
                        chunk_count += 1
                        # 工具调用失败，继续对话而不是抛出异常
                else:
                    logger.warning(f"最后工具调用参数不完整，跳过执行: {current_tool_call['name']}")

        except Exception as e:
            logger.error(f"普通模式流式响应失败: {e}")
            yield json.dumps({
                "type": "error",
                "content": f"响应失败: {str(e)}",
                "chunk_count": chunk_count
            })

    @classmethod
    async def execute_tool_call(
        cls,
        tool_name: str,
        arguments: Dict[str, Any],
        plan_id: int
    ) -> Dict[str, Any]:
        """
        执行工具调用

        Args:
            tool_name: 工具名称
            arguments: 工具参数
            plan_id: 计划ID

        Returns:
            工具执行结果
        """
        try:
            tools = get_all_tools()
            if tool_name not in tools:
                return {
                    "success": False,
                    "error": f"工具不存在: {tool_name}"
                }

            tool = tools[tool_name]

            # 添加plan_id到参数中
            arguments["plan_id"] = plan_id

            # 执行工具
            result = await tool.execute(**arguments)

            return {
                "success": True,
                "result": result
            }

        except Exception as e:
            logger.error(f"工具调用失败: {tool_name}, error: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    @classmethod
    async def _execute_tool_call_stream(
        cls,
        tool_name: str,
        arguments: Dict[str, Any],
        plan_id: int
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        执行工具调用并流式返回结果

        Args:
            tool_name: 工具名称
            arguments: 工具参数
            plan_id: 计划ID

        Yields:
            工具执行结果片段
        """
        try:
            from services.trading_tools import OKXTradingTools
            from database.models import TradingPlan

            # 检查工具是否存在
            tools = get_all_tools()
            if tool_name not in tools:
                yield {
                    "success": False,
                    "error": f"工具不存在: {tool_name}"
                }
                return

            # 获取计划信息以得到API凭据
            with get_db() as db:
                plan = db.query(TradingPlan).filter(TradingPlan.id == plan_id).first()
                if not plan:
                    yield {
                        "success": False,
                        "error": "计划不存在，无法执行工具"
                    }
                    return

                # 创建交易工具实例
                trading_tools = OKXTradingTools(
                    api_key=plan.okx_api_key,
                    secret_key=plan.okx_secret_key,
                    passphrase=plan.okx_passphrase,
                    is_demo=plan.is_demo
                )

            # 添加plan_id到参数中
            arguments["plan_id"] = plan_id

            # 根据工具名称执行相应的操作
            result = None
            if tool_name == "get_current_price":
                result = trading_tools.get_current_price(
                    inst_id=arguments.get("inst_id")
                )
            elif tool_name == "query_historical_kline_data":
                result = trading_tools.query_historical_kline_data(
                    inst_id=arguments.get("inst_id"),
                    interval=arguments.get("interval"),
                    start_time=arguments.get("start_time"),
                    end_time=arguments.get("end_time"),
                    limit=arguments.get("limit")
                )
            elif tool_name == "place_order":
                result = trading_tools.place_order(
                    inst_id=arguments.get("inst_id"),
                    side=arguments.get("side"),
                    order_type=arguments.get("order_type"),
                    size=arguments.get("size"),
                    price=arguments.get("price")
                )
            else:
                # 对于其他工具，使用通用调用方式
                tool = tools[tool_name]
                if hasattr(tool, 'execute'):
                    result = await tool.execute(**arguments)
                else:
                    yield {
                        "success": False,
                        "error": f"工具 {tool_name} 不支持执行"
                    }
                    return

            yield {
                "success": True,
                "result": result
            }

        except Exception as e:
            logger.error(f"工具调用失败: {tool_name}, error: {e}")
            yield {
                "success": False,
                "error": str(e)
            }


# 全局实例
enhanced_agent_stream_service = EnhancedAgentStreamService()