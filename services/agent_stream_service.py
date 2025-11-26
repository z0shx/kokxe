"""
AI Agent流式对话服务
专门用于Gradio chatbot的流式推理，支持thinking和工具调用
"""
import json
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, AsyncGenerator
from database.db import get_db
from database.models import TradingPlan, TrainingRecord, PredictionData, LLMConfig
from utils.logger import setup_logger
from services.agent_tools import get_all_tools
import gradio as gr

logger = setup_logger(__name__, "agent_stream_service.log")


class AgentStreamService:
    """AI Agent流式对话服务"""

    @classmethod
    async def chat_with_tools_stream(
        cls,
        message: str,
        history: List[Dict],
        plan_id: int,
        training_record_id: Optional[int] = None
    ) -> AsyncGenerator[str, None]:
        """
        流式对话，支持工具调用和thinking

        Args:
            message: 用户消息
            history: 对话历史
            plan_id: 交易计划ID
            training_record_id: 训练记录ID

        Yields:
            流式响应字符串（JSON格式）
        """
        try:
            # 获取计划信息
            with get_db() as db:
                plan = db.query(TradingPlan).filter(TradingPlan.id == plan_id).first()
                if not plan:
                    yield json.dumps({"type": "error", "content": "❌ 计划不存在"})
                    return

                # 获取LLM配置
                llm_config = None
                if plan.llm_config_id:
                    llm_config = db.query(LLMConfig).filter(
                        LLMConfig.id == plan.llm_config_id
                    ).first()

                if not llm_config:
                    yield json.dumps({"type": "error", "content": "❌ 未配置LLM"})
                    return

                # 获取训练记录和预测数据
                training_record = None
                prediction_data = []

                if training_record_id:
                    training_record = db.query(TrainingRecord).filter(
                        TrainingRecord.id == training_record_id
                    ).first()
                else:
                    # 获取最新的训练记录
                    training_record = db.query(TrainingRecord).filter(
                        TrainingRecord.plan_id == plan_id,
                        TrainingRecord.status == 'completed',
                        TrainingRecord.is_active == True
                    ).order_by(TrainingRecord.created_at.desc()).first()

                if training_record:
                    prediction_data = db.query(PredictionData).filter(
                        PredictionData.training_record_id == training_record.id
                    ).order_by(PredictionData.timestamp.desc()).limit(10).all()

            # 构建系统消息
            system_prompt = cls._build_system_prompt(plan, training_record, prediction_data)

            # 构建对话历史
            messages = []

            # 添加系统消息
            messages.append({"role": "system", "content": system_prompt})

            # 添加历史对话
            for msg in history:
                if isinstance(msg, dict):
                    messages.append({
                        "role": msg.get("role", "user"),
                        "content": msg.get("content", "")
                    })

            # 添加当前用户消息
            messages.append({"role": "user", "content": message})

            # 获取可用工具
            tools_config = plan.agent_tools_config or {}
            available_tools = {}
            enabled_tools = []

            for tool_name, tool_obj in get_all_tools().items():
                if tools_config.get(tool_name, False):
                    available_tools[tool_name] = tool_obj
                    # 转换为OpenAI工具格式
                    enabled_tools.append({
                        "type": "function",
                        "function": {
                            "name": tool_name,
                            "description": tool_obj.description,
                            "parameters": tool_obj.parameters
                        }
                    })

            # 发送开始思考消息
            yield json.dumps({
                "type": "thinking_start",
                "content": "🧠 正在思考..."
            })

            # 调用LLM进行流式推理
            if llm_config.provider == 'qwen':
                async for chunk in cls._stream_qwen_response(
                    llm_config, messages, enabled_tools, available_tools, plan_id
                ):
                    yield chunk
            elif llm_config.provider == 'openai':
                async for chunk in cls._stream_openai_response(
                    llm_config, messages, enabled_tools, available_tools, plan_id
                ):
                    yield chunk
            elif llm_config.provider == 'anthropic':
                async for chunk in cls._stream_claude_response(
                    llm_config, messages, enabled_tools, available_tools, plan_id
                ):
                    yield chunk
            else:
                yield json.dumps({
                    "type": "error",
                    "content": f"❌ 不支持的LLM提供商: {llm_config.provider}"
                })

        except Exception as e:
            logger.error(f"流式对话失败: {e}")
            yield json.dumps({"type": "error", "content": f"❌ 对话失败: {str(e)}"})

    @classmethod
    def _build_system_prompt(
        cls,
        plan: TradingPlan,
        training_record: Optional[TrainingRecord],
        prediction_data: List[PredictionData]
    ) -> str:
        """构建系统提示词"""

        # 基础系统提示
        system_prompt = f"""你是一个专业的加密货币交易AI助手，负责分析市场数据并做出交易决策。

**交易计划信息**:
- 交易对: {plan.inst_id}
- 时间周期: {plan.interval}
- 环境: {'🧪 模拟盘' if plan.is_demo else '💰 实盘'}
- 计划状态: {plan.status}

**推理任务**:
基于Kronos模型的预测数据，使用ReAct模式进行思考和决策：
1. **思考** (Thought): 分析市场状况和预测数据
2. **行动** (Action): 调用工具获取更多信息或执行交易
3. **观察** (Observation): 分析工具返回的结果
4. **重复** 直到得出最终结论

**可用工具**:
你可以调用以下工具来获取信息和执行操作：
"""

        # 添加工具说明
        tools_config = plan.agent_tools_config or {}
        for tool_name, tool_obj in get_all_tools().items():
            if tools_config.get(tool_name, False):
                description = tool_obj.description
                system_prompt += f"- {tool_name}: {description}\n"

        # 添加预测数据
        if prediction_data:
            latest_prediction = prediction_data[0]  # 最新的预测数据

            # 安全处理数据
            current_price = latest_prediction.close or 0
            upward_prob = latest_prediction.upward_probability or 0
            volatility_prob = latest_prediction.volatility_amplification_probability or 0
            close_min = latest_prediction.close_min or 0
            close_max = latest_prediction.close_max or 0

            prediction_info = f"""

**最新预测数据**:
- 当前价格: ${current_price:.4f}
- 预测区间: ${close_min:.4f} ~ ${close_max:.4f}
- 上涨概率: {upward_prob:.2%}
- 波动放大概率: {volatility_prob:.2%}
- 预测时间: {latest_prediction.timestamp.strftime('%Y-%m-%d %H:%M:%S')}
- 模型版本: {training_record.version if training_record else 'N/A'}
"""

            system_prompt += prediction_info

        # 添加自定义提示词
        if plan.agent_prompt:
            system_prompt += f"""

**额外指示**:
{plan.agent_prompt}
"""

        system_prompt += """

**重要提醒**:
- 始终谨慎决策，控制风险
- 在模拟盘环境中可以大胆尝试策略
- 所有交易操作都会被记录用于分析
- 使用限价单而非市价单以避免价格滑点

现在请基于以上信息进行分析和推理。"""

        return system_prompt

    @classmethod
    async def _stream_qwen_response(
        cls,
        llm_config: LLMConfig,
        messages: List[Dict],
        tools: List[Dict],
        available_tools: Dict,
        plan_id: int
    ) -> AsyncGenerator[str, None]:
        """流式调用通义千问"""
        try:
            import openai

            client = openai.AsyncOpenAI(
                api_key=llm_config.api_key,
                base_url=llm_config.api_base_url
            )

            response = await client.chat.completions.create(
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
                        "content": thinking_content
                    })

                # 处理正常内容
                if delta.content:
                    content += delta.content
                    yield json.dumps({
                        "type": "content",
                        "content": content
                    })

                # 处理工具调用
                if delta.tool_calls:
                    for tool_call in delta.tool_calls:
                        if not current_tool_call or current_tool_call.get("id") != tool_call.id:
                            if current_tool_call:
                                # 执行前一个工具调用
                                async for chunk in cls._execute_and_stream_tool(
                                    current_tool_call, available_tools, plan_id
                                ):
                                    yield chunk

                            current_tool_call = {
                                "id": tool_call.id,
                                "name": tool_call.function.name,
                                "arguments": tool_call.function.arguments or ""
                            }

                            yield json.dumps({
                                "type": "tool_call_start",
                                "tool_name": tool_call.function.name,
                                "tool_id": tool_call.id
                            })
                        else:
                            current_tool_call["arguments"] += tool_call.function.arguments or ""

            # 执行最后的工具调用
            if current_tool_call:
                async for chunk in cls._execute_and_stream_tool(current_tool_call, available_tools, plan_id):
                    yield chunk

        except Exception as e:
            logger.error(f"Qwen流式调用失败: {e}")
            yield json.dumps({"type": "error", "content": f"❌ 调用失败: {str(e)}"})

    @classmethod
    async def _stream_openai_response(
        cls,
        llm_config: LLMConfig,
        messages: List[Dict],
        tools: List[Dict],
        available_tools: Dict,
        plan_id: int
    ) -> AsyncGenerator[str, None]:
        """流式调用OpenAI"""
        try:
            import openai

            client = openai.AsyncOpenAI(api_key=llm_config.api_key)

            response = await client.chat.completions.create(
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
                        "content": content
                    })

                # 处理工具调用
                if delta.tool_calls:
                    for tool_call in delta.tool_calls:
                        if not current_tool_call or current_tool_call.get("index") != tool_call.index:
                            if current_tool_call:
                                async for chunk in cls._execute_and_stream_tool(
                                    current_tool_call, available_tools, plan_id
                                ):
                                    yield chunk

                            current_tool_call = {
                                "id": tool_call.id,
                                "name": tool_call.function.name,
                                "arguments": tool_call.function.arguments or "",
                                "index": tool_call.index
                            }

                            yield json.dumps({
                                "type": "tool_call_start",
                                "tool_name": tool_call.function.name,
                                "tool_id": tool_call.id
                            })
                        else:
                            current_tool_call["arguments"] += tool_call.function.arguments or ""

            # 执行最后的工具调用
            if current_tool_call:
                async for chunk in cls._execute_and_stream_tool(current_tool_call, available_tools, plan_id):
                    yield chunk

        except Exception as e:
            logger.error(f"OpenAI流式调用失败: {e}")
            yield json.dumps({"type": "error", "content": f"❌ 调用失败: {str(e)}"})

    @classmethod
    async def _stream_claude_response(
        cls,
        llm_config: LLMConfig,
        messages: List[Dict],
        tools: List[Dict],
        available_tools: Dict,
        plan_id: int
    ) -> AsyncGenerator[str, None]:
        """流式调用Claude"""
        try:
            import anthropic

            client = anthropic.AsyncAnthropic(api_key=llm_config.api_key)

            # 过滤出非系统消息
            non_system_messages = [msg for msg in messages if msg["role"] != "system"]
            system_content = " ".join([msg["content"] for msg in messages if msg["role"] == "system"])

            response = await client.messages.create(
                model=llm_config.model_name,
                max_tokens=llm_config.max_tokens,
                temperature=llm_config.temperature,
                system=system_content,
                messages=non_system_messages,
                tools=tools if tools else None,
                stream=True
            )

            content = ""
            current_tool_call = None

            async for chunk in response:
                if chunk.type == "content_block_delta":
                    if chunk.delta.type == "text_delta":
                        content += chunk.delta.text
                        yield json.dumps({
                            "type": "content",
                            "content": content
                        })

                elif chunk.type == "content_block_start":
                    if hasattr(chunk, 'content_block') and chunk.content_block.type == "tool_use":
                        tool_block = chunk.content_block
                        current_tool_call = {
                            "id": tool_block.id,
                            "name": tool_block.name,
                            "arguments": ""
                        }

                        yield json.dumps({
                            "type": "tool_call_start",
                            "tool_name": tool_block.name,
                            "tool_id": tool_block.id
                        })

                elif chunk.type == "content_block_delta" and current_tool_call:
                    if hasattr(chunk.delta, 'partial_json'):
                        current_tool_call["arguments"] += chunk.delta.partial_json

                elif chunk.type == "content_block_stop" and current_tool_call:
                    async for chunk in cls._execute_and_stream_tool(current_tool_call, available_tools, plan_id):
                        yield chunk
                    current_tool_call = None

        except Exception as e:
            logger.error(f"Claude流式调用失败: {e}")
            yield json.dumps({"type": "error", "content": f"❌ 调用失败: {str(e)}"})

    @classmethod
    async def _execute_and_stream_tool(
        cls,
        tool_call: Dict,
        available_tools: Dict,
        plan_id: int
    ) -> AsyncGenerator[str, None]:
        """执行工具并流式返回结果"""
        try:
            # 确保tool_name是字符串
            tool_name = str(tool_call.get("name", ""))
            if not tool_name:
                yield json.dumps({
                    "type": "tool_error",
                    "tool_name": "unknown",
                    "tool_id": tool_call.get("id", "unknown"),
                    "error": "工具名称为空"
                })
                return

            tool_id = tool_call.get("id", "")
            arguments_str = tool_call.get("arguments", "")

            # 解析参数
            try:
                arguments = json.loads(arguments_str) if arguments_str else {}
            except json.JSONDecodeError:
                arguments = {}
                yield json.dumps({
                    "type": "tool_error",
                    "tool_name": tool_name,
                    "tool_id": tool_id,
                    "error": f"工具参数解析失败: {arguments_str}"
                })

            yield json.dumps({
                "type": "tool_call_arguments",
                "tool_name": tool_name,
                "tool_id": tool_id,
                "arguments": arguments
            })

            # 执行工具
            from services.trading_tools import OKXTradingTools
            from database.models import TradingPlan
            from database.db import get_db

            # 获取计划信息以得到API凭据
            with get_db() as db:
                plan = db.query(TradingPlan).filter(TradingPlan.id == plan_id).first()
                if not plan:
                    yield json.dumps({
                        "type": "tool_error",
                        "tool_name": tool_name,
                        "tool_id": tool_id,
                        "error": "计划不存在，无法执行工具"
                    })
                    return

                trading_tools = OKXTradingTools(
                    api_key=plan.okx_api_key,
                    secret_key=plan.okx_secret_key,
                    passphrase=plan.okx_passphrase,
                    is_demo=plan.is_demo,
                    trading_limits=plan.trading_limits
                )
            tool_func = getattr(trading_tools, tool_name, None)

            if tool_func and callable(tool_func):
                # 过滤参数，只传递工具函数期望的参数
                import inspect
                try:
                    sig = inspect.signature(tool_func)
                    valid_params = {}
                    for param_name, param in sig.parameters.items():
                        if param_name in arguments:
                            valid_params[param_name] = arguments[param_name]
                        elif param.default == inspect.Parameter.empty and param.kind != inspect.Parameter.VAR_KEYWORD:
                            # 必需参数缺失
                            valid_params[param_name] = None

                    # 在线程池中执行同步工具
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(None, lambda: tool_func(**valid_params))
                except TypeError as te:
                    logger.error(f"工具参数错误: {te}")
                    result = {"success": False, "error": f"工具参数错误: {str(te)}"}
                except Exception as exec_error:
                    logger.error(f"工具执行错误: {exec_error}")
                    result = {"success": False, "error": f"工具执行错误: {str(exec_error)}"}
            else:
                result = {"success": False, "error": f"工具 '{tool_name}' 不存在或不可调用"}

            yield json.dumps({
                "type": "tool_result",
                "tool_name": tool_name,
                "tool_id": tool_id,
                "result": result
            })

        except Exception as e:
            logger.error(f"工具执行失败: {e}")
            yield json.dumps({
                "type": "tool_error",
                "tool_name": tool_call.get("name", "unknown"),
                "tool_id": tool_call.get("id", "unknown"),
                "error": str(e)
            })

  