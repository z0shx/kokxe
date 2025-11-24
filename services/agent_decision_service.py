"""
AI Agent 决策服务
负责基于预测数据进行智能决策
"""
import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from database.db import get_db
from database.models import TradingPlan, TrainingRecord, PredictionData, AgentDecision, LLMConfig
from utils.logger import setup_logger
from sqlalchemy import desc
from config import config

logger = setup_logger(__name__, "agent_decision_service.log")


class AgentDecisionService:
    """AI Agent决策服务"""

    @classmethod
    async def trigger_decision(cls, plan_id: int, training_id: int) -> Optional[int]:
        """
        触发AI Agent决策

        Args:
            plan_id: 计划ID
            training_id: 训练记录ID

        Returns:
            决策记录ID，失败返回None
        """
        try:
            logger.info(f"触发AI Agent决策: plan_id={plan_id}, training_id={training_id}")

            # 在线程池中执行决策（避免阻塞）
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                cls._make_decision_sync,
                plan_id,
                training_id
            )

            if result['success']:
                logger.info(f"✅ 决策完成: decision_id={result['decision_id']}")
                return result['decision_id']
            else:
                logger.error(f"❌ 决策失败: {result.get('error')}")
                return None

        except Exception as e:
            logger.error(f"触发决策失败: {e}")
            import traceback
            traceback.print_exc()
            return None

    @classmethod
    def _make_decision_sync(cls, plan_id: int, training_id: int) -> Dict:
        """
        同步执行决策（在线程池中运行）

        Returns:
            结果字典: {
                'success': bool,
                'decision_id': int,
                'error': str
            }
        """
        try:
            with get_db() as db:
                # 获取计划信息
                plan = db.query(TradingPlan).filter(TradingPlan.id == plan_id).first()
                if not plan:
                    return {'success': False, 'error': '计划不存在'}

                # 获取训练记录
                training_record = db.query(TrainingRecord).filter(
                    TrainingRecord.id == training_id
                ).first()
                if not training_record:
                    return {'success': False, 'error': '训练记录不存在'}

                # 获取预测数据
                predictions = db.query(PredictionData).filter(
                    PredictionData.training_record_id == training_id
                ).order_by(PredictionData.timestamp).all()

                # 如果指定的训练记录没有预测数据，查找最新的可用预测数据
                if not predictions:
                    logger.info(f"训练记录 {training_id} 没有预测数据，查找最新的可用预测数据...")

                    # 查找该计划最新的预测数据
                    latest_prediction = db.query(PredictionData).join(
                        TrainingRecord, PredictionData.training_record_id == TrainingRecord.id
                    ).filter(
                        TrainingRecord.plan_id == plan_id,
                        TrainingRecord.status == 'completed'
                    ).order_by(PredictionData.timestamp.desc()).first()

                    if not latest_prediction:
                        return {'success': False, 'error': '无预测数据，请先执行推理生成预测数据'}

                    # 获取该批次的所有预测数据
                    predictions = db.query(PredictionData).filter(
                        PredictionData.training_record_id == latest_prediction.training_record_id
                    ).order_by(PredictionData.timestamp).all()

                    logger.info(f"找到最新预测数据批次: training_record_id={latest_prediction.training_record_id}, 预测数量={len(predictions)}")

                if not predictions:
                    return {'success': False, 'error': '无预测数据'}

                # 获取历史K线数据（用于上下文）
                from services.database_dataset import get_kline_dataframe
                from datetime import timedelta

                hist_end = datetime.now()
                hist_start = hist_end - timedelta(days=7)  # 最近7天数据
                historical_df = get_kline_dataframe(
                    inst_id=plan.inst_id,
                    interval=plan.interval,
                    start_time=hist_start,
                    end_time=hist_end
                )

                # 获取LLM配置
                llm_config = None
                if plan.llm_config_id:
                    llm_config = db.query(LLMConfig).filter(
                        LLMConfig.id == plan.llm_config_id
                    ).first()

                # 构建决策上下文
                decision_context = cls._build_decision_context(
                    plan=plan,
                    predictions=predictions,
                    historical_df=historical_df
                )

                logger.info(f"决策上下文: {len(historical_df) if historical_df is not None else 0}条历史数据, {len(predictions) if predictions is not None else 0}条预测数据")

                # 调用LLM进行决策
                llm_response = cls._call_llm_for_decision(
                    llm_config=llm_config,
                    context=decision_context,
                    agent_prompt=plan.agent_prompt
                )

                # 先创建决策记录（用于关联工具确认）
                decision = AgentDecision(
                    plan_id=plan_id,
                    training_record_id=training_id,
                    decision_time=datetime.utcnow(),
                    decision_type=llm_response.get('decision_type', 'analysis'),
                    reasoning=llm_response.get('reasoning', ''),
                    llm_model=llm_config.model_name if llm_config else 'default',
                    llm_input=decision_context,
                    llm_output=json.dumps(llm_response, ensure_ascii=False),  # 转换为JSON字符串
                    tool_calls=[],  # 稍后填充
                    tool_results=[],  # 稍后填充
                    order_ids=[],  # 从tool_results中提取
                    status='processing'  # 初始状态为处理中
                )

                db.add(decision)
                db.commit()
                db.refresh(decision)

                logger.info(f"决策记录已创建: decision_id={decision.id}")

                # 执行工具调用（同步版本）
                tool_calls, tool_results = cls._execute_tools_sync(
                    plan=plan,
                    llm_response=llm_response,
                    agent_decision_id=decision.id
                )

                # 更新决策记录
                decision.tool_calls = tool_calls
                decision.tool_results = tool_results
                decision.status = 'completed' if tool_results else 'no_action'
                db.commit()

                logger.info(
                    f"决策记录已保存: decision_id={decision.id}, "
                    f"type={decision.decision_type}, tools={len(tool_calls)}"
                )

                return {
                    'success': True,
                    'decision_id': decision.id
                }

        except Exception as e:
            logger.error(f"决策执行失败: {e}")
            import traceback
            traceback.print_exc()
            return {
                'success': False,
                'error': str(e)
            }

    @classmethod
    def _build_decision_context(
        cls,
        plan: TradingPlan,
        predictions: List[PredictionData],
        historical_df
    ) -> Dict:
        """
        构建决策上下文

        Args:
            plan: 交易计划
            predictions: 预测数据列表
            historical_df: 历史数据DataFrame

        Returns:
            上下文字典
        """
        import pandas as pd

        # 最近N条历史数据
        recent_history = []
        if historical_df is not None and len(historical_df) > 0:
            for _, row in historical_df.tail(20).iterrows():
                record = {}
                for col in row.index:
                    value = row[col]
                    # 处理 Timestamp 类型
                    if isinstance(value, pd.Timestamp):
                        record[col] = value.isoformat()
                    # 处理 numpy/pandas 数值类型
                    elif hasattr(value, 'item'):
                        record[col] = value.item()
                    else:
                        record[col] = value
                recent_history.append(record)

        # 预测数据
        pred_data = []
        for pred in predictions:
            pred_data.append({
                'timestamp': pred.timestamp.isoformat(),
                'open': float(pred.open),
                'high': float(pred.high),
                'low': float(pred.low),
                'close': float(pred.close),
                'volume': float(pred.volume)
            })

        # 市场分析
        if historical_df is not None and len(historical_df) > 0:
            last_close = float(historical_df['close'].iloc[-1])
            pred_first_close = float(predictions[0].close) if predictions else last_close
            price_change_pct = ((pred_first_close - last_close) / last_close) * 100
        else:
            price_change_pct = 0.0

        context = {
            'trading_pair': plan.inst_id,
            'interval': plan.interval,
            'current_time': datetime.now().isoformat(),
            'recent_history': recent_history,
            'predictions': pred_data,
            'market_analysis': {
                'predicted_price_change_pct': round(price_change_pct, 2)
            },
            'trading_limits': plan.trading_limits
        }

        return context

    @classmethod
    async def _call_llm_for_conversation(
        cls,
        llm_config: LLMConfig,
        context: Dict,
        conversation_prompt: str
    ) -> Dict:
        """
        调用LLM进行对话（非决策模式）

        Args:
            llm_config: LLM配置
            context: 对话上下文，包含用户消息和历史记录
            conversation_prompt: 对话专用的提示词

        Returns:
            LLM响应字典，包含success和content字段
        """
        try:
            if not llm_config:
                return {
                    'success': False,
                    'error': '未配置LLM'
                }

            # 构建对话消息
            messages = []

            # 添加系统消息
            system_message = conversation_prompt
            if context.get('prediction_data'):
                system_message += f"\n\n{context['prediction_data']}"

            messages.append({"role": "system", "content": system_message})

            # 添加历史对话
            chat_history = context.get('chat_history', [])
            for msg in chat_history:
                if msg.get('role') in ['user', 'assistant']:
                    messages.append({
                        "role": msg['role'],
                        "content": msg['content']
                    })

            # 添加当前用户消息
            user_message = context.get('user_message', '')
            if user_message:
                messages.append({"role": "user", "content": user_message})

            # 根据模型类型调用不同的API
            if llm_config.provider == 'anthropic':
                return await cls._call_claude_conversation(llm_config, messages)
            elif llm_config.provider == 'openai':
                return await cls._call_openai_conversation(llm_config, messages)
            elif llm_config.provider == 'qwen':
                return await cls._call_qwen_conversation(llm_config, messages)
            elif llm_config.provider == 'ollama':
                return await cls._call_ollama_conversation(llm_config, messages)
            else:
                return {
                    'success': False,
                    'error': f'不支持的LLM提供商: {llm_config.provider}'
                }

        except Exception as e:
            logger.error(f"调用LLM对话失败: {e}")
            return {
                'success': False,
                'error': f'调用LLM失败: {str(e)}'
            }

    @classmethod
    def _call_llm_for_decision(
        cls,
        llm_config: Optional[LLMConfig],
        context: Dict,
        agent_prompt: str
    ) -> Dict:
        """
        调用LLM进行决策

        Args:
            llm_config: LLM配置
            context: 决策上下文
            agent_prompt: Agent提示词

        Returns:
            LLM响应字典
        """
        try:
            # 如果没有配置LLM，使用规则决策
            if not llm_config:
                logger.warning("未配置LLM，使用规则决策")
                return cls._rule_based_decision(context)

            # 根据模型类型调用不同的API
            if llm_config.provider == 'anthropic':
                return cls._call_claude(llm_config, context, agent_prompt)
            elif llm_config.provider == 'openai':
                return cls._call_openai(llm_config, context, agent_prompt)
            elif llm_config.provider == 'qwen':
                return cls._call_qwen(llm_config, context, agent_prompt)
            elif llm_config.provider == 'ollama':
                return cls._call_ollama(llm_config, context, agent_prompt)
            else:
                logger.warning(f"不支持的LLM提供商: {llm_config.provider}")
                return cls._rule_based_decision(context)

        except Exception as e:
            logger.error(f"调用LLM失败: {e}")
            # 降级到规则决策
            return cls._rule_based_decision(context)

    @classmethod
    def _rule_based_decision(cls, context: Dict) -> Dict:
        """
        基于规则的决策（当LLM不可用时）

        Args:
            context: 决策上下文

        Returns:
            决策结果
        """
        price_change = context['market_analysis']['predicted_price_change_pct']

        # 简单的规则：涨幅>2%买入，跌幅>2%卖出
        if price_change > 2.0:
            decision_type = 'buy'
            reasoning = f"预测价格上涨 {price_change:.2f}%，建议买入"
        elif price_change < -2.0:
            decision_type = 'sell'
            reasoning = f"预测价格下跌 {price_change:.2f}%，建议卖出"
        else:
            decision_type = 'hold'
            reasoning = f"预测价格变化 {price_change:.2f}%，建议持有"

        return {
            'decision_type': decision_type,
            'reasoning': reasoning,
            'confidence': 0.7,
            'tool_calls': []
        }

    @classmethod
    def _call_claude(cls, llm_config: LLMConfig, context: Dict, agent_prompt: str) -> Dict:
        """调用Claude API"""
        try:
            import anthropic
            import httpx

            # 配置代理
            http_client = None
            if config.PROXY_ENABLED and config.PROXY_URL:
                http_client = httpx.Client(
                    proxy=config.PROXY_URL,
                    timeout=60.0
                )
                logger.info(f"使用代理调用Claude API: {config.PROXY_URL}")

            client = anthropic.Anthropic(
                api_key=llm_config.api_key,
                http_client=http_client
            )

            # 构建消息
            system_prompt = agent_prompt or "你是一个专业的加密货币交易助手，基于历史数据和预测结果提供交易建议。"

            # 格式化上下文
            context_str = cls._format_context_for_llm(context)

            # 定义工具
            tools = [
                {
                    "name": "place_order",
                    "description": "下单（买入或卖出）",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "side": {
                                "type": "string",
                                "enum": ["buy", "sell"],
                                "description": "交易方向"
                            },
                            "size": {
                                "type": "number",
                                "description": "交易数量"
                            },
                            "order_type": {
                                "type": "string",
                                "enum": ["market", "limit"],
                                "description": "订单类型"
                            },
                            "price": {
                                "type": "number",
                                "description": "限价单价格（市价单不需要）"
                            }
                        },
                        "required": ["side", "size", "order_type"]
                    }
                }
            ]

            # 调用API
            response = client.messages.create(
                model=llm_config.model_name,
                max_tokens=llm_config.max_tokens,
                temperature=llm_config.temperature,
                system=system_prompt,
                messages=[{
                    "role": "user",
                    "content": f"请分析以下市场数据并提供交易建议：\n\n{context_str}"
                }],
                tools=tools
            )

            # 解析响应
            return cls._parse_claude_response(response)

        except ImportError:
            logger.warning("anthropic库未安装，降级到规则决策")
            return cls._rule_based_decision(context)
        except Exception as e:
            logger.error(f"调用Claude API失败: {e}")
            return cls._rule_based_decision(context)

    @classmethod
    def _call_openai(cls, llm_config: LLMConfig, context: Dict, agent_prompt: str) -> Dict:
        """调用OpenAI API"""
        try:
            import openai
            import httpx

            # 配置代理
            http_client = None
            if config.PROXY_ENABLED and config.PROXY_URL:
                http_client = httpx.Client(
                    proxy=config.PROXY_URL,
                    timeout=60.0
                )
                logger.info(f"使用代理调用OpenAI API: {config.PROXY_URL}")

            client = openai.OpenAI(
                api_key=llm_config.api_key,
                http_client=http_client
            )

            # 构建消息
            system_prompt = agent_prompt or "你是一个专业的加密货币交易助手，基于历史数据和预测结果提供交易建议。"
            context_str = cls._format_context_for_llm(context)

            # 定义工具
            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "place_order",
                        "description": "下单（买入或卖出）",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "side": {
                                    "type": "string",
                                    "enum": ["buy", "sell"],
                                    "description": "交易方向"
                                },
                                "size": {
                                    "type": "number",
                                    "description": "交易数量"
                                },
                                "order_type": {
                                    "type": "string",
                                    "enum": ["market", "limit"],
                                    "description": "订单类型"
                                },
                                "price": {
                                    "type": "number",
                                    "description": "限价单价格（市价单不需要）"
                                }
                            },
                            "required": ["side", "size", "order_type"]
                        }
                    }
                }
            ]

            # 调用API
            response = client.chat.completions.create(
                model=llm_config.model_name,
                max_tokens=llm_config.max_tokens,
                temperature=llm_config.temperature,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"请分析以下市场数据并提供交易建议：\n\n{context_str}"}
                ],
                tools=tools,
                extra_body={"enable_thinking": False}  # Qwen 非流式调用必须禁用思考
            )

            # 解析响应
            return cls._parse_openai_response(response)

        except ImportError:
            logger.warning("openai库未安装，降级到规则决策")
            return cls._rule_based_decision(context)
        except Exception as e:
            logger.error(f"调用OpenAI API失败: {e}")
            return cls._rule_based_decision(context)

    @classmethod
    def _call_qwen(cls, llm_config: LLMConfig, context: Dict, agent_prompt: str) -> Dict:
        """调用Qwen API（通兼容OpenAI格式）"""
        try:
            import openai
            import httpx

            # 配置代理
            http_client = None
            if config.PROXY_ENABLED and config.PROXY_URL:
                http_client = httpx.Client(
                    proxy=config.PROXY_URL,
                    timeout=60.0
                )
                logger.info(f"使用代理调用Qwen API: {config.PROXY_URL}")

            # Qwen使用OpenAI兼容的API
            client = openai.OpenAI(
                api_key=llm_config.api_key,
                base_url=llm_config.api_base_url or "https://dashscope.aliyuncs.com/compatible-mode/v1",
                http_client=http_client
            )

            system_prompt = agent_prompt or "你是一个专业的加密货币交易助手，基于历史数据和预测结果提供交易建议。"
            context_str = cls._format_context_for_llm(context)

            # 定义工具
            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "place_order",
                        "description": "下单（买入或卖出）",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "side": {
                                    "type": "string",
                                    "enum": ["buy", "sell"],
                                    "description": "交易方向"
                                },
                                "size": {
                                    "type": "number",
                                    "description": "交易数量"
                                },
                                "order_type": {
                                    "type": "string",
                                    "enum": ["market", "limit"],
                                    "description": "订单类型"
                                },
                                "price": {
                                    "type": "number",
                                    "description": "限价单价格"
                                }
                            },
                            "required": ["side", "size", "order_type"]
                        }
                    }
                }
            ]

            # 调用API（Qwen 要求非流式调用时禁用 thinking）
            response = client.chat.completions.create(
                model=llm_config.model_name,
                max_tokens=llm_config.max_tokens,
                temperature=llm_config.temperature,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"请分析以下市场数据并提供交易建议：\n\n{context_str}"}
                ],
                tools=tools,
                extra_body={"enable_thinking": False}  # Qwen 非流式调用必须禁用思考
            )

            # 解析响应
            return cls._parse_openai_response(response)

        except ImportError:
            logger.warning("openai库未安装，降级到规则决策")
            return cls._rule_based_decision(context)
        except Exception as e:
            logger.error(f"调用Qwen API失败: {e}")
            return cls._rule_based_decision(context)

    @classmethod
    def _call_ollama(cls, llm_config: LLMConfig, context: Dict, agent_prompt: str) -> Dict:
        """调用Ollama API（使用OpenAI兼容格式）"""
        try:
            import openai
            import httpx

            # Ollama使用OpenAI兼容的API，默认端点是 http://localhost:11434/v1
            base_url = llm_config.api_base_url or "http://localhost:11434/v1"

            # 配置代理（如果使用远程Ollama服务）
            http_client = None
            if config.PROXY_ENABLED and config.PROXY_URL and not base_url.startswith("http://localhost"):
                http_client = httpx.Client(
                    proxy=config.PROXY_URL,
                    timeout=60.0
                )
                logger.info(f"使用代理调用Ollama API: {config.PROXY_URL}")

            client = openai.OpenAI(
                api_key=llm_config.api_key or "ollama",  # Ollama不需要真实的API key
                base_url=base_url,
                http_client=http_client
            )

            system_prompt = agent_prompt or "你是一个专业的加密货币交易助手，基于历史数据和预测结果提供交易建议。"
            context_str = cls._format_context_for_llm(context)

            # 定义工具
            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "place_order",
                        "description": "下单（买入或卖出）",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "side": {
                                    "type": "string",
                                    "enum": ["buy", "sell"],
                                    "description": "交易方向"
                                },
                                "size": {
                                    "type": "number",
                                    "description": "交易数量"
                                },
                                "order_type": {
                                    "type": "string",
                                    "enum": ["market", "limit"],
                                    "description": "订单类型"
                                },
                                "price": {
                                    "type": "number",
                                    "description": "限价单价格"
                                }
                            },
                            "required": ["side", "size", "order_type"]
                        }
                    }
                }
            ]

            # 调用API
            response = client.chat.completions.create(
                model=llm_config.model_name,
                max_tokens=llm_config.max_tokens,
                temperature=llm_config.temperature,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"请分析以下市场数据并提供交易建议：\n\n{context_str}"}
                ],
                tools=tools,
                extra_body={"enable_thinking": False}  # Qwen 非流式调用必须禁用思考
            )

            # 解析响应（使用与OpenAI相同的解析方法）
            return cls._parse_openai_response(response)

        except ImportError:
            logger.warning("openai库未安装，降级到规则决策")
            return cls._rule_based_decision(context)
        except Exception as e:
            logger.error(f"调用Ollama API失败: {e}")
            import traceback
            traceback.print_exc()
            return cls._rule_based_decision(context)

    @classmethod
    def _format_context_for_llm(cls, context: Dict) -> str:
        """
        格式化上下文为LLM可读格式

        Args:
            context: 决策上下文

        Returns:
            格式化后的字符串
        """
        lines = []
        lines.append(f"交易对: {context['trading_pair']}")
        lines.append(f"时间周期: {context['interval']}")
        lines.append(f"当前时间: {context['current_time']}")
        lines.append("")

        # 最近历史数据
        lines.append("最近历史数据（最后5条）:")
        recent = context['recent_history'][-5:] if context['recent_history'] else []
        for item in recent:
            ts = item.get('timestamps', 'N/A')
            close = item.get('close', 0)
            lines.append(f"  - {ts}: 收盘价 {close:.2f}")
        lines.append("")

        # 预测数据
        lines.append(f"AI预测数据（未来{len(context['predictions'])}个周期）:")
        for pred in context['predictions'][:5]:  # 只显示前5条
            ts = pred['timestamp']
            close = pred['close']
            lines.append(f"  - {ts}: 预测收盘价 {close:.2f}")
        lines.append("")

        # 市场分析
        analysis = context['market_analysis']
        price_change = analysis['predicted_price_change_pct']
        lines.append(f"市场分析:")
        lines.append(f"  - 预测价格变化: {price_change:+.2f}%")
        lines.append("")

        # 交易限制
        limits = context['trading_limits']
        lines.append(f"交易限制:")
        lines.append(f"  - 可用账户资金 (USDT): {limits.get('available_usdt_amount', 'N/A')}")
        lines.append(f"  - 可用账户资金比例: {limits.get('available_usdt_percentage', 'N/A')}%")
        lines.append(f"  - 平摊操作单量: {limits.get('avg_order_count', 'N/A')} 笔")
        lines.append(f"  - 止损比例: {limits.get('stop_loss_percentage', 'N/A')}%")
        lines.append(f"  - 最大持仓: {limits.get('max_position_size', 'N/A')}")
        lines.append(f"  - 最大订单金额: {limits.get('max_order_amount', 'N/A')}")

        return "\n".join(lines)

    @classmethod
    def _parse_claude_response(cls, response) -> Dict:
        """
        解析Claude API响应

        Args:
            response: Claude API响应对象

        Returns:
            决策结果字典
        """
        try:
            # 提取文本内容
            reasoning = ""
            tool_calls = []

            for content_block in response.content:
                if content_block.type == "text":
                    reasoning = content_block.text
                elif content_block.type == "tool_use":
                    tool_calls.append({
                        "name": content_block.name,
                        "arguments": content_block.input
                    })

            # 判断决策类型
            if tool_calls:
                first_tool = tool_calls[0]
                if first_tool['name'] == 'place_order':
                    side = first_tool['arguments'].get('side', 'hold')
                    decision_type = side
                else:
                    decision_type = 'analysis'
            else:
                decision_type = 'hold'

            return {
                'decision_type': decision_type,
                'reasoning': reasoning,
                'confidence': 0.8,
                'tool_calls': tool_calls
            }

        except Exception as e:
            logger.error(f"解析Claude响应失败: {e}")
            return {
                'decision_type': 'hold',
                'reasoning': '解析失败',
                'confidence': 0.0,
                'tool_calls': []
            }

    @classmethod
    def _parse_openai_response(cls, response) -> Dict:
        """
        解析OpenAI/Qwen API响应

        Args:
            response: OpenAI API响应对象

        Returns:
            决策结果字典
        """
        try:
            message = response.choices[0].message
            reasoning = message.content or ""
            tool_calls = []

            # 提取工具调用
            if message.tool_calls:
                for tool_call in message.tool_calls:
                    import json
                    tool_calls.append({
                        "name": tool_call.function.name,
                        "arguments": json.loads(tool_call.function.arguments)
                    })

            # 判断决策类型
            if tool_calls:
                first_tool = tool_calls[0]
                if first_tool['name'] == 'place_order':
                    side = first_tool['arguments'].get('side', 'hold')
                    decision_type = side
                else:
                    decision_type = 'analysis'
            else:
                decision_type = 'hold'

            return {
                'decision_type': decision_type,
                'reasoning': reasoning,
                'confidence': 0.8,
                'tool_calls': tool_calls
            }

        except Exception as e:
            logger.error(f"解析OpenAI响应失败: {e}")
            return {
                'decision_type': 'hold',
                'reasoning': '解析失败',
                'confidence': 0.0,
                'tool_calls': []
            }

    @classmethod
    async def _execute_tools(cls, plan: TradingPlan, llm_response: Dict, agent_decision_id: int) -> tuple:
        """
        执行工具调用（集成确认流程）

        Args:
            plan: 交易计划
            llm_response: LLM响应
            agent_decision_id: Agent决策记录ID

        Returns:
            (tool_calls, tool_results)
        """
        from services.agent_confirmation_service import confirmation_service

        tool_calls = llm_response.get('tool_calls', [])
        tool_results = []

        if not tool_calls:
            return [], []

        logger.info(f"处理工具调用: {len(tool_calls)}个工具")

        # 获取确认模式
        confirmation_mode = confirmation_service.get_confirmation_mode(plan)
        logger.info(f"计划 {plan.id} 确认模式: {confirmation_mode.value}")

        for tool_call in tool_calls:
            tool_name = tool_call.get('name', 'unknown')
            tool_args = tool_call.get('arguments', {})

            logger.info(f"  处理工具: {tool_name}, 参数: {tool_args}")

            # 生成预期效果和风险提示
            expected_effect, risk_warning = cls._generate_tool_info(tool_name, tool_args, plan)

            if confirmation_mode.value == "auto":
                # 自动执行模式
                logger.info(f"  自动执行工具: {tool_name}")
                result = await cls._execute_tool_directly(plan, tool_name, tool_args)
                tool_results.append(result)

            elif confirmation_mode.value == "manual":
                # 手动确认模式 - 创建待确认工具
                logger.info(f"  创建待确认工具: {tool_name}")
                try:
                    pending_tool_id = await confirmation_service.create_pending_tool_call(
                        plan_id=plan.id,
                        agent_decision_id=agent_decision_id,
                        tool_name=tool_name,
                        tool_args=tool_args,
                        expected_effect=expected_effect,
                        risk_warning=risk_warning
                    )

                    result = {
                        'tool_name': tool_name,
                        'success': True,
                        'message': f'工具调用已创建待确认记录，ID: {pending_tool_id}',
                        'pending_tool_id': pending_tool_id,
                        'status': 'pending_confirmation',
                        'requires_confirmation': True
                    }
                    tool_results.append(result)

                except Exception as e:
                    logger.error(f"  创建待确认工具失败: {e}")
                    result = {
                        'tool_name': tool_name,
                        'success': False,
                        'error': f'创建待确认失败: {str(e)}',
                        'status': 'error'
                    }
                    tool_results.append(result)

            else:  # disabled
                # 禁用工具调用
                logger.info(f"  工具调用已禁用: {tool_name}")
                result = {
                    'tool_name': tool_name,
                    'success': False,
                    'error': '工具调用已禁用',
                    'status': 'disabled'
                }
                tool_results.append(result)

        return tool_calls, tool_results

    @classmethod
    def _execute_tools_sync(cls, plan: TradingPlan, llm_response: Dict, agent_decision_id: int) -> tuple:
        """
        执行工具调用（同步版本，集成确认流程）

        Args:
            plan: 交易计划
            llm_response: LLM响应
            agent_decision_id: Agent决策记录ID

        Returns:
            (tool_calls, tool_results)
        """
        from services.agent_confirmation_service import confirmation_service

        tool_calls = llm_response.get('tool_calls', [])
        tool_results = []

        if not tool_calls:
            return [], []

        logger.info(f"处理工具调用(同步): {len(tool_calls)}个工具")

        # 获取确认模式
        confirmation_mode = confirmation_service.get_confirmation_mode(plan)
        logger.info(f"计划 {plan.id} 确认模式: {confirmation_mode.value}")

        for tool_call in tool_calls:
            tool_name = tool_call.get('name', 'unknown')
            tool_args = tool_call.get('arguments', {})

            logger.info(f"  处理工具: {tool_name}, 参数: {tool_args}")

            # 生成预期效果和风险提示
            expected_effect, risk_warning = cls._generate_tool_info(tool_name, tool_args, plan)

            if confirmation_mode.value == "auto":
                # 自动执行模式 - 使用同步版本
                logger.info(f"  自动执行工具: {tool_name}")
                result = cls._execute_tool_directly_sync(plan, tool_name, tool_args)
                tool_results.append(result)

            elif confirmation_mode.value == "manual":
                # 手动确认模式 - 创建待确认工具（同步）
                logger.info(f"  创建待确认工具: {tool_name}")
                try:
                    pending_tool_id = asyncio.run(confirmation_service.create_pending_tool_call(
                        plan_id=plan.id,
                        agent_decision_id=agent_decision_id,
                        tool_name=tool_name,
                        tool_args=tool_args,
                        expected_effect=expected_effect,
                        risk_warning=risk_warning
                    ))

                    result = {
                        'tool_name': tool_name,
                        'success': True,
                        'message': f'工具调用已创建待确认记录，ID: {pending_tool_id}',
                        'pending_tool_id': pending_tool_id,
                        'status': 'pending_confirmation',
                        'requires_confirmation': True
                    }
                    tool_results.append(result)

                except Exception as e:
                    logger.error(f"  创建待确认工具失败: {e}")
                    result = {
                        'tool_name': tool_name,
                        'success': False,
                        'error': f'创建待确认失败: {str(e)}',
                        'status': 'error'
                    }
                    tool_results.append(result)

            else:  # disabled
                # 禁用工具调用
                logger.info(f"  工具调用已禁用: {tool_name}")
                result = {
                    'tool_name': tool_name,
                    'success': False,
                    'error': '工具调用已禁用',
                    'status': 'disabled'
                }
                tool_results.append(result)

        return tool_calls, tool_results

    @classmethod
    def _execute_tool_directly_sync(cls, plan: TradingPlan, tool_name: str, tool_args: Dict) -> Dict[str, Any]:
        """
        直接执行工具（同步版本）

        Args:
            plan: 交易计划
            tool_name: 工具名称
            tool_args: 工具参数

        Returns:
            执行结果
        """
        try:
            from services.trading_tools import OKXTradingTools

            # 创建交易工具实例
            trading_tools = OKXTradingTools(
                api_key=plan.okx_api_key,
                secret_key=plan.okx_secret_key,
                passphrase=plan.okx_passphrase,
                is_demo=plan.is_demo,
                trading_limits=plan.trading_limits
            )

            # 执行具体工具（同步调用）
            if tool_name == 'place_order':
                result = asyncio.run(trading_tools.place_order(**tool_args))
            elif tool_name == 'place_limit_order':
                result = asyncio.run(trading_tools.place_limit_order(**tool_args))
            elif tool_name == 'place_stop_loss_order':
                result = asyncio.run(trading_tools.place_stop_loss_order(**tool_args))
            elif tool_name == 'cancel_order':
                result = asyncio.run(trading_tools.cancel_order(**tool_args))
            elif tool_name == 'get_positions':
                result = asyncio.run(trading_tools.get_positions(**tool_args))
            elif tool_name == 'get_current_price':
                result = asyncio.run(trading_tools.get_current_price(**tool_args))
            elif tool_name == 'get_trading_limits':
                result = asyncio.run(trading_tools.get_trading_limits(**tool_args))
            else:
                result = {
                    'success': False,
                    'error': f'未知工具: {tool_name}'
                }

            # 记录自动执行结果
            logger.info(f"自动执行工具完成: {tool_name}, result: {result.get('success', False)}")

            return {
                'tool_name': tool_name,
                'success': result.get('success', False),
                'result': result,
                'status': 'executed',
                'executed_at': datetime.utcnow(),
                'execution_mode': 'auto'
            }

        except Exception as e:
            logger.error(f"直接执行工具失败: {tool_name}, error: {e}")
            return {
                'tool_name': tool_name,
                'success': False,
                'error': str(e),
                'status': 'error',
                'executed_at': datetime.utcnow(),
                'execution_mode': 'auto'
            }

    @classmethod
    def _generate_tool_info(cls, tool_name: str, tool_args: Dict, plan: TradingPlan) -> Tuple[str, str]:
        """
        生成工具的预期效果和风险提示

        Args:
            tool_name: 工具名称
            tool_args: 工具参数
            plan: 交易计划

        Returns:
            (预期效果, 风险提示)
        """
        expected_effect = ""
        risk_warning = ""

        try:
            if tool_name in ['place_order', 'place_limit_order']:
                side = tool_args.get('side', 'unknown')
                size = tool_args.get('size', 0)
                price = tool_args.get('price', 'market')
                inst_id = tool_args.get('inst_id', plan.inst_id)

                expected_effect = f"执行{side}订单，交易对{inst_id}，数量{size}，价格{price}"
                risk_warning = "⚠️ 市场风险：订单执行后可能产生盈亏，请确认交易参数"

            elif tool_name == 'place_stop_loss_order':
                side = tool_args.get('side', 'sell')
                size = tool_args.get('size', 0)
                stop_price = tool_args.get('stop_price', 0)
                inst_id = tool_args.get('inst_id', plan.inst_id)

                expected_effect = f"设置止损订单，交易对{inst_id}，{side}方向，数量{size}，止损价格{stop_price}"
                risk_warning = "⚠️ 止损风险：止损订单触发时将以市价执行，可能与预期价格有差异"

            elif tool_name == 'cancel_order':
                order_id = tool_args.get('order_id', 'unknown')
                expected_effect = f"取消订单 {order_id}"
                risk_warning = "ℹ️ 取消操作：取消未成交的订单，无直接资金风险"

            elif tool_name == 'get_positions':
                expected_effect = "查询当前持仓信息"
                risk_warning = "ℹ️ 查询操作：仅获取信息，无资金风险"

            elif tool_name == 'get_current_price':
                inst_id = tool_args.get('inst_id', plan.inst_id)
                expected_effect = f"获取 {inst_id} 当前价格"
                risk_warning = "ℹ️ 查询操作：仅获取价格信息，无资金风险"

            else:
                expected_effect = f"执行工具 {tool_name}"
                risk_warning = "⚠️ 请仔细确认工具参数和风险"

        except Exception as e:
            logger.error(f"生成工具信息失败: {e}")
            expected_effect = f"执行工具 {tool_name}"
            risk_warning = "⚠️ 请仔细确认工具参数和风险"

        return expected_effect, risk_warning

    @classmethod
    async def _execute_tool_directly(cls, plan: TradingPlan, tool_name: str, tool_args: Dict) -> Dict[str, Any]:
        """
        直接执行工具（自动模式）

        Args:
            plan: 交易计划
            tool_name: 工具名称
            tool_args: 工具参数

        Returns:
            执行结果
        """
        try:
            from services.trading_tools import OKXTradingTools

            # 创建交易工具实例
            trading_tools = OKXTradingTools(
                api_key=plan.okx_api_key,
                secret_key=plan.okx_secret_key,
                passphrase=plan.okx_passphrase,
                is_demo=plan.is_demo,
                trading_limits=plan.trading_limits
            )

            # 执行具体工具
            if tool_name == 'place_order':
                result = await trading_tools.place_order(**tool_args)
            elif tool_name == 'place_limit_order':
                result = await trading_tools.place_limit_order(**tool_args)
            elif tool_name == 'place_stop_loss_order':
                result = await trading_tools.place_stop_loss_order(**tool_args)
            elif tool_name == 'cancel_order':
                result = await trading_tools.cancel_order(**tool_args)
            elif tool_name == 'get_positions':
                result = await trading_tools.get_positions(**tool_args)
            elif tool_name == 'get_current_price':
                result = await trading_tools.get_current_price(**tool_args)
            elif tool_name == 'get_trading_limits':
                result = await trading_tools.get_trading_limits(**tool_args)
            else:
                result = {
                    'success': False,
                    'error': f'未知工具: {tool_name}'
                }

            # 记录自动执行结果
            logger.info(f"自动执行工具完成: {tool_name}, result: {result.get('success', False)}")

            return {
                'tool_name': tool_name,
                'success': result.get('success', False),
                'result': result,
                'status': 'executed',
                'executed_at': datetime.utcnow(),
                'execution_mode': 'auto'
            }

        except Exception as e:
            logger.error(f"直接执行工具失败: {tool_name}, error: {e}")
            return {
                'tool_name': tool_name,
                'success': False,
                'error': str(e),
                'status': 'error',
                'executed_at': datetime.utcnow(),
                'execution_mode': 'auto'
            }

        return tool_calls, tool_results

    @classmethod
    def _execute_place_order(
        cls,
        trading_tools,
        plan: TradingPlan,
        tool_args: Dict
    ) -> Dict:
        """
        执行下单工具

        Args:
            trading_tools: 交易工具实例
            plan: 交易计划
            tool_args: 工具参数

        Returns:
            执行结果
        """
        try:
            side = tool_args.get('side')
            size = tool_args.get('size')
            order_type = tool_args.get('order_type', 'market')
            price = tool_args.get('price')
            total_amount = tool_args.get('total_amount')

            # 检查参数
            if not side or not price:
                return {
                    'success': False,
                    'error': '缺少必要参数: side或price'
                }

            # 获取交易限制配置
            trading_limits = plan.trading_limits or {}
            available_usdt_amount = trading_limits.get('available_usdt_amount', 1000.0)
            available_usdt_percentage = trading_limits.get('available_usdt_percentage', 30.0)
            avg_order_count = trading_limits.get('avg_order_count', 1)

            # 如果没有指定size和total_amount，使用默认交易限制
            if not size and not total_amount:
                total_amount = available_usdt_amount

            # 如果指定了总金额，计算size
            if total_amount and not size:
                size = float(total_amount) / float(price) if price else 0

            # 如果没有指定平摊数量，使用默认值
            if avg_order_count <= 0:
                avg_order_count = 1

            # 计算单笔订单大小
            single_order_size = float(size) / avg_order_count if size else 0

            # 估算总订单金额
            estimated_total_amount = float(size) * float(price) if size and price else 0

            # 检查交易限制
            if estimated_total_amount > available_usdt_amount:
                logger.warning(f"订单金额超过可用USDT限制: {estimated_total_amount} > {available_usdt_amount}")
                # 使用百分比限制重新计算
                try:
                    # 查询账户总余额
                    account_balance = trading_tools.get_account_balance('USDT')
                    total_balance = float(account_balance.get('available', 0))
                    max_amount_by_percentage = total_balance * (available_usdt_percentage / 100.0)

                    if estimated_total_amount > max_amount_by_percentage:
                        return {
                            'success': False,
                            'error': f'订单金额超过限制(固定:{available_usdt_amount}USDT, 百分比:{available_usdt_percentage}%={max_amount_by_percentage:.2f}USDT): {estimated_total_amount}'
                        }
                    else:
                        logger.info(f"使用百分比限制: {estimated_total_amount} <= {max_amount_by_percentage:.2f}USDT")
                except Exception as e:
                    logger.error(f"查询账户余额失败: {e}")
                    return {
                        'success': False,
                        'error': f'无法查询账户余额，无法使用百分比限制: {str(e)}'
                    }

            # 执行下单（支持平摊操作）
            order_results = []
            successful_orders = 0
            total_filled_size = 0

            for i in range(int(avg_order_count)):
                try:
                    # 为每笔订单生成唯一ID
                    client_order_id = f"{plan.inst_id}_{side}_{i+1}of{avg_order_count}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

                    # 执行单笔订单
                    result = trading_tools.place_order(
                        inst_id=plan.inst_id,
                        side=side,
                        order_type=order_type,
                        size=single_order_size,
                        price=price,
                        client_order_id=client_order_id
                    )

                    order_results.append({
                        'order_index': i + 1,
                        'total_orders': avg_order_count,
                        'size': single_order_size,
                        'result': result
                    })

                    if result.get('success'):
                        successful_orders += 1
                        total_filled_size += result.get('filled_size', single_order_size)

                except Exception as e:
                    logger.error(f"执行第{i+1}笔订单失败: {e}")
                    order_results.append({
                        'order_index': i + 1,
                        'total_orders': avg_order_count,
                        'size': single_order_size,
                        'error': str(e)
                    })

            # 返回执行结果
            return {
                'success': successful_orders > 0,
                'message': f"平摊下单完成: {successful_orders}/{avg_order_count} 成功",
                'avg_order_count': avg_order_count,
                'successful_orders': successful_orders,
                'total_filled_size': total_filled_size,
                'order_results': order_results
            }

        except Exception as e:
            logger.error(f"执行下单失败: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    @classmethod
    def _execute_stop_loss_order(
        cls,
        trading_tools,
        plan: TradingPlan,
        tool_args: Dict
    ) -> Dict:
        """
        执行止损订单工具

        Args:
            trading_tools: 交易工具实例
            plan: 交易计划
            tool_args: 工具参数

        Returns:
            执行结果
        """
        try:
            inst_id = tool_args.get('inst_id', plan.inst_id)
            stop_loss_percentage = float(tool_args.get('stop_loss_percentage', 0))

            # 获取交易限制配置
            trading_limits = plan.trading_limits or {}
            default_stop_loss_percentage = trading_limits.get('stop_loss_percentage', 20.0)

            # 如果没有指定止损百分比，使用默认值
            if stop_loss_percentage <= 0:
                stop_loss_percentage = default_stop_loss_percentage

            # 查询当前持仓
            try:
                positions = trading_tools.get_account_positions(inst_id)
                if not positions:
                    return {
                        'success': False,
                        'error': f'当前没有 {inst_id} 的持仓，无需设置止损'
                    }

                # 找到指定交易对的持仓
                position = None
                for pos in positions:
                    if pos.get('instId') == inst_id and float(pos.get('pos', 0)) != 0:
                        position = pos
                        break

                if not position:
                    return {
                        'success': False,
                        'error': f'当前没有 {inst_id} 的有效持仓'
                    }

                # 计算持仓信息
                pos_size = float(position.get('pos', 0))
                avg_cost = float(position.get('avgCost', 0))
                mark_price = float(position.get('markPx', 0))

                if pos_size == 0 or avg_cost == 0:
                    return {
                        'success': False,
                        'error': '持仓数据无效，无法计算止损价格'
                    }

                # 计算止损价格
                # 对于多头持仓：止损价 = 成本价 * (1 - 止损百分比)
                # 对于空头持仓：止损价 = 成本价 * (1 + 止损百分比)
                if pos_size > 0:  # 多头
                    stop_loss_price = avg_cost * (1 - stop_loss_percentage / 100.0)
                    side = 'sell'
                else:  # 空头
                    stop_loss_price = avg_cost * (1 + stop_loss_percentage / 100.0)
                    side = 'buy'

                # 检查是否已经达到止损条件
                current_pnl_percentage = ((mark_price - avg_cost) / avg_cost) * 100
                if pos_size > 0:  # 多头
                    is_stop_loss_triggered = current_pnl_percentage <= -stop_loss_percentage
                else:  # 空头
                    is_stop_loss_triggered = current_pnl_percentage >= stop_loss_percentage

                # 如果已经触发止损，立即市价卖出
                if is_stop_loss_triggered:
                    logger.warning(f"止损条件已触发！当前盈亏: {current_pnl_percentage:.2f}%, 止损线: -{stop_loss_percentage}%")

                    # 生成客户端订单ID
                    client_order_id = f"STOP_LOSS_{inst_id}_{side}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

                    # 执行市价止损
                    result = trading_tools.place_order(
                        inst_id=inst_id,
                        side=side,
                        order_type='market',
                        size=abs(pos_size),  # 使用持仓的绝对数量
                        client_order_id=client_order_id
                    )

                    return {
                        'success': result.get('success', False),
                        'message': f"止损已触发并执行市价{side}，数量: {abs(pos_size)}",
                        'stop_loss_triggered': True,
                        'stop_loss_price': stop_loss_price,
                        'current_price': mark_price,
                        'pnl_percentage': current_pnl_percentage,
                        'result': result
                    }
                else:
                    # 未触发止损，返回信息
                    return {
                        'success': True,
                        'message': f"止损条件未触发，当前盈亏: {current_pnl_percentage:.2f}%, 止损线: -{stop_loss_percentage}%",
                        'stop_loss_triggered': False,
                        'stop_loss_price': stop_loss_price,
                        'current_price': mark_price,
                        'pnl_percentage': current_pnl_percentage,
                        'position_size': pos_size,
                        'avg_cost': avg_cost
                    }

            except Exception as e:
                logger.error(f"查询持仓信息失败: {e}")
                return {
                    'success': False,
                    'error': f'查询持仓信息失败: {str(e)}'
                }

        except Exception as e:
            logger.error(f"执行止损订单失败: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    @classmethod
    def _execute_query_prediction_data(
        cls,
        plan: TradingPlan,
        tool_args: Dict
    ) -> Dict:
        """
        执行查询预测数据工具

        Args:
            plan: 交易计划
            tool_args: 工具参数

        Returns:
            执行结果
        """
        try:
            from database.models import PredictionData
            from sqlalchemy import and_, desc, asc
            from datetime import datetime

            # 解析参数
            training_id = tool_args.get('training_id')
            inference_batch_id = tool_args.get('inference_batch_id')
            start_time = tool_args.get('start_time')
            end_time = tool_args.get('end_time')
            limit = min(tool_args.get('limit', 50), 200)  # 最大200条
            order_by = tool_args.get('order_by', 'time_asc')

            with get_db() as db:
                # 构建查询
                query = db.query(PredictionData).filter(
                    PredictionData.plan_id == plan.id
                )

                # 如果指定了training_id，筛选特定训练版本
                if training_id:
                    query = query.filter(PredictionData.training_record_id == training_id)

                # 如果指定了inference_batch_id，筛选特定批次
                if inference_batch_id:
                    query = query.filter(PredictionData.inference_batch_id == inference_batch_id)

                # 如果指定了时间范围
                if start_time:
                    try:
                        start_dt = datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S')
                        query = query.filter(PredictionData.timestamp >= start_dt)
                    except ValueError:
                        return {
                            'success': False,
                            'error': f'开始时间格式错误: {start_time}，请使用 YYYY-MM-DD HH:MM:SS 格式'
                        }

                if end_time:
                    try:
                        end_dt = datetime.strptime(end_time, '%Y-%m-%d %H:%M:%S')
                        query = query.filter(PredictionData.timestamp <= end_dt)
                    except ValueError:
                        return {
                            'success': False,
                            'error': f'结束时间格式错误: {end_time}，请使用 YYYY-MM-DD HH:MM:SS 格式'
                        }

                # 应用排序
                if order_by == 'time_asc':
                    query = query.order_by(PredictionData.timestamp.asc())
                elif order_by == 'time_desc':
                    query = query.order_by(PredictionData.timestamp.desc())
                elif order_by == 'created_asc':
                    query = query.order_by(PredictionData.created_at.asc())
                elif order_by == 'created_desc':
                    query = query.order_by(PredictionData.created_at.desc())
                else:
                    query = query.order_by(PredictionData.timestamp.asc())

                # 应用限制
                predictions = query.limit(limit).all()

                if not predictions:
                    return {
                        'success': True,
                        'message': '查询成功，但没有找到符合条件的预测数据',
                        'data_count': 0,
                        'predictions': []
                    }

                # 格式化预测数据
                formatted_predictions = []
                for pred in predictions:
                    pred_data = {
                        'id': pred.id,
                        'training_record_id': pred.training_record_id,
                        'inference_batch_id': pred.inference_batch_id,
                        'timestamp': pred.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                        'created_at': pred.created_at.strftime('%Y-%m-%d %H:%M:%S'),
                        'open': pred.open,
                        'high': pred.high,
                        'low': pred.low,
                        'close': pred.close,
                        'volume': pred.volume
                    }

                    # 添加不确定性数据（如果有）
                    if pred.close_min is not None and pred.close_max is not None:
                        pred_data['close_min'] = pred.close_min
                        pred_data['close_max'] = pred.close_max
                        pred_data['price_range'] = f"{pred.close_min:.4f} - {pred.close_max:.4f}"

                    # 添加概率指标（如果有）
                    if pred.upward_probability is not None and pred.volatility_amplification_probability is not None:
                        pred_data['upward_probability'] = pred.upward_probability
                        pred_data['volatility_amplification_probability'] = pred.volatility_amplification_probability

                    formatted_predictions.append(pred_data)

                # 统计信息
                if formatted_predictions:
                    close_prices = [p['close'] for p in formatted_predictions]
                    min_close = min(close_prices)
                    max_close = max(close_prices)
                    avg_close = sum(close_prices) / len(close_prices)
                    first_close = close_prices[0]
                    last_close = close_prices[-1]
                    change_pct = ((last_close - first_close) / first_close) * 100

                    stats = {
                        'data_count': len(formatted_predictions),
                        'price_range': f"{min_close:.4f} - {max_close:.4f}",
                        'average_price': f"{avg_close:.4f}",
                        'price_change': f"{change_pct:+.2f}%",
                        'time_range': {
                            'start': formatted_predictions[0]['timestamp'],
                            'end': formatted_predictions[-1]['timestamp']
                        }
                    }

                    # 批次统计
                    batch_ids = list(set(p['inference_batch_id'] for p in formatted_predictions))
                    stats['batch_count'] = len(batch_ids)
                    stats['batches'] = batch_ids
                else:
                    stats = {}

                return {
                    'success': True,
                    'message': f"查询成功，找到 {len(formatted_predictions)} 条预测数据",
                    'data_count': len(formatted_predictions),
                    'statistics': stats,
                    'predictions': formatted_predictions
                }

        except Exception as e:
            logger.error(f"查询预测数据失败: {e}")
            return {
                'success': False,
                'error': f"查询失败: {str(e)}"
            }

    @classmethod
    def _execute_get_prediction_history(
        cls,
        plan: TradingPlan,
        tool_args: Dict
    ) -> Dict:
        """
        执行获取预测历史工具

        Args:
            plan: 交易计划
            tool_args: 工具参数

        Returns:
            执行结果
        """
        try:
            from database.models import PredictionData, TrainingRecord
            from sqlalchemy import and_, func, desc

            training_id = tool_args.get('training_id')
            inference_batch_id = tool_args.get('inference_batch_id')
            limit = min(tool_args.get('limit', 10), 50)

            with get_db() as db:
                # 如果没有指定training_id，使用最新的已完成训练记录
                if not training_id:
                    latest_training = db.query(TrainingRecord).filter(
                        and_(
                            TrainingRecord.plan_id == plan.id,
                            TrainingRecord.status == 'completed',
                            TrainingRecord.is_active == True
                        )
                    ).order_by(desc(TrainingRecord.created_at)).first()

                    if not latest_training:
                        return {
                            'success': False,
                            'error': '没有找到已完成的训练记录'
                        }
                    training_id = latest_training.id

                # 如果指定了inference_batch_id，返回该批次的详细数据
                if inference_batch_id:
                    predictions = db.query(PredictionData).filter(
                        and_(
                            PredictionData.training_record_id == training_id,
                            PredictionData.inference_batch_id == inference_batch_id
                        )
                    ).order_by(PredictionData.timestamp).all()

                    if predictions:
                        formatted_predictions = []
                        for pred in predictions:
                            pred_data = {
                                'timestamp': pred.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                                'open': pred.open,
                                'high': pred.high,
                                'low': pred.low,
                                'close': pred.close,
                                'volume': pred.volume
                            }

                            # 添加不确定性数据（如果有）
                            if pred.close_min is not None and pred.close_max is not None:
                                pred_data['close_min'] = pred.close_min
                                pred_data['close_max'] = pred.close_max

                            formatted_predictions.append(pred_data)

                        return {
                            'success': True,
                            'message': f"找到批次 {inference_batch_id} 的 {len(formatted_predictions)} 条预测数据",
                            'batch_id': inference_batch_id,
                            'training_id': training_id,
                            'data_count': len(formatted_predictions),
                            'predictions': formatted_predictions
                        }
                    else:
                        return {
                            'success': False,
                            'error': f"未找到批次 {inference_batch_id} 的预测数据"
                        }

                # 否则返回批次列表
                batches = db.query(
                    PredictionData.inference_batch_id,
                    func.min(PredictionData.created_at).label('inference_time'),
                    func.count(PredictionData.id).label('data_count')
                ).filter(
                    PredictionData.training_record_id == training_id
                ).group_by(
                    PredictionData.inference_batch_id
                ).order_by(
                    func.min(PredictionData.created_at).desc()
                ).limit(limit).all()

                if not batches:
                    return {
                        'success': False,
                        'error': '训练记录ID没有关联的预测数据'
                    }

                batch_list = []
                for batch in batches:
                    batch_list.append({
                        'inference_batch_id': batch.inference_batch_id,
                        'inference_time': batch.inference_time.strftime('%Y-%m-%d %H:%M:%S'),
                        'data_count': batch.data_count
                    })

                return {
                    'success': True,
                    'message': f"找到 {len(batch_list)} 个推理批次",
                    'training_id': training_id,
                    'batch_count': len(batch_list),
                    'batches': batch_list
                }

        except Exception as e:
            logger.error(f"获取预测历史失败: {e}")
            return {
                'success': False,
                'error': f"获取失败: {str(e)}"
            }

    @classmethod
    def get_decision_history(
        cls,
        plan_id: int,
        limit: int = 50
    ) -> List[Dict]:
        """
        获取决策历史

        Args:
            plan_id: 计划ID
            limit: 返回数量限制

        Returns:
            决策记录列表
        """
        try:
            with get_db() as db:
                decisions = db.query(AgentDecision).filter(
                    AgentDecision.plan_id == plan_id
                ).order_by(desc(AgentDecision.decision_time)).limit(limit).all()

                result = []
                for decision in decisions:
                    result.append({
                        'id': decision.id,
                        'decision_time': decision.decision_time,
                        'decision_type': decision.decision_type,
                        'reasoning': decision.reasoning,
                        'status': decision.status,
                        'tool_calls': decision.tool_calls,
                        'order_ids': decision.order_ids
                    })

                return result

        except Exception as e:
            logger.error(f"获取决策历史失败: {e}")
            return []


    # ==================== 流式决策方法 ====================

    @classmethod
    async def trigger_decision_stream(cls, plan_id: int, training_id: int):
        """
        触发AI Agent决策（流式版本）

        Args:
            plan_id: 计划ID
            training_id: 训练记录ID

        Yields:
            消息字典: {
                'type': 'thinking' | 'tool_call' | 'tool_result' | 'error' | 'completed',
                'content': str,  # 思考内容（逐token）
                'tool_name': str,  # 工具名称
                'tool_arguments': dict,  # 工具参数
                'tool_result': dict,  # 工具执行结果
                'decision_id': int  # 最终决策ID
            }
        """
        try:
            logger.info(f"触发流式AI Agent决策: plan_id={plan_id}, training_id={training_id}")

            # 准备决策上下文
            with get_db() as db:
                # 获取计划信息
                plan = db.query(TradingPlan).filter(TradingPlan.id == plan_id).first()
                if not plan:
                    yield {'type': 'error', 'content': '❌ 计划不存在'}
                    return

                # 获取训练记录
                training_record = db.query(TrainingRecord).filter(
                    TrainingRecord.id == training_id
                ).first()
                if not training_record:
                    yield {'type': 'error', 'content': '❌ 训练记录不存在'}
                    return

                # 获取预测数据
                predictions = db.query(PredictionData).filter(
                    PredictionData.training_record_id == training_id
                ).order_by(PredictionData.timestamp).all()

                if not predictions:
                    yield {'type': 'error', 'content': '❌ 无预测数据'}
                    return

                # 获取历史K线数据
                from services.database_dataset import get_kline_dataframe
                from datetime import timedelta

                hist_end = datetime.now()
                hist_start = hist_end - timedelta(days=7)
                historical_df = get_kline_dataframe(
                    inst_id=plan.inst_id,
                    interval=plan.interval,
                    start_time=hist_start,
                    end_time=hist_end
                )

                # 获取LLM配置
                llm_config = None
                if plan.llm_config_id:
                    llm_config = db.query(LLMConfig).filter(
                        LLMConfig.id == plan.llm_config_id
                    ).first()

                if not llm_config:
                    yield {'type': 'error', 'content': '❌ 未配置LLM'}
                    return

                # 构建决策上下文
                decision_context = cls._build_decision_context(
                    plan=plan,
                    predictions=predictions,
                    historical_df=historical_df
                )

            # 流式调用LLM
            reasoning_text = ""
            tool_calls_list = []

            async for chunk in cls._call_llm_for_decision_stream(
                llm_config, decision_context, plan.agent_prompt
            ):
                if chunk['type'] == 'thinking':
                    # 逐token输出思考内容
                    reasoning_text += chunk['content']
                    yield chunk

                elif chunk['type'] == 'tool_call':
                    # 工具调用
                    tool_calls_list.append(chunk)
                    yield chunk

            # 执行工具调用
            tool_results = []
            decision_id_temp = None  # 先保存决策ID，用于关联待确认工具

            # 先创建决策记录（以便关联待确认工具）
            with get_db() as db:
                decision = AgentDecision(
                    plan_id=plan_id,
                    training_record_id=training_id,
                    decision_time=datetime.utcnow(),
                    decision_type='analysis',
                    reasoning=reasoning_text,
                    llm_model=llm_config.model_name if llm_config else 'default',
                    llm_input=decision_context,
                    llm_output=reasoning_text,
                    tool_calls=tool_calls_list,
                    tool_results=[],  # 稍后更新
                    order_ids=[],
                    status='processing'
                )

                db.add(decision)
                db.commit()
                db.refresh(decision)
                decision_id_temp = decision.id

            # 处理工具调用
            for tool_call in tool_calls_list:
                tool_name = tool_call.get('tool_name', 'unknown')
                tool_args = tool_call.get('tool_arguments', {})

                logger.info(f"处理工具调用: {tool_name}, 参数: {tool_args}")

                # 检查是否启用自动执行
                if not plan.auto_tool_execution_enabled:
                    # 未启用自动执行，创建待确认工具记录
                    logger.info(f"工具 {tool_name} 需要手动确认")

                    from database.models import PendingToolCall
                    from datetime import timedelta

                    with get_db() as db:
                        # 创建待确认工具记录
                        pending_tool = PendingToolCall(
                            plan_id=plan_id,
                            agent_decision_id=decision_id_temp,
                            tool_name=tool_name,
                            tool_arguments=tool_args,
                            expected_effect=f"将执行 {tool_name} 操作",
                            risk_warning="请谨慎确认工具调用参数",
                            status='pending',
                            expires_at=datetime.utcnow() + timedelta(hours=24)  # 24小时后过期
                        )
                        db.add(pending_tool)
                        db.commit()
                        db.refresh(pending_tool)

                        logger.info(f"已创建待确认工具: pending_tool_id={pending_tool.id}")

                    result = {
                        'tool_name': tool_name,
                        'status': 'pending_confirmation',
                        'message': '⏸️ 工具调用需要手动确认，请在气泡中点击确认按钮',
                        'pending_tool_id': pending_tool.id
                    }
                    tool_results.append(result)

                    yield {
                        'type': 'tool_pending',
                        'tool_name': tool_name,
                        'tool_arguments': tool_args,
                        'pending_tool_id': pending_tool.id,
                        'tool_result': result
                    }
                else:
                    # 自动执行
                    logger.info(f"自动执行工具: {tool_name}")
                    result = await cls._execute_single_tool_async(plan, tool_name, tool_args)
                    tool_results.append(result)

                    yield {
                        'type': 'tool_result',
                        'tool_name': tool_name,
                        'tool_result': result
                    }

            # 更新决策记录
            with get_db() as db:
                decision = db.query(AgentDecision).filter(
                    AgentDecision.id == decision_id_temp
                ).first()
                if decision:
                    decision.tool_results = tool_results
                    decision.status = 'completed'
                    db.commit()

                yield {
                    'type': 'completed',
                    'decision_id': decision.id,
                    'content': '✅ 决策完成'
                }

        except Exception as e:
            logger.error(f"流式决策失败: {e}")
            import traceback
            traceback.print_exc()
            yield {
                'type': 'error',
                'content': f'❌ 决策失败: {str(e)}'
            }


    @classmethod
    async def _call_llm_for_decision_stream(
        cls,
        llm_config: LLMConfig,
        context: Dict,
        agent_prompt: str
    ):
        """
        流式调用LLM进行决策

        Yields:
            消息字典
        """
        try:
            # 根据模型类型调用不同的流式API
            if llm_config.provider == 'anthropic':
                async for chunk in cls._call_claude_stream(llm_config, context, agent_prompt):
                    yield chunk
            elif llm_config.provider == 'openai':
                async for chunk in cls._call_openai_stream(llm_config, context, agent_prompt):
                    yield chunk
            elif llm_config.provider == 'qwen':
                async for chunk in cls._call_qwen_stream(llm_config, context, agent_prompt):
                    yield chunk
            elif llm_config.provider == 'ollama':
                async for chunk in cls._call_ollama_stream(llm_config, context, agent_prompt):
                    yield chunk
            else:
                yield {'type': 'error', 'content': f'不支持的LLM提供商: {llm_config.provider}'}

        except Exception as e:
            logger.error(f"流式LLM调用失败: {e}")
            yield {'type': 'error', 'content': f'LLM调用失败: {str(e)}'}


    @classmethod
    async def _call_claude_stream(cls, llm_config: LLMConfig, context: Dict, agent_prompt: str):
        """Claude 流式调用"""
        try:
            import anthropic
            import httpx

            # 配置代理
            http_client = None
            if config.PROXY_ENABLED and config.PROXY_URL:
                http_client = httpx.AsyncClient(
                    proxy=config.PROXY_URL,
                    timeout=60.0
                )

            client = anthropic.AsyncAnthropic(
                api_key=llm_config.api_key,
                http_client=http_client
            )

            system_prompt = agent_prompt or "你是一个专业的加密货币交易助手。"
            context_str = cls._format_context_for_llm(context)

            # 定义工具
            tools = [
                {
                    "name": "place_order",
                    "description": "下单（买入或卖出）",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "side": {"type": "string", "enum": ["buy", "sell"]},
                            "size": {"type": "number"},
                            "order_type": {"type": "string", "enum": ["market", "limit"]},
                            "price": {"type": "number"}
                        },
                        "required": ["side", "size", "order_type"]
                    }
                }
            ]

            # 流式调用
            async with client.messages.stream(
                model=llm_config.model_name,
                max_tokens=llm_config.max_tokens,
                temperature=llm_config.temperature,
                system=system_prompt,
                messages=[{
                    "role": "user",
                    "content": f"请分析以下市场数据并提供交易建议：\n\n{context_str}"
                }],
                tools=tools
            ) as stream:
                async for event in stream:
                    if event.type == "content_block_delta":
                        if hasattr(event.delta, 'text'):
                            yield {
                                'type': 'thinking',
                                'content': event.delta.text
                            }
                    elif event.type == "message_stop":
                        # 检查是否有工具调用
                        message = await stream.get_final_message()
                        for block in message.content:
                            if block.type == "tool_use":
                                yield {
                                    'type': 'tool_call',
                                    'tool_name': block.name,
                                    'tool_arguments': block.input
                                }

        except ImportError:
            yield {'type': 'error', 'content': '❌ 未安装 anthropic 库'}
        except Exception as e:
            logger.error(f"Claude 流式调用失败: {e}")
            yield {'type': 'error', 'content': f'Claude 调用失败: {str(e)}'}


    @classmethod
    async def _call_openai_stream(cls, llm_config: LLMConfig, context: Dict, agent_prompt: str):
        """OpenAI 流式调用"""
        try:
            import openai
            import httpx

            # 配置代理
            http_client = None
            if config.PROXY_ENABLED and config.PROXY_URL:
                http_client = httpx.AsyncClient(
                    proxy=config.PROXY_URL,
                    timeout=60.0
                )

            client = openai.AsyncOpenAI(
                api_key=llm_config.api_key,
                base_url=llm_config.api_base_url,
                http_client=http_client
            )

            system_prompt = agent_prompt or "你是一个专业的加密货币交易助手。"
            context_str = cls._format_context_for_llm(context)

            # 定义工具
            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "place_order",
                        "description": "下单（买入或卖出）",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "side": {"type": "string", "enum": ["buy", "sell"]},
                                "size": {"type": "number"},
                                "order_type": {"type": "string", "enum": ["market", "limit"]},
                                "price": {"type": "number"}
                            },
                            "required": ["side", "size", "order_type"]
                        }
                    }
                }
            ]

            # 流式调用
            stream = await client.chat.completions.create(
                model=llm_config.model_name,
                max_tokens=llm_config.max_tokens,
                temperature=llm_config.temperature,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"请分析以下市场数据并提供交易建议：\n\n{context_str}"}
                ],
                tools=tools,
                stream=True,
                extra_body={"enable_thinking": True}  # Qwen 流式支持思考
            )

            async for chunk in stream:
                if chunk.choices and len(chunk.choices) > 0:
                    delta = chunk.choices[0].delta

                    # 文本内容
                    if delta.content:
                        yield {
                            'type': 'thinking',
                            'content': delta.content
                        }

                    # 工具调用
                    if delta.tool_calls:
                        for tool_call in delta.tool_calls:
                            if tool_call.function.name and tool_call.function.arguments:
                                try:
                                    args = json.loads(tool_call.function.arguments)
                                    yield {
                                        'type': 'tool_call',
                                        'tool_name': tool_call.function.name,
                                        'tool_arguments': args
                                    }
                                except json.JSONDecodeError:
                                    pass

        except ImportError:
            yield {'type': 'error', 'content': '❌ 未安装 openai 库'}
        except Exception as e:
            logger.error(f"OpenAI 流式调用失败: {e}")
            yield {'type': 'error', 'content': f'OpenAI 调用失败: {str(e)}'}


    @classmethod
    async def _call_qwen_stream(cls, llm_config: LLMConfig, context: Dict, agent_prompt: str):
        """Qwen 流式调用"""
        try:
            import openai
            import httpx

            # 配置代理
            http_client = None
            if config.PROXY_ENABLED and config.PROXY_URL:
                http_client = httpx.AsyncClient(
                    proxy=config.PROXY_URL,
                    timeout=60.0
                )

            client = openai.AsyncOpenAI(
                api_key=llm_config.api_key,
                base_url=llm_config.api_base_url or "https://dashscope.aliyuncs.com/compatible-mode/v1",
                http_client=http_client
            )

            system_prompt = agent_prompt or "你是一个专业的加密货币交易助手。"
            context_str = cls._format_context_for_llm(context)

            # 定义工具
            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "place_order",
                        "description": "下单（买入或卖出）",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "side": {"type": "string", "enum": ["buy", "sell"]},
                                "size": {"type": "number"},
                                "order_type": {"type": "string", "enum": ["market", "limit"]},
                                "price": {"type": "number"}
                            },
                            "required": ["side", "size", "order_type"]
                        }
                    }
                }
            ]

            # 流式调用 (Qwen 支持流式+thinking)
            stream = await client.chat.completions.create(
                model=llm_config.model_name,
                max_tokens=llm_config.max_tokens,
                temperature=llm_config.temperature,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"请分析以下市场数据并提供交易建议：\n\n{context_str}"}
                ],
                tools=tools,
                stream=True,
                extra_body={"enable_thinking": True}  # Qwen 流式支持思考
            )

            async for chunk in stream:
                if chunk.choices and len(chunk.choices) > 0:
                    delta = chunk.choices[0].delta

                    # 文本内容
                    if delta.content:
                        yield {
                            'type': 'thinking',
                            'content': delta.content
                        }

                    # 工具调用
                    if delta.tool_calls:
                        for tool_call in delta.tool_calls:
                            if tool_call.function.name and tool_call.function.arguments:
                                try:
                                    args = json.loads(tool_call.function.arguments)
                                    yield {
                                        'type': 'tool_call',
                                        'tool_name': tool_call.function.name,
                                        'tool_arguments': args
                                    }
                                except json.JSONDecodeError:
                                    pass

        except ImportError:
            yield {'type': 'error', 'content': '❌ 未安装 openai 库'}
        except Exception as e:
            logger.error(f"Qwen 流式调用失败: {e}")
            yield {'type': 'error', 'content': f'Qwen 调用失败: {str(e)}'}


    @classmethod
    async def _call_ollama_stream(cls, llm_config: LLMConfig, context: Dict, agent_prompt: str):
        """Ollama 流式调用"""
        try:
            import openai

            client = openai.AsyncOpenAI(
                api_key="ollama",  # Ollama 不需要真实API key
                base_url=llm_config.api_base_url or "http://localhost:11434/v1"
            )

            system_prompt = agent_prompt or "你是一个专业的加密货币交易助手。"
            context_str = cls._format_context_for_llm(context)

            # 定义工具
            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "place_order",
                        "description": "下单（买入或卖出）",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "side": {"type": "string", "enum": ["buy", "sell"]},
                                "size": {"type": "number"},
                                "order_type": {"type": "string", "enum": ["market", "limit"]},
                                "price": {"type": "number"}
                            },
                            "required": ["side", "size", "order_type"]
                        }
                    }
                }
            ]

            # 流式调用
            stream = await client.chat.completions.create(
                model=llm_config.model_name,
                max_tokens=llm_config.max_tokens,
                temperature=llm_config.temperature,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"请分析以下市场数据并提供交易建议：\n\n{context_str}"}
                ],
                tools=tools,
                stream=True,
                extra_body={"enable_thinking": True}  # Qwen 流式支持思考
            )

            async for chunk in stream:
                if chunk.choices and len(chunk.choices) > 0:
                    delta = chunk.choices[0].delta

                    # 文本内容
                    if delta.content:
                        yield {
                            'type': 'thinking',
                            'content': delta.content
                        }

                    # 工具调用
                    if delta.tool_calls:
                        for tool_call in delta.tool_calls:
                            if tool_call.function.name and tool_call.function.arguments:
                                try:
                                    args = json.loads(tool_call.function.arguments)
                                    yield {
                                        'type': 'tool_call',
                                        'tool_name': tool_call.function.name,
                                        'tool_arguments': args
                                    }
                                except json.JSONDecodeError:
                                    pass

        except ImportError:
            yield {'type': 'error', 'content': '❌ 未安装 openai 库'}
        except Exception as e:
            logger.error(f"Ollama 流式调用失败: {e}")
            yield {'type': 'error', 'content': f'Ollama 调用失败: {str(e)}'}


    # ==================== 工具执行辅助方法 ====================

    @classmethod
    async def _execute_single_tool_async(cls, plan: TradingPlan, tool_name: str, tool_args: Dict) -> Dict:
        """
        异步执行单个工具

        Args:
            plan: 交易计划
            tool_name: 工具名称
            tool_args: 工具参数

        Returns:
            执行结果
        """
        try:
            # 在线程池中执行同步工具调用
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                cls._execute_single_tool_sync,
                plan,
                tool_name,
                tool_args
            )
            return result

        except Exception as e:
            logger.error(f"工具执行失败: {e}")
            return {
                'tool_name': tool_name,
                'status': 'error',
                'message': str(e)
            }


    @classmethod
    def _execute_single_tool_sync(cls, plan: TradingPlan, tool_name: str, tool_args: Dict) -> Dict:
        """
        同步执行单个工具

        Args:
            plan: 交易计划
            tool_name: 工具名称
            tool_args: 工具参数

        Returns:
            执行结果
        """
        try:
            # 创建交易工具实例
            from services.trading_tools import OKXTradingTools

            trading_tools = OKXTradingTools(
                api_key=plan.okx_api_key,
                secret_key=plan.okx_secret_key,
                passphrase=plan.okx_passphrase,
                is_demo=plan.is_demo
            )

            logger.info(f"执行工具: {tool_name}, 参数: {tool_args}")

            # 执行工具
            if tool_name == 'place_order':
                result = cls._execute_place_order(trading_tools, plan, tool_args)
            elif tool_name == 'place_limit_order':
                result = cls._execute_place_order(trading_tools, plan, tool_args)
            elif tool_name == 'place_stop_loss_order':
                result = cls._execute_stop_loss_order(trading_tools, plan, tool_args)
            elif tool_name == 'query_prediction_data':
                result = cls._execute_query_prediction_data(plan, tool_args)
            elif tool_name == 'get_prediction_history':
                result = cls._execute_get_prediction_history(plan, tool_args)
            elif tool_name == 'cancel_order':
                # TODO: 实现取消订单
                result = {
                    'success': True,
                    'message': f"取消订单功能开发中",
                    'simulated': True
                }
            elif tool_name == 'modify_order':
                # TODO: 实现修改订单
                result = {
                    'success': True,
                    'message': f"修改订单功能开发中",
                    'simulated': True
                }
            else:
                result = {
                    'success': False,
                    'message': f"未知工具: {tool_name}"
                }

            return {
                'tool_name': tool_name,
                'status': 'success' if result.get('success') else 'error',
                'message': result.get('message', ''),
                'result': result
            }

        except Exception as e:
            logger.error(f"工具执行失败: {e}")
            import traceback
            traceback.print_exc()
            return {
                'tool_name': tool_name,
                'status': 'error',
                'message': str(e)
            }

    # ==================== 对话专用的LLM调用方法 ====================

    @classmethod
    async def _call_claude_conversation(cls, llm_config: LLMConfig, messages: List[Dict]) -> Dict:
        """调用Claude进行对话"""
        try:
            import anthropic

            client = anthropic.AsyncAnthropic(
                api_key=llm_config.api_key,
                base_url=llm_config.api_base_url
            )

            # 过滤掉system消息，Claude使用单独的system参数
            system_message = ""
            filtered_messages = []
            for msg in messages:
                if msg['role'] == 'system':
                    system_message = msg['content']
                else:
                    filtered_messages.append(msg)

            response = await client.messages.create(
                model=llm_config.model_name,
                max_tokens=llm_config.max_tokens or 2000,
                temperature=llm_config.temperature or 0.7,
                system=system_message,
                messages=filtered_messages
            )

            return {
                'success': True,
                'content': response.content[0].text
            }

        except ImportError:
            return {
                'success': False,
                'error': 'anthropic库未安装'
            }
        except Exception as e:
            logger.error(f"调用Claude对话失败: {e}")
            return {
                'success': False,
                'error': f'调用Claude失败: {str(e)}'
            }

    @classmethod
    async def _call_openai_conversation(cls, llm_config: LLMConfig, messages: List[Dict]) -> Dict:
        """调用OpenAI进行对话"""
        try:
            import openai

            client = openai.AsyncOpenAI(
                api_key=llm_config.api_key,
                base_url=llm_config.api_base_url
            )

            response = await client.chat.completions.create(
                model=llm_config.model_name,
                max_tokens=llm_config.max_tokens or 2000,
                temperature=llm_config.temperature or 0.7,
                messages=messages,
                extra_body={"enable_thinking": False}  # Qwen 非流式调用必须禁用思考
            )

            return {
                'success': True,
                'content': response.choices[0].message.content
            }

        except ImportError:
            return {
                'success': False,
                'error': 'openai库未安装'
            }
        except Exception as e:
            logger.error(f"调用OpenAI对话失败: {e}")
            return {
                'success': False,
                'error': f'调用OpenAI失败: {str(e)}'
            }

    @classmethod
    async def _call_qwen_conversation(cls, llm_config: LLMConfig, messages: List[Dict]) -> Dict:
        """调用通义千问进行对话"""
        try:
            import openai

            # 通义千问使用OpenAI兼容接口
            client = openai.AsyncOpenAI(
                api_key=llm_config.api_key,
                base_url=llm_config.api_base_url or "https://dashscope.aliyuncs.com/compatible-mode/v1"
            )

            response = await client.chat.completions.create(
                model=llm_config.model_name,
                max_tokens=llm_config.max_tokens or 2000,
                temperature=llm_config.temperature or 0.7,
                messages=messages,
                extra_body={"enable_thinking": False}  # Qwen 非流式调用必须禁用思考
            )

            return {
                'success': True,
                'content': response.choices[0].message.content
            }

        except ImportError:
            return {
                'success': False,
                'error': 'openai库未安装'
            }
        except Exception as e:
            logger.error(f"调用通义千问对话失败: {e}")
            return {
                'success': False,
                'error': f'调用通义千问失败: {str(e)}'
            }

    @classmethod
    async def _call_ollama_conversation(cls, llm_config: LLMConfig, messages: List[Dict]) -> Dict:
        """调用Ollama进行对话"""
        try:
            import openai

            client = openai.AsyncOpenAI(
                api_key="ollama",  # Ollama不需要真实的API key
                base_url=llm_config.api_base_url or "http://localhost:11434/v1"
            )

            response = await client.chat.completions.create(
                model=llm_config.model_name,
                max_tokens=llm_config.max_tokens or 2000,
                temperature=llm_config.temperature or 0.7,
                messages=messages,
                extra_body={"enable_thinking": False}  # Qwen 非流式调用必须禁用思考
            )

            return {
                'success': True,
                'content': response.choices[0].message.content
            }

        except ImportError:
            return {
                'success': False,
                'error': 'openai库未安装'
            }
        except Exception as e:
            logger.error(f"调用Ollama对话失败: {e}")
            return {
                'success': False,
                'error': f'调用Ollama失败: {str(e)}'
            }

    @classmethod
    async def react_tool_use_stream(
        cls,
        plan_id: int,
        training_id: int = None,
        manual_tool_approval: bool = True,
        progress=None
    ):
        """
        ReAct + Tool Use 流式推理

        Args:
            plan_id: 计划ID
            training_id: 训练记录ID（可选）
            manual_tool_approval: 是否需要手动审批工具调用
            progress: Gradio进度条

        Yields:
            流式消息，包含思考过程、工具调用、工具结果等
        """
        try:
            # 设置当前计划ID供工具确认使用
            cls._current_plan_id = plan_id

            with get_db() as db:
                plan = db.query(TradingPlan).filter(TradingPlan.id == plan_id).first()
                if not plan:
                    yield [{"role": "assistant", "content": "❌ 计划不存在"}]
                    return

                # 获取预测数据
                prediction_records = []
                if training_id:
                    prediction_records = db.query(PredictionData).filter(
                        PredictionData.training_record_id == training_id
                    ).order_by(PredictionData.timestamp).all()

                # 获取历史数据
                historical_df = None
                try:
                    from database.models import KlineData
                    import pandas as pd

                    hist_start = datetime.utcnow() - timedelta(days=7)
                    hist_end = datetime.utcnow()

                    with get_db() as db:
                        # 从数据库查询历史K线数据
                        kline_records = db.query(KlineData).filter(
                            KlineData.inst_id == plan.inst_id,
                            KlineData.interval == plan.interval,
                            KlineData.timestamp >= hist_start,
                            KlineData.timestamp <= hist_end
                        ).order_by(KlineData.timestamp.asc()).all()

                        if kline_records:
                            # 转换为DataFrame
                            data = []
                            for record in kline_records:
                                data.append({
                                    'timestamp': record.timestamp,
                                    'open': record.open,
                                    'high': record.high,
                                    'low': record.low,
                                    'close': record.close,
                                    'volume': record.volume,
                                    'amount': record.amount
                                })
                            historical_df = pd.DataFrame(data)
                            historical_df['timestamp'] = pd.to_datetime(historical_df['timestamp'])
                            historical_df = historical_df.set_index('timestamp')
                            logger.info(f"成功获取历史数据: {len(historical_df)} 条记录")
                        else:
                            logger.warning(f"未找到历史数据: {plan.inst_id} {plan.interval}")

                except Exception as e:
                    logger.warning(f"获取历史数据失败: {e}")

                # 获取LLM配置
                llm_config = None
                if plan.llm_config_id:
                    llm_config = db.query(LLMConfig).filter(
                        LLMConfig.id == plan.llm_config_id
                    ).first()

                if not llm_config:
                    yield [{"role": "assistant", "content": "❌ 未配置LLM，请先在Agent配置中设置"}]
                    return

                # 构建初始上下文（使用原始的PredictionData对象，而不是转换后的字典）
                context = cls._build_decision_context(
                    plan=plan,
                    predictions=prediction_records,
                    historical_df=historical_df
                )

                # 构建ReAct系统提示词
                system_prompt = cls._build_react_prompt(plan.agent_prompt or "")

                # 初始化对话历史
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"请分析以下市场数据并使用工具进行交易决策：\n\n{context}"}
                ]

                yield [{"role": "assistant", "content": "🤖 开始分析市场数据..."}]

                # 获取可用工具
                available_tools = cls._get_enabled_tools(plan)

                # ReAct循环
                # 从计划配置中获取ReAct参数
                react_config = plan.react_config or {}
                max_iterations = int(react_config.get('max_iterations', 10))
                enable_thinking = bool(react_config.get('enable_thinking', True))
                tool_approval = bool(react_config.get('tool_approval', False))
                thinking_style = react_config.get('thinking_style', '详细')

                iteration = 0

                while iteration < max_iterations:
                    iteration += 1

                    # 调用LLM进行思考
                    if enable_thinking:
                        if thinking_style == '详细':
                            yield [{"role": "assistant", "content": f"\n---\n**第{iteration}轮思考**\n"}]
                        elif thinking_style == '简洁':
                            yield [{"role": "assistant", "content": f"\n**思考({iteration}/{max_iterations})**: "}]
                        else:  # 极简
                            yield [{"role": "assistant", "content": ""}]  # 不显示思考标题

                    if llm_config.provider == 'openai':
                        async for chunk in cls._call_openai_react_stream(
                            llm_config, messages, available_tools, enable_thinking, thinking_style
                        ):
                            yield chunk
                    elif llm_config.provider == 'qwen':
                        async for chunk in cls._call_qwen_react_stream(
                            llm_config, messages, available_tools, enable_thinking, thinking_style
                        ):
                            yield chunk
                    elif llm_config.provider == 'ollama':
                        async for chunk in cls._call_ollama_react_stream(
                            llm_config, messages, available_tools, enable_thinking, thinking_style
                        ):
                            yield chunk
                    else:
                        yield [{"role": "assistant", "content": "❌ 不支持的LLM提供商"}]
                        return

        except Exception as e:
            logger.error(f"ReAct推理失败: {e}")
            yield [{"role": "assistant", "content": f"❌ 推理过程出错: {str(e)}"}]

    @classmethod
    def _format_thinking_content(cls, thinking_content: str, thinking_style: str = "详细") -> str:
        """格式化思考内容为可显示的格式"""
        if not thinking_content:
            return ""

        if thinking_style == "详细":
            return f"🧔 **思考过程：**\n```\n{thinking_content}\n```"
        elif thinking_style == "简洁":
            return f"💭 **思考：** {thinking_content}"
        else:  # 极简
            return thinking_content

    @classmethod
    def _build_react_prompt(cls, custom_prompt: str = "") -> str:
        """构建ReAct系统提示词"""
        base_prompt = """你是一个专业的量化交易AI助手，使用ReAct模式进行思考和工具调用。

你的工作流程是：
1. **Thought** (思考): 分析当前情况，决定下一步行动
2. **Action** (行动): 调用相应的工具获取信息或执行操作
3. **Observation** (观察): 分析工具返回的结果
4. **重复**直到可以做出最终决策

可用工具：
"""

        # 添加工具描述
        tool_descriptions = cls._get_tool_descriptions()
        base_prompt += tool_descriptions

        base_prompt += """

**重要规则：**
- 每次思考都必须以 "Thought:" 开头
- 每次行动都必须以 "Action:" 开头，格式为 "Action: tool_name(arguments)"
- 在获得结果后，以 "Observation:" 开头进行分析
- 必须调用工具查询最新信息（账户余额、持仓、价格等）才能做决策
- 严格遵守交易限制配置进行操作

""" + custom_prompt

        return base_prompt

    @classmethod
    def _get_enabled_tools(cls, plan) -> Dict:
        """获取启用的工具"""
        try:
            from services.agent_tools import AGENT_TOOLS

            # 检查 plan 参数类型
            from database.models import TradingPlan
            from database.db import get_db

            logger.debug(f"_get_enabled_tools 调用，plan 类型: {type(plan)}, plan 值: {plan}")

            if isinstance(plan, int):
                # 如果传入了 plan_id，需要从数据库获取 plan 对象
                logger.debug(f"传入的是 plan_id: {plan}，开始查询数据库")

                with get_db() as db:
                    plan_obj = db.query(TradingPlan).filter(TradingPlan.id == plan).first()
                    if not plan_obj:
                        logger.error(f"找不到计划: plan_id={plan}")
                        return {}
                    logger.debug(f"成功获取 plan 对象: {plan_obj}")
                    plan = plan_obj
            elif not isinstance(plan, TradingPlan):
                logger.error(f"plan 参数类型错误: {type(plan)}, 期望 TradingPlan 或 int，plan 值: {plan}")
                # 打印调用堆栈以帮助调试
                import traceback
                logger.error(f"调用堆栈: {traceback.format_exc()}")
                return {}

            logger.debug(f"最终 plan 对象类型: {type(plan)}")
            # 双重检查 plan 是否是正确的 TradingPlan 对象
            if not hasattr(plan, 'agent_tools_config'):
                logger.error(f"plan 对象没有 agent_tools_config 属性，plan: {plan}")
                return {}

            tools_config = plan.agent_tools_config or {}
            enabled_tools = {}

            for tool_name, tool_def in AGENT_TOOLS.items():
                if tools_config.get(tool_name, True):  # 默认启用
                    enabled_tools[tool_name] = tool_def

            return enabled_tools

        except Exception as e:
            logger.error(f"获取启用工具失败: {e}")
            import traceback
            logger.error(f"详细错误信息: {traceback.format_exc()}")
            # 返回默认工具配置
            from services.agent_tools import AGENT_TOOLS
            return AGENT_TOOLS

    @classmethod
    def _get_tool_descriptions(cls) -> str:
        """获取工具描述"""
        from services.agent_tools import AGENT_TOOLS

        descriptions = []
        for tool_name, tool_def in AGENT_TOOLS.items():
            desc = f"- **{tool_name}**: {tool_def.description}"
            if tool_def.parameters:
                params = []
                for param_name, param_info in tool_def.parameters.items():
                    required = "必需" if param_name in tool_def.required_params else "可选"
                    params.append(f"  - {param_name}: {param_info.get('description', 'N/A')} ({required})")
                if params:
                    desc += "\n" + "\n".join(params)
            descriptions.append(desc)

        return "\n".join(descriptions)

    @classmethod
    async def _call_openai_react_stream(
        cls,
        llm_config: LLMConfig,
        messages: List[Dict],
        available_tools: Dict,
        enable_thinking: bool = True,
        thinking_style: str = "详细"
    ):
        """OpenAI流式ReAct调用"""
        try:
            import openai

            client = openai.AsyncOpenAI(
                api_key=llm_config.api_key,
                base_url=llm_config.api_base_url
            )

            # 转换工具为OpenAI格式
            tools = []
            for tool_name, tool_def in available_tools.items():
                tool_schema = {
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "description": tool_def.description,
                        "parameters": {
                            "type": "object",
                            "properties": tool_def.parameters,
                            "required": tool_def.required_params
                        }
                    }
                }
                tools.append(tool_schema)

            response = await client.chat.completions.create(
                model=llm_config.model_name,
                max_tokens=llm_config.max_tokens or 2000,
                temperature=llm_config.temperature or 0.7,
                messages=messages,
                tools=tools,
                stream=True
            )

            current_content = ""
            thinking_content = ""  # 用于累积思考内容
            tool_calls = []

            async for chunk in response:
                if chunk.choices and len(chunk.choices) > 0:
                    delta = chunk.choices[0].delta

                    if hasattr(delta, 'content') and delta.content:
                        content_piece = delta.content
                        current_content += content_piece
                        thinking_content += content_piece

                        # 累积输出完整内容，而不是只输出最新片段
                        yield [{"role": "assistant", "content": thinking_content}]

                    elif hasattr(delta, 'tool_calls') and delta.tool_calls:
                        for tool_call in delta.tool_calls:
                            if hasattr(tool_call, 'function'):
                                func_name = tool_call.function.name
                                func_args = tool_call.function.arguments or "{}"

                                if hasattr(tool_call, 'id') and tool_call.id:
                                    tool_call_id = tool_call.id
                                else:
                                    tool_call_id = f"call_{len(tool_calls)}"

                                # 完成工具调用
                                if func_name:
                                    # 添加到当前内容中
                                    action_text = f"\n\n**Action:** {func_name}({func_args})"
                                    current_content += action_text
                                    yield [{"role": "assistant", "content": current_content}]

                                    # 这里应该等待用户确认
                                    # 暂时模拟工具执行
                                    result = await cls._simulate_tool_execution(func_name, func_args)
                                    observation_text = f"\n\n**Observation:** {result}"
                                    current_content += observation_text
                                    yield [{"role": "assistant", "content": current_content}]

        except Exception as e:
            logger.error(f"OpenAI ReAct调用失败: {e}")
            yield [{"role": "assistant", "content": f"❌ 调用失败: {str(e)}"}]

    @classmethod
    async def _call_qwen_react_stream(
        cls,
        llm_config: LLMConfig,
        messages: List[Dict],
        available_tools: Dict,
        enable_thinking: bool = True,
        thinking_style: str = "详细"
    ):
        """通义千问流式ReAct调用"""
        try:
            import openai

            client = openai.AsyncOpenAI(
                api_key=llm_config.api_key,
                base_url=llm_config.api_base_url
            )

            # 转换工具为OpenAI格式
            tools = []
            for tool_name, tool_def in available_tools.items():
                tool_schema = {
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "description": tool_def.description,
                        "parameters": {
                            "type": "object",
                            "properties": tool_def.parameters,
                            "required": tool_def.required_params
                        }
                    }
                }
                tools.append(tool_schema)

            response = await client.chat.completions.create(
                model=llm_config.model_name,
                max_tokens=llm_config.max_tokens or 2000,
                temperature=llm_config.temperature or 0.7,
                messages=messages,
                tools=tools,
                stream=True,
                extra_body={"enable_thinking": True}
            )

            current_content = ""
            thinking_content = ""  # 用于累积思考内容
            reasoning_content = ""  # 用于累积推理内容

            async for chunk in response:
                if chunk.choices and len(chunk.choices) > 0:
                    delta = chunk.choices[0].delta

                    # 处理推理内容（思考过程）
                    if hasattr(delta, 'reasoning_content') and delta.reasoning_content:
                        reasoning_piece = delta.reasoning_content
                        reasoning_content += reasoning_piece
                        # 累积输出完整的思考内容
                        if enable_thinking:
                            formatted_thinking = cls._format_thinking_content(reasoning_content, thinking_style)
                            yield [{"role": "assistant", "content": formatted_thinking}]

                    # 处理常规内容
                    elif hasattr(delta, 'content') and delta.content:
                        content_piece = delta.content
                        current_content += content_piece
                        thinking_content += content_piece

                        # 如果没有推理内容，累积输出完整内容
                        if not reasoning_content and enable_thinking:
                            yield [{"role": "assistant", "content": thinking_content}]

                    elif hasattr(delta, 'tool_calls') and delta.tool_calls:
                        for tool_call in delta.tool_calls:
                            if hasattr(tool_call, 'function'):
                                func_name = tool_call.function.name
                                func_args = tool_call.function.arguments or "{}"

                                if func_name:
                                    # 添加到当前内容中
                                    action_text = f"\n\n**🔧 工具调用:** {func_name}({func_args})"
                                    current_content += action_text
                                    yield [{"role": "assistant", "content": current_content}]

                                    # 检查是否需要工具确认
                                    from services.agent_confirmation_service import confirmation_service

                                    # 获取计划ID（从上下文中传递，这里需要修改）
                                    plan_id = getattr(cls, '_current_plan_id', None)
                                    if plan_id:
                                        confirmation_mode = confirmation_service.get_confirmation_mode(plan_id)

                                        if confirmation_mode == 'manual':
                                            # 创建待确认工具调用
                                            tool_call_id = await confirmation_service.create_pending_tool_call(
                                                plan_id=plan_id,
                                                agent_decision_id=0,  # 这里需要传递实际的决策ID
                                                tool_name=func_name,
                                                tool_args=eval(func_args) if func_args != "{}" else {},
                                                expected_effect=f"执行工具 {func_name}",
                                                risk_warning="请确认是否执行此工具调用",
                                                timeout_minutes=5
                                            )

                                            # 生成确认消息，等待用户确认
                                            confirmation_msg = f"""
⏳ **等待用户确认**

发现工具调用需要您的确认：

**工具名称:** {func_name}
**参数:** {func_args}

请在工具确认面板中操作：
- ✅ 同意执行：点击"同意执行"按钮
- ❌ 拒绝执行：点击"拒绝执行"按钮

系统将等待您的确认后继续执行...
"""
                                            yield [{"role": "assistant", "content": confirmation_msg}]
                                            return  # 结束当前推理，等待确认后继续
                                        elif confirmation_mode == 'disabled':
                                            # 禁用确认模式，直接执行
                                            pass

                                    # 自动或禁用模式下直接执行
                                    result = await cls._simulate_tool_execution(func_name, func_args)
                                    observation_text = f"\n\n**📋 工具结果:** {result}"
                                    current_content += observation_text
                                    yield [{"role": "assistant", "content": current_content}]

        except Exception as e:
            logger.error(f"Qwen ReAct调用失败: {e}")
            yield [{"role": "assistant", "content": f"❌ 调用失败: {str(e)}"}]

    @classmethod
    async def _call_ollama_react_stream(
        cls,
        llm_config: LLMConfig,
        messages: List[Dict],
        available_tools: Dict,
        enable_thinking: bool = True,
        thinking_style: str = "详细"
    ):
        """Ollama流式ReAct调用"""
        try:
            import openai

            client = openai.AsyncOpenAI(
                api_key="ollama",
                base_url=llm_config.api_base_url or "http://localhost:11434/v1"
            )

            # 转换工具为OpenAI格式
            tools = []
            for tool_name, tool_def in available_tools.items():
                tool_schema = {
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "description": tool_def.description,
                        "parameters": {
                            "type": "object",
                            "properties": tool_def.parameters,
                            "required": tool_def.required_params
                        }
                    }
                }
                tools.append(tool_schema)

            response = await client.chat.completions.create(
                model=llm_config.model_name,
                max_tokens=llm_config.max_tokens or 2000,
                temperature=llm_config.temperature or 0.7,
                messages=messages,
                tools=tools,
                stream=True
            )

            current_content = ""
            thinking_content = ""  # 用于累积思考内容

            async for chunk in response:
                if chunk.choices and len(chunk.choices) > 0:
                    delta = chunk.choices[0].delta

                    if hasattr(delta, 'content') and delta.content:
                        content_piece = delta.content
                        current_content += content_piece
                        thinking_content += content_piece

                        # 累积输出完整内容，而不是只输出最新片段
                        yield [{"role": "assistant", "content": thinking_content}]

                    elif hasattr(delta, 'tool_calls') and delta.tool_calls:
                        for tool_call in delta.tool_calls:
                            if hasattr(tool_call, 'function'):
                                func_name = tool_call.function.name
                                func_args = tool_call.function.arguments or "{}"

                                if func_name:
                                    # 添加到当前内容中
                                    action_text = f"\n\n**Action:** {func_name}({func_args})"
                                    current_content += action_text
                                    yield [{"role": "assistant", "content": current_content}]

                                    result = await cls._simulate_tool_execution(func_name, func_args)
                                    observation_text = f"\n\n**Observation:** {result}"
                                    current_content += observation_text
                                    yield [{"role": "assistant", "content": current_content}]

        except Exception as e:
            logger.error(f"Ollama ReAct调用失败: {e}")
            yield [{"role": "assistant", "content": f"❌ 调用失败: {str(e)}"}]

    @classmethod
    async def _simulate_tool_execution(cls, tool_name: str, args_json: str) -> str:
        """模拟工具执行（用于演示）"""
        try:
            import json
            args = json.loads(args_json)

            # 根据工具名称返回模拟结果
            if tool_name == "get_account_balance":
                return f"账户余额查询成功：USDT可用余额 1000.0，冻结余额 0.0"
            elif tool_name == "get_positions":
                return f"持仓查询成功：当前无持仓"
            elif tool_name == "query_prediction_data":
                return f"预测数据查询成功：找到20条最新预测数据，显示上涨趋势"
            elif tool_name == "get_prediction_history":
                return f"历史预测查询成功：最近5个批次的预测记录"
            elif tool_name == "place_limit_order":
                side = args.get('side', 'unknown')
                size = args.get('size', 0)
                price = args.get('price', 0)
                return f"下单请求：{side} {size} BTC @ ${price} (等待用户确认)"
            else:
                return f"工具 {tool_name} 执行完成，参数: {args}"

        except Exception as e:
            return f"工具执行失败: {str(e)}"

