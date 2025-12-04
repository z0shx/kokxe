"""
实时流式消息处理器
提供真正的流式输出功能，解决异步转同步的性能问题
"""
import asyncio
import json
from typing import AsyncGenerator, List, Dict, Any
from ui.custom_chatbot import process_streaming_messages
from utils.logger import setup_logger

logger = setup_logger(__name__, "streaming_handler.log")


class StreamingHandler:
    """
    实时流式消息处理器

    主要功能：
    - 提供真正的实时流式输出
    - 避免消息批量处理导致的延迟
    - 支持工具调用和结果的独立处理
    """

    def __init__(self):
        self.message_queue = asyncio.Queue()
        self.is_running = False
        self.session_id = None

    async def process_agent_stream_realtime(
        self,
        agent_stream: AsyncGenerator,
        session_id: str
    ) -> AsyncGenerator[List[Dict[str, Any]], None]:
        """
        实时处理Agent流式输出，确保消息立即显示

        Args:
            agent_stream: Agent的异步生成器
            session_id: 会话ID，用于日志追踪

        Yields:
            List[Dict]: 格式化后的消息列表，每批只包含一个消息确保实时显示
        """
        self.is_running = True
        self.session_id = session_id

        logger.info(f"SESSION {session_id} - 开始实时流式处理")

        async def message_processor():
            """异步消息处理器"""
            try:
                message_count = 0

                async for message_batch in agent_stream:
                    if not self.is_running:
                        logger.warning(f"SESSION {session_id} - 流式处理被中断")
                        break

                    message_count += 1
                    logger.debug(f"SESSION {session_id} - 处理消息批次 #{message_count}: {type(message_batch)}")

                    # 立即处理单个消息批次，不等待累积
                    try:
                        processed_messages = process_streaming_messages([message_batch])
                        logger.debug(f"SESSION {session_id} - 处理后得到 {len(processed_messages)} 条消息")

                        # 逐个消息yield，确保实时显示
                        for i, message in enumerate(processed_messages):
                            if not self.is_running:
                                break

                            logger.debug(f"SESSION {session_id} - yield 消息 #{i+1}: {message.get('role', 'unknown')}")
                            yield [message]

                    except Exception as e:
                        logger.error(f"SESSION {session_id} - 消息处理失败: {e}")
                        # 发送错误消息，但不中断流式处理
                        yield [{
                            "role": "assistant",
                            "content": f"❌ 消息处理错误: {str(e)}"
                        }]

            except Exception as e:
                logger.error(f"SESSION {session_id} - 流式处理器异常: {e}")
                yield [{
                    "role": "assistant",
                    "content": f"❌ 流式处理错误: {str(e)}"
                }]
            finally:
                self.is_running = False
                logger.info(f"SESSION {session_id} - 流式处理结束")

        # 返回实时处理的消息流
        async for message in message_processor():
            yield message

    def stop_streaming(self):
        """停止流式处理"""
        logger.info(f"SESSION {self.session_id} - 请求停止流式处理")
        self.is_running = False

    def is_streaming_active(self) -> bool:
        """检查流式处理是否活跃"""
        return self.is_running