"""
新的Agent工具执行方法 - 替换旧的execute_tool方法
"""
from services.agent_error_handler import AgentErrorHandler


async def execute_tool_with_error_handling(self, tool_name: str, params):
    """
    执行工具（带完整错误处理和记录）

    Args:
        tool_name: 工具名称
        params: 工具参数

    Returns:
        执行结果（始终包含continue_conversation字段）
    """
    from services.agent_tools import get_tool, validate_tool_params

    logger = self.logger
    logger.info(f"[{self.environment}] 执行工具: {tool_name}, 参数: {params}")

    try:
        # 验证工具是否存在
        tool = get_tool(tool_name)
        if not tool:
            error_msg = f"工具 {tool_name} 不存在"
            logger.error(f"[{self.environment}] {error_msg}")

            # 记录错误到数据库
            if self.plan_id:
                AgentErrorHandler.record_tool_error(
                    plan_id=self.plan_id,
                    tool_name=tool_name,
                    error_message=error_msg,
                    tool_params=params,
                    conversation_id=self.conversation_id
                )

            return AgentErrorHandler.create_fallback_response(
                tool_name=tool_name,
                error_message=error_msg,
                plan_context={"plan_id": self.plan_id}
            )

        # 验证参数
        is_valid, error_msg = validate_tool_params(tool_name, params)
        if not is_valid:
            logger.error(f"[{self.environment}] 参数验证失败: {error_msg}")

            # 记录错误到数据库
            if self.plan_id:
                AgentErrorHandler.record_tool_error(
                    plan_id=self.plan_id,
                    tool_name=tool_name,
                    error_message=error_msg,
                    tool_params=params,
                    conversation_id=self.conversation_id
                )

            return AgentErrorHandler.create_fallback_response(
                tool_name=tool_name,
                error_message=error_msg,
                plan_context={"plan_id": self.plan_id}
            )

        # 检查交易限制
        is_allowed, limit_msg = self._check_trading_limits(tool_name, params)
        if not is_allowed:
            logger.warning(f"[{self.environment}] 交易限制: {limit_msg}")

            # 记录交易限制错误
            if self.plan_id:
                AgentErrorHandler.record_tool_error(
                    plan_id=self.plan_id,
                    tool_name=tool_name,
                    error_message=limit_msg,
                    tool_params=params,
                    conversation_id=self.conversation_id
                )

            return AgentErrorHandler.create_fallback_response(
                tool_name=tool_name,
                error_message=limit_msg,
                plan_context={"plan_id": self.plan_id}
            )

        # 执行工具
        try:
            # 映射到实际的方法
            method_map = {
                "get_account_balance": self._get_account_balance,
                "get_account_positions": self._get_account_positions,
                "get_order_info": self._get_order_info,
                "get_pending_orders": self._get_pending_orders,
                "get_order_history": self._get_order_history,
                "get_fills": self._get_fills,
                "get_current_price": self._get_current_price,
                "place_limit_order": self._place_limit_order,
                "cancel_order": self._cancel_order,
                "amend_order": self._amend_order,
                "get_prediction_history": self._get_prediction_history,
                "query_prediction_data": self._query_prediction_data,
                "query_historical_kline_data": self._query_historical_kline_data,
                "run_latest_model_inference": self._run_latest_model_inference,
                "get_current_utc_time": self._get_current_utc_time,
            }

            method = method_map.get(tool_name)
            if not method:
                error_msg = f"工具 {tool_name} 未实现"
                logger.error(f"[{self.environment}] {error_msg}")

                if self.plan_id:
                    AgentErrorHandler.record_tool_error(
                        plan_id=self.plan_id,
                        tool_name=tool_name,
                        error_message=error_msg,
                        tool_params=params,
                        conversation_id=self.conversation_id
                    )

                return AgentErrorHandler.create_fallback_response(
                    tool_name=tool_name,
                    error_message=error_msg,
                    plan_context={"plan_id": self.plan_id}
                )

            # 执行工具方法
            result = await method(params)

            # 如果执行失败，记录错误
            if not result.get("success", False):
                error_message = result.get("error", "未知错误")

                # 记录错误到数据库
                if self.plan_id:
                    AgentErrorHandler.record_tool_error(
                        plan_id=self.plan_id,
                        tool_name=tool_name,
                        error_message=error_message,
                        tool_params=params,
                        conversation_id=self.conversation_id
                    )

                # 判断是否应该继续对话
                recent_errors = AgentErrorHandler.get_recent_error_count(self.plan_id) if self.plan_id else 0
                should_continue = AgentErrorHandler.should_continue_conversation(
                    error_message=error_message,
                    tool_name=tool_name,
                    consecutive_errors=recent_errors
                )

                # 创建包含继续对话建议的响应
                fallback_response = AgentErrorHandler.create_fallback_response(
                    tool_name=tool_name,
                    error_message=error_message,
                    plan_context={"plan_id": self.plan_id}
                )
                fallback_response["continue_conversation"] = should_continue

                return fallback_response

            # 成功执行，确保包含continue_conversation字段
            result["continue_conversation"] = True
            return result

        except Exception as e:
            error_msg = f"工具执行异常: {str(e)}"
            logger.error(f"[{self.environment}] {error_msg}", exc_info=True)

            # 记录异常到数据库
            if self.plan_id:
                AgentErrorHandler.record_tool_error(
                    plan_id=self.plan_id,
                    tool_name=tool_name,
                    error_message=error_msg,
                    tool_params=params,
                    conversation_id=self.conversation_id
                )

            # 异常情况下的响应，默认继续对话
            recent_errors = AgentErrorHandler.get_recent_error_count(self.plan_id) if self.plan_id else 0
            should_continue = AgentErrorHandler.should_continue_conversation(
                error_message=error_msg,
                tool_name=tool_name,
                consecutive_errors=recent_errors
            )

            return AgentErrorHandler.create_fallback_response(
                tool_name=tool_name,
                error_message=error_msg,
                plan_context={"plan_id": self.plan_id}
            )

    except Exception as e:
        error_msg = f"工具处理流程异常: {str(e)}"
        logger.error(f"[{self.environment}] {error_msg}", exc_info=True)

        # 记录异常到数据库
        if self.plan_id:
            AgentErrorHandler.record_tool_error(
                plan_id=self.plan_id,
                tool_name=tool_name,
                error_message=error_msg,
                tool_params=params,
                conversation_id=self.conversation_id
            )

        # 默认继续对话
        return AgentErrorHandler.create_fallback_response(
            tool_name=tool_name,
            error_message=error_msg,
            plan_context={"plan_id": self.plan_id}
        )