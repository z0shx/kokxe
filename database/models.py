"""
数据库模型定义
"""
from datetime import datetime
import pytz
from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, Text, UniqueConstraint, Index, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.dialects.postgresql import JSONB

Base = declarative_base()

# 统一时区：北京时区 UTC+8
BEIJING_TZ = pytz.timezone('Asia/Shanghai')

def now_beijing():
    """获取北京时间"""
    return datetime.now(BEIJING_TZ)


class KlineData(Base):
    """K线数据表"""
    __tablename__ = 'kline_data'

    id = Column(Integer, primary_key=True, autoincrement=True)
    inst_id = Column(String(50), nullable=False, comment='交易对，如 ETH-USDT')
    interval = Column(String(10), nullable=False, comment='时间颗粒度：30m/1H/2H/4H')
    timestamp = Column(DateTime, nullable=False, comment='K线时间戳')
    open = Column(Float, nullable=False, comment='开盘价')
    high = Column(Float, nullable=False, comment='最高价')
    low = Column(Float, nullable=False, comment='最低价')
    close = Column(Float, nullable=False, comment='收盘价')
    volume = Column(Float, nullable=False, comment='成交量')
    amount = Column(Float, nullable=False, comment='成交额')
    created_at = Column(DateTime, default=now_beijing, comment='创建时间')

    # 唯一约束：交易对+时间颗粒度+时间戳
    __table_args__ = (
        UniqueConstraint('inst_id', 'interval', 'timestamp', name='uq_kline_data'),
        Index('idx_inst_interval_timestamp', 'inst_id', 'interval', 'timestamp'),
        Index('idx_timestamp', 'timestamp'),
    )

    def __repr__(self):
        return f"<KlineData({self.inst_id}, {self.interval}, {self.timestamp})>"


class TradingPlan(Base):
    """交易计划表"""
    __tablename__ = 'trading_plans'

    id = Column(Integer, primary_key=True, autoincrement=True)
    plan_name = Column(String(100), nullable=False, comment='计划名称')
    inst_id = Column(String(50), nullable=False, comment='交易对')
    interval = Column(String(10), nullable=False, comment='时间颗粒度')

    # Kronos 模型配置
    model_version = Column(String(50), comment='模型版本')
    data_start_time = Column(DateTime, comment='训练数据开始时间')
    data_end_time = Column(DateTime, comment='训练数据结束时间')
    finetune_params = Column(JSONB, comment='微调参数（JSON）')
    auto_finetune_schedule = Column(JSONB, comment='自动微调时间表（JSON数组），如 ["00:00", "12:00"]')
    auto_inference_interval_hours = Column(Integer, default=4, comment='自动预测间隔时间（小时），如 3, 6, 12, 24')

    # 自动化配置
    auto_finetune_enabled = Column(Boolean, default=False, comment='是否启用自动微调')
    auto_inference_enabled = Column(Boolean, default=False, comment='是否启用自动推理')
    auto_agent_enabled = Column(Boolean, default=False, comment='是否启用自动Agent触发')
    auto_tool_execution_enabled = Column(Boolean, default=False, comment='已废弃：工具确认功能已移除，AI Agent现在直接使用启用的工具')
    latest_training_record_id = Column(Integer, comment='最新的训练记录ID')

    # AI Agent 配置
    llm_config_id = Column(Integer, comment='LLM 配置 ID')
    agent_prompt = Column(Text, comment='AI Agent 提示词')
    agent_tools_config = Column(JSONB, comment='Agent 工具配置（JSON）')
    trading_limits = Column(JSONB, comment='交易限制配置（JSON）')
    
    # 资金管理配置
    initial_capital = Column(Float, default=1000.0, comment='初始本金（USDT）')
    avg_orders_per_batch = Column(Integer, default=10, comment='平均每批订单数（用于平摊策略）')
    max_single_order_ratio = Column(Float, default=0.2, comment='单次订单最大占总资金比例')
    capital_management_enabled = Column(Boolean, default=True, comment='是否启用资金管理策略')

    # OKX API 配置
    okx_api_key = Column(String(100), comment='OKX API Key')
    okx_secret_key = Column(String(200), comment='OKX Secret Key')
    okx_passphrase = Column(String(100), comment='OKX Passphrase')
    is_demo = Column(Boolean, default=True, comment='是否模拟盘')

    # 状态
    status = Column(String(20), default='created', comment='状态：created/running/paused/stopped')
    ws_connected = Column(Boolean, default=False, comment='WebSocket 是否连接')
    last_sync_time = Column(DateTime, comment='最后同步时间')
    last_finetune_time = Column(DateTime, comment='最后微调时间')

    # 时间戳
    created_at = Column(DateTime, default=now_beijing, comment='创建时间')
    updated_at = Column(DateTime, default=now_beijing, onupdate=now_beijing, comment='更新时间')

    # 关联关系
    task_executions = relationship("TaskExecution", back_populates="plan")

    def __repr__(self):
        return f"<TradingPlan({self.plan_name}, {self.inst_id}, {self.interval})>"


class TradeOrder(Base):
    """交易订单表"""
    __tablename__ = 'trade_orders'

    id = Column(Integer, primary_key=True, autoincrement=True)
    plan_id = Column(Integer, nullable=False, comment='关联的交易计划ID')
    order_id = Column(String(100), unique=True, comment='OKX订单ID')

    inst_id = Column(String(50), nullable=False, comment='交易对')
    side = Column(String(10), nullable=False, comment='买卖方向：buy/sell')
    order_type = Column(String(20), nullable=False, comment='订单类型：market/limit等')
    price = Column(Float, comment='价格')
    size = Column(Float, nullable=False, comment='数量')

    status = Column(String(20), comment='订单状态')
    filled_size = Column(Float, default=0.0, comment='已成交数量')
    avg_price = Column(Float, comment='成交均价')

    is_demo = Column(Boolean, default=True, comment='是否模拟盘')
    is_from_agent = Column(Boolean, default=False, comment='是否来自Agent操作的订单')

    # 与Agent对话的关联
    agent_message_id = Column(Integer, comment='触发此订单的Agent消息ID')
    conversation_id = Column(Integer, comment='关联的Agent对话会话ID')
    tool_call_id = Column(String(100), comment='工具调用ID')

    created_at = Column(DateTime, default=now_beijing, comment='创建时间')
    updated_at = Column(DateTime, default=now_beijing, onupdate=now_beijing, comment='更新时间')

    __table_args__ = (
        Index('idx_trade_order_plan_id', 'plan_id'),
        Index('idx_trade_order_order_id', 'order_id'),
    )

    def __repr__(self):
        return f"<TradeOrder({self.order_id}, {self.inst_id}, {self.side})>"


class WebSocketSubscription(Base):
    """WebSocket 订阅状态表"""
    __tablename__ = 'ws_subscriptions'

    id = Column(Integer, primary_key=True, autoincrement=True)
    inst_id = Column(String(50), nullable=False, comment='交易对')
    interval = Column(String(10), nullable=False, comment='时间颗粒度')
    is_demo = Column(Boolean, default=True, comment='是否模拟盘')

    # 连接状态
    status = Column(String(20), default='stopped', comment='状态：running/stopped/error')
    is_connected = Column(Boolean, default=False, comment='是否连接')

    # 数据统计
    total_received = Column(Integer, default=0, comment='总接收消息数')
    total_saved = Column(Integer, default=0, comment='总保存数据条数')
    last_data_time = Column(DateTime, comment='最后接收数据时间')
    last_message = Column(Text, comment='最后一条消息')

    # 订单频道相关
    subscribed_channels = Column(JSONB, comment='订阅的频道列表（JSON）')
    last_order_update = Column(DateTime, comment='最后订单更新时间')
    order_count = Column(Integer, default=0, comment='接收订单数量')

    # 错误信息
    error_count = Column(Integer, default=0, comment='错误次数')
    last_error = Column(Text, comment='最后一次错误')
    last_error_time = Column(DateTime, comment='最后错误时间')

    # 时间戳
    started_at = Column(DateTime, comment='启动时间')
    stopped_at = Column(DateTime, comment='停止时间')
    created_at = Column(DateTime, default=now_beijing, comment='创建时间')
    updated_at = Column(DateTime, default=now_beijing, onupdate=now_beijing, comment='更新时间')

    __table_args__ = (
        UniqueConstraint('inst_id', 'interval', 'is_demo', name='uq_ws_subscription'),
        Index('idx_ws_subscription_status', 'status'),
    )

    def __repr__(self):
        return f"<WebSocketSubscription({self.inst_id}, {self.interval}, {self.status})>"


class SystemLog(Base):
    """系统日志表"""
    __tablename__ = 'system_logs'

    id = Column(Integer, primary_key=True, autoincrement=True)
    plan_id = Column(Integer, comment='关联的交易计划ID（可选）')
    log_type = Column(String(50), nullable=False, comment='日志类型：ws/api/finetune/agent/order')
    level = Column(String(10), nullable=False, comment='日志级别：info/warning/error')
    environment = Column(String(10), comment='交易环境：LIVE/DEMO')
    message = Column(Text, nullable=False, comment='日志消息')
    details = Column(JSONB, comment='详细信息（JSON）')
    created_at = Column(DateTime, default=now_beijing, comment='创建时间')

    __table_args__ = (
        Index('idx_system_log_plan_id', 'plan_id'),
        Index('idx_system_log_log_type', 'log_type'),
        Index('idx_system_log_created_at', 'created_at'),
    )

    def __repr__(self):
        return f"<SystemLog({self.log_type}, {self.level}, {self.created_at})>"


class LLMConfig(Base):
    """LLM 配置表"""
    __tablename__ = 'llm_configs'

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(100), nullable=False, unique=True, comment='配置名称')
    provider = Column(String(50), nullable=False, comment='LLM 提供商：claude/qwen/ollama/openai')

    # API 配置
    api_key = Column(String(200), comment='API Key（加密存储）')
    api_base_url = Column(String(200), comment='API 基础 URL')
    model_name = Column(String(100), comment='模型名称，如 claude-3-sonnet-20240229')

    # 高级配置
    max_tokens = Column(Integer, default=4096, comment='最大 token 数')
    temperature = Column(Float, default=0.7, comment='温度参数')
    top_p = Column(Float, default=1.0, comment='Top P 参数')
    extra_params = Column(JSONB, comment='其他参数（JSON）')

    # 状态
    is_active = Column(Boolean, default=True, comment='是否启用')
    is_default = Column(Boolean, default=False, comment='是否默认配置')

    # 时间戳
    created_at = Column(DateTime, default=now_beijing, comment='创建时间')
    updated_at = Column(DateTime, default=now_beijing, onupdate=now_beijing, comment='更新时间')

    __table_args__ = (
        Index('idx_llm_config_provider', 'provider'),
        Index('idx_llm_config_is_active', 'is_active'),
    )

    def __repr__(self):
        return f"<LLMConfig({self.name}, {self.provider})>"


class AgentPromptTemplate(Base):
    """Agent 提示词模版表"""
    __tablename__ = 'agent_prompt_templates'

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(100), nullable=False, unique=True, comment='模版名称')
    description = Column(Text, comment='模版描述')
    content = Column(Text, nullable=False, comment='提示词内容')

    # 分类和标签
    category = Column(String(50), comment='分类：conservative/aggressive/balanced/custom')
    tags = Column(JSONB, comment='标签（JSON数组）')

    # 状态
    is_active = Column(Boolean, default=True, comment='是否启用')
    is_default = Column(Boolean, default=False, comment='是否默认模版')

    # 时间戳
    created_at = Column(DateTime, default=now_beijing, comment='创建时间')
    updated_at = Column(DateTime, default=now_beijing, onupdate=now_beijing, comment='更新时间')

    __table_args__ = (
        Index('idx_agent_prompt_template_category', 'category'),
        Index('idx_agent_prompt_template_is_active', 'is_active'),
    )

    def __repr__(self):
        return f"<AgentPromptTemplate({self.name}, {self.category})>"


class TrainingRecord(Base):
    """训练记录表"""
    __tablename__ = 'training_records'

    id = Column(Integer, primary_key=True, autoincrement=True)
    plan_id = Column(Integer, nullable=False, comment='关联的交易计划ID')
    version = Column(String(50), nullable=False, comment='版本号，如 v1, v2, v3')

    # 训练状态
    status = Column(String(20), default='waiting', comment='状态：waiting/training/completed/failed')
    is_active = Column(Boolean, default=True, comment='是否启用（可手动关闭版本）')

    # 训练参数
    train_params = Column(JSONB, comment='训练参数（JSON）')
    data_start_time = Column(DateTime, comment='训练数据开始时间')
    data_end_time = Column(DateTime, comment='训练数据结束时间')
    data_count = Column(Integer, comment='训练数据条数')

    # 训练过程
    train_start_time = Column(DateTime, comment='训练开始时间')
    train_end_time = Column(DateTime, comment='训练结束时间')
    train_duration = Column(Integer, comment='训练时长（秒）')

    # 训练结果
    train_metrics = Column(JSONB, comment='训练指标（JSON），如 loss, accuracy等')
    tokenizer_path = Column(String(500), comment='Tokenizer模型保存路径')
    predictor_path = Column(String(500), comment='Predictor模型保存路径')

    # 错误信息
    error_message = Column(Text, comment='失败时的错误信息')

    # 时间戳
    created_at = Column(DateTime, default=now_beijing, comment='创建时间')
    updated_at = Column(DateTime, default=now_beijing, onupdate=now_beijing, comment='更新时间')

    __table_args__ = (
        Index('idx_training_record_plan_id', 'plan_id'),
        Index('idx_training_record_status', 'status'),
        Index('idx_training_record_version', 'version'),
        UniqueConstraint('plan_id', 'version', name='uq_training_record_plan_version'),
    )

    def __repr__(self):
        return f"<TrainingRecord(plan_id={self.plan_id}, version={self.version}, status={self.status})>"


class PredictionData(Base):
    """预测数据表"""
    __tablename__ = 'prediction_data'

    id = Column(Integer, primary_key=True, autoincrement=True)
    plan_id = Column(Integer, nullable=False, comment='关联的交易计划ID')
    training_record_id = Column(Integer, nullable=False, comment='关联的训练记录ID')
    inference_batch_id = Column(String(50), nullable=False, comment='推理批次ID，用于区分同一训练记录的不同推理')

    # 预测数据
    timestamp = Column(DateTime, nullable=False, comment='预测的时间点')
    open = Column(Float, nullable=False, comment='预测开盘价')
    high = Column(Float, nullable=False, comment='预测最高价')
    low = Column(Float, nullable=False, comment='预测最低价')
    close = Column(Float, nullable=False, comment='预测收盘价（平均值）')
    volume = Column(Float, comment='预测成交量')
    amount = Column(Float, comment='预测成交额')

    # 不确定性范围（基于多条蒙特卡罗路径）
    close_min = Column(Float, comment='预测收盘价最小值')
    close_max = Column(Float, comment='预测收盘价最大值')
    close_std = Column(Float, comment='预测收盘价标准差')
    open_min = Column(Float, comment='预测开盘价最小值')
    open_max = Column(Float, comment='预测开盘价最大值')
    high_min = Column(Float, comment='预测最高价最小值')
    high_max = Column(Float, comment='预测最高价最大值')
    low_min = Column(Float, comment='预测最低价最小值')
    low_max = Column(Float, comment='预测最低价最大值')

    # 概率指标
    upward_probability = Column(Float, comment='上涨概率（0-1）')
    volatility_amplification_probability = Column(Float, comment='波动性放大概率（0-1）')

    # 元数据
    prediction_time = Column(DateTime, default=now_beijing, comment='何时生成的预测')
    inference_params = Column(JSONB, comment='推理参数（JSON）')

    # 时间戳
    created_at = Column(DateTime, default=now_beijing, comment='创建时间')

    __table_args__ = (
        Index('idx_prediction_data_plan_id', 'plan_id'),
        Index('idx_prediction_data_training_record_id', 'training_record_id'),
        Index('idx_prediction_data_inference_batch_id', 'inference_batch_id'),
        Index('idx_prediction_data_timestamp', 'timestamp'),
        UniqueConstraint('training_record_id', 'inference_batch_id', 'timestamp', name='uq_prediction_data_batch_timestamp'),
    )

    def __repr__(self):
        return f"<PredictionData(training_record_id={self.training_record_id}, batch={self.inference_batch_id}, timestamp={self.timestamp})>"


class AgentDecision(Base):
    """AI Agent 决策记录表"""
    __tablename__ = 'agent_decisions'

    id = Column(Integer, primary_key=True, autoincrement=True)
    plan_id = Column(Integer, nullable=False, comment='关联的交易计划ID')
    training_record_id = Column(Integer, comment='关联的训练记录ID（触发时的模型版本）')

    # 决策时间
    decision_time = Column(DateTime, default=now_beijing, comment='决策时间')

    # LLM 输入输出
    llm_input = Column(JSONB, comment='LLM输入上下文（JSON）')
    llm_output = Column(Text, comment='LLM原始输出')
    llm_model = Column(String(100), comment='使用的LLM模型')

    # 决策内容
    reasoning = Column(Text, comment='决策理由')
    decision_type = Column(String(50), comment='决策类型：buy/sell/hold/adjust/cancel')

    # 工具调用
    tool_calls = Column(JSONB, comment='工具调用记录（JSON数组）')
    tool_results = Column(JSONB, comment='工具调用结果（JSON数组）')

    # 关联订单
    order_ids = Column(JSONB, comment='关联的OKX订单ID列表（JSON数组）')

    # 执行状态
    status = Column(String(20), default='completed', comment='执行状态：completed/failed/partial')
    error_message = Column(Text, comment='错误信息（如有）')

    # 时间戳
    created_at = Column(DateTime, default=now_beijing, comment='创建时间')

    __table_args__ = (
        Index('idx_agent_decision_plan_id', 'plan_id'),
        Index('idx_agent_decision_training_record_id', 'training_record_id'),
        Index('idx_agent_decision_time', 'decision_time'),
        Index('idx_agent_decision_type', 'decision_type'),
    )

    def __repr__(self):
        return f"<AgentDecision(plan_id={self.plan_id}, decision_type={self.decision_type}, time={self.decision_time})>"


# 工具确认功能已废弃 - PendingToolCall 模型已移除


class AgentConversation(Base):
    """AI Agent 对话会话表"""
    __tablename__ = 'agent_conversations'

    id = Column(Integer, primary_key=True, autoincrement=True)
    plan_id = Column(Integer, ForeignKey('trading_plans.id', ondelete='CASCADE'), nullable=False, comment='关联的交易计划ID')
    training_record_id = Column(Integer, comment='关联的训练记录ID（触发时的模型版本）')
    session_name = Column(String(200), comment='会话名称')

    # 会话类型和状态
    conversation_type = Column(String(50), default='auto_inference', comment='对话类型：auto_inference, manual_chat, analysis')
    status = Column(String(20), default='active', comment='状态：active, completed, archived')

    # 统计信息
    total_messages = Column(Integer, default=0, comment='总消息数')
    total_tool_calls = Column(Integer, default=0, comment='总工具调用数')

    # 时间戳
    started_at = Column(DateTime, default=now_beijing, comment='开始时间')
    last_message_at = Column(DateTime, default=now_beijing, comment='最后消息时间')
    completed_at = Column(DateTime, comment='完成时间')
    created_at = Column(DateTime, default=now_beijing, comment='创建时间')

    __table_args__ = (
        Index('idx_agent_conversation_plan_id', 'plan_id'),
        Index('idx_agent_conversation_training_record_id', 'training_record_id'),
        Index('idx_agent_conversation_type', 'conversation_type'),
        Index('idx_agent_conversation_status', 'status'),
        Index('idx_agent_conversation_started_at', 'started_at'),
    )

    def __repr__(self):
        return f"<AgentConversation(plan_id={self.plan_id}, type={self.conversation_type}, status={self.status})>"


class AgentMessage(Base):
    """AI Agent 对话消息表"""
    __tablename__ = 'agent_messages'

    id = Column(Integer, primary_key=True, autoincrement=True)
    conversation_id = Column(Integer, ForeignKey('agent_conversations.id', ondelete='CASCADE'), nullable=False, comment='关联的对话会话ID')

    # 消息基本信息
    role = Column(String(20), nullable=False, comment='角色：user, assistant, system, tool')
    content = Column(Text, comment='消息内容')
    message_type = Column(String(50), default='text', comment='消息类型：text, thinking, tool_call, tool_result, system')

    # React 循环信息
    react_iteration = Column(Integer, comment='ReAct循环迭代次数')
    react_stage = Column(String(50), comment='ReAct阶段：thought, action, observation')

    # 工具调用信息（当message_type为tool_call时）
    tool_call_id = Column(String(100), comment='工具调用唯一ID')
    tool_name = Column(String(100), comment='工具名称')
    tool_arguments = Column(JSONB, comment='工具参数')
    tool_result = Column(JSONB, comment='工具执行结果')
    tool_status = Column(String(20), default='pending', comment='工具状态：pending, success, failed')
    tool_execution_time = Column(Float, comment='工具执行耗时（秒）')
    related_order_id = Column(String(100), comment='关联的订单ID（如果有）')

    # 模型信息
    llm_model = Column(String(100), comment='使用的LLM模型')

    # 时间戳
    timestamp = Column(DateTime, default=now_beijing, comment='消息时间')
    created_at = Column(DateTime, default=now_beijing, comment='创建时间')

    __table_args__ = (
        Index('idx_agent_message_conversation_id', 'conversation_id'),
        Index('idx_agent_message_role', 'role'),
        Index('idx_agent_message_type', 'message_type'),
        Index('idx_agent_message_timestamp', 'timestamp'),
        Index('idx_agent_message_react_iteration', 'react_iteration'),
    )

    def __repr__(self):
        return f"<AgentMessage(conversation_id={self.conversation_id}, role={self.role}, type={self.message_type})>"


class TaskExecution(Base):
    """任务执行记录"""
    __tablename__ = 'task_executions'

    id = Column(Integer, primary_key=True)
    plan_id = Column(Integer, ForeignKey('trading_plans.id', ondelete='CASCADE'), nullable=False, comment='关联的交易计划ID')
    task_type = Column(String(50), nullable=False, comment='任务类型：auto_finetune, auto_inference, auto_agent')
    task_name = Column(String(200), nullable=False, comment='任务名称')
    task_description = Column(Text, comment='任务描述')
    status = Column(String(20), nullable=False, default='pending', comment='状态：pending, running, completed, failed, cancelled')
    priority = Column(Integer, default=1, comment='任务优先级')
    scheduled_time = Column(DateTime, comment='计划执行时间')
    started_at = Column(DateTime, comment='实际开始时间')
    completed_at = Column(DateTime, comment='完成时间')
    duration_seconds = Column(Integer, comment='执行时长（秒）')
    trigger_type = Column(String(20), nullable=False, comment='触发类型：scheduled, manual')
    trigger_source = Column(String(100), comment='触发源（计划ID等）')
    input_data = Column(JSONB, comment='输入参数')
    output_data = Column(JSONB, comment='输出结果')
    error_message = Column(Text, comment='错误信息')
    progress_percentage = Column(Integer, default=0, comment='进度百分比')
    task_metadata = Column(JSONB, comment='额外元数据')
    created_at = Column(DateTime, default=now_beijing, comment='创建时间')
    updated_at = Column(DateTime, default=now_beijing, onupdate=now_beijing, comment='更新时间')

    # 关联关系
    plan = relationship("TradingPlan", back_populates="task_executions")

    __table_args__ = (
        Index('idx_task_executions_plan_id', 'plan_id'),
        Index('idx_task_executions_status', 'status'),
        Index('idx_task_executions_scheduled_time', 'scheduled_time'),
        Index('idx_task_executions_task_type', 'task_type'),
    )

    def __repr__(self):
        return f"<TaskExecution(id={self.id}, plan_id={self.plan_id}, task_type={self.task_type}, status={self.status})>"


class OrderEventLog(Base):
    """订单事件日志表"""
    __tablename__ = 'order_event_logs'

    # 基本信息
    id = Column(Integer, primary_key=True, autoincrement=True)
    plan_id = Column(Integer, ForeignKey('trading_plans.id'), nullable=False, comment='关联的交易计划ID')
    event_type = Column(String(50), nullable=False, comment='事件类型：buy_order_done, sell_order_done等')
    order_id = Column(String(100), nullable=False, comment='OKX订单ID')
    inst_id = Column(String(50), nullable=False, comment='交易对')
    side = Column(String(10), nullable=False, comment='买卖方向：buy/sell')

    # 事件数据
    event_data = Column(JSONB, nullable=False, comment='订单事件的完整数据（JSON格式）')

    # 时间戳
    processed_at = Column(DateTime, default=now_beijing, comment='处理时间')

    # Agent关联
    agent_conversation_id = Column(Integer, ForeignKey('agent_conversations.id'), comment='关联的Agent对话会话ID')

    # 关联关系
    plan = relationship("TradingPlan")
    agent_conversation = relationship("AgentConversation")

    __table_args__ = (
        # 索引
        Index('idx_order_event_logs_plan_id', 'plan_id'),
        Index('idx_order_event_logs_order_id', 'order_id'),
        Index('idx_order_event_logs_event_type', 'event_type'),
        Index('idx_order_event_logs_processed_at', 'processed_at'),
        Index('idx_order_event_logs_conversation_id', 'agent_conversation_id'),
        Index('idx_order_event_logs_plan_order_event', 'plan_id', 'order_id', 'event_type'),
        # 唯一约束
        UniqueConstraint('plan_id', 'order_id', 'event_type', name='uq_plan_order_event'),
    )

    def __repr__(self):
        return f"<OrderEventLog(id={self.id}, plan_id={self.plan_id}, event_type={self.event_type}, order_id={self.order_id})>"
