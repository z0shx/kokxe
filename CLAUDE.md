# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

KOKEX is an AI-powered cryptocurrency trading platform built on top of the Kronos time series forecasting foundation model. It combines advanced machine learning predictions with AI agent decision-making to provide fully automated trading capabilities. The platform integrates real-time market data processing, model training/inference, and trade execution in a comprehensive end-to-end solution.

## Development Commands

### Installation and Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Configure environment variables
cp .env.example .env
# Edit .env with your database and API credentials

# Initialize database
python -c "from database.db import init_db; init_db()"

# Start the application
python app.py
# OR use the startup script
./start.sh
```

### Running the Application
```bash
# Main application (Gradio web UI)
python app.py

# Access the web interface
# Default URL: http://localhost:7881 (port configured via GRADIO_SERVER_PORT)
```

### Testing
```bash
# Run WebSocket connection tests
python tests/test_ws_header.py

# Test agent tools functionality
python tests/test_real_agent_tools.py
python tests/test_all_langchain_tools.py

# Test prediction analysis tool
python tests/test_prediction_tool.py

# Test LangChain agent integration
python tests/test_real_langchain_agent.py
python tests/test_langchain_agent_final.py

# Test automation workflow
python tests/test_agent_auto_inference.py

# Test configuration center
python scripts/test_config_center.py

# Run specific test cases
python tests/test_latest_prediction_analysis.py
python tests/test_inference_data_offset.py

# Test streaming functionality
python tests/test_streaming_fixes.py
```

### Database Operations
```bash
# Initialize database
python -c "from database.db import init_db; init_db()"

# Run database migrations
python scripts/run_migration.py

# Check database connection
python -c "from database.db import get_db; db = next(get_db()); print('Database connected successfully')"

# Initialize database schema
python sql/schema.sql  # Run SQL schema file

# Fix stuck training records
python scripts/fix_stuck_training.py

# Update agent tools configuration
python scripts/update_agent_tools_config.py [plan_id]

# Update agent configuration
python scripts/update_agent_config.py
```

## Architecture & Code Structure

### Core Architecture

KOKEX follows a **microservices architecture** with the following key layers:

1. **Web UI Layer (`ui/`)**: Gradio-based interface for user interaction
2. **Service Layer (`services/`)**: Business logic and orchestration
3. **Model Layer (`model/`, `services/kronos_trainer.py`)**: Kronos ML model implementation and training orchestration
4. **Database Layer (`database/`)**: Data persistence and management
5. **API Layer (`api/`)**: External integrations (OKX exchange)

### Key Components

#### **Kronos ML Model (`model/`)**
- **Two-Stage Architecture**: Tokenizer (BSQ) + Predictor (Autoregressive Transformer)
- **Binary Spherical Quantization**: Hierarchical tokenization for OHLCV data
- **Foundation Model**: Pre-trained on extensive financial time series data
- **Fine-tuning Support**: Customizable for specific trading pairs and timeframes

#### **Core Services (`services/`)**
- `training_service.py`: Orchestrates two-stage model training pipeline
- `inference_service.py`: Handles model predictions with Monte Carlo sampling
- `langchain_agent.py`: AI-powered trading decisions using LLMs with LangChain
- `agent_tools.py`: AI agent tool definitions and configurations
- `agent_tool_executor.py`: Tool execution engine for AI agents
- `prediction_analysis_service.py`: Multi-batch prediction data analysis
- `trading_tools.py`: OKX trading API integration (order placement, cancellation)
- `automation_service.py`: End-to-end automated trading workflow orchestration
- `unified_scheduler.py`: Unified task scheduling system (replaces deprecated schedulers)
- `conversation_service.py`: Agent conversation and message management
- `ws_connection_manager.py`: Real-time market data streaming
- `plan_service.py`: Trading plan management operations
- `capital_management_service.py`: Risk and capital management
- `kline_event_service.py`: K-line data event handling
- `order_event_service.py`: Order status event processing

#### **Web Interface (`ui/`)**
- `plan_create.py`: Trading plan creation and configuration
- `plan_detail.py`: Real-time monitoring and control
- `config_center.py`: System configuration management
- `plan_list.py`: Plan overview and management

#### **Database Models (`database/models.py`)**
Key entities:
- `TradingPlan`: Trading strategy configurations and automation settings
- `KlineData`: Historical OHLCV market data with UTC+0 timestamps
- `TrainingRecord`: Model training history and performance metrics
- `AgentDecision`: AI agent decision logs (legacy, mostly replaced by AgentMessage)
- `AgentConversation`: Agent conversation sessions and metadata
- `AgentMessage`: Individual agent messages with tool calls and results
- `PredictionData`: Model inference predictions and analysis data
- `TradeOrder`: Executed trade records and order tracking
- `LLMConfig`: LLM provider configurations and credentials
- `SystemLog`: System-level logging and monitoring events

### Data Flow Architecture

```
Real-time Market Data → Model Training → Price Prediction → AI Agent Decision → Trade Execution
        ↓                    ↓                ↓                   ↓              ↓
WebSocket Streaming   Two-stage Pipeline  Monte Carlo       LLM Analysis    OKX API
OKX API              Tokenizer+Predictor  Uncertainty      Risk Management  Order Management
PostgreSQL           Async Training       Quantification   Tool Execution   Database Logging
```

### Trading Workflow

1. **Data Synchronization**: WebSocket connections stream real-time K-line data from OKX
2. **Model Training**: Scheduled two-stage training (Tokenizer → Predictor)
3. **Prediction Generation**: Monte Carlo inference with uncertainty quantification
4. **AI Decision Making**: LLM agents analyze predictions and decide on trades
5. **Trade Execution**: Automated order placement with risk management
6. **Performance Monitoring**: Real-time tracking and analysis

## Configuration Management

### Environment Configuration (`config.py`)
Key configuration sections:
- **Database**: PostgreSQL connection settings
- **OKX API**: Exchange credentials and endpoints
- **Proxy Settings**: Network configuration for API access
- **Model Paths**: Pretrained and fine-tuned model storage
- **LLM Settings**: Anthropic, OpenAI, Qwen API configurations

### Trading Plan Configuration
Each trading plan includes:
- **Instrument**: Trading pair (e.g., ETH-USDT)
- **Timeframe**: K-line interval (1H, 4H, etc.)
- **Training Schedule**: Automated training times (cron-based)
- **LLM Selection**: AI agent provider
- **Risk Parameters**: Trading limits and controls

## Kronos Model Details

### Model Architecture
- **Tokenizer**: Encoder-decoder with Binary Spherical Quantization (10+10 bit tokens)
- **Predictor**: 12-layer autoregressive transformer (832-dim, 16 heads)
- **Context Windows**: Configurable lookback (default 512) and prediction (default 48)
- **Hierarchical Tokens**: Coarse (s1) and fine (s2) token prediction

### Training Pipeline
- **Stage 1**: Tokenizer training (25 epochs, reconstruction + BSQ entropy loss)
- **Stage 2**: Predictor training (50 epochs, frozen tokenizer, autoregressive loss)
- **Data Source**: PostgreSQL KlineData with time-based splitting
- **Multi-GPU**: PyTorch DistributedDataParallel support

### Inference Characteristics
- **Monte Carlo Sampling**: Configurable sample trajectories for uncertainty
- **Temperature Control**: Adjustable randomness in predictions
- **Probability Metrics**: Upward/volatility probabilities from prediction distribution
- **Batch Processing**: Efficient multiple prediction handling

## AI Agent Integration

### Supported LLM Providers
- **Anthropic Claude**: Advanced reasoning and decision-making
- **OpenAI GPT**: GPT-4/GPT-3.5 for trade analysis
- **Alibaba Qwen**: Multi-language support and cost efficiency

### Agent Capabilities
- **Decision Making**: Analyze predictions and recommend trades
- **Tool Calling**: Execute trade orders via OKX API
- **Risk Management**: Respect trading limits and controls
- **Decision Logging**: Complete audit trail for all decisions

### Tool Integration
The AI agent has access to 17+ tools across three categories:

**Query Tools** (9 tools):
- `get_account_balance`: Query account balances and available funds
- `get_account_positions`: Query current positions and holdings
- `get_order_info`: Query specific order details and status
- `get_pending_orders`: Query all active/limit orders
- `get_order_history`: Query historical order records
- `get_fills`: Query trade execution details
- `get_current_price`: Get real-time market prices
- `get_latest_prediction_analysis`: Analyze latest multi-batch prediction data
- `get_prediction_history`: Query historical model predictions
- `query_prediction_data`: Search stored prediction data by criteria
- `query_historical_kline_data`: Query real historical K-line data
- `get_current_utc_time`: Get current Beijing time

**Trade Tools** (3 tools):
- `place_limit_order`: Execute limit orders with risk management
- `cancel_order`: Cancel pending orders
- `amend_order`: Modify existing order parameters
- `place_order`: General order execution (market/limit)

**Monitor Tools** (1 tool):
- `run_latest_model_inference`: Trigger model inference on latest data

### Agent Context Persistence
The system maintains complete agent context through:
- **AgentConversation**: Session-level conversation metadata
- **AgentMessage**: Individual messages with tool calls, results, and execution times
- **Tool Call Tracking**: Complete audit trail with tool_call_id linking calls to results
- **Error Handling**: Comprehensive error logging and recovery mechanisms
- **Conversation History**: Persistent context loading for continued conversations

## Development Guidelines

### Database Operations
- Always use SQLAlchemy sessions from `database.db.get_db()`
- Commit transactions explicitly; use try/catch for error handling
- Use custom migration system in `database/migrate.py` (not Alembic)
- Implement proper indexing for time-series queries
- All timestamps use Beijing timezone (Asia/Shanghai) via `database.models.now_beijing()`

### WebSocket Management
- Use `WebSocketConnectionManager` for connection lifecycle
- Each trading plan maintains independent WebSocket connections
- Connections are automatically managed and health-checked
- Status is synchronized between connection manager and database

### Model Service Integration
- Models are automatically downloaded from Hugging Face Hub
- Use `ModelService` for model loading and management
- Training operations are asynchronous with progress tracking
- Model versioning is handled automatically (v1, v2, etc.)
- Multi-GPU training support via PyTorch DistributedDataParallel
- CUDA memory cleanup after training completion

### API Integration Best Practices
- OKX API credentials are encrypted using AES-256 in database
- All API operations support demo/live mode switching
- Implement proper rate limiting and error handling
- Use proxy configuration for network access
- WebSocket connections support automatic reconnection and health monitoring

### Agent Tool Development
- Tool definitions are centralized in `services/agent_tools.py`
- Tools are categorized as QUERY, TRADE, or MONITOR
- Each tool must implement proper parameter validation
- Tool execution is tracked with unique tool_call_id for audit trails
- Tool results are automatically persisted to AgentMessage table
- Use `services/agent_tool_executor.py` for tool execution logic

### Error Handling
- Implement comprehensive logging with structured format using `utils.logger`
- Use `services/agent_error_handler.py` for agent-specific error handling
- All services should handle database connection errors gracefully
- WebSocket reconnection is automatic but should be monitored via logs
- Agent execution failures are logged with full context to SystemLog table
- Training and inference operations have timeout and retry mechanisms

## Important Implementation Notes

### Security Considerations
- API keys are encrypted using AES-256 encryption
- Demo trading is enforced for testing environments
- Trading limits are strictly enforced
- All operations are logged for audit trails

### Performance Optimization
- Database queries should use proper indexing
- WebSocket connections are pooled and reused per trading plan
- Model inference supports batch processing
- Async/await patterns used throughout the service layer for concurrency
- Training operations use global locks to prevent resource conflicts

### Testing Strategy
- Unit tests for individual components
- Integration tests for API connections
- End-to-end workflow validation
- WebSocket lifecycle testing

### Monitoring and Observability
- Real-time WebSocket connection status
- Training progress and model metrics
- Agent decision logging and analysis
- Trade execution monitoring and reporting

This platform represents a sophisticated integration of advanced ML capabilities with practical trading automation, providing a complete solution for AI-powered cryptocurrency trading.