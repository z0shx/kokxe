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
python -c "from database.db import init_database; init_database()"

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
# Default URL: http://localhost:7860
```

### Testing
```bash
# Run WebSocket connection tests
python tests/test_ws_header.py

# Test agent tools functionality
python scripts/test_agent_tools.py

# Test configuration center
python scripts/test_config_center.py

# Run inference batch tests
python test_inference_batches.py
```

### Database Operations
```bash
# Run database migrations
python -c "from database.migrate import main; main()"

# Check database connection
python -c "from database.db import get_db; db = next(get_db()); print('Database connected successfully')"

# Initialize database schema
python sql/schema.sql  # Run SQL schema file
```

## Architecture & Code Structure

### Core Architecture

KOKEX follows a **microservices architecture** with the following key layers:

1. **Web UI Layer (`ui/`)**: Gradio-based interface for user interaction
2. **Service Layer (`services/`)**: Business logic and orchestration
3. **Model Layer (`model/`)**: Kronos ML model implementation
4. **Database Layer (`database/`)**: Data persistence and management
5. **API Layer (`api/`)**: External integrations (OKX exchange)

### Key Components

#### **Kronos ML Model (`model/`)**
- **Two-Stage Architecture**: Tokenizer (BSQ) + Predictor (Autoregressive Transformer)
- **Binary Spherical Quantization**: Hierarchical tokenization for OHLCV data
- **Foundation Model**: Pre-trained on extensive financial time series data
- **Fine-tuning Support**: Customizable for specific trading pairs and timeframes

#### **Core Services (`services/`)**
- `TrainingService`: Orchestrates two-stage model training pipeline
- `InferenceService`: Handles model predictions with Monte Carlo sampling
- `AgentDecisionService`: AI-powered trading decisions using LLMs
- `WebSocketConnectionManager`: Real-time market data streaming
- `OKXRestService`: Exchange integration for trade execution
- `ScheduleService`: Automated task scheduling and management

#### **Web Interface (`ui/`)**
- `plan_create.py`: Trading plan creation and configuration
- `plan_detail.py`: Real-time monitoring and control
- `config_center.py`: System configuration management
- `plan_list.py`: Plan overview and management

#### **Database Models (`database/models.py`)**
Key entities:
- `TradingPlan`: Trading strategy configurations
- `KlineData`: Historical OHLCV market data
- `TrainingRecord`: Model training history and metrics
- `AgentDecision`: AI agent decision logs
- `TradeOrder`: Executed trade records

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
- **Training Schedule**: Automated training times
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
The AI agent has access to:
- `get_current_price`: Real-time price data
- `place_order`: Order execution (market/limit orders)
- `cancel_order`: Order cancellation
- `get_positions`: Current position information
- `get_trading_limits`: Risk constraint checking

## Development Guidelines

### Database Operations
- Always use SQLAlchemy sessions from `database.db.get_db()`
- Commit transactions explicitly; use try/catch for error handling
- Use Alembic for schema migrations
- Implement proper indexing for time-series queries

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

### API Integration Best Practices
- OKX API credentials are encrypted in database
- All API operations support demo/live mode switching
- Implement proper rate limiting and error handling
- Use proxy configuration for network access

### Error Handling
- Implement comprehensive logging with structured format
- Use the global logger from `utils.logger`
- All services should handle database connection errors
- WebSocket reconnection is automatic but should be monitored

## Important Implementation Notes

### Security Considerations
- API keys are encrypted using AES-256 encryption
- Demo trading is enforced for testing environments
- Trading limits are strictly enforced
- All operations are logged for audit trails

### Performance Optimization
- Database queries should use proper indexing
- WebSocket connections are pooled and reused
- Model inference supports batch processing
- Async operations are used throughout the service layer

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