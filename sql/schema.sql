-- KOKEX Database Schema
-- Generated at: 2025-12-16 14:51:09.775395


-- Table: agent_decisions

CREATE TABLE agent_decisions (
	id SERIAL NOT NULL, 
	plan_id INTEGER NOT NULL, 
	training_record_id INTEGER, 
	decision_time TIMESTAMP WITHOUT TIME ZONE, 
	llm_input JSONB, 
	llm_output TEXT, 
	llm_model VARCHAR(100), 
	reasoning TEXT, 
	decision_type VARCHAR(50), 
	tool_calls JSONB, 
	tool_results JSONB, 
	order_ids JSONB, 
	status VARCHAR(20), 
	error_message TEXT, 
	created_at TIMESTAMP WITHOUT TIME ZONE, 
	PRIMARY KEY (id)
)

;

-- Table: agent_prompt_templates

CREATE TABLE agent_prompt_templates (
	id SERIAL NOT NULL, 
	name VARCHAR(100) NOT NULL, 
	description TEXT, 
	content TEXT NOT NULL, 
	category VARCHAR(50), 
	tags JSONB, 
	is_active BOOLEAN, 
	is_default BOOLEAN, 
	created_at TIMESTAMP WITHOUT TIME ZONE, 
	updated_at TIMESTAMP WITHOUT TIME ZONE, 
	PRIMARY KEY (id), 
	UNIQUE (name)
)

;

-- Table: kline_data

CREATE TABLE kline_data (
	id SERIAL NOT NULL, 
	inst_id VARCHAR(50) NOT NULL, 
	interval VARCHAR(10) NOT NULL, 
	timestamp TIMESTAMP WITHOUT TIME ZONE NOT NULL, 
	open FLOAT NOT NULL, 
	high FLOAT NOT NULL, 
	low FLOAT NOT NULL, 
	close FLOAT NOT NULL, 
	volume FLOAT NOT NULL, 
	amount FLOAT NOT NULL, 
	created_at TIMESTAMP WITHOUT TIME ZONE, 
	PRIMARY KEY (id), 
	CONSTRAINT uq_kline_data UNIQUE (inst_id, interval, timestamp)
)

;

-- Table: llm_configs

CREATE TABLE llm_configs (
	id SERIAL NOT NULL, 
	name VARCHAR(100) NOT NULL, 
	provider VARCHAR(50) NOT NULL, 
	api_key VARCHAR(200), 
	api_base_url VARCHAR(200), 
	model_name VARCHAR(100), 
	max_tokens INTEGER, 
	temperature FLOAT, 
	top_p FLOAT, 
	extra_params JSONB, 
	is_active BOOLEAN, 
	is_default BOOLEAN, 
	created_at TIMESTAMP WITHOUT TIME ZONE, 
	updated_at TIMESTAMP WITHOUT TIME ZONE, 
	PRIMARY KEY (id), 
	UNIQUE (name)
)

;

-- Table: prediction_data

CREATE TABLE prediction_data (
	id SERIAL NOT NULL, 
	plan_id INTEGER NOT NULL, 
	training_record_id INTEGER NOT NULL, 
	inference_batch_id VARCHAR(50) NOT NULL, 
	timestamp TIMESTAMP WITHOUT TIME ZONE NOT NULL, 
	open FLOAT NOT NULL, 
	high FLOAT NOT NULL, 
	low FLOAT NOT NULL, 
	close FLOAT NOT NULL, 
	volume FLOAT, 
	amount FLOAT, 
	close_min FLOAT, 
	close_max FLOAT, 
	close_std FLOAT, 
	open_min FLOAT, 
	open_max FLOAT, 
	high_min FLOAT, 
	high_max FLOAT, 
	low_min FLOAT, 
	low_max FLOAT, 
	upward_probability FLOAT, 
	volatility_amplification_probability FLOAT, 
	prediction_time TIMESTAMP WITHOUT TIME ZONE, 
	inference_params JSONB, 
	created_at TIMESTAMP WITHOUT TIME ZONE, 
	PRIMARY KEY (id), 
	CONSTRAINT uq_prediction_data_batch_timestamp UNIQUE (training_record_id, inference_batch_id, timestamp)
)

;

-- Table: system_logs

CREATE TABLE system_logs (
	id SERIAL NOT NULL, 
	plan_id INTEGER, 
	log_type VARCHAR(50) NOT NULL, 
	level VARCHAR(10) NOT NULL, 
	environment VARCHAR(10), 
	message TEXT NOT NULL, 
	details JSONB, 
	created_at TIMESTAMP WITHOUT TIME ZONE, 
	PRIMARY KEY (id)
)

;

-- Table: trade_orders

CREATE TABLE trade_orders (
	id SERIAL NOT NULL, 
	plan_id INTEGER NOT NULL, 
	order_id VARCHAR(100), 
	inst_id VARCHAR(50) NOT NULL, 
	side VARCHAR(10) NOT NULL, 
	order_type VARCHAR(20) NOT NULL, 
	price FLOAT, 
	size FLOAT NOT NULL, 
	status VARCHAR(20), 
	filled_size FLOAT, 
	avg_price FLOAT, 
	is_demo BOOLEAN, 
	is_from_agent BOOLEAN, 
	agent_message_id INTEGER, 
	conversation_id INTEGER, 
	tool_call_id VARCHAR(100), 
	created_at TIMESTAMP WITHOUT TIME ZONE, 
	updated_at TIMESTAMP WITHOUT TIME ZONE, 
	PRIMARY KEY (id), 
	UNIQUE (order_id)
)

;

-- Table: trading_plans

CREATE TABLE trading_plans (
	id SERIAL NOT NULL, 
	plan_name VARCHAR(100) NOT NULL, 
	inst_id VARCHAR(50) NOT NULL, 
	interval VARCHAR(10) NOT NULL, 
	model_version VARCHAR(50), 
	data_start_time TIMESTAMP WITHOUT TIME ZONE, 
	data_end_time TIMESTAMP WITHOUT TIME ZONE, 
	finetune_params JSONB, 
	auto_finetune_schedule JSONB, 
	auto_inference_interval_hours INTEGER, 
	auto_finetune_enabled BOOLEAN, 
	auto_inference_enabled BOOLEAN, 
	auto_agent_enabled BOOLEAN, 
	auto_tool_execution_enabled BOOLEAN, 
	latest_training_record_id INTEGER, 
	llm_config_id INTEGER, 
	agent_prompt TEXT, 
	agent_tools_config JSONB, 
	trading_limits JSONB, 
	initial_capital FLOAT, 
	avg_orders_per_batch INTEGER, 
	max_single_order_ratio FLOAT, 
	capital_management_enabled BOOLEAN, 
	okx_api_key VARCHAR(100), 
	okx_secret_key VARCHAR(200), 
	okx_passphrase VARCHAR(100), 
	is_demo BOOLEAN, 
	status VARCHAR(20), 
	ws_connected BOOLEAN, 
	last_sync_time TIMESTAMP WITHOUT TIME ZONE, 
	last_finetune_time TIMESTAMP WITHOUT TIME ZONE, 
	created_at TIMESTAMP WITHOUT TIME ZONE, 
	updated_at TIMESTAMP WITHOUT TIME ZONE, 
	PRIMARY KEY (id)
)

;

-- Table: training_records

CREATE TABLE training_records (
	id SERIAL NOT NULL, 
	plan_id INTEGER NOT NULL, 
	version VARCHAR(50) NOT NULL, 
	status VARCHAR(20), 
	is_active BOOLEAN, 
	train_params JSONB, 
	data_start_time TIMESTAMP WITHOUT TIME ZONE, 
	data_end_time TIMESTAMP WITHOUT TIME ZONE, 
	data_count INTEGER, 
	train_start_time TIMESTAMP WITHOUT TIME ZONE, 
	train_end_time TIMESTAMP WITHOUT TIME ZONE, 
	train_duration INTEGER, 
	train_metrics JSONB, 
	tokenizer_path VARCHAR(500), 
	predictor_path VARCHAR(500), 
	error_message TEXT, 
	created_at TIMESTAMP WITHOUT TIME ZONE, 
	updated_at TIMESTAMP WITHOUT TIME ZONE, 
	PRIMARY KEY (id), 
	CONSTRAINT uq_training_record_plan_version UNIQUE (plan_id, version)
)

;

-- Table: ws_subscriptions

CREATE TABLE ws_subscriptions (
	id SERIAL NOT NULL, 
	inst_id VARCHAR(50) NOT NULL, 
	interval VARCHAR(10) NOT NULL, 
	is_demo BOOLEAN, 
	status VARCHAR(20), 
	is_connected BOOLEAN, 
	total_received INTEGER, 
	total_saved INTEGER, 
	last_data_time TIMESTAMP WITHOUT TIME ZONE, 
	last_message TEXT, 
	subscribed_channels JSONB, 
	last_order_update TIMESTAMP WITHOUT TIME ZONE, 
	order_count INTEGER, 
	error_count INTEGER, 
	last_error TEXT, 
	last_error_time TIMESTAMP WITHOUT TIME ZONE, 
	started_at TIMESTAMP WITHOUT TIME ZONE, 
	stopped_at TIMESTAMP WITHOUT TIME ZONE, 
	created_at TIMESTAMP WITHOUT TIME ZONE, 
	updated_at TIMESTAMP WITHOUT TIME ZONE, 
	PRIMARY KEY (id), 
	CONSTRAINT uq_ws_subscription UNIQUE (inst_id, interval, is_demo)
)

;

-- Table: agent_conversations

CREATE TABLE agent_conversations (
	id SERIAL NOT NULL, 
	plan_id INTEGER NOT NULL, 
	training_record_id INTEGER, 
	session_name VARCHAR(200), 
	conversation_type VARCHAR(50), 
	status VARCHAR(20), 
	total_messages INTEGER, 
	total_tool_calls INTEGER, 
	started_at TIMESTAMP WITHOUT TIME ZONE, 
	last_message_at TIMESTAMP WITHOUT TIME ZONE, 
	completed_at TIMESTAMP WITHOUT TIME ZONE, 
	created_at TIMESTAMP WITHOUT TIME ZONE, 
	PRIMARY KEY (id), 
	FOREIGN KEY(plan_id) REFERENCES trading_plans (id) ON DELETE CASCADE
)

;

-- Table: task_executions

CREATE TABLE task_executions (
	id SERIAL NOT NULL, 
	plan_id INTEGER NOT NULL, 
	task_type VARCHAR(50) NOT NULL, 
	task_name VARCHAR(200) NOT NULL, 
	task_description TEXT, 
	status VARCHAR(20) NOT NULL, 
	priority INTEGER, 
	scheduled_time TIMESTAMP WITHOUT TIME ZONE, 
	started_at TIMESTAMP WITHOUT TIME ZONE, 
	completed_at TIMESTAMP WITHOUT TIME ZONE, 
	duration_seconds INTEGER, 
	trigger_type VARCHAR(20) NOT NULL, 
	trigger_source VARCHAR(100), 
	input_data JSONB, 
	output_data JSONB, 
	error_message TEXT, 
	progress_percentage INTEGER, 
	task_metadata JSONB, 
	created_at TIMESTAMP WITHOUT TIME ZONE, 
	updated_at TIMESTAMP WITHOUT TIME ZONE, 
	PRIMARY KEY (id), 
	FOREIGN KEY(plan_id) REFERENCES trading_plans (id) ON DELETE CASCADE
)

;

-- Table: agent_messages

CREATE TABLE agent_messages (
	id SERIAL NOT NULL, 
	conversation_id INTEGER NOT NULL, 
	role VARCHAR(20) NOT NULL, 
	content TEXT, 
	message_type VARCHAR(50), 
	react_iteration INTEGER, 
	react_stage VARCHAR(50), 
	tool_call_id VARCHAR(100), 
	tool_name VARCHAR(100), 
	tool_arguments JSONB, 
	tool_result JSONB, 
	tool_status VARCHAR(20), 
	tool_execution_time FLOAT, 
	related_order_id VARCHAR(100), 
	llm_model VARCHAR(100), 
	timestamp TIMESTAMP WITHOUT TIME ZONE, 
	created_at TIMESTAMP WITHOUT TIME ZONE, 
	PRIMARY KEY (id), 
	FOREIGN KEY(conversation_id) REFERENCES agent_conversations (id) ON DELETE CASCADE
)

;

-- Table: order_event_logs

CREATE TABLE order_event_logs (
	id SERIAL NOT NULL, 
	plan_id INTEGER NOT NULL, 
	event_type VARCHAR(50) NOT NULL, 
	order_id VARCHAR(100) NOT NULL, 
	inst_id VARCHAR(50) NOT NULL, 
	side VARCHAR(10) NOT NULL, 
	event_data JSONB NOT NULL, 
	processed_at TIMESTAMP WITHOUT TIME ZONE, 
	agent_conversation_id INTEGER, 
	PRIMARY KEY (id), 
	CONSTRAINT uq_plan_order_event UNIQUE (plan_id, order_id, event_type), 
	FOREIGN KEY(plan_id) REFERENCES trading_plans (id), 
	FOREIGN KEY(agent_conversation_id) REFERENCES agent_conversations (id)
)

;
