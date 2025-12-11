"""
模型推理服务
负责Kronos模型的推理预测
"""
import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Dict, List
from database.db import get_db
from database.models import TradingPlan, TrainingRecord, PredictionData, KlineData
from utils.logger import setup_logger
from utils.timezone_helper import format_datetime_full_beijing, format_time_range_utc8
from sqlalchemy import and_

logger = setup_logger(__name__, "inference_service.log")


def _convert_numpy_to_python(obj):
    """
    递归转换numpy类型为Python原生类型

    Args:
        obj: 任意对象

    Returns:
        转换后的Python原生类型对象
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: _convert_numpy_to_python(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_convert_numpy_to_python(item) for item in obj]
    else:
        return obj


class InferenceService:
    """模型推理服务"""

    @classmethod
    def _compute_multi_path_statistics(
        cls,
        all_predictions: list,
        historical_df: pd.DataFrame,
        predict_window: int
    ) -> pd.DataFrame:
        """
        计算多条蒙特卡罗路径的统计量和概率指标

        Args:
            all_predictions: List[DataFrame] - 多条预测路径
            historical_df: DataFrame - 历史K线数据
            predict_window: int - 预测窗口大小

        Returns:
            DataFrame - 包含平均值、不确定性范围、概率指标的预测结果
        """
        import pandas as pd

        logger.info(f"计算 {len(all_predictions)} 条路径的统计量...")

        # 初始化结果容器
        timestamps = all_predictions[0].index
        result_data = []

        # 对每个时间点计算统计量
        for timestamp in timestamps:
            # 收集所有路径在该时间点的值
            close_values = [df.loc[timestamp, 'close'] for df in all_predictions]
            open_values = [df.loc[timestamp, 'open'] for df in all_predictions]
            high_values = [df.loc[timestamp, 'high'] for df in all_predictions]
            low_values = [df.loc[timestamp, 'low'] for df in all_predictions]
            volume_values = [df.loc[timestamp, 'volume'] for df in all_predictions if 'volume' in df.columns]
            amount_values = [df.loc[timestamp, 'amount'] for df in all_predictions if 'amount' in df.columns]

            # 计算统计量
            result_data.append({
                'timestamp': timestamp,
                # 平均值
                'close': float(np.mean(close_values)),
                'open': float(np.mean(open_values)),
                'high': float(np.mean(high_values)),
                'low': float(np.mean(low_values)),
                'volume': float(np.mean(volume_values)) if volume_values else 0.0,
                'amount': float(np.mean(amount_values)) if amount_values else 0.0,
                # 不确定性范围
                'close_min': float(np.min(close_values)),
                'close_max': float(np.max(close_values)),
                'close_std': float(np.std(close_values)),
                'open_min': float(np.min(open_values)),
                'open_max': float(np.max(open_values)),
                'high_min': float(np.min(high_values)),
                'high_max': float(np.max(high_values)),
                'low_min': float(np.min(low_values)),
                'low_max': float(np.max(low_values)),
            })

        # 转换为 DataFrame
        result_df = pd.DataFrame(result_data)
        result_df.set_index('timestamp', inplace=True)

        # 计算概率指标
        last_known_price = float(historical_df['close'].iloc[-1])
        logger.info(f"最后已知价格: {last_known_price}")

        # 1. 计算上涨概率（未来预测期末价格 > 最后已知价格）
        upward_probability = cls._calculate_upward_probability(
            all_predictions,
            last_known_price
        )
        logger.info(f"上涨概率: {upward_probability:.2%}")

        # 2. 计算波动性放大概率
        volatility_amplification_probability = cls._calculate_volatility_amplification_probability(
            all_predictions,
            historical_df
        )
        logger.info(f"波动性放大概率: {volatility_amplification_probability:.2%}")

        # 将概率指标添加到每一行（作为全局指标）
        result_df['upward_probability'] = upward_probability
        result_df['volatility_amplification_probability'] = volatility_amplification_probability

        return result_df

    @classmethod
    def _calculate_upward_probability(
        cls,
        all_predictions: list,
        last_known_price: float
    ) -> float:
        """
        计算上涨概率

        Args:
            all_predictions: List[DataFrame] - 多条预测路径
            last_known_price: float - 最后已知价格

        Returns:
            float - 上涨概率 (0-1)
        """
        upward_count = 0
        for pred_df in all_predictions:
            # 取预测期末的收盘价
            future_price = pred_df['close'].iloc[-1]
            if future_price > last_known_price:
                upward_count += 1

        return upward_count / len(all_predictions)

    @classmethod
    def _calculate_volatility_amplification_probability(
        cls,
        all_predictions: list,
        historical_df: pd.DataFrame
    ) -> float:
        """
        计算波动性放大概率

        Args:
            all_predictions: List[DataFrame] - 多条预测路径
            historical_df: DataFrame - 历史K线数据

        Returns:
            float - 波动性放大概率 (0-1)
        """
        # 1. 计算历史波动率（使用收盘价的收益率标准差）
        historical_returns = historical_df['close'].pct_change().dropna()
        historical_volatility = float(historical_returns.std())

        # 2. 统计预测波动率超过历史波动率的路径数量
        amplified_count = 0
        for pred_df in all_predictions:
            pred_returns = pred_df['close'].pct_change().dropna()
            pred_volatility = float(pred_returns.std())

            if pred_volatility > historical_volatility:
                amplified_count += 1

        return amplified_count / len(all_predictions)

    @classmethod
    async def run_inference_async(
        cls,
        training_record_id: int,
        temperature: float = None,
        top_p: float = None,
        sample_count: int = None,
        lookback_window: int = None,
        predict_window: int = None,
        data_offset: int = None
    ) -> Dict:
        """
        异步执行推理，支持自定义参数

        Args:
            training_record_id: 训练记录ID
            temperature: 温度参数（可选，覆盖计划配置）
            top_p: Top-p参数（可选，覆盖计划配置）
            sample_count: 采样数量（可选，覆盖计划配置）
            lookback_window: 回溯窗口（可选，覆盖计划配置）
            predict_window: 预测窗口（可选，覆盖计划配置）
            data_offset: 数据偏移（可选，覆盖计划配置）

        Returns:
            推理结果字典
        """
        try:
            # 在线程池中执行推理
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                cls._inference_sync_with_params,
                training_record_id,
                temperature,
                top_p,
                sample_count,
                lookback_window,
                predict_window,
                data_offset
            )
            return result
        except Exception as e:
            logger.error(f"异步推理失败: training_record_id={training_record_id}, error={e}")
            return {'success': False, 'error': str(e)}

    @classmethod
    async def start_inference(cls, training_id: int) -> bool:
        """
        启动推理任务

        Args:
            training_id: 训练记录ID

        Returns:
            是否成功
        """
        try:
            # 在线程池中执行推理
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                cls._inference_sync,
                training_id
            )

            # 如果推理成功且启用了自动Agent，触发Agent
            if result['success']:
                with get_db() as db:
                    record = db.query(TrainingRecord).filter(
                        TrainingRecord.id == training_id
                    ).first()

                    if record:
                        plan = db.query(TradingPlan).filter(
                            TradingPlan.id == record.plan_id
                        ).first()

                        if plan and plan.auto_agent_enabled:
                            logger.info(f"自动触发Agent: training_id={training_id}")
                            from services.langchain_agent import agent_service
                            asyncio.create_task(agent_service.auto_decision_wrapper(
                                plan.id,
                                training_id
                            ))

            return result['success']

        except Exception as e:
            logger.error(f"启动推理失败: {e}")
            import traceback
            traceback.print_exc()
            return False

    @classmethod
    def _inference_sync_with_params(
        cls,
        training_id: int,
        temperature: float = None,
        top_p: float = None,
        sample_count: int = None,
        lookback_window: int = None,
        predict_window: int = None,
        data_offset: int = None
    ) -> Dict:
        """
        同步推理（在线程池中执行），支持自定义参数

        Args:
            training_id: 训练记录ID
            temperature: 温度参数（可选，覆盖计划配置）
            top_p: Top-p参数（可选，覆盖计划配置）
            sample_count: 采样数量（可选，覆盖计划配置）
            lookback_window: 回溯窗口（可选，覆盖计划配置）
            predict_window: 预测窗口（可选，覆盖计划配置）
            data_offset: 数据偏移（可选，覆盖计划配置）

        Returns:
            结果字典: {
                'success': bool,
                'predictions_count': int,
                'error': str  # 错误信息（如有）
            }
        """
        try:
            import pandas as pd
            from services.kronos_trainer import KronosInferencer

            with get_db() as db:
                # 获取训练记录
                training_record = db.query(TrainingRecord).filter(
                    TrainingRecord.id == training_id
                ).first()

                if not training_record or training_record.status != 'completed':
                    return {'success': False, 'error': '训练记录不存在或未完成'}

                # 获取计划信息
                plan = db.query(TradingPlan).filter(
                    TradingPlan.id == training_record.plan_id
                ).first()

                if not plan:
                    return {'success': False, 'error': '计划不存在'}

                # 获取推理参数 - 优先使用传入参数，然后使用计划配置
                finetune_params = plan.finetune_params or {}

                # 确保参数结构正确，处理可能的结构不匹配问题
                if 'data' not in finetune_params:
                    finetune_params['data'] = {}
                if 'inference' not in finetune_params:
                    finetune_params['inference'] = {}

                # 处理扁平结构参数（兼容性）
                # 如果参数在顶层，移动到对应的嵌套结构中
                if 'lookback_window' in finetune_params and 'lookback_window' not in finetune_params['data']:
                    finetune_params['data']['lookback_window'] = finetune_params['lookback_window']
                if 'predict_window' in finetune_params and 'predict_window' not in finetune_params['data']:
                    finetune_params['data']['predict_window'] = finetune_params['predict_window']

                # 使用传入的参数覆盖计划配置（如果提供）
                final_lookback_window = lookback_window if lookback_window is not None else finetune_params['data'].get('lookback_window', 512)
                final_predict_window = predict_window if predict_window is not None else finetune_params['data'].get('predict_window', 48)
                final_T = temperature if temperature is not None else finetune_params['inference'].get('temperature', 1.0)
                final_top_p = top_p if top_p is not None else finetune_params['inference'].get('top_p', 0.9)
                final_sample_count = sample_count if sample_count is not None else finetune_params['inference'].get('sample_count', 30)
                final_data_offset = data_offset if data_offset is not None else finetune_params['inference'].get('data_offset', 0)

                logger.info(
                    f"开始推理: training_id={training_id}, "
                    f"lookback={final_lookback_window}, predict={final_predict_window}, data_offset={final_data_offset}, "
                    f"temperature={final_T}, top_p={final_top_p}, sample_count={final_sample_count}"
                )

                # 获取最新的历史数据作为输入
                # 如果有数据偏移，则需要获取更多数据以便向前偏移
                data_end_time = datetime.now()
                data_start_time = data_end_time - timedelta(days=365)  # 最多取一年数据

                # 计算实际需要的数据量：final_lookback_window + final_data_offset
                total_data_needed = final_lookback_window + final_data_offset

                historical_klines = db.query(KlineData).filter(
                    and_(
                        KlineData.inst_id == plan.inst_id,
                        KlineData.interval == plan.interval,
                        KlineData.timestamp >= data_start_time,
                        KlineData.timestamp <= data_end_time
                    )
                ).order_by(KlineData.timestamp.desc()).limit(total_data_needed).all()

                if len(historical_klines) < total_data_needed:
                    return {
                        'success': False,
                        'error': f'历史数据不足: 需要{total_data_needed}条（lookback={final_lookback_window} + offset={final_data_offset}），实际{len(historical_klines)}条'
                    }

                # 反转数据（从旧到新）
                historical_klines = list(reversed(historical_klines))

                # 如果有偏移，跳过最后 final_data_offset 条数据（最新的数据）
                # 这样可以使用更早的历史数据进行预测
                if final_data_offset > 0:
                    historical_klines = historical_klines[:-final_data_offset]
                    logger.info(f"应用数据偏移: 跳过最新的{final_data_offset}条数据，使用更早的历史数据")

                df_data = []
                for kline in historical_klines:
                    df_data.append({
                        'timestamps': kline.timestamp,
                        'open': kline.open,
                        'high': kline.high,
                        'low': kline.low,
                        'close': kline.close,
                        'volume': kline.volume,
                        'amount': kline.amount
                    })

                historical_df = pd.DataFrame(df_data)
                historical_df['timestamps'] = pd.to_datetime(historical_df['timestamps'])
                historical_df.set_index('timestamps', inplace=True)

                logger.info(f"历史数据: {len(historical_df)} 条，时间范围: {historical_df.index[0]} 到 {historical_df.index[-1]}")

                # 执行推理
                inferencer = KronosInferencer(
                    tokenizer_path=training_record.tokenizer_path,
                    predictor_path=training_record.predictor_path,
                    lookback_window=final_lookback_window,
                    predict_window=final_predict_window
                )

                # 生成推理批次ID
                import uuid
                inference_batch_id = datetime.now().strftime('%Y%m%d%H%M%S') + '_' + uuid.uuid4().hex[:8]

                # 执行蒙特卡罗多路径采样
                all_predictions = []
                for i in range(final_sample_count):
                    try:
                        prediction_df = inferencer.predict(historical_df, temperature=final_T, top_p=final_top_p)
                        all_predictions.append(prediction_df)
                        logger.debug(f"完成第 {i+1}/{final_sample_count} 条路径采样")
                    except Exception as e:
                        logger.warning(f"第 {i+1} 条路径采样失败: {e}")
                        continue

                if not all_predictions:
                    return {'success': False, 'error': '所有采样路径都失败'}

                logger.info(f"成功生成 {len(all_predictions)} 条预测路径")

                # 计算统计量
                result_df = cls._compute_multi_path_statistics(
                    all_predictions,
                    historical_df,
                    final_predict_window
                )

                # 保存预测数据到数据库
                predictions_count = 0
                for timestamp, row in result_df.iterrows():
                    prediction = PredictionData(
                        plan_id=plan.id,
                        training_record_id=training_record.id,
                        inference_batch_id=inference_batch_id,
                        timestamp=timestamp,
                        open=float(row['open']),
                        high=float(row['high']),
                        low=float(row['low']),
                        close=float(row['close']),
                        volume=float(row['volume']) if 'volume' in row else None,
                        amount=float(row['amount']) if 'amount' in row else None,
                        close_min=float(row['close_min']),
                        close_max=float(row['close_max']),
                        close_std=float(row['close_std']),
                        open_min=float(row['open_min']),
                        open_max=float(row['open_max']),
                        high_min=float(row['high_min']),
                        high_max=float(row['high_max']),
                        low_min=float(row['low_min']),
                        low_max=float(row['low_max']),
                        upward_probability=float(row.get('upward_probability', None)) if pd.notna(row.get('upward_probability')) else None,
                        volatility_amplification_probability=float(row.get('volatility_amplification_probability', None)) if pd.notna(row.get('volatility_amplification_probability')) else None,
                        # 推理参数 - 保存实际使用的参数
                        inference_params=_convert_numpy_to_python({
                            'lookback_window': final_lookback_window,
                            'predict_window': final_predict_window,
                            'temperature': final_T,
                            'top_p': final_top_p,
                            'sample_count': final_sample_count,
                            'data_offset': final_data_offset
                        })
                    )

                    db.add(prediction)
                    predictions_count += 1

                db.commit()
                logger.info(f"推理成功: training_id={training_id}, 保存 {predictions_count} 条预测记录")

                return {
                    'success': True,
                    'predictions_count': predictions_count,
                    'inference_batch_id': inference_batch_id
                }

        except Exception as e:
            logger.error(f"推理失败: training_id={training_id}, error={e}")
            import traceback
            traceback.print_exc()
            return {'success': False, 'error': str(e)}

    @classmethod
    def _inference_sync(cls, training_id: int) -> Dict:
        """
        同步推理（在线程池中执行）

        Returns:
            结果字典: {
                'success': bool,
                'predictions_count': int,
                'error': str  # 错误信息（如有）
            }
        """
        try:
            import pandas as pd
            from services.kronos_trainer import KronosInferencer

            with get_db() as db:
                # 获取训练记录
                training_record = db.query(TrainingRecord).filter(
                    TrainingRecord.id == training_id
                ).first()

                if not training_record or training_record.status != 'completed':
                    return {'success': False, 'error': '训练记录不存在或未完成'}

                # 获取计划信息
                plan = db.query(TradingPlan).filter(
                    TradingPlan.id == training_record.plan_id
                ).first()

                if not plan:
                    return {'success': False, 'error': '计划不存在'}

                # 获取推理参数
                finetune_params = plan.finetune_params or {}

                # 确保参数结构正确，处理可能的结构不匹配问题
                if 'data' not in finetune_params:
                    finetune_params['data'] = {}
                if 'inference' not in finetune_params:
                    finetune_params['inference'] = {}

                # 处理扁平结构参数（兼容性）
                # 如果参数在顶层，移动到对应的嵌套结构中
                if 'lookback_window' in finetune_params and 'lookback_window' not in finetune_params['data']:
                    finetune_params['data']['lookback_window'] = finetune_params['lookback_window']
                if 'predict_window' in finetune_params and 'predict_window' not in finetune_params['data']:
                    finetune_params['data']['predict_window'] = finetune_params['predict_window']

                lookback_window = finetune_params['data'].get('lookback_window', 512)
                predict_window = finetune_params['data'].get('predict_window', 48)
                T = finetune_params['inference'].get('temperature', 1.0)
                top_p = finetune_params['inference'].get('top_p', 0.9)
                sample_count = finetune_params['inference'].get('sample_count', 30)  # 默认30条蒙特卡罗路径
                data_offset = finetune_params['inference'].get('data_offset', 0)  # 数据偏移量

                logger.info(
                    f"开始推理: training_id={training_id}, "
                    f"lookback={lookback_window}, predict={predict_window}, data_offset={data_offset}"
                )

                # 获取最新的历史数据作为输入
                # 如果有数据偏移，则需要获取更多数据以便向前偏移
                data_end_time = datetime.now()
                data_start_time = data_end_time - timedelta(days=365)  # 最多取一年数据

                # 计算实际需要的数据量：lookback_window + data_offset
                total_data_needed = lookback_window + data_offset

                historical_klines = db.query(KlineData).filter(
                    and_(
                        KlineData.inst_id == plan.inst_id,
                        KlineData.interval == plan.interval,
                        KlineData.timestamp >= data_start_time,
                        KlineData.timestamp <= data_end_time
                    )
                ).order_by(KlineData.timestamp.desc()).limit(total_data_needed).all()

                if len(historical_klines) < total_data_needed:
                    return {
                        'success': False,
                        'error': f'历史数据不足: 需要{total_data_needed}条（lookback={lookback_window} + offset={data_offset}），实际{len(historical_klines)}条'
                    }

                # 反转数据（从旧到新）
                historical_klines = list(reversed(historical_klines))

                # 如果有偏移，跳过最后 data_offset 条数据（最新的数据）
                # 这样可以使用更早的历史数据进行预测
                if data_offset > 0:
                    historical_klines = historical_klines[:-data_offset]
                    logger.info(f"应用数据偏移: 跳过最新的{data_offset}条数据，使用更早的历史数据")

                df_data = []
                for kline in historical_klines:
                    df_data.append({
                        'timestamps': kline.timestamp,
                        'open': kline.open,
                        'high': kline.high,
                        'low': kline.low,
                        'close': kline.close,
                        'volume': kline.volume,
                        'amount': kline.amount
                    })

                historical_df = pd.DataFrame(df_data)
                # 明确指定 timestamps 为 UTC 时间，防止本地时区解释
                historical_df['timestamps'] = pd.to_datetime(historical_df['timestamps'], utc=True)

                logger.info(f"历史数据准备完成: {len(historical_df)}条")

                # 加载推理器
                logger.info(f"加载模型: tokenizer={training_record.tokenizer_path}, predictor={training_record.predictor_path}")

                inferencer = KronosInferencer(
                    tokenizer_path=training_record.tokenizer_path,
                    predictor_path=training_record.predictor_path,
                    interval=plan.interval,
                    device="cuda:0" if __import__('torch').cuda.is_available() else "cpu"
                )

                # 执行推理
                logger.info(f"开始推理预测...（采样 {sample_count} 条路径）")

                # 多路径蒙特卡罗采样
                if sample_count > 1:
                    # 执行多次采样
                    all_predictions = []
                    for i in range(sample_count):
                        logger.info(f"执行第 {i+1}/{sample_count} 次采样...")
                        pred_df_sample = inferencer.predict(
                            historical_data=historical_df,
                            pred_len=predict_window,
                            T=T,
                            top_p=top_p,
                            sample_count=1  # 每次只采样1条路径
                        )
                        if pred_df_sample is not None and len(pred_df_sample) > 0:
                            all_predictions.append(pred_df_sample)

                    if len(all_predictions) == 0:
                        return {
                            'success': False,
                            'error': '推理未生成有效预测数据'
                        }

                    logger.info(f"多路径采样完成，共 {len(all_predictions)} 条路径")

                    # 计算统计量
                    pred_df = cls._compute_multi_path_statistics(
                        all_predictions,
                        historical_df,
                        predict_window
                    )
                else:
                    # 单次采样
                    pred_df = inferencer.predict(
                        historical_data=historical_df,
                        pred_len=predict_window,
                        T=T,
                        top_p=top_p,
                        sample_count=sample_count
                    )

                if pred_df is None or len(pred_df) == 0:
                    return {
                        'success': False,
                        'error': '推理未生成有效预测数据'
                    }

                logger.info(f"推理成功，生成 {len(pred_df)} 条预测数据")

                # 生成推理批次ID（时间戳格式）
                import uuid
                inference_batch_id = datetime.now().strftime('%Y%m%d%H%M%S') + '_' + str(uuid.uuid4())[:8]
                logger.info(f"推理批次ID: {inference_batch_id}")

                # 保存预测数据到数据库
                # pred_df 的 index 是时间戳，包含平均值、不确定性范围和概率指标
                for timestamp, row in pred_df.iterrows():
                    # 存储UTC时间戳，在UI显示时转换为北京时间
                    # 确保时间戳处理的一致性，避免双重时区转换

                    # 转换所有numpy类型为Python原生类型
                    prediction = PredictionData(
                        plan_id=plan.id,
                        training_record_id=training_id,
                        inference_batch_id=inference_batch_id,
                        timestamp=timestamp,
                        open=float(row['open']),
                        high=float(row['high']),
                        low=float(row['low']),
                        close=float(row['close']),
                        volume=float(row.get('volume', 0.0)),
                        amount=float(row.get('amount', 0.0)),
                        # 不确定性范围
                        close_min=float(row.get('close_min', row['close'])),
                        close_max=float(row.get('close_max', row['close'])),
                        close_std=float(row.get('close_std', 0.0)),
                        open_min=float(row.get('open_min', row['open'])),
                        open_max=float(row.get('open_max', row['open'])),
                        high_min=float(row.get('high_min', row['high'])),
                        high_max=float(row.get('high_max', row['high'])),
                        low_min=float(row.get('low_min', row['low'])),
                        low_max=float(row.get('low_max', row['low'])),
                        # 概率指标
                        upward_probability=float(row.get('upward_probability', None)) if pd.notna(row.get('upward_probability')) else None,
                        volatility_amplification_probability=float(row.get('volatility_amplification_probability', None)) if pd.notna(row.get('volatility_amplification_probability')) else None,
                        # 推理参数
                        inference_params=_convert_numpy_to_python({
                            'lookback_window': lookback_window,
                            'predict_window': predict_window,
                            'temperature': T,
                            'top_p': top_p,
                            'sample_count': sample_count
                        })
                    )
                    db.add(prediction)

                db.commit()

                logger.info(f"推理完成: training_id={training_id}, batch_id={inference_batch_id}, predictions={len(pred_df)}条")

                return {
                    'success': True,
                    'predictions_count': len(pred_df)
                }

        except Exception as e:
            logger.error(f"推理执行失败: {e}")
            import traceback
            traceback.print_exc()
            return {
                'success': False,
                'error': str(e)
            }

    @classmethod
    def get_prediction_data(
        cls,
        training_id: int,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[Dict]:
        """
        获取预测数据

        Args:
            training_id: 训练记录ID
            start_time: 开始时间（可选）
            end_time: 结束时间（可选）

        Returns:
            预测数据列表
        """
        try:
            with get_db() as db:
                query = db.query(PredictionData).filter(
                    PredictionData.training_record_id == training_id
                )

                if start_time:
                    query = query.filter(PredictionData.timestamp >= start_time)
                if end_time:
                    query = query.filter(PredictionData.timestamp <= end_time)

                predictions = query.order_by(PredictionData.timestamp).all()

                result = []
                for pred in predictions:
                    result.append({
                        'timestamp': pred.timestamp,
                        'open': pred.open,
                        'high': pred.high,
                        'low': pred.low,
                        'close': pred.close,
                        'volume': pred.volume,
                        'amount': pred.amount
                    })

                return result

        except Exception as e:
            logger.error(f"获取预测数据失败: {e}")
            return []

    @classmethod
    def get_latest_predictions(cls, plan_id: int) -> List[Dict]:
        """
        获取计划的最新预测数据（最新训练版本）

        Args:
            plan_id: 计划ID

        Returns:
            预测数据列表
        """
        try:
            with get_db() as db:
                # 获取最新的已完成训练记录
                latest_training = db.query(TrainingRecord).filter(
                    and_(
                        TrainingRecord.plan_id == plan_id,
                        TrainingRecord.status == 'completed',
                        TrainingRecord.is_active == True
                    )
                ).order_by(TrainingRecord.created_at.desc()).first()

                if not latest_training:
                    return []

                return cls.get_prediction_data(latest_training.id)

        except Exception as e:
            logger.error(f"获取最新预测数据失败: {e}")
            return []

    @classmethod
    def list_inference_batches(cls, training_id: int) -> List[Dict]:
        """
        列出指定训练记录的所有推理批次

        Args:
            training_id: 训练记录ID

        Returns:
            推理批次列表，每条记录包含：
            - inference_batch_id: 推理批次ID
            - inference_time: 推理时间（该批次最早的创建时间）
            - predictions_count: 该批次的预测数据条数
            - time_range: 预测的时间范围 (start, end)
        """
        try:
            with get_db() as db:
                # 查询该训练记录的所有批次
                batches = db.query(
                    PredictionData.inference_batch_id,
                    func.min(PredictionData.created_at).label('inference_time'),
                    func.count(PredictionData.id).label('predictions_count'),
                    func.min(PredictionData.timestamp).label('time_start'),
                    func.max(PredictionData.timestamp).label('time_end')
                ).filter(
                    PredictionData.training_record_id == training_id
                ).group_by(
                    PredictionData.inference_batch_id
                ).order_by(
                    func.min(PredictionData.created_at).desc()
                ).all()

                result = []
                for batch in batches:
                    result.append({
                        'inference_batch_id': batch.inference_batch_id,
                        'inference_time': batch.inference_time,
                        'predictions_count': batch.predictions_count,
                        'time_range': {
                            'start': batch.time_start,
                            'end': batch.time_end
                        }
                    })

                return result

        except Exception as e:
            logger.error(f"列出推理批次失败: {e}")
            return []

    @classmethod
    def get_prediction_data_by_batch(
        cls,
        training_id: int,
        inference_batch_id: str
    ) -> List[Dict]:
        """
        获取指定批次的预测数据

        Args:
            training_id: 训练记录ID
            inference_batch_id: 推理批次ID

        Returns:
            预测数据列表
        """
        try:
            with get_db() as db:
                predictions = db.query(PredictionData).filter(
                    and_(
                        PredictionData.training_record_id == training_id,
                        PredictionData.inference_batch_id == inference_batch_id
                    )
                ).order_by(PredictionData.timestamp).all()

                result = []
                for pred in predictions:
                    result.append({
                        'timestamp': pred.timestamp,
                        'open': pred.open,
                        'high': pred.high,
                        'low': pred.low,
                        'close': pred.close,
                        'volume': pred.volume,
                        'amount': pred.amount,
                        'close_min': pred.close_min,
                        'close_max': pred.close_max,
                        'upward_probability': pred.upward_probability,
                        'volatility_amplification_probability': pred.volatility_amplification_probability
                    })

                return result

        except Exception as e:
            logger.error(f"获取批次预测数据失败: {e}")
            return []

    @classmethod
    def list_inference_records(cls, plan_id: int) -> List[Dict]:
        """
        列出计划的所有推理记录

        Args:
            plan_id: 计划ID

        Returns:
            推理记录列表，每条记录包含：
            - training_record_id: 训练记录ID
            - version: 训练版本
            - inference_time: 推理时间（最早的预测数据创建时间）
            - predictions_count: 预测数据条数
            - has_predictions: 是否有预测数据
        """
        try:
            with get_db() as db:
                # 获取所有已完成的训练记录
                training_records = db.query(TrainingRecord).filter(
                    and_(
                        TrainingRecord.plan_id == plan_id,
                        TrainingRecord.status == 'completed'
                    )
                ).order_by(TrainingRecord.created_at.desc()).all()

                result = []
                for record in training_records:
                    # 查询该训练记录对应的预测数据
                    predictions = db.query(PredictionData).filter(
                        PredictionData.training_record_id == record.id
                    ).order_by(PredictionData.created_at.asc()).all()

                    has_predictions = len(predictions) > 0
                    inference_time = predictions[0].created_at if has_predictions else None

                    # 计算回看数据日期时间范围
                    data_start_time = record.data_start_time
                    data_end_time = record.data_end_time

                    # 格式化日期范围显示
                    if data_start_time and data_end_time:
                        date_range = format_time_range_utc8(data_start_time, data_end_time, '%m-%d')
                        datetime_range = format_time_range_utc8(data_start_time, data_end_time, '%Y-%m-%d %H:%M')
                    else:
                        date_range = "N/A"
                        datetime_range = "N/A"

                    result.append({
                        'training_record_id': record.id,
                        'version': record.version,
                        'inference_time': inference_time,
                        'predictions_count': len(predictions),
                        'has_predictions': has_predictions,
                        'train_end_time': record.train_end_time,
                        'data_start_time': data_start_time,
                        'data_end_time': data_end_time,
                        'date_range': date_range,
                        'datetime_range': datetime_range
                    })

                return result

        except Exception as e:
            logger.error(f"列出推理记录失败: {e}")
            return []

    @classmethod
    def validate_prediction_data(cls, predictions: List[Dict]) -> Dict:
        """
        验证预测数据的合理性

        Args:
            predictions: 预测数据列表

        Returns:
            验证结果: {
                'valid': bool,
                'errors': List[str],
                'warnings': List[str]
            }
        """
        errors = []
        warnings = []

        if not predictions or len(predictions) == 0:
            errors.append("预测数据为空")
            return {'valid': False, 'errors': errors, 'warnings': warnings}

        # 检查数据条数
        if len(predictions) < 10:
            warnings.append(f"预测数据条数较少: {len(predictions)}条")

        # 检查价格字段
        for i, pred in enumerate(predictions):
            # 检查必要字段
            required_fields = ['timestamp', 'open', 'high', 'low', 'close']
            for field in required_fields:
                if field not in pred or pred[field] is None:
                    errors.append(f"第{i+1}条数据缺少字段: {field}")
                    continue

            # 检查价格关系
            if 'open' in pred and 'high' in pred and 'low' in pred and 'close' in pred:
                open_price = pred['open']
                high_price = pred['high']
                low_price = pred['low']
                close_price = pred['close']

                if high_price < max(open_price, close_price):
                    errors.append(f"第{i+1}条: 最高价 {high_price} 小于开盘价/收盘价")

                if low_price > min(open_price, close_price):
                    errors.append(f"第{i+1}条: 最低价 {low_price} 大于开盘价/收盘价")

                if high_price < low_price:
                    errors.append(f"第{i+1}条: 最高价 {high_price} 小于最低价 {low_price}")

            # 检查价格是否为正
            for field in ['open', 'high', 'low', 'close']:
                if field in pred and pred[field] <= 0:
                    errors.append(f"第{i+1}条: {field} 价格必须为正数")

        # 检查时间戳是否递增
        timestamps = [pred['timestamp'] for pred in predictions if 'timestamp' in pred]
        for i in range(1, len(timestamps)):
            if timestamps[i] <= timestamps[i-1]:
                warnings.append(f"时间戳未递增: 第{i}条 -> 第{i+1}条")

        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings
        }

    
    @classmethod
    async def start_inference_by_plan(cls, plan_id: int, manual: bool = False) -> Optional[int]:
        """
        通过计划ID启动推理任务（自动获取最新完成的训练记录）

        Args:
            plan_id: 计划ID
            manual: 是否手动触发

        Returns:
            推理批次ID或None
        """
        try:
            with get_db() as db:
                # 获取最新的已完成训练记录
                latest_training = db.query(TrainingRecord).filter(
                    and_(
                        TrainingRecord.plan_id == plan_id,
                        TrainingRecord.status == 'completed',
                        TrainingRecord.is_active == True
                    )
                ).order_by(TrainingRecord.created_at.desc()).first()

                if not latest_training:
                    logger.warning(f"计划 {plan_id} 没有找到可用的训练记录")
                    return None

                logger.info(f"为计划 {plan_id} 启动推理，使用训练记录 {latest_training.id}")

                # 启动推理
                success = await cls.start_inference(latest_training.id)

                if success:
                    # 返回推理批次ID（这里简化处理，实际可能需要从推理结果中获取）
                    return f"auto_{plan_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                else:
                    return None

        except Exception as e:
            logger.error(f"通过计划ID启动推理失败: plan_id={plan_id}, error={e}")
            import traceback
            traceback.print_exc()
            return None
