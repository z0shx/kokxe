"""
KOKEX 项目配置管理
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()


class Config:
    """项目配置类"""

    # 项目路径
    BASE_DIR = Path(__file__).parent
    MODEL_SAVE_PATH = BASE_DIR / "models"
    LOG_DIR = BASE_DIR / "logs"
    SQL_DIR = BASE_DIR / "sql"

    # 数据库配置
    DB_HOST = os.getenv("DB_HOST", "127.0.0.1")
    DB_PORT = os.getenv("DB_PORT", "5432")
    DB_NAME = os.getenv("DB_NAME", "kronos_db")
    DB_USER = os.getenv("DB_USER", "kronos")
    DB_PASSWORD = os.getenv("DB_PASSWORD", "kronos")

    @property
    def DATABASE_URL(self):
        return f"postgresql://{self.DB_USER}:{self.DB_PASSWORD}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"

    # 网络代理
    PROXY_ENABLED = os.getenv("PROXY_ENABLED", "true").lower() == "true"
    PROXY_URL = os.getenv("PROXY_URL", "http://127.0.0.1:20170")

    @property
    def PROXIES(self):
        if self.PROXY_ENABLED:
            return {
                "http": self.PROXY_URL,
                "https": self.PROXY_URL
            }
        return None

    # OKX API 配置
    OKX_API_KEY = os.getenv("OKX_API_KEY", "")
    OKX_SECRET_KEY = os.getenv("OKX_SECRET_KEY", "")
    OKX_PASSPHRASE = os.getenv("OKX_PASSPHRASE", "")

    # OKX API 地址 - 实盘
    OKX_REST_URL = os.getenv("OKX_REST_URL", "https://www.okx.com")
    OKX_WS_PUBLIC = os.getenv("OKX_WS_PUBLIC", "wss://ws.okx.com:8443/ws/v5/public")
    OKX_WS_PRIVATE = os.getenv("OKX_WS_PRIVATE", "wss://ws.okx.com:8443/ws/v5/private")
    OKX_WS_BUSINESS = os.getenv("OKX_WS_BUSINESS", "wss://ws.okx.com:8443/ws/v5/business")

    # OKX API 地址 - 模拟盘
    OKX_DEMO_REST_URL = os.getenv("OKX_DEMO_REST_URL", "https://www.okx.com")
    OKX_DEMO_WS_PUBLIC = os.getenv("OKX_DEMO_WS_PUBLIC", "wss://wspap.okx.com:8443/ws/v5/public")
    OKX_DEMO_WS_PRIVATE = os.getenv("OKX_DEMO_WS_PRIVATE", "wss://wspap.okx.com:8443/ws/v5/private")
    OKX_DEMO_WS_BUSINESS = os.getenv("OKX_DEMO_WS_BUSINESS", "wss://wspap.okx.com:8443/ws/v5/business")

    # 模型配置
    LOOKBACK_WINDOW = int(os.getenv("LOOKBACK_WINDOW", "400"))
    PREDICT_WINDOW = int(os.getenv("PREDICT_WINDOW", "48"))
    CANDLE_INTERVAL = os.getenv("CANDLE_INTERVAL", "1H")

    # Gradio 配置
    GRADIO_SERVER_PORT = int(os.getenv("GRADIO_SERVER_PORT", "7860"))
    GRADIO_SERVER_NAME = os.getenv("GRADIO_SERVER_NAME", "0.0.0.0")

    # 时间颗粒度映射
    INTERVAL_MAPPING = {
        "30m": "30m",
        "1H": "1H",
        "2H": "2H",
        "4H": "4H"
    }

    # WebSocket 频道映射（现货K线）
    WS_CHANNEL_MAPPING = {
        "30m": "candle30m",
        "1H": "candle1H",
        "2H": "candle2H",
        "4H": "candle4H"
    }

    # 数据完整性验证配置
    DATA_VALIDATION_ENABLED = os.getenv("DATA_VALIDATION_ENABLED", "true").lower() == "true"
    DATA_VALIDATION_INTERVAL_HOURS = int(os.getenv("DATA_VALIDATION_INTERVAL_HOURS", "12"))  # 验证间隔（小时）
    DATA_VALIDATION_MAX_MISSING_POINTS = int(os.getenv("DATA_VALIDATION_MAX_MISSING_POINTS", "10"))  # 最大允许缺失数据点数
    DATA_VALIDATION_BATCH_SIZE = int(os.getenv("DATA_VALIDATION_BATCH_SIZE", "100"))  # 批量处理大小

    # Agent 推理详细日志配置
    AGENT_DETAILED_LOGGING = os.getenv("AGENT_DETAILED_LOGGING", "true").lower() == "true"

    def __init__(self):
        # 确保必要目录存在
        self.MODEL_SAVE_PATH.mkdir(parents=True, exist_ok=True)
        self.LOG_DIR.mkdir(parents=True, exist_ok=True)
        self.SQL_DIR.mkdir(parents=True, exist_ok=True)


# 全局配置实例
config = Config()
