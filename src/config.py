import os
from pathlib import Path
from dotenv import load_dotenv

# ========== 1. 自动定位项目根目录的 .env 并加载 ==========
# __file__ 是当前 config.py 的路径，假设它在 src/ 下
# parents[0] = src/, parents[1] = 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = PROJECT_ROOT / ".env"

if ENV_PATH.exists():
    # override=True 保证每次都以 .env 为准，避免系统环境变量污染
    load_dotenv(dotenv_path=ENV_PATH, override=True)
else:
    raise FileNotFoundError(
        f"❌ 未找到 .env 文件，请确认它位于项目根目录: {ENV_PATH}"
    )

# ========== 2. 封装 LLM 配置 ==========
class LLMConfig:
    """统一管理 LLM 相关配置"""
    def __init__(self):
        self.api_key = os.getenv("LLM_API_KEY")
        self.base_url = os.getenv("LLM_BASE_URL")
        self.llm_model = os.getenv("LLM_MODEL_ID", "deepseek-chat")
        self.timeout = int(os.getenv("LLM_TIMEOUT", "60"))

        # 关键配置缺失时尽早报错，避免运行时才出问题
        if not self.api_key:
            raise ValueError("❌ LLM_API_KEY 未在 .env 中配置")
        if not self.base_url:
            raise ValueError("❌ LLM_BASE_URL 未在 .env 中配置")

    def __repr__(self):
        # 隐藏 key，方便打印调试
        masked = f"{self.api_key[:6]}...{self.api_key[-4:]}" if self.api_key else "None"
        return f"LLMConfig(model={self.llm_model}, base_url={self.base_url}, key={masked})"

    def show(self):
        print(f"Currently using MODEL: {self.llm_model}")
        print(f"Base URL: {self.base_url}")

# 全局单例，全项目共享
llm_config = LLMConfig()