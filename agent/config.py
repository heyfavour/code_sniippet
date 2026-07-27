from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

# .env 固定在 config.py 所在目录，不受运行时工作目录影响
ENV_FILE = Path(__file__).resolve().parent / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ENV_FILE,
        env_file_encoding="utf-8",
        case_sensitive=False,   # 不区分大小写：DEEPSEEK_APIKEY == deepseek_apikey
        extra="ignore",         # .env 里多余的字段不报错
    )

    # —— 敏感字段，字段名对应 .env 中的 deepseek_apikey ——
    deepseek_apikey: SecretStr

# 单例模式：整个项目 import 这一份就够了
settings = Settings()