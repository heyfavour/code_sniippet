from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,   # 不区分大小写：DEEPSEEK_APIKEY == deepseek_apikey
        extra="ignore",         # .env 里多余的字段不报错
    )

    # —— 敏感字段，字段名对应 .env 中的 deepseek_apikey ——
    deepseek_apikey: SecretStr

# 单例模式：整个项目 import 这一份就够了
settings = Settings()