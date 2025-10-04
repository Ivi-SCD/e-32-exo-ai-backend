from pydantic import Field
from pydantic_settings import BaseSettings
from typing import Optional
import os


class Settings(BaseSettings):
    """Configurações da aplicação usando Pydantic Settings"""
    
    # Configurações da API
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=8000, env="API_PORT")
    api_reload: bool = Field(default=False, env="API_RELOAD")
    debug: bool = Field(default=False, env="DEBUG")
    
    # Configurações do Banco de Dados
    database_url: str = Field(
        default="postgresql://nasa_user:nasa_password@localhost:5433/nasa_exoplanets",
        env="DATABASE_URL"
    )
    
    # Configurações de Log
    log_level: str = Field(default="info", env="LOG_LEVEL")
    
    # Configurações de CORS
    cors_origins: list[str] = Field(default=["*"], env="CORS_ORIGINS")
    cors_allow_credentials: bool = Field(default=True, env="CORS_ALLOW_CREDENTIALS")
    cors_allow_methods: list[str] = Field(default=["*"], env="CORS_ALLOW_METHODS")
    cors_allow_headers: list[str] = Field(default=["*"], env="CORS_ALLOW_HEADERS")
    
    # Configurações de Modelo
    model_path: str = Field(default="./notebooks/models", env="MODEL_PATH")
    data_path: str = Field(default="./notebooks/data", env="DATA_PATH")

    # Configurações de Rate Limiting
    rate_limit_requests: int = Field(default=100, env="RATE_LIMIT_REQUESTS")
    rate_limit_window: int = Field(default=60, env="RATE_LIMIT_WINDOW")  # segundos
    
    # Configurações de Cache
    cache_ttl: int = Field(default=300, env="CACHE_TTL")  # 5 minutos
    
    # Configurações de Upload
    max_upload_size: int = Field(default=10 * 1024 * 1024, env="MAX_UPLOAD_SIZE")  # 10MB
    allowed_file_types: list[str] = Field(
        default=["csv", "xlsx", "xls"], 
        env="ALLOWED_FILE_TYPES"
    )
    
    # Configurações de Processamento
    max_batch_size: int = Field(default=1000, env="MAX_BATCH_SIZE")
    processing_timeout: int = Field(default=300, env="PROCESSING_TIMEOUT")  # 5 minutos
    
    # Configurações de Desenvolvimento
    enable_docs: bool = Field(default=True, env="ENABLE_DOCS")
    enable_redoc: bool = Field(default=True, env="ENABLE_REDOC")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "ignore"  # Ignorar campos extras
    
    @property
    def is_development(self) -> bool:
        """Verifica se está em modo de desenvolvimento"""
        return self.debug or self.api_reload
    
    @property
    def is_production(self) -> bool:
        """Verifica se está em modo de produção"""
        return not self.is_development
    
    
    def get_cors_config(self) -> dict:
        """Retorna configuração do CORS"""
        return {
            "allow_origins": self.cors_origins,
            "allow_credentials": self.cors_allow_credentials,
            "allow_methods": self.cors_allow_methods,
            "allow_headers": self.cors_allow_headers,
        }


# Instância global das configurações
settings = Settings()


def get_settings() -> Settings:
    """Dependency para obter configurações"""
    return settings
