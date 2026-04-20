from pydantic import BaseModel


class Settings(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8765
    cors_origins: list[str] = ["http://localhost:5173", "http://localhost:8765"]
    ws_interval_ms: int = 200
    metric_buffer_size: int = 100
    histogram_bins: int = 50


settings = Settings()
