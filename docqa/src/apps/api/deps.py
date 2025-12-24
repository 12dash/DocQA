from docqa.config import Settings
from docqa.pipeline.engine import QAEngine

settings = Settings()
settings.validate()

_engine = QAEngine(settings)

def get_engine() -> QAEngine:
    return _engine
