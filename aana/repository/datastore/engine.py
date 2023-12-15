from aana.configs.db import create_database_engine
from aana.configs.settings import settings

engine = create_database_engine(settings.db_config)
