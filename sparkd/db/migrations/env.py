from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

from sparkd.db.models import Base

config = context.config
if config.config_file_name is not None:
    # disable_existing_loggers=False is critical: sparkd runs migrations
    # in-process during app startup (see db.engine.init_engine). With the
    # default True, fileConfig() would disable the already-configured
    # uvicorn loggers, silently swallowing "Application startup complete"
    # and "Uvicorn running on ..." — making a healthy server look hung.
    fileConfig(config.config_file_name, disable_existing_loggers=False)
target_metadata = Base.metadata


def run_migrations_online() -> None:
    cfg = config.get_section(config.config_ini_section, {})
    cfg["sqlalchemy.url"] = cfg["sqlalchemy.url"].replace("+aiosqlite", "")
    connectable = engine_from_config(cfg, prefix="sqlalchemy.", poolclass=pool.NullPool)
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


run_migrations_online()
