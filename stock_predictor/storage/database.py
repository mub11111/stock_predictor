from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker
from config import AppConfig
from storage.models import Base

_factory = None


def init_database(cfg: AppConfig):
    global _factory
    if _factory is not None:
        return _factory
    engine = create_engine(
        f"sqlite:///{cfg.data.db_path}",
        echo=False,
        connect_args={"check_same_thread": False},
    )

    @event.listens_for(engine, "connect")
    def _set_sqlite_pragma(dbapi_connection, _connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA page_size=16384")
        cursor.execute("PRAGMA cache_size=-64000")
        cursor.close()

    Base.metadata.create_all(engine)
    _factory = sessionmaker(bind=engine)
    return _factory


def get_session(cfg: AppConfig):
    """Return a new SQLAlchemy session."""
    factory = init_database(cfg)
    return factory()
