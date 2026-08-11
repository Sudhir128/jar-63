"""Tests for database foundation (engine, base, session, health-check).

Uses an in-memory SQLite URL configured in conftest to avoid requiring a
running PostgreSQL during unit tests.
"""

from __future__ import annotations

from contextlib import suppress

from sqlalchemy import String, text
from sqlalchemy.orm import Mapped, mapped_column

from app.config import get_settings
from app.database import Base, check_db_connection, engine, get_db


def test_database_url_is_sqlite_in_tests() -> None:
    assert get_settings().database.effective_url == "sqlite:///:memory:"


def test_engine_created() -> None:
    assert engine is not None


def test_base_is_declarative() -> None:
    assert hasattr(Base, "metadata")
    assert Base.metadata is not None


def test_get_db_yields_session() -> None:
    gen = get_db()
    session = next(gen)
    assert session is not None
    with suppress(StopIteration):
        next(gen)


def test_health_check_sqlite_memory() -> None:
    # In-memory SQLite engine per process: build tables on the same engine.
    class _Sample(Base):
        __tablename__ = "db_test_sample"
        id: Mapped[int] = mapped_column(primary_key=True)
        name: Mapped[str] = mapped_column(String(50))

    Base.metadata.create_all(engine)
    assert check_db_connection() is True
    with engine.connect() as conn:
        assert conn.execute(text("SELECT 1")).scalar() == 1
    Base.metadata.drop_all(engine)
