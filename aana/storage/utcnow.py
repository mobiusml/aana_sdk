from sqlalchemy import DateTime
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.sql.functions import FunctionElement


class utcnow(FunctionElement):
    """UTCNOW() expression for multiple dialects."""

    inherit_cache = True
    type = DateTime()


@compiles(utcnow)
def default_sql_utcnow(element, compiler, **kw):
    """Assume, by default, time zones work correctly.

    Note:
        This is a valid assumption for PostgreSQL and Oracle.
    """
    return "CURRENT_TIMESTAMP"


@compiles(utcnow, "sqlite")
def sqlite_sql_utcnow(element, compiler, **kw):
    """SQLite DATETIME('NOW') returns a correct `datetime.datetime` but does not add milliseconds to it.

    Directly call STRFTIME with the final %f modifier in order to get those.
    """
    return r"(STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW'))"


@compiles(utcnow, "snowflake")
def snowflake_sql_utcnow(element, compiler, **kw):
    """In Snowflake, SYSDATE() returns the current timestamp for the system in the UTC time zone."""
    return "SYSDATE()"
