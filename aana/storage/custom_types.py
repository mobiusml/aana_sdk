import orjson
from snowflake.sqlalchemy.custom_types import VARIANT as SnowflakeVariantType
from sqlalchemy import func
from sqlalchemy.types import JSON as SqlAlchemyJSON
from sqlalchemy.types import TypeDecorator


class VARIANT(SnowflakeVariantType):
    """Extends VARIANT type for better SqlAlchemy support."""

    def bind_expression(self, bindvalue):
        """Wraps value with PARSE_JSON for Snowflake."""
        return func.PARSE_JSON(bindvalue)

    def result_processor(self, dialect, coltype):
        """Convert JSON string to Python dictionary when retrieving."""

        def process(value):
            if value is None:
                return None
            try:
                return orjson.loads(value)
            except (ValueError, TypeError):
                return value  # Return raw value if not valid JSON

        return process


JSON = VARIANT

# class JSON(TypeDecorator):
#     """Custom JSON type that supports Snowflake-specific and standard dialects."""

#     impl = SqlAlchemyJSON  # Default to standard SQLAlchemy JSON

#     def load_dialect_impl(self, dialect):
#         """Load dialect-specific implementation."""
#         if dialect.name == "snowflake":
#             return SnowflakeVariantType()
#         return self.impl

#     def bind_expression(self, bindvalue):
#         """Handle binding expressions dynamically."""
#         if hasattr(
#             bindvalue.type, "bind_expression"
#         ):  # Check if impl has bind_expression
#             return bindvalue.type.bind_expression(bindvalue)
#         return bindvalue  # Default binding behavior

#     def process_result_value(self, value, dialect):
#         """Process the result based on dialect."""
#         if dialect.name == "snowflake":
#             if value is None:
#                 return None
#             try:
#                 return orjson.loads(value)
#             except (ValueError, TypeError):
#                 return value  # Return raw value if not valid JSON
#         # For other dialects, call the default implementation
#         return self.impl.process_result_value(value, dialect)
