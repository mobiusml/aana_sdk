from fastapi import FastAPI
from pydantic import ValidationError

from aana.api.exception_handler import (
    aana_exception_handler,
    validation_exception_handler,
)

app = FastAPI()

app.add_exception_handler(ValidationError, validation_exception_handler)
app.add_exception_handler(Exception, aana_exception_handler)
