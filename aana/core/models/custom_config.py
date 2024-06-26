import pickle
from typing import Annotated

from pydantic import BeforeValidator, PlainSerializer

CustomConfig = Annotated[
    dict,
    PlainSerializer(lambda x: pickle.dumps(x).decode("latin1"), return_type=str),
    BeforeValidator(
        lambda x: x if isinstance(x, dict) else pickle.loads(x.encode("latin1"))  # noqa: S301
    ),
]
