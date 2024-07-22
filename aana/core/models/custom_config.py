import pickle
from typing import Annotated

from pydantic import BeforeValidator, PlainSerializer

__all__ = ["CustomConfig"]

CustomConfig = Annotated[
    dict,
    PlainSerializer(lambda x: pickle.dumps(x).decode("latin1"), return_type=str),
    BeforeValidator(
        lambda x: x if isinstance(x, dict) else pickle.loads(x.encode("latin1"))  # noqa: S301
    ),
]
"""
A custom configuration field that can be used to pass arbitrary configuration to the deployment.

For example, you can define a custom configuration field in a deployment configuration like this:

```python
class HfPipelineConfig(BaseModel):
    model_id: str
    task: str | None = None
    model_kwargs: CustomConfig = {}
    pipeline_kwargs: CustomConfig = {}
    generation_kwargs: CustomConfig = {}
```

Then you can use the custom configuration field to pass a configuration to the deployment:

```python
HfPipelineConfig(
    model_id="Salesforce/blip2-opt-2.7b",
    model_kwargs={
        "quantization_config": BitsAndBytesConfig(
            load_in_8bit=False, load_in_4bit=True
        ),
    },
)
```
"""
