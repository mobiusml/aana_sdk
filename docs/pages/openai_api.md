# OpenAI-compatible API

Aana SDK provides an OpenAI-compatible Chat Completions API that allows you to integrate Aana with any OpenAI-compatible application.

Chat Completions API is available at the `/chat/completions` endpoint.

It is compatible with the OpenAI client libraries and can be used as a drop-in replacement for OpenAI API.

```python
from openai import OpenAI

client = OpenAI(
    api_key="token", # Any non empty string will work, we don't require an API key
    base_url="http://localhost:8000",
)

messages = [
    {"role": "user", "content": "What is the capital of France?"}
]

completion = client.chat.completions.create(
    messages=messages,
    model="llm_deployment",
)

print(completion.choices[0].message.content)
```

The API also supports streaming:

```python
from openai import OpenAI

client = OpenAI(
    api_key="token", # Any non empty string will work, we don't require an API key
    base_url="http://localhost:8000",
)

messages = [
    {"role": "user", "content": "What is the capital of France?"}
]

stream = client.chat.completions.create(
    messages=messages,
    model="llm_deployment",
    stream=True,
)
for chunk in stream:
    print(chunk.choices[0].delta.content or "", end="")
```

The API requires an LLM deployment. Aana SDK provides support for [vLLM](integrations.md#vllm) and [Hugging Face Transformers](integrations.md#hugging-face-transformers).

The name of the model matches the name of the deployment. For example, if you registered a vLLM deployment with the name `llm_deployment`, you can use it with the OpenAI API as `model="llm_deployment"`.

```python
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from aana.core.models.sampling import SamplingParams
from aana.core.models.types import Dtype
from aana.deployments.vllm_deployment import VLLMConfig, VLLMDeployment
from aana.sdk import AanaSDK

llm_deployment = VLLMDeployment.options(
    num_replicas=1,
    ray_actor_options={"num_gpus": 1},
    user_config=VLLMConfig(
        model="TheBloke/Llama-2-7b-Chat-AWQ",
        dtype=Dtype.AUTO,
        quantization="awq",
        gpu_memory_reserved=13000,
        enforce_eager=True,
        default_sampling_params=SamplingParams(
            temperature=0.0, top_p=1.0, top_k=-1, max_tokens=1024
        ),
        chat_template="llama2",
    ).model_dump(mode="json"),
)

aana_app = AanaSDK(name="llm_app")
aana_app.register_deployment(name="llm_deployment", instance=llm_deployment)

if __name__ == "__main__":
    aana_app.connect()
    aana_app.migrate()
    aana_app.deploy()
```

You can also use the example project `llama2` to deploy Llama-2-7b Chat model.

```bash
CUDA_VISIBLE_DEVICES=0 aana deploy aana.projects.llama2.app:aana_app
```
