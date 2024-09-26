from enum import Enum
from typing import Any

import torch
from hqq.core.quantize import BaseQuantizeConfig
from hqq.core.quantize import HQQBackend as HQQBackendKernel
from hqq.models.hf.base import AutoHQQHFModel
from hqq.utils.generation_hf import patch_model_for_compiled_runtime
from hqq.utils.patching import (
    HQQLinear,
    patch_add_quant_config,
    patch_linearlayers,
    prepare_for_inference,
)
from pydantic import BaseModel, ConfigDict, Field
from ray import serve
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

from aana.core.models.base import pydantic_protected_fields
from aana.core.models.sampling import SamplingParams
from aana.core.models.types import Dtype
from aana.deployments.hf_pipeline_deployment import CustomConfig
from aana.deployments.hf_text_generation_deployment import (
    BaseHfTextGenerationDeployment,
)


class HqqBackend(str, Enum):
    """HQQ Backend types.

    Possible values are "torchao_int4" and "bitblas".

    Attributes:
        TORCHAO_INT4 (str): torchao_int4
        BITBLAS (str): bitblas
    """

    TORCHAO_INT4 = "torchao_int4"
    BITBLAS = "bitblas"


class HqqTexGenerationConfig(BaseModel):
    """The configuration for the HQQ text generation deployment.

    Attributes:
        model_id (str): The model ID on Hugging Face.
        quantize_on_fly: Whether to quantize the model or it is already pre-quantized. Defaults to False.
        backend (HqqBackend): The backend library to use. Defaults to HqqBackend.BITBLAS.
        compile (bool): Whether to compile the model with torch.compile. Defaults to False.
        dtype (Dtype): The data type. Defaults to Dtype.AUTO.
        quantization_config (dict): The quantization configuration.
        model_kwargs (CustomConfig): The extra model keyword arguments. Defaults to {}.
        default_sampling_params (SamplingParams): The default sampling parameters.
            Defaults to SamplingParams(temperature=0, max_tokens=256).
        chat_template (str | None): The name of the chat template. If not provided, the chat template
            from the model will be used. Some models may not have a chat template. Defaults to None.
    """

    model_id: str
    quantize_on_fly: bool = False
    backend: HqqBackend = HqqBackend.BITBLAS
    compile: bool = True
    dtype: Dtype = Field(default=Dtype.AUTO)
    quantization_config: CustomConfig = BaseQuantizeConfig()
    model_kwargs: CustomConfig = {}

    default_sampling_params: SamplingParams = SamplingParams(
        temperature=0, max_tokens=512
    )
    chat_template: str | None = None

    model_config = ConfigDict(protected_namespaces=(*pydantic_protected_fields,))


@serve.deployment
class HqqTextGenerationDeployment(BaseHfTextGenerationDeployment):
    """Deployment to serve Hugging Face text generation models with Half-Quadratic Quantization (HQQ)."""

    async def apply_config(self, config: dict[str, Any]):
        """Apply the configuration.

        The method is called when the deployment is created or updated.

        It loads the model and tokenizer from HuggingFace and if needed quantizes the model using HQQ on the fly.

        The configuration should conform to the HqqTexGenerationConfig schema.
        """
        config_obj = HqqTexGenerationConfig(**config)
        self.model_id = config_obj.model_id
        self.backend = config_obj.backend
        self.quantization_config = config_obj.quantization_config
        self.dtype = config_obj.dtype
        self.model_kwargs = config_obj.model_kwargs
        self.default_sampling_params = config_obj.default_sampling_params

        default_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = self.model_kwargs.get("device_map", default_device)

        if self.backend == HqqBackend.BITBLAS:
            try:
                import bitblas  # noqa: F401
            except ImportError as e:
                raise ImportError(  # noqa: TRY003
                    "Failed to import the BitBLAS but HQQ is configured to use BitBLAS backend. "
                    "Check if BitBlas is correctly installed "
                    "if you want to use the bitblas backend (https://github.com/microsoft/BitBLAS)."
                ) from e

        if self.dtype == Dtype.AUTO:
            if self.backend == HqqBackend.TORCHAO_INT4:
                self.dtype = Dtype.BFLOAT16
            elif self.backend == HqqBackend.BITBLAS:
                self.dtype = Dtype.FLOAT16
            else:
                self.dtype = Dtype.BFLOAT16

        if config_obj.quantize_on_fly:
            self.model_kwargs["device_map"] = "cpu"
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id, torch_dtype=self.dtype.to_torch(), **self.model_kwargs
            )
            AutoHQQHFModel.quantize_model(
                self.model,
                quant_config=self.quantization_config,
                device=self.device,
                compute_dtype=self.dtype.to_torch(),
            )
        else:
            self.model_kwargs["device"] = self.device
            self.model = AutoHQQHFModel.from_quantized(
                self.model_id, compute_dtype=self.dtype.to_torch(), **self.model_kwargs
            )
            patch_linearlayers(
                self.model, patch_add_quant_config, self.quantization_config
            )

        HQQLinear.set_backend(HQQBackendKernel.PYTORCH)
        self.model.generation_config.cache_implementation = "static"
        prepare_for_inference(self.model, backend=self.backend)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.chat_template_name = config_obj.chat_template
        if config_obj.compile:
            patch_model_for_compiled_runtime(self.model, self.tokenizer, warmup=True)
