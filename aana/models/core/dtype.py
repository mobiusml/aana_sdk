from enum import Enum
from typing import Union

import torch


class Dtype(str, Enum):
    """
    Data types.

    Possible values are "auto", "float32", "float16", and "int8".

    Attributes:
        AUTO (str): auto
        FLOAT32 (str): float32
        FLOAT16 (str): float16
        INT8 (str): int8
    """

    AUTO = "auto"
    FLOAT32 = "float32"
    FLOAT16 = "float16"
    INT8 = "int8"

    @classmethod
    def to_torch(cls, dtype: "Dtype") -> Union[torch.dtype, str]:
        """
        Convert a dtype to a torch dtype.

        Args:
            dtype (Dtype): the dtype

        Returns:
            Union[torch.dtype, str]: the torch dtype or "auto"

        Raises:
            ValueError: if the dtype is unknown
        """

        if dtype == cls.AUTO:
            return "auto"
        elif dtype == cls.FLOAT32:
            return torch.float32
        elif dtype == cls.FLOAT16:
            return torch.float16
        elif dtype == cls.INT8:
            return torch.int8
        else:
            raise ValueError(f"Unknown dtype: {dtype}")
