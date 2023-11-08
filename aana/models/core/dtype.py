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

    def to_torch(self) -> Union[torch.dtype, str]:
        """
        Convert the instance's dtype to a torch dtype.

        Returns:
            Union[torch.dtype, str]: the torch dtype or "auto"

        Raises:
            ValueError: if the dtype is unknown
        """
        if self.value == self.AUTO:
            return "auto"
        elif self.value == self.FLOAT32:
            return torch.float32
        elif self.value == self.FLOAT16:
            return torch.float16
        elif self.value == self.INT8:
            return torch.int8
        else:
            raise ValueError(f"Unknown dtype: {self}")
