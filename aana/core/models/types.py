from enum import Enum

import torch


class Dtype(str, Enum):
    """Data types.

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

    def to_torch(self) -> torch.dtype | str:
        """Convert the instance's dtype to a torch dtype.

        Returns:
            Union[torch.dtype, str]: the torch dtype or "auto"

        Raises:
            ValueError: if the dtype is unknown
        """
        match self.value:
            case self.AUTO:
                return "auto"
            case self.FLOAT32:
                return torch.float32
            case self.FLOAT16:
                return torch.float16
            case self.INT8:
                return torch.int8
            case _:
                raise ValueError(f"Unknown dtype: {self}")  # noqa: TRY003
