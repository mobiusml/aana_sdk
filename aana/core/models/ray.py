from pydantic import BaseModel, Field


class Resources(BaseModel):
    """Resources of a video.

    Attributes:
        CPU (float): Available CPU.
        GPU (float): Available GPU.
        memory (float): Available Ram.
        object_store_memory (float): Available object_store_memory.
    """

    CPU: float = Field(0, description="Available CPU.")
    GPU: float = Field(0, description="Available GPU.")
    memory: float = Field(0, description="Available Ram.")
    object_store_memory: float = Field(0, description="Available object store memory.")

    def __add__(self, other: "Resources") -> "Resources":
        """Sum two Resources."""
        return Resources(
            CPU=self.CPU + other.CPU,
            GPU=self.GPU + other.GPU,
            memory=self.memory + other.memory,
            object_store_memory=self.object_store_memory + other.object_store_memory,
        )

    def __sub__(self, other: "Resources") -> "Resources":
        """Subtract two Resources."""
        return Resources(
            CPU=self.CPU - other.CPU,
            GPU=self.GPU - other.GPU,
            memory=self.memory - other.memory,
            object_store_memory=self.object_store_memory - other.object_store_memory,
        )

    @classmethod
    def from_dict(self, resources: dict) -> "Resources":
        """Create a Resources instance from a dict.

        Args:
            resources (dict): the resources as dict

        Returns:
            Resources: the resources as class
        """

        return Resources(
            CPU=resources.get("num_cpus", 0),
            GPU=resources.get("num_gpus", 0),
            memory=resources.get("memory", 0),
            object_store_memory=resources.get("object_store_memory", 0)
        )
