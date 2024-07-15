from pydantic import BaseModel, Field


class Resources(BaseModel):
    """Resources of a video.

    Attributes:
        CPU (float): Available CPU.
        GPU (float): Available GPU.
        memory (float): Available Ram.
    """

    CPU: float = Field(0, description="Available CPU.")
    GPU: float = Field(0, description="Available GPU.")
    memory: float = Field(0, description="Available Ram.")

    def __add__(self, other: "Resources") -> "Resources":
        """Sum two Resources."""
        return Resources(
            CPU=self.CPU + other.CPU,
            GPU=self.GPU + other.GPU,
            memory=self.memory + other.memory
        )

    def __sub__(self, other: "Resources") -> "Resources":
        """Subtract two Resources."""
        return Resources(
            CPU=self.CPU - other.CPU,
            GPU=self.GPU - other.GPU,
            memory=self.memory - other.memory,
        )

    @classmethod
    def from_dict(self, resources: dict, num_replicas: int = 1) -> "Resources":
        """Create a Resources instance from a dict.

        Args:
            resources (dict): the resources as dict
            num_replicas (int): the number of replicas

        Returns:
            Resources: the resources as class
        """
        return Resources(
            CPU=resources.get("num_cpus", 0) * num_replicas,
            GPU=resources.get("num_gpus", 0) * num_replicas,
            memory=resources.get("memory", 0) * num_replicas,
        )
