
from aana.core.models.ray import Resources


def test_resources_initiation():
    """Test resources initiation."""
    resources  = Resources()
    assert resources.CPU == 0
    assert resources.GPU == 0
    assert resources.memory == 0

    resources  = Resources(**{"CPU": 5, "GPU": 3, "memory": 20_000})
    assert resources.CPU == 5
    assert resources.GPU == 3
    assert resources.memory == 20_000

    resources  = Resources.from_dict({"num_cpus": 5, "num_gpus": 3, "memory": 20_000})
    assert resources.CPU == 5
    assert resources.GPU == 3
    assert resources.memory == 20_000

    resources  = Resources.from_dict({"num_cpus": 5, "num_gpus": 3, "memory": 20_000}, num_replicas=3)
    assert resources.CPU == 5 * 3
    assert resources.GPU == 3 * 3
    assert resources.memory == 20_000 * 3

def test_resources_operations():
    """Test resources operation."""
    resources_1  = Resources(CPU=5, GPU=3, memory=20_000)
    assert resources_1.CPU == 5
    assert resources_1.GPU == 3
    assert resources_1.memory == 20_000

    resources_2  = Resources(CPU=7, GPU=4, memory=10_000)
    assert resources_2.CPU == 7
    assert resources_2.GPU == 4
    assert resources_2.memory == 10_000

    resources = resources_1 + resources_2
    assert resources.CPU == 12
    assert resources.GPU == 7
    assert resources.memory == 30_000

    resources = resources_2 - resources_1
    assert resources.CPU == 2
    assert resources.GPU == 1
    assert resources.memory == -10_000
