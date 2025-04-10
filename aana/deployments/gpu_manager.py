from typing import Any

from pydantic import BaseModel, Field
from ray import serve
from ray.serve.api import _get_global_client

from aana.deployments.base_deployment import BaseDeployment


class GpuManagerConfig(BaseModel):
    """Configuration for the GPU manager deployment."""

    # Define any configuration fields here.
    available_gpus: list[int] = Field(
        default_factory=list,
        description="List of available GPUs for allocation.",
    )


@serve.deployment
class GpuManagerDeployment(BaseDeployment):
    """GPU manager deployment for managing GPU allocations."""

    async def apply_config(self, config: dict[str, Any]):
        """Apply the configuration settings to the deployment."""
        self.allocations = {}
        config_obj = GpuManagerConfig(**config)
        self.available_gpus = config_obj.available_gpus

    def _get_available_gpus(self) -> list[int]:
        """Get the list of available GPUs."""
        return self.available_gpus

    def _cleanup_inactive_replicas(self, active_replica_ids: set):
        """Remove allocations for replicas that are no longer active (e.g. STOPPING or INACTIVE)."""
        for deployment in self.allocations:
            for rep in list(self.allocations[deployment].keys()):
                if rep not in active_replica_ids:
                    print(f"Cleaning up replica {rep} for deployment {deployment}")
                    del self.allocations[deployment][rep]

    def _compute_gpu_load(self) -> dict:
        """Computes the total fraction allocated on each GPU for this deployment.

        Returns:
            dict: A dictionary mapping GPU IDs to their current load (fraction of allocated capacity).
        """
        load = {gpu: 0.0 for gpu in self.available_gpus}
        for deployment in self.allocations:
            for replica_alloc in self.allocations[deployment].values():
                for gpu, fraction in replica_alloc.items():
                    load[gpu] += fraction
        return load

    def _get_active_replica_ids(self) -> set:
        """Get the IDs of active replicas."""
        client = _get_global_client(raise_if_no_controller_running=False)
        if client is None:
            raise RuntimeError(  # noqa: TRY003
                "Serve has not started yet. Cannot get active replica IDs."
            )
        status = client.get_serve_details()

        active_replica_ids = set()
        for app in status["applications"].values():
            for deployment in app["deployments"]:
                for replica in app["deployments"][deployment]["replicas"]:
                    state = replica["state"]
                    if state not in ["STOPPING", "INACTIVE"]:
                        active_replica_ids.add(replica["replica_id"])
                    print(f"Replica ID: {replica['replica_id']}, State: {state}")
        return active_replica_ids

    async def request_gpu(
        self, deployment: str, replica_id: str, num_gpus: float
    ) -> list[int]:
        """Allocate GPUs for a specific deployment and replica.

        Returns:
            list[int]: A list of GPU IDs that have been allocated.
        """
        print(
            f"Requesting {num_gpus} GPUs for deployment {deployment} with replica ID {replica_id}"
        )
        if num_gpus > 1:
            raise NotImplementedError(
                "Multi-GPU allocation is not yet supported. Please request only one GPU or less."
            )

        # Identify active replicas for this deployment.
        active_replica_ids = self._get_active_replica_ids()

        # Clean up any allocations for replicas that are no longer active.
        self._cleanup_inactive_replicas(active_replica_ids)

        # Ensure allocation state exists for this deployment and replica.
        if deployment not in self.allocations:
            self.allocations[deployment] = {}
        if replica_id not in self.allocations[deployment]:
            self.allocations[deployment][replica_id] = {}

        # Get list of GPUs and compute current load.
        available_gpus = self._get_available_gpus()
        print(f"Available GPUs for deployment '{deployment}': {available_gpus}")
        gpu_load = self._compute_gpu_load()
        print(f"Current GPU load: {gpu_load}")

        # Find GPUs already used by other replicas in this deployment.
        used_gpus = set()
        for rep, alloc in self.allocations[deployment].items():
            # Skip the current replica's state.
            if rep == replica_id:
                continue
            used_gpus.update(alloc.keys())
        print(f"Used GPUs for other replicas in deployment '{deployment}': {used_gpus}")

        allocated_gpus = []
        sorted_gpus = sorted(
            available_gpus,
            key=lambda gpu: (gpu in used_gpus, -(1.0 - gpu_load[gpu])),
        )
        for gpu in sorted_gpus:
            free_capacity = 1.0 - gpu_load[gpu]
            if num_gpus > free_capacity:
                continue  # Skip GPU if not enough capacity.
            print(
                f"Allocating {num_gpus:.2f} on GPU {gpu} for replica {replica_id} (free capacity {free_capacity:.2f})"
            )
            self.allocations[deployment][replica_id][gpu] = num_gpus
            allocated_gpus.append(gpu)
            break

        if allocated_gpus == []:
            raise Exception("Insufficient GPU capacity to fulfill the request")

        print(f"Allocated GPUs for replica {replica_id}: {allocated_gpus}")
        print(self.allocations)
        return allocated_gpus
