import asyncio
import random
from typing import List, Dict, Any, TypeVar
import logging
from .wrapper import QAWrapper

T = TypeVar('T')
logger = logging.getLogger(__name__)


class APIPool:
    """
    Asynchronous API pool that maximizes parallelism by allowing
    multiple concurrent requests per API key.
    """

    def __init__(
            self,
            model_name: str,
            api_keys: List[str],
            max_concurrent_per_key: int = 300,
    ):
        """
        Initialize an async API pool.

        Args:
            model_name: Name of the model to use
            api_keys: List of API keys to use
            max_concurrent_per_key: Maximum number of concurrent requests per API key
        """
        if not api_keys:
            raise ValueError("At least one API key is required")

        self.api_instances = [
            QAWrapper(model_name, api_key)
            for api_key in api_keys
        ]

        self.active_requests = [0] * len(self.api_instances)

        self.max_concurrent_per_key = max_concurrent_per_key

        # Semaphores to control concurrency per API key
        self.semaphores = [
            asyncio.Semaphore(max_concurrent_per_key)
            for _ in range(len(self.api_instances))
        ]

        self.stats = {
            "total_calls": 0,
            "total_errors": 0,
            "total_retries": 0,
            "api_instances": len(self.api_instances),
            "api_distribution": [0] * len(self.api_instances)
        }

        # Lock for updating stats
        self.stats_lock = asyncio.Lock()

        logger.info(f"Initialized API pool with {len(api_keys)} API keys, "
                    f"{max_concurrent_per_key} max concurrent requests per key")

    @property
    def len_keys(self) -> int:
        """
        Returns the number of API keys in the pool.

        Returns:
            int: The number of API keys
        """
        return len(self.api_instances)

    async def execute(self, method_name: str, *args, **kwargs) -> Any:
        """
        Execute a method on the least busy API instance.

        Args:
            method_name: Name of the method to call (e.g. 'qa')
            *args, **kwargs: Arguments to pass to the method

        Returns:
            Result from the API call
        """
        api_index = await self._select_optimal_api_instance()

        async with self.stats_lock:
            self.stats["api_distribution"][api_index] += 1

        async with self.semaphores[api_index]:
            self.active_requests[api_index] += 1

            try:
                api_instance = self.api_instances[api_index]
                method = getattr(api_instance, method_name)

                result = await method(*args, **kwargs)

                async with self.stats_lock:
                    self.stats["total_calls"] += 1

                return result

            except Exception as e:
                async with self.stats_lock:
                    self.stats["total_errors"] += 1

                # Re-raise the exception
                raise

            finally:
                self.active_requests[api_index] -= 1

    async def _select_optimal_api_instance(self) -> int:
        """
        Select the optimal API instance based on current load.

        Returns:
            Index of the selected API instance
        """
        min_active = min(self.active_requests)
        candidates = [
            i for i, count in enumerate(self.active_requests)
            if count == min_active
        ]

        # If multiple candidates, choose randomly to balance load
        return random.choice(candidates)

    async def get_stats(self) -> Dict[str, Any]:
        """Get aggregated statistics for all API instances."""
        for i, api in enumerate(self.api_instances):
            instance_stats = api.get_stats()
            async with self.stats_lock:
                self.stats[f"instance_{i}_calls"] = instance_stats["calls"]
                self.stats[f"instance_{i}_errors"] = instance_stats["errors"]

        async with self.stats_lock:
            return self.stats.copy()