import asyncio
from typing import AsyncGenerator, List, Dict, Any, Optional, Callable, Union
import logging
from .async_pool import APIPool

logger = logging.getLogger(__name__)


class StreamGenerator:
    """High-level API for streaming generation with automatic concurrency management."""

    def __init__(
            self,
            model_name: str,
            api_keys: List[str],
            max_concurrent_per_key: int = 300,
            max_retries: int = 5,
            rational: bool = False,
            with_unique_id: bool = False
    ):
        """
        Initialize the stream generator.

        Args:
            model_name: Name of the model to use
            api_keys: List of API keys
            max_concurrent_per_key: Max concurrent requests per key
            max_retries: Max retries per request
            rational: Whether to enable reasoning
            with_unique_id: Whether to use unique ID mode for maintaining order
        """
        self.api_pool = APIPool(
            model_name=model_name,
            api_keys=api_keys,
            max_concurrent_per_key=max_concurrent_per_key
        )
        self.max_retries = max_retries
        self.rational = rational
        self.with_unique_id = with_unique_id

        # Semaphore for total concurrency control
        self.total_concurrency = max_concurrent_per_key * len(api_keys)
        self.semaphore = asyncio.Semaphore(self.total_concurrency)

        logger.info(f"Initialized StreamGenerator with total concurrency: {self.total_concurrency}, "
                    f"unique_id_mode: {self.with_unique_id}")

    def _validate_input_format(self, prompts: List[Any]) -> None:
        """Validate input format based on unique_id mode."""
        if self.with_unique_id:
            for i, prompt in enumerate(prompts):
                if not isinstance(prompt, dict) or "id" not in prompt or "prompt" not in prompt:
                    raise ValueError(
                        f"When with_unique_id=True, each prompt must be a dict with 'id' and 'prompt' keys. "
                        f"Invalid item at index {i}: {prompt}"
                    )
        else:
            for i, prompt in enumerate(prompts):
                if not isinstance(prompt, (str, list)):
                    raise ValueError(
                        f"When with_unique_id=False, each prompt must be str or list. "
                        f"Invalid item at index {i}: {type(prompt)}"
                    )

    async def generate_stream(
            self,
            prompts: Union[List[str], List[List[Dict[str, Any]]], List[Dict[str, Any]]],
            system_prompt: str = "",
            validate_func: Optional[Callable[[str], Any]] = None
    ) -> AsyncGenerator[Union[str, Dict[str, Any]], None]:
        """
        Generate responses for a list of prompts in a streaming fashion.

        Args:
            prompts: List of prompts. Format depends on with_unique_id mode:
                    - If with_unique_id=False: List[str] or List[List[Dict[str, Any]]] (for video prompts)
                    - If with_unique_id=True: List[Dict[str, Any]] with keys 'id' and 'prompt'
            system_prompt: System prompt (same for all requests)
            validate_func: Optional function to validate the response

        Yields:
            Response data:
            - If with_unique_id=False: str (direct answer)
            - If with_unique_id=True: Dict[str, Any] with keys 'id' and 'result'
        """
        # Validate input format
        self._validate_input_format(prompts)

        tasks = set()
        processed_tasks = set()

        try:
            for prompt_item in prompts:
                if self.with_unique_id:
                    prompt_id = prompt_item["id"]
                    prompt_content = prompt_item["prompt"]
                    task = asyncio.create_task(
                        self._generate_single_with_id(
                            prompt_id=prompt_id,
                            system_prompt=system_prompt,
                            user_prompt=prompt_content,
                            validate_func=validate_func
                        )
                    )
                else:
                    task = asyncio.create_task(
                        self._generate_single(
                            system_prompt=system_prompt,
                            user_prompt=prompt_item,
                            validate_func=validate_func
                        )
                    )

                tasks.add(task)
                task.add_done_callback(tasks.discard)

                # Wait if we've reached max concurrency
                if len(tasks) >= self.total_concurrency:
                    done, _ = await asyncio.wait(
                        tasks,
                        return_when=asyncio.FIRST_COMPLETED
                    )
                    for completed_task in done:
                        if completed_task not in processed_tasks:
                            processed_tasks.add(completed_task)
                            result = await completed_task
                            if result is not None:
                                yield result
                        else:
                            logger.debug(f"Task already processed in first loop, skipping")

            # Process remaining tasks
            while tasks:
                done, _ = await asyncio.wait(
                    tasks,
                    return_when=asyncio.FIRST_COMPLETED
                )
                for completed_task in done:
                    if completed_task not in processed_tasks:
                        processed_tasks.add(completed_task)
                        result = await completed_task
                        if result is not None:
                            yield result
                    else:
                        logger.debug(f"Task already processed in second loop, skipping")

        finally:
            # Clean up any remaining tasks
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _generate_single_with_id(
            self,
            prompt_id: str,
            system_prompt: str,
            user_prompt: Union[str, List[Dict[str, Any]]],
            validate_func: Optional[Callable[[str], Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """Generate a single response with ID tracking."""
        result = await self._generate_single(system_prompt, user_prompt, validate_func)
        if result is not None:
            return {"id": prompt_id, "result": result}
        return None

    async def _generate_single(
            self,
            system_prompt: str,
            user_prompt: Union[str, List[Dict[str, Any]]],
            validate_func: Optional[Callable[[str], Any]] = None
    ) -> Optional[str]:
        """
        Generate a single response with retry mechanism.

        Args:
            system_prompt: System prompt
            user_prompt: User prompt (str or list for video prompts)
            validate_func: Optional function to validate the response

        Returns:
            Generated answer string or None if failed
        """
        retry_count = 0
        while retry_count < self.max_retries:
            async with self.semaphore:
                try:
                    response = await self.api_pool.execute(
                        "qa",
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        rational=self.rational
                    )

                    answer = response["answer"]

                    # If validation function exists, validate the answer
                    if validate_func is not None:
                        if not validate_func(answer):
                            logger.warning(f"Answer validation failed, retrying (attempt {retry_count + 1})")
                            retry_count += 1
                            continue
                        else:
                            validated_answer = validate_func(answer)
                            if validated_answer is not None:
                                answer = validated_answer

                    return answer

                except Exception as e:
                    logger.warning(f"Generation error: {e}, retrying (attempt {retry_count + 1})")
                    retry_count += 1

        logger.error(f"Max retries reached for prompt")
        return None