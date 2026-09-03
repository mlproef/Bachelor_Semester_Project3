from typing import List, Dict, Tuple, Iterator

import openai
import time
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from openai.error import (
    APIError,
    RateLimitError,
    ServiceUnavailableError,
    Timeout as OpenAITimeout,
    TryAgain,
)

from src.configs import ModelConfig
from src.prompts import Prompt, Conversation

from .model import BaseModel


def _is_retryable_openai_error(exc: BaseException) -> bool:
    """Retry transient uni/litellm failures (500, connection drops, rate limits)."""
    if isinstance(
        exc, (RateLimitError, ServiceUnavailableError, OpenAITimeout, TryAgain)
    ):
        return True
    if isinstance(exc, APIError):
        code = getattr(exc, "http_status", None) or getattr(exc, "code", None)
        if code in (408, 429, 500, 502, 503, 504):
            return True
        msg = str(exc).lower()
        needles = (
            "500",
            "502",
            "503",
            "504",
            "connection error",
            "internalservererror",
            "timeout",
            "temporarily unavailable",
            "overloaded",
        )
        return any(n in msg for n in needles)
    msg = str(exc).lower()
    return "connection error" in msg or "internalservererror" in msg


class OpenAIGPT(BaseModel):
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.config = config

        if "temperature" not in self.config.args.keys():
            self.config.args["temperature"] = 0.0
        if "max_tokens" not in self.config.args.keys():
            self.config.args["max_tokens"] = 2000
        # Retries for transient API failures (override via ModelConfig.args).
        self.max_retries = int(self.config.args.get("max_retries", 8))
        self.retry_base_delay = float(self.config.args.get("retry_base_delay", 5.0))

    def _predict_call(self, input: List[Dict[str, str]]) -> str:
        # Do not forward retry knobs to the OpenAI API.
        call_args = {
            k: v
            for k, v in self.config.args.items()
            if k not in ("max_retries", "retry_base_delay")
        }
        delay = self.retry_base_delay
        last_exc: BaseException | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                if self.config.provider == "azure":
                    response = openai.ChatCompletion.create(
                        engine=self.config.name, messages=input, **call_args
                    )
                else:
                    response = openai.ChatCompletion.create(
                        model=self.config.name, messages=input, **call_args
                    )
                return response["choices"][0]["message"]["content"]
            except Exception as e:
                last_exc = e
                if attempt >= self.max_retries or not _is_retryable_openai_error(e):
                    raise
                print(
                    f"[openai retry {attempt}/{self.max_retries}] {type(e).__name__}: "
                    f"{str(e)[:180]} — sleep {delay:.0f}s",
                    flush=True,
                )
                time.sleep(delay)
                delay = min(delay * 2, 120.0)
        assert last_exc is not None
        raise last_exc

    def predict(self, input: Prompt, **kwargs) -> str:
        messages: List[Dict[str, str]] = []

        if input.system_prompt is not None:
            messages.append(
                {
                    "role": "system",
                    "content": input.system_prompt,
                }
            )
        else:
            messages = [
                {
                    "role": "system",
                    "content": "You are an expert investigator and detective with years of experience in online profiling and text analysis.",
                }
            ]

        messages += [
            {"role": "user", "content": self.apply_model_template(input.get_prompt())}
        ]

        guess = self._predict_call(messages)

        # response = openai.ChatCompletion.create(
        #     model=self.config.name, messages=messages, **self.config.args
        # )

        # guess = response["choices"][0]["message"]["content"]

        return guess

    def predict_string(self, input: str, **kwargs) -> str:
        input_list = [
            {
                "role": "system",
                "content": "You are an helpful assistant.",
            },
            {"role": "user", "content": input},
        ]

        guess = self._predict_call(input_list)

        # response = openai.ChatCompletion.create(
        #     model=self.config.name, messages=input_list, **self.config.args
        # )

        # guess = response["choices"][0]["message"]["content"]

        return guess

    def predict_multi(
        self, inputs: List[Prompt], **kwargs
    ) -> Iterator[Tuple[Prompt, str]]:
        max_workers = kwargs["max_workers"] if "max_workers" in kwargs else 4
        base_timeout = kwargs["timeout"] if "timeout" in kwargs else 120

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            ids_to_do = list(range(len(inputs)))
            retry_ctr = 0
            timeout = base_timeout

            while len(ids_to_do) > 0 and retry_ctr <= len(inputs):
                # executor.map will apply the function to every item in the iterable (prompts), returning a generator that yields the results
                results = executor.map(
                    lambda id: (id, inputs[id], self.predict(inputs[id])),
                    ids_to_do,
                    timeout=timeout,
                )
                try:
                    for res in tqdm(
                        results,
                        total=len(ids_to_do),
                        desc="Profiles",
                        position=1,
                        leave=False,
                    ):
                        id, orig, answer = res
                        yield (orig, answer)
                        # answered_prompts.append()
                        ids_to_do.remove(id)
                except TimeoutError:
                    print(f"Timeout: {len(ids_to_do)} prompts remaining")
                except RateLimitError as r:
                    print(f"Rate_limit {r}")
                    time.sleep(30)
                    continue
                except Exception as e:
                    print(f"Exception: {e}")
                    time.sleep(10)
                    continue

                if len(ids_to_do) == 0:
                    break

                time.sleep(2 * retry_ctr)
                timeout *= 2
                timeout = min(120, timeout)
                retry_ctr += 1

        # return answered_prompts

    def continue_conversation(self, input: Conversation, **kwargs) -> str:
        input_list: List[Dict[str, str]] = [
            {
                "role": "system",
                "content": input.system_prompt,
            }
        ]

        for message in input.prompts:
            assert message.role is not None
            input_list.append(
                {
                    "role": message.role,
                    "content": message.get_prompt(),  # Simply returns the intermediate text
                }
            )

        guess = None
        while guess is None:
            try:
                guess = self._predict_call(input_list)
            except RateLimitError as r:
                time.sleep(30)
                continue

        return guess
