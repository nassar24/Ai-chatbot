"""Shared base for LLM providers that expose an OpenAI-compatible chat
completions endpoint. OpenRouter and Alibaba's DashScope both do — this
class holds the request/response handling once; concrete subclasses only
declare their endpoint URL, default model, and env var names.
"""

from __future__ import annotations

import os
import time

import requests

from app.llm.base import ChatMessage, LLMProvider


class EmptyCompletionError(RuntimeError):
    """The provider answered 200 OK with a well-formed body whose
    `content` was null or blank.

    Its own class rather than a bare RuntimeError so the retry loop can
    tell it apart from a malformed response shape, which is a bug and
    must not be retried.
    """


class OpenAICompatibleLLMProvider(LLMProvider):
    # Subclasses must set these.
    DEFAULT_MODEL: str
    API_KEY_ENV_VAR: str
    MODEL_ENV_VAR: str
    PROVIDER_LABEL: str  # used only in error messages

    # One retry on top of the initial attempt — a transient network
    # blip (observed in practice against dashscope-intl.aliyuncs.com)
    # shouldn't fail the whole chat turn if a second attempt would
    # succeed. Not retried indefinitely: a genuinely down/misconfigured
    # upstream should still surface as an error rather than hang the
    # request for multiples of the timeout.
    _MAX_ATTEMPTS = 2
    _RETRY_BACKOFF_SECONDS = 1.5

    # A completion this short that also does not end on terminal
    # punctuation was cut off mid-thought. Observed in the faithfulness
    # eval: the answer to "Do you offer ongoing content creation?" came
    # back as the complete string "Yes, PixNo" and was served as-is,
    # because every check downstream only asks whether an answer is
    # EMPTY. It was not empty, so it sailed through the provider guard,
    # all six guardrails, and out to the visitor.
    #
    # Both halves of the condition are needed. Length alone would reject
    # "Yes, we do." — a perfectly good answer. Missing punctuation alone
    # would reject long answers that legitimately end on a bullet or a
    # URL, which is a real formatting choice the model makes. Together
    # they describe truncation and very little else: real answers to
    # these questions measured 472-862 characters.
    #
    # It still has known false positives — "Yes, we do" and "Visit
    # apexcreative.example" are both short and unpunctuated, and both would be
    # retried. That is precisely why a failed retry returns the better
    # answer instead of raising: this heuristic is allowed to be wrong,
    # and when it is, the visitor pays one extra second and still gets
    # their answer. Compare the emptiness check above, which CAN raise
    # because an empty string is unambiguous.
    _TRUNCATION_MAX_CHARS = 80
    _SENTENCE_ENDINGS = (".", "!", "?", ":", "…", "؟", "۔", ")", "\"", "'", "”")

    def __init__(
        self,
        api_url: str,
        model: str | None = None,
        api_key: str | None = None,
        timeout_seconds: int = 60,
    ):
        resolved_key = api_key or os.environ.get(self.API_KEY_ENV_VAR)
        if not resolved_key:
            raise RuntimeError(
                f"{self.API_KEY_ENV_VAR} is not set. Provide it as an "
                "environment variable — never hardcode API keys in source."
            )

        self._api_url = api_url
        self._model = model or os.environ.get(self.MODEL_ENV_VAR, self.DEFAULT_MODEL)
        self._api_key = resolved_key
        self._timeout_seconds = timeout_seconds

    @property
    def model_name(self) -> str:
        return self._model

    @classmethod
    def _looks_truncated(cls, content: str) -> bool:
        stripped = content.strip()
        return (
            len(stripped) < cls._TRUNCATION_MAX_CHARS
            and not stripped.endswith(cls._SENTENCE_ENDINGS)
        )

    def generate(
        self,
        system_prompt: str,
        messages: list[ChatMessage],
        max_tokens: int = 600,
    ) -> str:
        if not system_prompt or not system_prompt.strip():
            raise ValueError("system_prompt must be non-empty.")
        if not messages:
            raise ValueError("messages must contain at least one entry.")

        payload_messages = [{"role": "system", "content": system_prompt}]
        payload_messages += [{"role": m.role, "content": m.content} for m in messages]

        last_exception: Exception | None = None
        truncated_candidate = ""
        for attempt in range(1, self._MAX_ATTEMPTS + 1):
            try:
                response = requests.post(
                    self._api_url,
                    headers={
                        "Authorization": f"Bearer {self._api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self._model,
                        "messages": payload_messages,
                        "max_tokens": max_tokens,
                    },
                    timeout=self._timeout_seconds,
                )
                response.raise_for_status()
                data = response.json()

                try:
                    content = data["choices"][0]["message"]["content"]
                except (KeyError, IndexError) as exc:
                    raise RuntimeError(
                        f"Unexpected {self.PROVIDER_LABEL} response shape "
                        f"(missing choices[0].message.content): {data}"
                    ) from exc

                # A present-but-null/blank `content` is a real provider
                # behavior (content filter trips, or finish_reason=length
                # with nothing emitted) and slips past the KeyError guard
                # above. Returning it would hand None to the guardrails,
                # which raise on an empty answer — surfacing as a 500 to
                # the visitor instead of anything useful. Raise here so
                # the caller sees a provider failure for what it is.
                if not content or not content.strip():
                    raise EmptyCompletionError(
                        f"{self.PROVIDER_LABEL} returned an empty completion "
                        f"(finish_reason="
                        f"{data.get('choices', [{}])[0].get('finish_reason')!r})."
                    )

                # A truncated completion is retried like an empty one —
                # same transient cause — but it deliberately does NOT
                # raise when the retry is no better. Raising would turn
                # a merely-short answer into a 502 for the visitor, and
                # this check is a heuristic: unlike emptiness, it can be
                # wrong. So the worst case is bounded at "you get the
                # better of the two attempts", never "you get an error
                # where an answer existed".
                if self._looks_truncated(content):
                    if len(content.strip()) > len(truncated_candidate.strip()):
                        truncated_candidate = content
                    if attempt < self._MAX_ATTEMPTS:
                        time.sleep(self._RETRY_BACKOFF_SECONDS)
                        continue
                    return truncated_candidate
                return content
            except (
                requests.exceptions.Timeout,
                requests.exceptions.ConnectionError,
                EmptyCompletionError,
            ) as exc:
                # Network-layer failures and empty completions are
                # retried; an HTTP error status (4xx/5xx from
                # raise_for_status) or a malformed response shape is not
                # a transient blip and retrying it would just waste the
                # timeout twice for the same guaranteed failure.
                #
                # An empty completion belongs in the retryable set
                # because it is a SAMPLING outcome, not a deterministic
                # one: the same request sent again usually produces a
                # normal answer. Measured at roughly 3% of turns against
                # DashScope, each one surfacing to the visitor as
                # "Something went wrong on our side" — a 502 on a
                # question the model can answer perfectly well. Retrying
                # costs one extra request on 3% of turns and removes
                # nearly all of them, since two independent empties in a
                # row is ~0.1%.
                #
                # The cost of being wrong is bounded: if the emptiness
                # IS deterministic (a content filter tripping on the
                # same input, or max_tokens set so low nothing can be
                # emitted), the second attempt fails identically and the
                # error surfaces exactly as it does today, one backoff
                # later.
                last_exception = exc
                if attempt < self._MAX_ATTEMPTS:
                    time.sleep(self._RETRY_BACKOFF_SECONDS)
                    continue
                if truncated_candidate:
                    # An earlier attempt produced a short answer and this
                    # one failed outright. A partial answer beats a 502.
                    return truncated_candidate
                raise

        # Unreachable in practice (the loop above always returns or
        # raises), but keeps type checkers happy and fails loudly
        # rather than returning None if that ever changes.
        raise RuntimeError(
            f"{self.PROVIDER_LABEL} request failed after {self._MAX_ATTEMPTS} attempts."
        ) from last_exception

