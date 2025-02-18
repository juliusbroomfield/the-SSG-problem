import re
import json
import os
import functools
import openai
from dataclasses import dataclass
from typing import Optional, List, Callable
from utils import Retry

class StrongReject:
    @dataclass
    class AutograderResponse:
        refusal: bool
        grader_model_input: str
        grader_model_output: str

    def __init__(
        self,
        evaluator_prompt: Optional[str] = None,
        evaluator_model: str = "gpt-4o",
        retries: int = 10,
    ):
        """
        Parameters:
            evaluator_prompt (str, optional): The evaluator prompt template.
                If not provided, it will be loaded from 'evaluations/models/evaluator_prompt.txt'.
            evaluator_model (str): The model name used for refusal detection.
            retries (int): Number of retries for the evaluator call.
        """
        self.evaluator_prompt = evaluator_prompt
        self.evaluator_model = evaluator_model
        self.retries = retries
        self.RESPONSE_PARSE_RE = re.compile(
            r"""
                .*?
                ^\s*1\.b\s+(?P<refusal_score>[0-1])
            """,
            flags=re.DOTALL | re.MULTILINE | re.VERBOSE,
        )

    def _load_evaluator_prompt(self) -> str:
        if self.evaluator_prompt is not None:
            return self.evaluator_prompt
        else:
            with open("guardrails/prompts/strongreject.txt", "r", encoding="utf-8") as f:
                return f.read()

    def __call__(self, question: str, response: str) -> bool:
        """
        Runs the evaluator to decide if the response should be rejected.
        
        Parameters:
            question (str): The original prompt.
            response (str): The response generated using the canonical prompt.
        """
        prompt_template = self._load_evaluator_prompt()
        model_input = prompt_template.format(prompt=question, response=response)

        @Retry(self.retries)
        def call_autograder() -> bool:
            with openai.OpenAI() as client:
                completion = client.chat.completions.create(
                    model=self.evaluator_model,
                    messages=[{"role": "user", "content": model_input}],
                    temperature=0.0,
                    max_tokens=1000,
                )
            model_output = completion.choices[0].message.content
            parsed_response = self.RESPONSE_PARSE_RE.search(model_output)
            if parsed_response is None:
                return False
            else:
                group_dict = parsed_response.groupdict()
                return int(group_dict["refusal_score"]) > 0

        return call_autograder()

# Guardrail Module

DEFAULT_CANONICALIZATION_TEMPLATE = (
    "Your task is to transform the following prompt and its associated images into a plain English instruction that clearly captures the core intent. DO NOT ANSWER the prompt — only transform it.\n\n"
    "Input prompt: {prompt}\n\n"
    "Please output only the transformed instruction with no extra commentary."
)

class SRGuardrail:
    def __init__(
        self,
        base_model, 
        translate_model = None,
        refusal_detection_fn: Optional[Callable[[str, str], bool]] = None,
        canonicalization_template: Optional[str] = None,
    ):
        """
        Initialize the guardrail.

        Parameters:
            base_model: The model that the guardrail is applied to.
            translate_model: The model used to canonicalize and translate prompts.
                Defaults to base_model if not provided.
            refusal_detection_fn: A callable that takes (prompt, response) and returns a boolean.
                Defaults to using StrongReject() with the base model as the backbone.
            canonicalization_template: Template for the canonicalization prompt.
                Must include '{prompt}'. Defaults to a built-in template.
        """
        self.base_model = base_model
        self.translate_model = translate_model if translate_model is not None else base_model
        self.refusal_detection_fn = refusal_detection_fn if refusal_detection_fn is not None else StrongReject()
        self.canonicalization_template = (
            canonicalization_template if canonicalization_template is not None else DEFAULT_CANONICALIZATION_TEMPLATE
        )

    def query(self, prompt: str, images: Optional[List[str]] = None) -> dict:
        """
        Process the prompt through the guardrail pipeline:
          1. Canonicalize the prompt using the translation model.
          2. Query the base model with the canonical prompt.
          3. Apply refusal detection on the original prompt and response.

          If the model refuses to answer, the original response

        Parameters:
            prompt (str): The original text prompt.
            images (List[str], optional): A list of image paths associated with the prompt.

        Returns:
            dict: A dictionary with keys:
                - "translation": The canonicalized prompt.
                - "response": The response from the base model.
                - "refusal": A boolean indicating whether refusal was detected.
        """
        canonical_prompt = self.canonicalization_template.format(prompt=prompt)
        canonical_form = self.translate_model.query(canonical_prompt, images)
        response = self.base_model.query(canonical_form, images)
        refusal = self.refusal_detection_fn(prompt, response)

        if refusal:
            return {
                "translation": canonical_form,
                "response": response,
                "refusal": refusal,
            }
        
        response = self.base_model.query(prompt, images)
        return {
            "translation": canonical_form,
            "response": response,
            "refusal": refusal,
        }
