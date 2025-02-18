import os
import base64
import mimetypes
import logging
from typing import AsyncGenerator, Union, Optional, List
from litellm import acompletion, supports_vision

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class LLM:
    def __init__(self,
                 model: Optional[str] = None,
                 api_key: Optional[str] = None,
                 temperature: float = 1.0,
                 max_tokens: int = 500,
                 top_p: float = 1.0):
        """
        Initialize the AsyncLiteLLM instance

        Parameters:
            model (str, optional): The name of the model to use; defaults to the environment variable 'LITELLM_MODEL'
            api_key (str, optional): Your API key. Defaults to the environment variable 'API_KEY'.
        """

        self.api_key = api_key or os.getenv("LLM_API_KEY")
        self.model = model or os.getenv("LITELLM_MODEL")

        if (not self.api_key) or (not self.model):
            raise ValueError("API key and model name must be set")
        
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p

        os.environ["OPENAI_API_KEY"] = self.api_key

    def _encode_image(image_path: str) -> str:
        try:
            with open(image_path, "rb") as f:
                encoded = base64.b64encode(f.read()).decode("utf-8")
            mime_type, _ = mimetypes.guess_type(image_path)
            return f"data:{mime_type};base64,{encoded}"
        except Exception as e:
            logger.error("Error encoding image %s: %s", image_path, e)
            raise

    async def generate(
        self,
        text: Optional[str] = None,
        images: Optional[Union[str, List[str]]] = None,
        stream: bool = False
    ) -> str:
        """
        Asynchronously generate a response using text and/or image inputs
        
        Parameters:
            text (Optional[str]): A text prompt
            images (Optional[Union[str, List[str]]]): A single image path or a list of image paths
        
        Returns:
            str: The complete response as a string.
        """
        if not text and not images:
            raise ValueError("At least one of 'text' or 'images' must be provided.")

        if images:
            if not supports_vision(model=self.model):
                raise ValueError(f"The current model ({self.model}) does not support vision input.")

        content_list = []
        if text:
            content_list.append({"type": "text", "text": text})
        if images:
            if isinstance(images, str):
                images = [images]
            for image_path in images:
                encoded_image = self._encode_image(image_path)
                content_list.append({
                    "type": "image_base64",
                    "image_base64": {"data": encoded_image}
                })

        messages = [{"role": "user", "content": content_list}]

        try:
            response = await acompletion(
                model=self.model,
                messages=messages,
                stream=stream
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                top_p=self.top_p
            )
        except Exception as e:
            logger.error("Error during async generation: %s", e)
            raise

        if stream:
            complete_response = ""
            async for chunk in response:
                delta = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
                complete_response += delta
            return complete_response
        else:
            try:
                return response["choices"][0]["message"]["content"]
            except (KeyError, IndexError) as e:
                logger.error("Unexpected response format: %s", e)
                raise ValueError("Unexpected response format") from e


def load_model(model: Optional[str] = None, api_key: Optional[str] = None) -> LiteLLM:
    """
    Convenience function to instantiate an AsyncLiteLLM object

    Parameters:
        model (str, optional): The model name.
        api_key (str, optional): Your API key.

    Returns:
        AsyncLiteLLM: An instance of AsyncLiteLLM.
    """
    return LLM(model=model, api_key=api_key)
