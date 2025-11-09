"""LLM interface module using LiteLLM for unified VLM access."""

import re
import json
import time
import logging
from pathlib import Path
from typing import List, Dict, Optional, Union
import litellm
from litellm import completion
import base64


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VLMError(Exception):
    """Base exception for VLM-related errors."""
    pass


class VLMAPIError(VLMError):
    """API request failed."""
    pass


class VLMResponseParseError(VLMError):
    """Failed to parse VLM response."""
    pass


class VLMRateLimitError(VLMError):
    """Hit rate limit."""
    pass


class VLMClient:
    """Unified interface for Vision-Language Models using LiteLLM."""

    def __init__(
        self,
        provider: str,
        model_name: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 16384
    ):
        """
        Initialize VLM client.

        Args:
            provider: Provider name ('gemini', 'ollama', 'openai', etc.)
            model_name: Full model name (e.g., 'gemini/gemini-2.0-flash-exp', 'ollama/qwen2-vl:8b')
            api_key: API key for cloud providers (optional for local models)
            base_url: Base URL for API (optional, for custom endpoints)
            temperature: Sampling temperature (0.0 - 1.0)
            max_tokens: Maximum tokens in response
        """
        self.provider = provider
        self.model_name = model_name
        self.api_key = api_key
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Set API key in environment if provided
        if api_key:
            if provider == 'gemini':
                import os
                os.environ['GEMINI_API_KEY'] = api_key
            elif provider == 'openai':
                import os
                os.environ['OPENAI_API_KEY'] = api_key

        # Configure LiteLLM
        if base_url:
            litellm.api_base = base_url

        logger.info(f"Initialized VLM client: {provider}/{model_name}")

    def _encode_image(self, image_path: Path) -> str:
        """
        Encode image to base64 string.

        Args:
            image_path: Path to image file

        Returns:
            Base64 encoded image string
        """
        with open(image_path, 'rb') as f:
            image_data = f.read()
            return base64.b64encode(image_data).decode('utf-8')

    def _prepare_messages(
        self,
        prompt: str,
        images: Optional[List[Path]] = None
    ) -> List[Dict]:
        """
        Prepare messages for LiteLLM API.

        Args:
            prompt: Text prompt
            images: Optional list of image paths

        Returns:
            List of message dictionaries
        """
        if not images:
            # Text-only message
            return [{"role": "user", "content": prompt}]

        # Multimodal message with images
        content = [{"type": "text", "text": prompt}]

        for image_path in images:
            image_path = Path(image_path)

            if not image_path.exists():
                logger.warning(f"Image not found: {image_path}")
                continue

            # For Gemini and other vision models
            if self.provider in ['gemini', 'openai']:
                # Encode image as base64
                image_data = self._encode_image(image_path)
                image_type = image_path.suffix.lower().replace('.', '')

                if image_type == 'jpg':
                    image_type = 'jpeg'

                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/{image_type};base64,{image_data}"
                    }
                })
            else:
                # For Ollama and others, use file path
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": str(image_path.absolute())
                    }
                })

        return [{"role": "user", "content": content}]

    def send_message(
        self,
        prompt: str,
        images: Optional[List[Path]] = None,
        max_retries: int = 3,
        backoff_factor: float = 2.0
    ) -> str:
        """
        Send a message to the VLM and get response.

        Args:
            prompt: Text prompt
            images: Optional list of image paths
            max_retries: Maximum number of retry attempts
            backoff_factor: Exponential backoff factor

        Returns:
            Response text from VLM

        Raises:
            VLMAPIError: If API request fails
            VLMRateLimitError: If rate limit is hit
        """
        messages = self._prepare_messages(prompt, images)

        for attempt in range(max_retries):
            try:
                logger.info(
                    f"Sending request to {self.model_name} (attempt {attempt + 1}/{max_retries})")

                response = completion(
                    model=self.model_name,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )

                result = response.choices[0].message.content

                # Check if result is None
                if result is None:
                    raise VLMAPIError(
                        "VLM returned empty response (content is None)")

                logger.info(f"Received response ({len(result)} characters)")

                return result

            except Exception as e:
                error_str = str(e).lower()

                if 'rate limit' in error_str or 'quota' in error_str:
                    if attempt < max_retries - 1:
                        sleep_time = backoff_factor ** attempt
                        logger.warning(
                            f"Rate limit hit, sleeping for {sleep_time}s")
                        time.sleep(sleep_time)
                        continue
                    else:
                        raise VLMRateLimitError(f"Rate limit exceeded: {e}")

                elif attempt < max_retries - 1:
                    logger.warning(f"Attempt {attempt + 1} failed: {e}")
                    time.sleep(backoff_factor ** attempt)
                    continue
                else:
                    raise VLMAPIError(f"Failed to get response from VLM: {e}")

    def send_message_with_json(
        self,
        prompt: str,
        images: Optional[List[Path]] = None,
        max_retries: int = 3
    ) -> Dict:
        """
        Send a message and parse JSON response.

        Args:
            prompt: Text prompt (should request JSON format)
            images: Optional list of image paths
            max_retries: Maximum number of retry attempts

        Returns:
            Parsed JSON response as dictionary

        Raises:
            VLMResponseParseError: If response is not valid JSON
        """
        response_text = self.send_message(prompt, images, max_retries)

        try:
            # Try to extract JSON from response (in case it's wrapped in markdown)
            # Look for ```json ... ``` blocks
            json_match = re.search(
                r'```json\s*(\{.*?\})\s*```', response_text, re.DOTALL)
            if json_match:
                response_text = json_match.group(1)
            else:
                # Look for any JSON object
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    response_text = json_match.group(0)

            return json.loads(response_text)

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.error(f"Response text: {response_text}")
            raise VLMResponseParseError(f"Invalid JSON response: {e}")

    def test_connection(self) -> bool:
        """
        Test connection to VLM.

        Returns:
            True if connection successful, False otherwise
        """
        try:
            response = self.send_message("Hello! Please respond with 'OK'.")
            return len(response) > 0
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False


# Import re for JSON extraction
