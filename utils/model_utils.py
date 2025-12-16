import base64
import logging
import time
from dataclasses import dataclass
from io import BytesIO
from typing import Optional, List, Any
from PIL import Image
import argparse
import requests
import io
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def read_markdown_file(file_path):
    """Read markdown file and return its content"""
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    return text

@dataclass
class Prompt:
    """Prompt class for model input
    
    Attributes:
        text: The text content of the prompt
        role: The role of the message sender (e.g., 'user', 'system')
        images: Optional list of images to include with the prompt
    """
    text: str
    role: str = "user"
    images: Optional[List[Image.Image]] = None

class ModelOutput:
    """Class for model generation output
    
    Attributes:
        text: The generated text response
        usage: Dictionary containing token usage statistics
        finish_reason: Reason why the model stopped generating
    """
    def __init__(self, text: str, usage: dict = None, finish_reason: str = None):
        self.text = text
        self.usage = usage or {}
        self.finish_reason = finish_reason

class BaseModelInterface:
    """Base interface for AI model implementations
    
    This class provides common functionality for all model interfaces
    and defines the interface that specific model implementations must follow.
    
    Attributes:
        config: Configuration for the model
        _client: The underlying client for the model API
    """
    
    def __init__(self, model_name: str, api_url: Optional[str] = None, api_key: Optional[str] = None, 
                 max_tokens: int = 8192, temperature: float = 0.7, retries: int = 3, retry_interval: int = 1):
        """Initialize the model interface
        
        Args:
            model_name: The name of the model
            api_url: Optional API endpoint URL
            api_key: Optional API key
            max_tokens: The maximum number of tokens to generate
            temperature: The sampling temperature for generation
            retries: Number of retries for an operation
            retry_interval: Initial interval between retries
        """
        self.model_name = model_name
        self.api_url = api_url
        self.api_key = api_key
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.retries = retries
        self.retry_interval = retry_interval
        self._client = None
        
    def _retry_operation(self, operation):
        """Execute operation with retry logic
        
        Args:
            operation: Function to execute with retries
            
        Returns:
            The result of the operation
            
        Raises:
            Exception: If all retry attempts fail
        """
        attempts = 0
        while attempts < self.retries:
            try:
                return operation()
            except Exception as e:
                attempts += 1
                delay = self.retry_interval * (2 ** attempts)
                logger.error(f"Operation failed: {e}. Attempt {attempts}/{self.retries}")
                if attempts < self.retries:
                    time.sleep(delay)
                else:
                    raise
                    
    def _process_image(self, image: Image.Image) -> str:
        """Convert image to base64 string
        
        Args:
            image: PIL Image to convert
            
        Returns:
            Base64 encoded string of the image
        """
        buffer = BytesIO()
        # Convert RGBA to RGB if needed
        if image.mode == 'RGBA':
            image = image.convert('RGB')
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode()

    def generate(self, prompt: Prompt) -> ModelOutput:
        """Generate response from model
        
        Args:
            prompt: The prompt to send to the model
            
        Returns:
            ModelOutput containing the generated response
            
        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        raise NotImplementedError

class OpenAIInterface(BaseModelInterface):
    """Interface for OpenAI models"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        from openai import OpenAI
        if self.api_url:
            self._client = OpenAI(api_key=self.api_key, base_url=self.api_url)
        else:
            self._client = OpenAI(api_key=self.api_key)
        
    def generate(self, prompt: Prompt) -> ModelOutput:
        def _generate():
            messages = [{"role": prompt.role, "content": []}]
            
            # Add text content
            if prompt.text:
                messages[0]["content"].append({"type": "text", "text": prompt.text})
            
            # Add image content if available
            if prompt.images:
                for image in prompt.images:
                    image_content = {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{self._process_image(image)}"}
                    }
                    messages[0]["content"].append(image_content)
            
            completion = self._client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )

            return ModelOutput(
                text=completion.choices[0].message.content,
                usage=completion.usage.model_dump(),
                finish_reason=completion.choices[0].finish_reason
            )
        return self._retry_operation(_generate)

class LocalInterface(BaseModelInterface):
    """Interface for locally deployed models (e.g., Qwen-2.5VL)"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.server_url = self.api_url
        if not self.server_url:
            raise ValueError("api_url is required for LocalInterface")

    def generate(self, prompt: Prompt) -> ModelOutput:
        def _generate():
            data = {}
            files = []
            
            # Add prompt text
            if prompt.text:
                data['prompt'] = prompt.text
            
            # Add images if available
            if prompt.images:
                for idx, image in enumerate(prompt.images):
                    # Convert RGBA to RGB if needed
                    if image.mode == 'RGBA':
                        image = image.convert('RGB')
                    
                    img_byte_arr = io.BytesIO()
                    image.save(img_byte_arr, format='JPEG')
                    image_data = img_byte_arr.getvalue()
                    
                    # Prepare request data
                    files.append(('images', (f'image_{idx}.jpg', image_data, 'image/jpeg')))
            
            response = requests.post(
                f"{self.server_url}",
                files=files,
                data=data
            )
            response.raise_for_status()
            result = response.json()
            
            return ModelOutput(
                text=result.get("response", result.get("text", "")),
                usage={},
                finish_reason="stop"
            )
        return self._retry_operation(_generate)


def create_model(model_name: str, api_url: Optional[str] = None, api_key: Optional[str] = None, **kwargs) -> BaseModelInterface:
    """Factory function to create model interface
    
    Args:
        model_name: The name of the model to create an interface for.
        api_url: Optional API endpoint URL.
        api_key: Optional API key.
        **kwargs: Additional keyword arguments for the model interface.
        
    Returns:
        An instance of the appropriate model interface
        
    Raises:
        ValueError: If the provider is not supported
    """
    if 'gpt-4o' in model_name:
        model_name = "gpt-4o"
    model_kwargs = {
        "model_name": model_name,
        "api_url": api_url,
        "api_key": api_key,
        **kwargs
    }
    model_name_lower = model_name.lower()
    
    if 'gpt' in model_name_lower or 'openai' in model_name_lower or 'o1' in model_name_lower or 'o3' in model_name_lower:
        return OpenAIInterface(**model_kwargs)
    else:
        return LocalInterface(**model_kwargs)

def generate_response(model_name: str, prompt_text: str, images: List[Image.Image] = None, 
                      api_url: str = None, api_key: str = None, **kwargs) -> str:
    """High-level function to generate response from model
    
    Args:
        model_name: Name of the model to use
        prompt_text: Text prompt to send to the model
        images: List of PIL Image objects to include with the prompt
        api_url: Optional API endpoint URL
        api_key: Optional API key
        **kwargs: Additional keyword arguments for model creation
        
    Returns:
        The generated text response
    """
    
    # Create model interface
    model = create_model(
        model_name=model_name,
        api_url=api_url,
        api_key=api_key,
        **kwargs
    )
    
    # Process images
    processed_images = []
    if images:
        for image in images:
            if image.mode == 'RGBA':
                image = image.convert('RGB')
            processed_images.append(image)
    
    # Create prompt
    prompt = Prompt(text=prompt_text, images=processed_images if processed_images else None)
    
    # Generate response
    response = model.generate(prompt)
    return response.text

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Model Interface for ImagineAffordance")
    parser.add_argument("--model", required=True, help="Model name")
    parser.add_argument("--prompt", required=False, help="Input prompt")
    parser.add_argument("--prompt_file", required=False, help="Input prompt file (markdown)")
    parser.add_argument("--images", nargs="+", help="Optional image paths")
    parser.add_argument("--api_url", help="Optional API endpoint")
    parser.add_argument("--api_key", help="Optional API key")
    
    args = parser.parse_args()
    
    # Read prompt from file if provided
    if args.prompt_file:
        args.prompt = read_markdown_file(args.prompt_file)
    
    # Load images if provided
    images = []
    if args.images:
        for image_path in args.images:
            images.append(Image.open(image_path))
    
    response = generate_response(
        args.model,
        args.prompt,
        images,
        args.api_url,
        args.api_key
    )
    
    print("\nModel Response:")
    print("=" * 50)
    print(response)
    print("=" * 50)

