"""OpenAI VLM (Vision Language Model) Implementation"""

import sys
from pathlib import Path

# Add project root to path when running as script (python agenticpay/models/openai_vlm.py)
if __name__ == "__main__":
    _project_root = Path(__file__).resolve().parent.parent.parent
    if str(_project_root) not in sys.path:
        sys.path.insert(0, str(_project_root))

from typing import Optional, Union, List, Any
import logging
import os
import time
import base64
import io
import requests
from agenticpay.models.base_vlm import BaseVLM

logger = logging.getLogger(__name__)

# Prefixed to the user prompt when all image(s) failed to load so the model still runs text-only.
_IMAGE_UNAVAILABLE_NOTE = (
    "[System note: Product image(s) could not be loaded. Continue using only the text below.]\n\n"
)


class OpenAIVLM(BaseVLM):
    """OpenAI VLM Implementation
    
    Uses official OpenAI API for vision language model inference.
    Supports models like GPT-4V (gpt-4-vision-preview), GPT-4o, etc.
    """
    
    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        """Initialize OpenAI VLM
        
        Args:
            model: Model name, e.g., "gpt-4o", "gpt-4-vision-preview", "gpt-4-turbo"
            api_key: API key, if not provided, will be obtained from environment variable
            base_url: API base URL (optional, defaults to official OpenAI API)
        """
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("OPENAI_URL")
        
        if not self.api_key:
            raise ValueError(
                "OpenAI API key is required. "
                "Set it via api_key parameter or OPENAI_API_KEY environment variable."
            )
        # if not self.base_url:
        #     raise ValueError(
        #         "OpenAI base URL is required. "
        #         "Set it via base_url parameter or OPENAI_URL environment variable."
        #     )
        
        # Lazy import openai to avoid errors when not installed
        try:
            import openai
            self.client = openai.OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
            )
        except ImportError:
            raise ImportError(
                "OpenAI package is required. Install it with: pip install openai"
            )
    
    @staticmethod
    def _image_input_repr(image_input: Any) -> str:
        if isinstance(image_input, (str, Path)):
            s = str(image_input)
            return s if len(s) < 120 else s[:117] + "..."
        return type(image_input).__name__

    def _encode_image_to_base64(self, image_input: Union[str, Path, Any]) -> str:
        """Encode image to base64 string for OpenAI API
        
        Args:
            image_input: Image input (path, URL, PIL Image, numpy array, bytes, etc.)
            
        Returns:
            Base64 encoded image string with data URL prefix
        """
        try:
            from PIL import Image
            import numpy as np
        except ImportError:
            raise ImportError(
                "PIL (Pillow) package is required. Install it with: pip install pillow"
            )
        
        # If it's a Path object, convert to string
        if isinstance(image_input, Path):
            image_input = str(image_input)
        
        # If it's a string, check if it's a URL or file path
        if isinstance(image_input, str):
            # If it's a URL, download and encode
            if image_input.startswith(("http://", "https://")):
                response = requests.get(image_input, timeout=30)
                response.raise_for_status()
                image_bytes = response.content
                image_base64 = base64.b64encode(image_bytes).decode('utf-8')
                # Determine image format from URL or content
                if image_input.lower().endswith(('.png', '.PNG')):
                    return f"data:image/png;base64,{image_base64}"
                elif image_input.lower().endswith(('.jpg', '.jpeg', '.JPG', '.JPEG')):
                    return f"data:image/jpeg;base64,{image_base64}"
                elif image_input.lower().endswith(('.gif', '.GIF')):
                    return f"data:image/gif;base64,{image_base64}"
                elif image_input.lower().endswith(('.webp', '.WEBP')):
                    return f"data:image/webp;base64,{image_base64}"
                else:
                    # Try to detect from content
                    img = Image.open(io.BytesIO(image_bytes))
                    format_map = {
                        'PNG': 'image/png',
                        'JPEG': 'image/jpeg',
                        'GIF': 'image/gif',
                        'WEBP': 'image/webp',
                    }
                    img_format = format_map.get(img.format, 'image/jpeg')
                    return f"data:{img_format};base64,{image_base64}"
            else:
                # File path
                if not os.path.exists(image_input):
                    raise FileNotFoundError(f"Image file not found: {image_input}")
                with open(image_input, "rb") as image_file:
                    image_bytes = image_file.read()
                    image_base64 = base64.b64encode(image_bytes).decode('utf-8')
                    # Determine format from file extension
                    if image_input.lower().endswith(('.png', '.PNG')):
                        return f"data:image/png;base64,{image_base64}"
                    elif image_input.lower().endswith(('.jpg', '.jpeg', '.JPG', '.JPEG')):
                        return f"data:image/jpeg;base64,{image_base64}"
                    elif image_input.lower().endswith(('.gif', '.GIF')):
                        return f"data:image/gif;base64,{image_base64}"
                    elif image_input.lower().endswith(('.webp', '.WEBP')):
                        return f"data:image/webp;base64,{image_base64}"
                    else:
                        return f"data:image/jpeg;base64,{image_base64}"  # Default to JPEG
        
        # If it's a PIL Image
        if isinstance(image_input, Image.Image):
            buffered = io.BytesIO()
            # Save as JPEG by default, or preserve format if available
            if image_input.format:
                image_input.save(buffered, format=image_input.format)
                format_map = {
                    'PNG': 'image/png',
                    'JPEG': 'image/jpeg',
                    'GIF': 'image/gif',
                    'WEBP': 'image/webp',
                }
                img_format = format_map.get(image_input.format, 'image/jpeg')
            else:
                image_input.save(buffered, format='JPEG')
                img_format = 'image/jpeg'
            image_bytes = buffered.getvalue()
            image_base64 = base64.b64encode(image_bytes).decode('utf-8')
            return f"data:{img_format};base64,{image_base64}"
        
        # If it's a numpy array
        try:
            import numpy as np
            if isinstance(image_input, np.ndarray):
                img = Image.fromarray(image_input)
                buffered = io.BytesIO()
                img.save(buffered, format='JPEG')
                image_bytes = buffered.getvalue()
                image_base64 = base64.b64encode(image_bytes).decode('utf-8')
                return f"data:image/jpeg;base64,{image_base64}"
        except ImportError:
            pass
        
        # If it's bytes
        if isinstance(image_input, bytes):
            # Try to determine if it's already base64
            try:
                # Check if it's a valid base64 string
                base64.b64decode(image_input, validate=True)
                # If it's valid base64, assume it's already encoded
                return f"data:image/jpeg;base64,{image_input.decode('utf-8')}"
            except Exception:
                # If not base64, treat as raw image bytes
                image_base64 = base64.b64encode(image_input).decode('utf-8')
                return f"data:image/jpeg;base64,{image_base64}"
        
        raise ValueError(f"Unsupported image input type: {type(image_input)}")
    
    def generate(
        self,
        prompt: str,
        images: Optional[Union[str, Path, List[Union[str, Path]], Any]] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> str:
        """Generate text from text prompt and optional images
        
        Args:
            prompt: Input text prompt
            images: Image input(s). Can be:
                - str: Path to image file or image URL
                - Path: Path object to image file
                - List[str/Path]: List of image paths/URLs
                - PIL.Image: PIL Image object
                - numpy.ndarray: Image as numpy array
                - bytes: Image as bytes (base64 encoded or raw)
                - List of any above types: Multiple images
                - None: No images (text-only generation)
            temperature: Temperature parameter (controls randomness, 0-2)
            max_tokens: Maximum number of tokens to generate
            **kwargs: Other parameters (e.g., top_p, frequency_penalty, presence_penalty, etc.)
            
        Returns:
            Generated text response

        If one or more images fail to load (network, HTTP error, missing file, invalid data),
        those inputs are skipped. If every image fails, generation continues as text-only and
        a short system note is prepended to the prompt.
        """
        # Validate image input
        if images is not None and not self._validate_image_input(images):
            raise ValueError("Invalid image input format")

        user_prompt = prompt
        content: list = []

        if images is not None:
            if not isinstance(images, list):
                image_list: List[Any] = [images]
            else:
                image_list = list(images)

            for image in image_list:
                try:
                    data_url = self._encode_image_to_base64(image)
                    content.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": data_url},
                        }
                    )
                except Exception as e:
                    logger.warning(
                        "Skipping unreadable image (%s): %s",
                        self._image_input_repr(image),
                        e,
                    )

            if not content and image_list:
                user_prompt = _IMAGE_UNAVAILABLE_NOTE + prompt
                logger.warning(
                    "All %d image(s) failed to load; continuing with text-only generation.",
                    len(image_list),
                )

        # Add text prompt (after any image parts, per OpenAI multimodal convention)
        content.append(
            {
                "type": "text",
                "text": user_prompt,
            }
        )

        messages = [{"role": "user", "content": content}]
        max_retries = 3
        for attempt in range(max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **kwargs,
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                if attempt >= max_retries:
                    raise RuntimeError(f"OpenAI VLM API error: {e}") from e
                logger.warning(
                    "OpenAI VLM API call failed (attempt %d/%d): %s; retrying in 1s",
                    attempt + 1,
                    max_retries + 1,
                    e,
                )
                time.sleep(1)
    
    def __repr__(self) -> str:
        """Return string representation of VLM"""
        return f"OpenAIVLM(model={self.model})"


if __name__ == "__main__":
    """Test OpenAI VLM API calls"""
    import sys
    
    print("=" * 50)
    print("Testing OpenAI VLM API calls")
    print("=" * 50)
    
    try:
        # Initialize VLM
        print("\n1. Initializing OpenAI VLM...")
        vlm = OpenAIVLM(model="gpt-4o")
        print(f"✓ Successfully initialized: {vlm}")
        
        # Test generation with image (URL or local path)
        print("\n2. Testing image-to-text generation...")
        test_image = "https://m.media-amazon.com/images/I/41IiEBGouZL.jpg"  # Eyeshadow/makeup image
        # Or use local: test_image = "path/to/your/image.jpg"
        response = vlm.generate(
            prompt="What's in this image? Describe it briefly.",
            images=test_image,
            temperature=0.7,
            max_tokens=256
        )
        print(f"Response: {response}")
        print("✓ Image-to-text generation test completed")
        
        print("\n" + "=" * 50)
        print("Test completed!")
        print("=" * 50)
        
    except ValueError as e:
        print(f"\n✗ Configuration error: {e}")
        print("Hint: Please set OPENAI_API_KEY environment variable")
        sys.exit(1)
    except ImportError as e:
        print(f"\n✗ Import error: {e}")
        print("Hint: Install required packages with: pip install openai pillow requests")
        sys.exit(1)
    except RuntimeError as e:
        print(f"\n✗ API call error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Unknown error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

