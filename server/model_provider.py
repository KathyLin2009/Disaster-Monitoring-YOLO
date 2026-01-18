import os
import base64
import time
from abc import ABC, abstractmethod
from typing import List
import google.generativeai as genai
from google.api_core import exceptions
import httpx

class ModelProvider(ABC):
    @abstractmethod
    async def analyze_image(self, image_b64: str, prompt_text: str = None) -> str:
        """Analyze an image and return a text description."""
        pass

    @abstractmethod
    async def discover_prompts(self, image_b64: str, current_prompts: List[str]) -> List[str]:
        """Analyze an image to discover new object prompts."""
        pass

class GeminiProvider(ModelProvider):
    def __init__(self, api_key: str):
        self.api_key = api_key
        if not self.api_key:
            print("Warning: GEMINI_API_KEY not set.")
        else:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel('gemini-2.0-flash-exp')

    async def analyze_image(self, image_b64: str, prompt_text: str = None) -> str:
        if not self.api_key:
            return "Error: Server missing GEMINI_API_KEY."

        try:
            # Clean base64
            if "," in image_b64:
                image_b64 = image_b64.split(",")[1]
            img_bytes = base64.b64decode(image_b64)

            if not prompt_text:
                prompt_text = """Analyze the provided image for any potential risks or hazards. Also give a brief description. 
                Do not mention the YOLO bounding boxes or percentages.
                Do not format the output, just use plain text in one paragraph."""

            # Retry logic
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    # Gemini 'generate_content' is sync, but we treat it as blocking for now or could wrap it
                    # The previous main.py implementation was using synchronous generate_content inside async def, 
                    # but had a commented out await generate_content_async. 
                    # We will use generate_content_async if available in this version of SDK, or wrap it.
                    # Current google-generativeai usually supports generate_content_async.
                    response = await self.model.generate_content_async([
                        prompt_text,
                        {"mime_type": "image/jpeg", "data": img_bytes}
                    ])
                    return response.text
                except exceptions.ResourceExhausted as e:
                    if attempt < max_retries - 1:
                        wait_time = (2 ** attempt) + 1
                        time.sleep(wait_time) 
                    else:
                        raise e
        except Exception as e:
            print(f"Gemini Error: {e}")
            return f"Failed to analyze image: {str(e)}"

    async def discover_prompts(self, image_b64: str, current_prompts: List[str]) -> List[str]:
        if not self.api_key:
            return []

        try:
            if "," in image_b64:
                image_b64 = image_b64.split(",")[1]
            img_bytes = base64.b64decode(image_b64)

            if not current_prompts:
                prompt = """Analyze this image and identify the 3 most important or distinct object categories present. 
                Focus on things that would be relevant for a drone monitoring an area after a natural disaster.
                Return only a comma-separated list of the 3 labels as lowercase text. No explanation."""
            else:
                prompt = f"""Analyze this image. The current detected objects are: {', '.join(current_prompts)}.
                Identify up to 3 additional UNIQUE objects of interest not already listed. These must be important for aiding a search and rescue team and must be related to the landscape after a natural disaster.
                DO NOT SUGGEST OBJECTS THAT ARE ALREADY LISTED.
                Evaluate the importance of these 3 objects in comparison to the current list and ONLY keep the most important ones.
                Priority: Only suggest objects that are more visually prominent or contextually significant than those currently listed. 
                Constraints: Do not exceed a total of 20 prompts including current ones. If the limit is hit, do not suggest additional prompts and return an empty string.
                Output: Return only a comma-separated list of new lowercase labels. 
                If no significant new objects are found, return an empty string. Do not include any introductory text or explanation."""

            response = await self.model.generate_content_async([
                prompt,
                {"mime_type": "image/jpeg", "data": img_bytes}
            ])

            text = response.text.strip()
            if not text:
                return []
            
            text = text.replace("```", "").replace("csv", "").strip()
            new_prompts = [p.strip().lower() for p in text.split(",") if p.strip()]
            return new_prompts
        except Exception as e:
            print(f"Gemini Discovery Error: {e}")
            return []

class GemmaProvider(ModelProvider):
    def __init__(self, api_base: str = "http://34.59.89.44:8000/v1", model_name: str = "google/gemma-3-12b-it"):
        self.api_base = api_base
        self.model_name = model_name
        self.client = httpx.AsyncClient(timeout=60.0)
        print(f"Initialized GemmaProvider with {api_base} for {model_name}")

    async def analyze_image(self, image_b64: str, prompt_text: str = None) -> str:
        if "," in image_b64:
            image_b64 = image_b64.split(",")[1]
            
        if not prompt_text:
            prompt_text = """Analyze the provided image for any potential risks or hazards. Also give a brief description. 
            Do not mention the YOLO bounding boxes or percentages.
            Do not format the output, just use plain text in one paragraph."""

        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_text},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_b64}"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 512,
            "temperature": 0.7
        }

        try:
            response = await self.client.post(
                f"{self.api_base}/chat/completions", 
                json=payload,
                headers={"Authorization": "Bearer EMPTY"}
            )
            response.raise_for_status()
            result = response.json()
            return result['choices'][0]['message']['content']
        except Exception as e:
            print(f"Gemma Error: {e}")
            return f"Failed to analyze image with Gemma: {str(e)}"

    async def discover_prompts(self, image_b64: str, current_prompts: List[str]) -> List[str]:
        if not current_prompts:
            prompt = """Analyze this image and identify the 3 most important or distinct object categories present. 
            Focus on things that would be relevant for a drone monitoring an area after a natural disaster.
            Return only a comma-separated list of the 3 labels as lowercase text. No explanation."""
        else:
            prompt = f"""Analyze this image. The current detected objects are: {', '.join(current_prompts)}.
            Identify up to 3 additional UNIQUE objects of interest not already listed. These must be important for aiding a search and rescue team and must be related to the landscape after a natural disaster.
            DO NOT SUGGEST OBJECTS THAT ARE ALREADY LISTED.
            Evaluate the importance of these 3 objects in comparison to the current list and ONLY keep the most important ones.
            Priority: Only suggest objects that are more visually prominent or contextually significant than those currently listed. 
            Constraints: Do not exceed a total of 20 prompts including current ones. If the limit is hit, do not suggest additional prompts and return an empty string.
            Output: Return only a comma-separated list of new lowercase labels. 
            If no significant new objects are found, return an empty string. Do not include any introductory text or explanation."""

        try:
            text = await self.analyze_image(image_b64, prompt)
            text = text.replace("```", "").replace("csv", "").strip()
            new_prompts = [p.strip().lower() for p in text.split(",") if p.strip()]
            return new_prompts
        except Exception as e:
            print(f"Gemma Discovery Error: {e}")
            return []
