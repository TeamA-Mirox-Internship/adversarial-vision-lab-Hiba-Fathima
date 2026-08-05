import logging
import re

from ollama import chat

logger = logging.getLogger(__name__)

_RESULT_PATTERN = re.compile(
    r"Object:\s*(?P<label>.+?)\s*,\s*Confidence:\s*(?P<confidence>[\d.]+)\s*%",
    re.IGNORECASE,
)


def classify_image(image_path):
    """
    Classify an image using the Ollama LLaVA vision model.

    Args:
        image_path (str): Path to the image file.

    Returns:
        dict: {"label": str, "confidence": float, "raw": str} on success,
              or {"label": "Error", "confidence": 0.0, "error": str} on failure.
    """
    try:
        response = chat(
            model="llava",
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Identify the main object in this image. "
                        "Respond ONLY in this format: Object: <object> , Confidence: <number>%"
                    ),
                    "images": [image_path],
                }
            ],
        )

        content = response["message"]["content"]

        match = _RESULT_PATTERN.search(content)

        if match:
            return {
                "label": match.group("label").strip(),
                "confidence": float(match.group("confidence")),
                "raw": content,
            }

        # Model didn't follow the format - surface the raw text instead of failing silently.
        return {"label": content.strip(), "confidence": 0.0, "raw": content}

    except Exception as exc:
        logger.exception("LLaVA classification failed")
        return {
            "label": "Error",
            "confidence": 0.0,
            "error": str(exc),
            "raw": "Model timeout - try again. Is Ollama running with the 'llava' model pulled?",
        }
