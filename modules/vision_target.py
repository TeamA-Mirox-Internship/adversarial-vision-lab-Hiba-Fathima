import logging

import streamlit as st
from transformers import pipeline

logger = logging.getLogger(__name__)


@st.cache_resource(show_spinner=False)
def load_classifier():
    """
    Load and cache the ViT image classification pipeline across reruns.

    Returns:
        transformers.Pipeline: The loaded image-classification pipeline.
    """
    return pipeline(
        "image-classification",
        model="google/vit-base-patch16-224"
    )


def classify_image(image):
    """
    Classify an image using the ViT model.

    Args:
        image (PIL.Image): Input image.

    Returns:
        dict: {"label": str, "confidence": float} on success,
              or {"label": "Error", "confidence": 0.0, "error": str} on failure.
    """
    try:
        classifier = load_classifier()

        result = classifier(image)

        prediction = result[0]

        label = prediction["label"]
        confidence = prediction["score"] * 100

        return {"label": label, "confidence": confidence}

    except Exception as exc:
        logger.exception("ViT classification failed")
        return {"label": "Error", "confidence": 0.0, "error": str(exc)}
