import logging

from PIL import Image, ImageFilter

logger = logging.getLogger(__name__)


def strip_exif(image):
    """
    Remove EXIF metadata from an image by rebuilding it from raw pixel data.

    Args:
        image (PIL.Image): Input image.

    Returns:
        PIL.Image: Image without EXIF metadata.
    """
    try:
        data = list(image.getdata())
        clean_image = Image.new(image.mode, image.size)
        clean_image.putdata(data)
        return clean_image

    except Exception:
        logger.exception("EXIF stripping failed")
        return image


def smooth_pixels(image, radius=2):
    """
    Apply Gaussian Blur to smooth image pixels.

    Args:
        image (PIL.Image): Input image.
        radius (float): Blur radius.

    Returns:
        PIL.Image: Smoothed image.
    """
    try:
        return image.filter(ImageFilter.GaussianBlur(radius=radius))

    except Exception:
        logger.exception("Smoothing failed")
        return image


def run_defense_pipeline(image, remove_exif=True, apply_blur=True, blur_radius=2):
    """
    Run the complete defense pipeline.

    Args:
        image (PIL.Image): Input image.
        remove_exif (bool): Remove EXIF metadata.
        apply_blur (bool): Apply smoothing filter.
        blur_radius (float): Radius used for the blur filter.

    Returns:
        PIL.Image: Defended image.
    """
    try:
        defended_image = image.copy()

        if remove_exif:
            defended_image = strip_exif(defended_image)

        if apply_blur:
            defended_image = smooth_pixels(defended_image, radius=blur_radius)

        return defended_image

    except Exception:
        logger.exception("Defense pipeline failed")
        return image
