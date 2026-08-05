import logging

import piexif
from PIL import Image

logger = logging.getLogger(__name__)


def text_to_binary(text):
    """
    Convert text into a binary string.
    """
    return "".join(format(ord(char), "08b") for char in text)


def inject_exif(image, prompt):
    """
    Inject a hidden prompt into the EXIF ImageDescription tag.

    Note: PNG does not natively support EXIF the way JPEG does. Pillow will
    only persist the EXIF bytes on save if they are passed explicitly to
    `Image.save(..., exif=...)`. Storing them in `image.info["exif"]` here
    means the caller MUST pass `image.info["exif"]` back in when saving,
    otherwise the injected payload is silently lost.

    Args:
        image (PIL.Image): Input image.
        prompt (str): Hidden prompt.

    Returns:
        PIL.Image: Image containing EXIF metadata (stored in image.info["exif"]).
    """
    try:
        modified_image = image.copy()

        exif_dict = {
            "0th": {},
            "Exif": {},
            "GPS": {},
            "Interop": {},
            "1st": {},
            "thumbnail": None
        }

        exif_dict["0th"][piexif.ImageIFD.ImageDescription] = prompt.encode("utf-8")

        exif_bytes = piexif.dump(exif_dict)

        modified_image.info["exif"] = exif_bytes

        return modified_image

    except Exception:
        logger.exception("EXIF injection failed")
        return image


def inject_lsb(image, prompt):
    """
    Hide a prompt inside the least significant bit of the Red channel.

    Args:
        image (PIL.Image): Input image.
        prompt (str): Hidden prompt.

    Returns:
        PIL.Image: Image containing embedded message.
    """
    try:
        modified_image = image.copy().convert("RGB")

        binary_message = text_to_binary(prompt)

        width, height = modified_image.size

        capacity = width * height

        if len(binary_message) > capacity:
            logger.warning(
                "Prompt too large for LSB capacity (%d bits needed, %d available)",
                len(binary_message),
                capacity,
            )
            return image

        pixels = modified_image.load()
        index = 0

        for y in range(height):

            for x in range(width):

                if index >= len(binary_message):
                    break

                r, g, b = pixels[x, y]

                bit = int(binary_message[index])

                r = (r & 254) | bit

                pixels[x, y] = (r, g, b)

                index += 1

            if index >= len(binary_message):
                break

        return modified_image

    except Exception:
        logger.exception("LSB injection failed")
        return image
