import numpy as np
from PIL import Image


def apply_gaussian(image, intensity):
    """
    Apply Gaussian noise to a PIL image.

    Args:
        image (PIL.Image): Input image.
        intensity (int): Noise intensity.

    Returns:
        PIL.Image: Noisy image.
    """
    try:
        img_array = np.array(image).astype(np.float32)

        mean = 0
        sigma = intensity * 3

        gaussian_noise = np.random.normal(mean, sigma, img_array.shape)

        noisy_image = img_array + gaussian_noise
        noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)

        return Image.fromarray(noisy_image)

    except Exception:
        return image


def apply_pixel_shift(image, intensity):
    """
    Shift image pixels to disrupt spatial structure, breaking the
    coherence a classifier relies on. More intense than a simple
    uniform roll: shifts each row (and optionally each color channel)
    by a different, intensity-scaled, randomized amount, and also
    applies a global vertical shift.
 
    Args:
        image (PIL.Image): Input image.
        intensity (int): Shift intensity. Higher = more disruptive.
            Roughly controls the max pixel displacement per row.
 
    Returns:
        PIL.Image: Shifted image.
    """
    try:
        img_array = np.array(image)
        h, w = img_array.shape[0], img_array.shape[1]
 
        # Scale max shift with intensity and image width so it stays
        # meaningful on both small and large images.
        max_shift = max(1, int(intensity * max(2, w * 0.02)))
 
        result = img_array.copy()
 
        # Per-row horizontal shifts (different random amount each row)
        # breaks horizontal edges/contours much more than a uniform roll.
        row_shifts = np.random.randint(-max_shift, max_shift + 1, size=h)
        for y in range(h):
            result[y] = np.roll(result[y], shift=row_shifts[y], axis=0)
 
        # Global vertical shift on top, so structure is broken in both axes.
        vertical_shift = np.random.randint(-max_shift, max_shift + 1)
        result = np.roll(result, shift=vertical_shift, axis=0)
 
        # If there are multiple color channels, shift each one slightly
        # differently to also scramble color-edge alignment.
        if result.ndim == 3:
            for c in range(result.shape[2]):
                channel_shift = np.random.randint(-max_shift // 2 - 1, max_shift // 2 + 2)
                result[:, :, c] = np.roll(result[:, :, c], shift=channel_shift, axis=1)
 
        return Image.fromarray(result.astype(np.uint8))
 
    except Exception:
        return image
 
   