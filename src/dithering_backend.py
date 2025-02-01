import numpy as np
import cv2
from numba import jit
from sklearn.cluster import KMeans

# ------------------ Helper Functions ------------------

@jit(nopython=True)
def clamp(val, low=0, high=255):
    """Clamp the value between low and high."""
    return max(low, min(high, val))

@jit(nopython=True)
def color_distance(r1, g1, b1, r2, g2, b2):
    """
    Return squared distance in RGB space, to avoid sqrt overhead:
    (r1-r2)^2 + (g1-g2)^2 + (b1-b2)^2
    """
    return (r1 - r2)**2 + (g1 - g2)**2 + (b1 - b2)**2

@jit(nopython=True)
def find_nearest_color_preserve(r, g, b, palette, preserve, preserve_threshold):
    """
    1) Find nearest color in 'palette' by squared distance in RGB space.
    2) If 'preserve' is True and the distance to that nearest color
       is <= preserve_threshold^2, we keep the original (r,g,b) instead.
    """
    min_dist = 1e12
    nearest = (0, 0, 0)
    for color in palette:
        R, G, B = color
        dist = color_distance(r, g, b, R, G, B)
        if dist < min_dist:
            min_dist = dist
            nearest = (R, G, B)

    if preserve:
        # Compare distance with threshold^2
        if min_dist <= preserve_threshold * preserve_threshold:
            # Keep original color
            return (r, g, b)
        else:
            return nearest
    else:
        return nearest

# ------------------ Dithering Functions (Color-based) ------------------

@jit(nopython=True)
def ordered_dithering(image, palette, strength=100.0, preserve=False, preserve_threshold=0.0):
    """
    Ordered dithering using a 2x2 Bayer matrix. Full RGB approach.
    - preserve: if True, check threshold to decide if we keep original color
    """
    bayer2 = np.array([[0, 128],
                       [192, 64]], dtype=np.float32)
    h, w, c = image.shape

    for y in range(h):
        for x in range(w):
            old_b = image[y, x, 0]
            old_g = image[y, x, 1]
            old_r = image[y, x, 2]

            # scale threshold by strength
            threshold = bayer2[y % 2, x % 2] * (strength / 100.0)
            t_b = clamp(old_b + threshold)
            t_g = clamp(old_g + threshold)
            t_r = clamp(old_r + threshold)

            nr, ng, nb = find_nearest_color_preserve(
                t_r, t_g, t_b, palette,
                preserve, preserve_threshold
            )

            image[y, x, 2] = clamp(nr)
            image[y, x, 1] = clamp(ng)
            image[y, x, 0] = clamp(nb)
    return image

@jit(nopython=True)
def atkinson_dithering(image, palette, strength=100.0, preserve=False, preserve_threshold=0.0):
    """
    Atkinson dithering, distributing error among 6 neighbors. Full RGB approach.
    - preserve: if True, keep the original pixel if within distance threshold.
    """
    h, w, c = image.shape

    for y in range(h):
        for x in range(w):
            old_b = image[y, x, 0]
            old_g = image[y, x, 1]
            old_r = image[y, x, 2]

            nr, ng, nb = find_nearest_color_preserve(
                old_r, old_g, old_b, palette,
                preserve, preserve_threshold
            )

            image[y, x, 2] = clamp(nr)
            image[y, x, 1] = clamp(ng)
            image[y, x, 0] = clamp(nb)

            # error
            err_b = old_b - nb
            err_g = old_g - ng
            err_r = old_r - nr

            e_b = err_b * (strength / 100.0) / 8.0
            e_g = err_g * (strength / 100.0) / 8.0
            e_r = err_r * (strength / 100.0) / 8.0

            neighbors = [(0, 1), (0, 2), (1, -1), (1, 0), (1, 1), (2, 0)]
            for dy, dx in neighbors:
                ny = y + dy
                nx = x + dx
                if 0 <= ny < h and 0 <= nx < w:
                    image[ny, nx, 0] = clamp(image[ny, nx, 0] + e_b)
                    image[ny, nx, 1] = clamp(image[ny, nx, 1] + e_g)
                    image[ny, nx, 2] = clamp(image[ny, nx, 2] + e_r)
    return image

@jit(nopython=True)
def floyd_steinberg_dither(image, palette, strength=100.0, preserve=False, preserve_threshold=0.0):
    """
    Floyd-Steinberg dithering with user-defined strength, full RGB.
    - preserve: if True, keep the original pixel if within distance threshold.
    """
    h, w, c = image.shape

    for y in range(h):
        for x in range(w):
            old_b = image[y, x, 0]
            old_g = image[y, x, 1]
            old_r = image[y, x, 2]

            nr, ng, nb = find_nearest_color_preserve(
                old_r, old_g, old_b, palette,
                preserve, preserve_threshold
            )

            image[y, x, 2] = clamp(nr)
            image[y, x, 1] = clamp(ng)
            image[y, x, 0] = clamp(nb)

            err_b = (old_b - nb) * (strength / 100.0)
            err_g = (old_g - ng) * (strength / 100.0)
            err_r = (old_r - nr) * (strength / 100.0)

            # Distribute error
            if x + 1 < w:
                image[y, x+1, 0] = clamp(image[y, x+1, 0] + err_b * 7/16)
                image[y, x+1, 1] = clamp(image[y, x+1, 1] + err_g * 7/16)
                image[y, x+1, 2] = clamp(image[y, x+1, 2] + err_r * 7/16)

            if x - 1 >= 0 and y + 1 < h:
                image[y+1, x-1, 0] = clamp(image[y+1, x-1, 0] + err_b * 3/16)
                image[y+1, x-1, 1] = clamp(image[y+1, x-1, 1] + err_g * 3/16)
                image[y+1, x-1, 2] = clamp(image[y+1, x-1, 2] + err_r * 3/16)

            if y + 1 < h:
                image[y+1, x, 0] = clamp(image[y+1, x, 0] + err_b * 5/16)
                image[y+1, x, 1] = clamp(image[y+1, x, 1] + err_g * 5/16)
                image[y+1, x, 2] = clamp(image[y+1, x, 2] + err_r * 5/16)

            if x + 1 < w and y + 1 < h:
                image[y+1, x+1, 0] = clamp(image[y+1, x+1, 0] + err_b * 1/16)
                image[y+1, x+1, 1] = clamp(image[y+1, x+1, 1] + err_g * 1/16)
                image[y+1, x+1, 2] = clamp(image[y+1, x+1, 2] + err_r * 1/16)
    return image

@jit(nopython=True)
def jarvis_judice_ninke_dither(image, palette, strength=100.0, preserve=False, preserve_threshold=0.0):
    """
    Jarvis-Judice-Ninke dithering with user-defined strength, full RGB.
    - preserve: if True, keep the original pixel if within distance threshold.
    """
    h, w, c = image.shape

    for y in range(h - 2):
        for x in range(2, w - 2):
            old_b = image[y, x, 0]
            old_g = image[y, x, 1]
            old_r = image[y, x, 2]

            nr, ng, nb = find_nearest_color_preserve(
                old_r, old_g, old_b, palette,
                preserve, preserve_threshold
            )

            image[y, x, 2] = clamp(nr)
            image[y, x, 1] = clamp(ng)
            image[y, x, 0] = clamp(nb)

            err_b = (old_b - nb) * (strength / 100.0)
            err_g = (old_g - ng) * (strength / 100.0)
            err_r = (old_r - nr) * (strength / 100.0)

            distribution = [
                (0, 1, 7/48), (0, 2, 5/48),
                (1, -2, 3/48), (1, -1, 5/48), (1, 0, 7/48),
                (1, 1, 5/48),  (1, 2, 3/48),
                (2, -2, 1/48), (2, -1, 3/48), (2, 0, 5/48),
                (2, 1, 3/48),  (2, 2, 1/48)
            ]
            for dy, dx, wgt in distribution:
                ny = y + dy
                nx = x + dx
                if 0 <= ny < h and 0 <= nx < w:
                    image[ny, nx, 0] = clamp(image[ny, nx, 0] + err_b * wgt)
                    image[ny, nx, 1] = clamp(image[ny, nx, 1] + err_g * wgt)
                    image[ny, nx, 2] = clamp(image[ny, nx, 2] + err_r * wgt)
    return image

@jit(nopython=True)
def stucki_dither(image, palette, strength=100.0, preserve=False, preserve_threshold=0.0):
    """
    Stucki dithering with user-defined strength, full RGB.
    - preserve: if True, keep the original pixel if within distance threshold.
    """
    h, w, c = image.shape

    for y in range(h - 2):
        for x in range(2, w - 2):
            old_b = image[y, x, 0]
            old_g = image[y, x, 1]
            old_r = image[y, x, 2]

            nr, ng, nb = find_nearest_color_preserve(
                old_r, old_g, old_b, palette,
                preserve, preserve_threshold
            )

            image[y, x, 2] = clamp(nr)
            image[y, x, 1] = clamp(ng)
            image[y, x, 0] = clamp(nb)

            err_b = (old_b - nb) * (strength / 100.0)
            err_g = (old_g - ng) * (strength / 100.0)
            err_r = (old_r - nr) * (strength / 100.0)

            distribution = [
                (0, 1, 8/42), (0, 2, 4/42),
                (1, -2, 2/42), (1, -1, 4/42), (1, 0, 8/42),
                (1, 1, 4/42),  (1, 2, 2/42),
                (2, -2, 1/42), (2, -1, 2/42), (2, 0, 4/42),
                (2, 1, 2/42),  (2, 2, 1/42)
            ]
            for dy, dx, wgt in distribution:
                ny = y + dy
                nx = x + dx
                if 0 <= ny < h and 0 <= nx < w:
                    image[ny, nx, 0] = clamp(image[ny, nx, 0] + err_b * wgt)
                    image[ny, nx, 1] = clamp(image[ny, nx, 1] + err_g * wgt)
                    image[ny, nx, 2] = clamp(image[ny, nx, 2] + err_r * wgt)
    return image

@jit(nopython=True)
def patterned_dithering(image, palette, strength=100.0, preserve=False, preserve_threshold=0.0):
    """
    Patterned dithering using a 4x4 Bayer matrix for more intricate patterns.
    Full RGB approach.
    - preserve: if True, keep the original pixel if within distance threshold.
    """
    bayer4 = np.array([
        [0,  8,  2, 10],
        [12, 4, 14, 6],
        [3, 11, 1, 9],
        [15, 7, 13, 5]
    ], dtype=np.float32)

    bayer4 = bayer4 * (strength / 100.0) * (255.0 / 16.0)

    h, w, c = image.shape
    for y in range(h):
        for x in range(w):
            old_b = image[y, x, 0]
            old_g = image[y, x, 1]
            old_r = image[y, x, 2]

            threshold = bayer4[y % 4, x % 4]
            t_b = clamp(old_b + threshold)
            t_g = clamp(old_g + threshold)
            t_r = clamp(old_r + threshold)

            nr, ng, nb = find_nearest_color_preserve(
                t_r, t_g, t_b, palette,
                preserve, preserve_threshold
            )

            image[y, x, 2] = clamp(nr)
            image[y, x, 1] = clamp(ng)
            image[y, x, 0] = clamp(nb)
    return image

# ------------------ Dithering Dictionary ------------------

DITHERING_TYPES = {
    "Ordered": ordered_dithering,
    "Atkinson": atkinson_dithering,
    "Floyd-Steinberg": floyd_steinberg_dither,
    "Jarvis-Judice-Ninke": jarvis_judice_ninke_dither,
    "Stucki": stucki_dither,
    "Patterned Dithering": patterned_dithering
}

# ------------------ Image Adjustment Functions ------------------

def adjust_brightness(image, value):
    """
    Adjust brightness by adding 'value' to all pixels.
    value: -100..100
    """
    return cv2.convertScaleAbs(image, alpha=1, beta=value)

def adjust_contrast(image, value):
    """
    Adjust contrast by multiplying all pixel values by 'value'.
    value: 0.5..3.0
    """
    return cv2.convertScaleAbs(image, alpha=value, beta=0)

def adjust_saturation(image, value):
    """
    Adjust saturation by scaling the S channel in HSV.
    value: 0.0..3.0
    """
    image_bgr = image.copy()
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] *= value
    hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
    image_bgr = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return image_bgr

def adjust_hue(image, value):
    """
    Shift hue by 'value' degrees in range -180..180.
    OpenCV hue range is [0, 179].
    """
    image_bgr = image.copy()
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 0] = (hsv[:, :, 0] + (value / 2)) % 180
    hsv[:, :, 0] = np.clip(hsv[:, :, 0], 0, 179)
    image_bgr = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return image_bgr

def apply_noise_reduction(image):
    """
    Apply a Gaussian Blur to reduce noise.
    """
    return cv2.GaussianBlur(image, (5, 5), 0)

def extract_palette_accurate_kmeans(image, num_colors=8, max_dim=128,
                                    kmeans_algorithm='elkan',
                                    kmeans_inits=1):
    """
    Runs scikit-learn KMeans on a downsampled image for color extraction.

    :param image: BGR image (NumPy array).
    :param num_colors: Number of colors to extract (k).
    :param max_dim: Max dimension to downscale the image (width or height).
    :param kmeans_algorithm: 'elkan' or 'full' for scikit-learn KMeans.
    :param kmeans_inits: n_init (number of times to run KMeans with different seeds).
    :return: List of (R, G, B) cluster centers (as int tuples).
    """
    h, w = image.shape[:2]

    # 1) Downsample if needed for performance
    scale_factor = 1.0
    if max(h, w) > max_dim:
        scale_factor = max_dim / float(max(h, w))

    if scale_factor < 1.0:
        new_w = int(w * scale_factor)
        new_h = int(h * scale_factor)
        small = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        small = image

    # 2) Flatten & convert to float
    reshaped = small.reshape((-1, 3)).astype(np.float32)

    # 3) Setup KMeans with user-chosen algorithm + n_init
    kmeans = KMeans(
        n_clusters=num_colors,
        algorithm=kmeans_algorithm,
        n_init=kmeans_inits,
        random_state=42
    )
    kmeans.fit(reshaped)

    # 4) Convert cluster centers to integer (B, G, R) 
    # NOTE: scikit-learn is typically (float32) in order (R,G,B).
    centers = kmeans.cluster_centers_
    palette = [tuple(map(int, c)) for c in centers]
    return palette

def palette_distance(c1, c2):
    """Return squared distance between two (B,G,R) tuples."""
    return (c1[0]-c2[0])**2 + (c1[1]-c2[1])**2 + (c1[2]-c2[2])**2

def find_nearest_color(target, palette):
    """Return the color in 'palette' that is nearest to 'target' by squared distance."""
    best = None
    best_dist = float('inf')
    for color in palette:
        dist = palette_distance(target, color)
        if dist < best_dist:
            best_dist = dist
            best = color
    return best

def blend_color(c_old, c_new, alpha=0.3):
    """
    Weighted blend of old and new color:
    final = (1-alpha)*c_old + alpha*c_new
    Each is (B,G,R).
    """
    b = int((1.0 - alpha)*c_old[0] + alpha*c_new[0])
    g = int((1.0 - alpha)*c_old[1] + alpha*c_new[1])
    r = int((1.0 - alpha)*c_old[2] + alpha*c_new[2])
    return (b, g, r)

def blend_palette(old_palette, new_palette, alpha=0.3):
    """
    For each color in old_palette, find its nearest match in new_palette and blend them.
    If old/new differ in length, or you want advanced matching, adapt accordingly.
    """
    if len(old_palette) != len(new_palette):
        # Fallback or partial approach
        return new_palette
    
    blended = []
    for old_color in old_palette:
        near_new = find_nearest_color(old_color, new_palette)
        bc = blend_color(old_color, near_new, alpha)
        blended.append(bc)
    return blended
