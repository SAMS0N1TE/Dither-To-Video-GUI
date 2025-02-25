import numpy as np
import cv2
from numba import jit
from sklearn.cluster import KMeans

# ------------------ Helper Functions ------------------
@jit
def to_gray(r, g, b):
    """
    Convert an RGB value to grayscale using the luminosity method.
    """
    return 0.299 * r + 0.587 * g + 0.114 * b

@jit(nopython=True)
def clamp(val, low=0, high=255):
    """Clamp the value between low and high."""
    return max(low, min(high, val))

@jit(nopython=True)
def color_distance(r1, g1, b1, r2, g2, b2):
    """
    Return squared distance in RGB space, to avoid sqrt overhead.
    """
    return (r1 - r2)**2 + (g1 - g2)**2 + (b1 - b2)**2

@jit(nopython=True)
def find_nearest_color_preserve(r, g, b, palette, preserve, preserve_threshold):
    """
    1) Find nearest color in 'palette' by squared distance.
    2) If 'preserve' is True and the distance is <= preserve_threshold^2,
       return the original color.
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
        if min_dist <= preserve_threshold * preserve_threshold:
            return (r, g, b)
        else:
            return nearest
    else:
        return nearest

# ------------------ Dithering Functions (Color-based) ------------------

@jit(nopython=True)
def ordered_dithering(image, palette, strength=100.0, preserve=False, preserve_threshold=0.0):
    """
    Ordered dithering using a 2x2 Bayer matrix.
    """
    bayer2 = np.array([[0, 128],
                       [192, 64]], dtype=np.float32)
    h, w, _ = image.shape
    for y in range(h):
        for x in range(w):
            old_b = image[y, x, 0]
            old_g = image[y, x, 1]
            old_r = image[y, x, 2]
            threshold = bayer2[y % 2, x % 2] * (strength / 100.0)
            t_b = clamp(old_b + threshold)
            t_g = clamp(old_g + threshold)
            t_r = clamp(old_r + threshold)
            nr, ng, nb = find_nearest_color_preserve(t_r, t_g, t_b, palette, preserve, preserve_threshold)
            image[y, x, 2] = clamp(nr)
            image[y, x, 1] = clamp(ng)
            image[y, x, 0] = clamp(nb)
    return image

@jit(nopython=True)
def atkinson_dithering(image, palette, strength=100.0, preserve=False, preserve_threshold=0.0):
    """
    Atkinson dithering with error diffusion among 6 neighbors.
    """
    h, w, _ = image.shape
    for y in range(h):
        for x in range(w):
            old_b = image[y, x, 0]
            old_g = image[y, x, 1]
            old_r = image[y, x, 2]
            nr, ng, nb = find_nearest_color_preserve(old_r, old_g, old_b, palette, preserve, preserve_threshold)
            image[y, x, 2] = clamp(nr)
            image[y, x, 1] = clamp(ng)
            image[y, x, 0] = clamp(nb)
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
    Floyd-Steinberg dithering with error diffusion.
    """
    h, w, _ = image.shape
    for y in range(h):
        for x in range(w):
            old_b = image[y, x, 0]
            old_g = image[y, x, 1]
            old_r = image[y, x, 2]
            nr, ng, nb = find_nearest_color_preserve(old_r, old_g, old_b, palette, preserve, preserve_threshold)
            image[y, x, 2] = clamp(nr)
            image[y, x, 1] = clamp(ng)
            image[y, x, 0] = clamp(nb)
            err_b = (old_b - nb) * (strength / 100.0)
            err_g = (old_g - ng) * (strength / 100.0)
            err_r = (old_r - nr) * (strength / 100.0)
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
    Jarvis-Judice-Ninke dithering with error diffusion.
    """
    h, w, _ = image.shape
    for y in range(h - 2):
        for x in range(2, w - 2):
            old_b = image[y, x, 0]
            old_g = image[y, x, 1]
            old_r = image[y, x, 2]
            nr, ng, nb = find_nearest_color_preserve(old_r, old_g, old_b, palette, preserve, preserve_threshold)
            image[y, x, 2] = clamp(nr)
            image[y, x, 1] = clamp(ng)
            image[y, x, 0] = clamp(nb)
            err_b = (old_b - nb) * (strength / 100.0)
            err_g = (old_g - ng) * (strength / 100.0)
            err_r = (old_r - nr) * (strength / 100.0)
            distribution = [
                (0, 1, 7/48), (0, 2, 5/48),
                (1, -2, 3/48), (1, -1, 5/48), (1, 0, 7/48),
                (1, 1, 5/48), (1, 2, 3/48),
                (2, -2, 1/48), (2, -1, 3/48), (2, 0, 5/48),
                (2, 1, 3/48), (2, 2, 1/48)
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
    Stucki dithering with error diffusion.
    """
    h, w, _ = image.shape
    for y in range(h - 2):
        for x in range(w - 2):
            old_b = image[y, x, 0]
            old_g = image[y, x, 1]
            old_r = image[y, x, 2]
            nr, ng, nb = find_nearest_color_preserve(old_r, old_g, old_b, palette, preserve, preserve_threshold)
            image[y, x, 2] = clamp(nr)
            image[y, x, 1] = clamp(ng)
            image[y, x, 0] = clamp(nb)
            err_b = (old_b - nb) * (strength / 100.0)
            err_g = (old_g - ng) * (strength / 100.0)
            err_r = (old_r - nr) * (strength / 100.0)
            distribution = [
                (0, 1, 8/42), (0, 2, 4/42),
                (1, -2, 2/42), (1, -1, 4/42), (1, 0, 8/42),
                (1, 1, 4/42), (1, 2, 2/42),
                (2, -2, 1/42), (2, -1, 2/42), (2, 0, 4/42),
                (2, 1, 2/42), (2, 2, 1/42)
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
    Patterned dithering using a 4x4 Bayer matrix.
    """
    bayer4 = np.array([
        [0,  8,  2, 10],
        [12, 4, 14, 6],
        [3, 11, 1, 9],
        [15, 7, 13, 5]
    ], dtype=np.float32)
    bayer4 = bayer4 * (strength / 100.0) * (255.0 / 16.0)
    h, w, _ = image.shape
    for y in range(h):
        for x in range(w):
            old_b = image[y, x, 0]
            old_g = image[y, x, 1]
            old_r = image[y, x, 2]
            threshold = bayer4[y % 4, x % 4]
            t_b = clamp(old_b + threshold)
            t_g = clamp(old_g + threshold)
            t_r = clamp(old_r + threshold)
            nr, ng, nb = find_nearest_color_preserve(t_r, t_g, t_b, palette, preserve, preserve_threshold)
            image[y, x, 2] = clamp(nr)
            image[y, x, 1] = clamp(ng)
            image[y, x, 0] = clamp(nb)
    return image

@jit(nopython=True)
def retro_shading_dithering(image, palette, strength=100.0, preserve=False, preserve_threshold=0.0, matrix=None):
    """
    Retro shading effect using a customizable dithering matrix.
    """
    if matrix is None:
        matrix = np.array([
            [ 0, 128,  32, 160],
            [192,  64, 224,  96],
            [ 48, 176,  16, 144],
            [240, 112, 208,  80]
        ], dtype=np.float32)
    matrix = matrix * (strength / 100.0) * (255.0 / (matrix.max() + 1e-5))
    h, w, _ = image.shape
    for y in range(h):
        for x in range(w):
            r = image[y, x, 2]
            g = image[y, x, 1]
            b = image[y, x, 0]
            gray = to_gray(r, g, b)
            threshold = matrix[y % matrix.shape[0], x % matrix.shape[1]]
            final_gray = gray + threshold
            # Here we simply find the nearest color based on gray level.
            # This function could be improved for full RGB.
            nearest_color = find_nearest_color_in_palette((final_gray, final_gray, final_gray), palette)
            image[y, x, 2] = clamp(nearest_color[2])
            image[y, x, 1] = clamp(nearest_color[1])
            image[y, x, 0] = clamp(nearest_color[0])
    return image

# ------------------ Dithering Dictionary ------------------
DITHERING_TYPES = {
    "Ordered": ordered_dithering,
    "Atkinson": atkinson_dithering,
    "Floyd-Steinberg": floyd_steinberg_dither,
    "Jarvis-Judice-Ninke": jarvis_judice_ninke_dither,
    "Stucki": stucki_dither,
    "Patterned Dithering": patterned_dithering,
    "Retro Shading": retro_shading_dithering
}

# ------------------ Image Adjustment Functions ------------------
def adjust_brightness(image, value):
    """Adjust brightness by adding 'value' to all pixels."""
    return cv2.convertScaleAbs(image, alpha=1, beta=value)

def adjust_contrast(image, value):
    """Adjust contrast by multiplying pixel values by 'value'."""
    return cv2.convertScaleAbs(image, alpha=value, beta=0)

def adjust_saturation(image, value):
    """Adjust saturation by scaling the S channel in HSV."""
    image_bgr = image.copy()
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] *= value
    hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

def adjust_hue(image, value):
    """
    Shift hue by 'value' degrees.
    """
    image_bgr = image.copy()
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 0] = (hsv[:, :, 0] + (value / 2)) % 180
    hsv[:, :, 0] = np.clip(hsv[:, :, 0], 0, 179)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

def apply_noise_reduction(image):
    """Apply Gaussian Blur to reduce noise."""
    return cv2.GaussianBlur(image, (5, 5), 0)

def extract_palette_accurate_kmeans(image, num_colors=8, max_dim=128,
                                    kmeans_algorithm='elkan', kmeans_inits=1,
                                    kmeans_max_iter=300, kmeans_tol=1e-4, random_state=42):
    """
    Runs scikit-learn KMeans on a downsampled image for color extraction.
    
    Parameters:
      image           : BGR image (NumPy array).
      num_colors      : Number of colors to extract.
      max_dim         : Maximum dimension for downscaling.
      kmeans_algorithm: 'elkan' or 'full' for scikit-learn KMeans.
      kmeans_inits    : Number of initializations.
      kmeans_max_iter : Maximum iterations for KMeans.
      kmeans_tol      : Tolerance for convergence.
      random_state    : Random seed for reproducibility.
      
    Returns:
      List of (R, G, B) tuples.
    """
    h, w = image.shape[:2]
    scale_factor = 1.0
    if max(h, w) > max_dim:
        scale_factor = max_dim / float(max(h, w))
        image = cv2.resize(image, (int(w * scale_factor), int(h * scale_factor)), interpolation=cv2.INTER_AREA)
    reshaped = image.reshape((-1, 3)).astype(np.float32)
    kmeans = KMeans(n_clusters=num_colors, algorithm=kmeans_algorithm, n_init=kmeans_inits,
                    max_iter=kmeans_max_iter, tol=kmeans_tol, random_state=random_state)
    kmeans.fit(reshaped)
    centers = kmeans.cluster_centers_
    palette = [tuple(map(int, c)) for c in centers]
    return palette

# New: Fast palette extraction using OpenCV's kmeans
def extract_palette_fast_opencv(frame, num_colors, max_dim):
    """
    Uses OpenCV's built-in kmeans for fast palette extraction.
    Returns a list of (R, G, B) tuples.
    """
    h, w = frame.shape[:2]
    scale_factor = 1.0
    if max(h, w) > max_dim:
        scale_factor = max_dim / float(max(h, w))
        frame = cv2.resize(frame, (int(w * scale_factor), int(h * scale_factor)), interpolation=cv2.INTER_AREA)
    pixels = frame.reshape((-1, 3)).astype(np.float32)
    # Define criteria and run kmeans; note OpenCV's kmeans does not accept a seed parameter.
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret, labels, centers = cv2.kmeans(pixels, num_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    palette = [tuple(int(c) for c in center) for center in centers]
    return palette

def palette_distance(c1, c2):
    """Return squared distance between two (B, G, R) tuples."""
    return (c1[0]-c2[0])**2 + (c1[1]-c2[1])**2 + (c1[2]-c2[2])**2

def find_nearest_color_in_palette(target, palette):
    """Return the color in 'palette' nearest to 'target'."""
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
    Blend two colors using a weighted average.
    """
    r = int((1 - alpha) * c_old[0] + alpha * c_new[0])
    g = int((1 - alpha) * c_old[1] + alpha * c_new[1])
    b = int((1 - alpha) * c_old[2] + alpha * c_new[2])
    return (b, g, r)

def blend_palette(old_palette, new_palette, alpha=0.3):
    """
    Blend two palettes by blending each color in old_palette with its nearest in new_palette.
    """
    if len(old_palette) != len(new_palette):
        return new_palette
    blended = []
    for old_color in old_palette:
        near_new = find_nearest_color_in_palette(old_color, new_palette)
        blended.append(blend_color(old_color, near_new, alpha))
    return blended
