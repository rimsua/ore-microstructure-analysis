import cv2
import matplotlib.pyplot as plt

# загрузка изображения с диска
def load_image(path):
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(f"Не удалось загрузить изображение: {path}")

    return image

# просмотр изображения
def show_image(image, title = 'Изображение'):
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(8,8))
    plt.imshow(img_rgb)
    plt.axis('off')
    plt.title(title)
    plt.show()

"""
нормализация освещения
"""

def lighting_normalization(image, kernel_size=101):
    img_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_chanell, a_chanell, b_chanell = cv2.split(img_lab)

    back_lightning = cv2.GaussianBlur(l_chanell, (kernel_size, kernel_size), 0)

    l_normalized = cv2.divide(l_chanell, back_lightning, scale=255)
    l_normalized = cv2.normalize(l_normalized, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")

    img_normalized_lab = cv2.merge([l_normalized, a_chanell, b_chanell])

    img_BGR = cv2.cvtColor(img_normalized_lab, cv2.COLOR_LAB2BGR)

    return img_BGR

"""
шумоподавление
"""

# 1. фильтр Гаусса
def denoise_gauss(image, kernel_size=5):
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

# 2. медианный фильтр (соль и перец)
def denoise_median(image, kernel_size=5):
    return cv2.medianBlur(image, kernel_size)

# 3. билатеральный (для сохранения четкости)
def denoise_bilateral(image, diam=9, sigm_clr=75, sigm_space=75):
    return cv2.bilateralFilter(image, diam, sigm_clr, sigm_space)

"""
контраст
"""

def contrast_correction(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    img_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_chanell, a_chanell, b_chanell = cv2.split(img_lab)
    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=tile_grid_size
    )
    enhanced_l = clahe.apply(l_chanell)

    enhanced_lab = cv2.merge([enhanced_l, a_chanell, b_chanell])

    result = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

    return result

"""
сохранение изображения
"""
def save_img(image, path):
    saved = cv2.imwrite(path, image)
    if not saved:
        raise IOError(f'Не удалось сохранить изображение: {path}')

"""
масштабирование (уменьшение, чтобы большая сторона была не больше заданного значения)
"""

def resize_image(image, max_size=4000):
    height, width = image.shape[:2]

    scale = min(
        max_size / width,
        max_size / height
    )

    # Если изображение уже достаточно маленькое, ничего не меняем
    if scale >= 1:
        return image

    new_width = int(width * scale)
    new_height = int(height * scale)

    resized = cv2.resize(
        image,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA
    )

    return resized