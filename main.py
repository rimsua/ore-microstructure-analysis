from pathlib import Path

from filters import (
    load_image,
    show_image,
    lighting_normalization,
    denoise_gauss,
    denoise_median,
    denoise_bilateral,
    contrast_correction,
    save_img,
    resize_image,
)


def main():
    # Папка, в которой находятся main.py и filters.py
    project_dir = Path(__file__).resolve().parent

    # Пути к исходному изображению и результатам
    input_path = project_dir / "output" / "DSCN3872.JPG"
    output_dir = project_dir / "output" / "processed"

    # папка для результатов
    output_dir.mkdir(parents=True, exist_ok=True)

    # загрузка изображения
    image = load_image(str(input_path))

    height, width = image.shape[:2]

    print("Исходное изображение")
    print(f"Размер: {width} × {height} пикселей")

    show_image(image, "Исходное изображение")

    # эксперименты
    experiments = [
        (
            "lighting_normalization",
            lambda: lighting_normalization(image, kernel_size=101),
        ),
        (
            "denoise_gauss",
            lambda: denoise_gauss(image, kernel_size=5),
        ),
        (
            "denoise_median",
            lambda: denoise_median(image, kernel_size=5),
        ),
        (
            "denoise_bilateral",
            lambda: denoise_bilateral(image),
        ),
        (
            "contrast_correction",
            lambda: contrast_correction(image),
        ),
        (
            "resize_image",
            lambda: resize_image(image, max_size=4000),
        ),
    ]

    for name, operation in experiments:
        print(f"\nЭксперимент: {name}")

        result = operation()

        result_height, result_width = result.shape[:2]
        print(
            f"Размер результата: "
            f"{result_width} × {result_height} пикселей"
        )

        # Сохранение
        output_path = output_dir / f"{name}.png"
        save_img(result, str(output_path))

        print(f"Результат сохранён: {output_path}")

        show_image(result, name)

        del result

    print("\nВсе эксперименты завершены")


main()