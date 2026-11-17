import random
import shutil
import string
from pathlib import Path

import pytesseract
from captcha.image import ImageCaptcha
from PIL import Image, ImageFilter, ImageOps


def configure_tesseract() -> None:
    if shutil.which("tesseract") is None:
        default_win_path = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
        if default_win_path.exists():
            pytesseract.pytesseract.tesseract_cmd = str(default_win_path)
        else:
            raise FileNotFoundError("Tesseract executable not found.")


def get_otsu_threshold(gray_img: Image.Image) -> int:
    """Compute optimal binary threshold using Otsu's method in pure Python."""
    hist = gray_img.histogram()
    total = sum(hist)
    sum_total = sum(i * h for i, h in enumerate(hist))

    sum_bg, weight_bg, max_var, threshold = 0, 0, 0.0, 128
    for i, h in enumerate(hist):
        weight_bg += h
        if weight_bg == 0:
            continue
        weight_fg = total - weight_bg
        if weight_fg == 0:
            break

        sum_bg += i * h
        mean_bg = sum_bg / weight_bg
        mean_fg = (sum_total - sum_bg) / weight_fg
        var_between = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2

        if var_between > max_var:
            max_var = var_between
            threshold = i
    return threshold


def generate_captcha(
    output_path: Path, length: int = 5, width: int = 200, height: int = 80
) -> str:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    characters = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
    captcha_text = "".join(random.choices(characters, k=length))

    generator = ImageCaptcha(width=width, height=height)
    generator.write(captcha_text, str(output_path))
    return captcha_text


def preprocess_image(image_path: Path) -> Image.Image:
    with Image.open(image_path) as img:
        gray = img.convert("L")

       
        upscaled = gray.resize(
            (gray.width * 2, gray.height * 2), Image.Resampling.BICUBIC
        )

        
        thresh = get_otsu_threshold(upscaled)
        binary = upscaled.point(lambda p: 255 if p > thresh else 0)

        
        hist = binary.histogram()
        if hist[0] > hist[255]:  
            binary = ImageOps.invert(binary)

        
        cleaned = binary.filter(ImageFilter.MaxFilter(3)).filter(
            ImageFilter.MinFilter(3)
        )

       
        return ImageOps.expand(cleaned, border=20, fill=255)


def recognize_captcha(image: Image.Image) -> str:
    whitelist = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
    config = f"--psm 7 -c tessedit_char_whitelist={whitelist}"
    raw_text = pytesseract.image_to_string(image, config=config)
    return "".join(raw_text.split())


def main() -> None:
    configure_tesseract()

    image_path = Path("captchas/sample.png")
    actual_text = generate_captcha(image_path)
    print(f"Actual CAPTCHA:    {actual_text}")

    processed_image = preprocess_image(image_path)
    predicted_text = recognize_captcha(processed_image)
    print(f"Predicted CAPTCHA: {predicted_text}")

    if predicted_text == actual_text:
        print("Result: CAPTCHA recognized correctly")
    else:
        print("Result: CAPTCHA recognition failed")

    processed_image.show(title="Preprocessed CAPTCHA")


if __name__ == "__main__":
    main()