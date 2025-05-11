#!/usr/bin/env python3
"""
ASCII Art Image Converter
A tool to convert images to ASCII art with multiple rendering options.
"""

import argparse
import os
import shutil
import sys
from pathlib import Path
from typing import Tuple, Union

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    print("Error: Pillow library not found. Install with: pip install pillow")
    sys.exit(1)

# Define ASCII character sets from dense to sparse
ASCII_SETS = {
    "standard": "@%#*+=-:. ",
    "detailed": "$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\\|()1{}[]?-_+~<>i!lI;:,\"^`'. ",
    "simple": "#@%=+*:-. ",
    "blocks": "█▓▒░ ",
}

def detect_terminal_size() -> Tuple[int, int]:
    """Detect terminal width and height."""
    return shutil.get_terminal_size((80, 24))

def resize_image(image: Image.Image, width: int, height: int, maintain_aspect: bool = True) -> Image.Image:
    """Resize an image to the specified dimensions."""
    if maintain_aspect:
        aspect_ratio = image.height / image.width
        corrected_aspect = aspect_ratio * 0.5
        new_height = int(width * corrected_aspect)
        
        if height and new_height > height:
            new_height = height
            width = int(new_height / corrected_aspect)
    
    return image.resize((width, height), Image.LANCZOS).convert("L")

def apply_brightness_contrast(image: Image.Image, brightness: float, contrast: float) -> Image.Image:
    """Apply brightness and contrast adjustments to an image."""
    if brightness == 0 and contrast == 0:
        return image
    
    if brightness != 0:
        factor = 255.0 * (brightness / 100.0)
        image = image.point(lambda x: max(0, min(255, x + factor)))
    
    if contrast != 0:
        factor = (259.0 * (contrast + 255)) / (255.0 * (259 - contrast))
        image = image.point(lambda x: max(0, min(255, 128 + factor * (x - 128))))
    
    return image

def image_to_ascii(
    image: Image.Image, 
    width: int, 
    height: int, 
    ascii_set: str = "standard",
    invert: bool = False,
    color: bool = False,
    brightness: float = 0,
    contrast: float = 0,
    maintain_aspect: bool = True
) -> Tuple[str, list]:
    """Convert an image to ASCII art and return both the ASCII string and color data."""
    chars = ASCII_SETS.get(ascii_set, ASCII_SETS["standard"])
    if invert:
        chars = chars[::-1]
    
    image = resize_image(image, width, height, maintain_aspect)
    image = apply_brightness_contrast(image, brightness, contrast)
    
    pixels = list(image.getdata())
    char_count = len(chars) - 1
    
    # Store ASCII chars and colors separately
    ascii_chars = []
    colors = []
    
    if color:
        color_image = image.convert("RGB")
        color_pixels = list(color_image.getdata())
    else:
        color_pixels = [(255, 255, 255)] * len(pixels)  # White text for non-color output
    
    for i, (pixel, color_pixel) in enumerate(zip(pixels, color_pixels)):
        char = chars[min(int(pixel * char_count / 255), char_count)]
        ascii_chars.append(char)
        colors.append(color_pixel)
        if (i + 1) % width == 0:
            ascii_chars.append('\n')
            colors.append((0, 0, 0))  # Add black color for newline
    
    return ''.join(ascii_chars), colors

def create_image_from_ascii(
    ascii_str: str,
    colors: list,
    font_size: int = 14,
    bg_color: Tuple[int, int, int] = (0, 0, 0)
) -> Image.Image:
    """Create a PIL Image from ASCII art and colors."""
    # Try to find a suitable font
    font = None
    font_paths = [
        "/System/Library/Fonts/Menlo.ttc",  # macOS
        "/System/Library/Fonts/Monaco.ttf",  # macOS
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",  # Linux
        "C:\\Windows\\Fonts\\consola.ttf",  # Windows
        "DejaVuSansMono.ttf",
    ]
    
    for font_path in font_paths:
        try:
            font = ImageFont.truetype(font_path, font_size)
            break
        except (OSError, IOError):
            continue
    
    if font is None:
        font = ImageFont.load_default()
        font_size = 10
    
    # Calculate image dimensions
    lines = ascii_str.split('\n')
    char_width = font_size * 0.6  # Approximate width of a monospace character
    char_height = font_size * 1.2  # Add some line spacing
    
    img_width = int(max(len(line) for line in lines) * char_width)
    img_height = int(len(lines) * char_height)
    
    # Create image and draw context
    image = Image.new('RGB', (img_width, img_height), bg_color)
    draw = ImageDraw.Draw(image)
    
    # Draw ASCII characters with colors
    x = y = 0
    color_idx = 0
    
    for char in ascii_str:
        if char == '\n':
            y += char_height
            x = 0
        else:
            color = colors[color_idx]
            draw.text((x, y), char, fill=color, font=font)
            x += char_width
        color_idx += 1
    
    return image

def save_to_file(content: Union[str, Image.Image], output_file: Path, is_image: bool = False) -> None:
    """Save ASCII art to a file."""
    if is_image:
        content.save(output_file)
    else:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(content)
    print(f"Output saved to {output_file}")

def main():
    """Main function for the ASCII art converter."""
    term_width, term_height = detect_terminal_size()
    
    parser = argparse.ArgumentParser(
        description="Convert images to ASCII art",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("-f", "--file", required=True, help="Input image file")
    parser.add_argument("-o", "--output", help="Output file")
    parser.add_argument("--width", type=int, default=term_width, help="ASCII output width")
    parser.add_argument("--height", type=int, default=term_height-2, help="ASCII output height")
    parser.add_argument("--ascii-set", choices=list(ASCII_SETS.keys()), default="standard",
                      help="ASCII character set to use")
    parser.add_argument("--invert", action="store_true", help="Invert brightness")
    parser.add_argument("--color", action="store_true", help="Enable color output")
    parser.add_argument("--brightness", type=float, default=0, help="Adjust brightness (-100 to 100)")
    parser.add_argument("--contrast", type=float, default=0, help="Adjust contrast (-100 to 100)")
    parser.add_argument("--maintain-aspect", action="store_true", default=True,
                      help="Maintain aspect ratio of the original image")
    parser.add_argument("--font-size", type=int, default=14, help="Font size for image output")
    
    args = parser.parse_args()
    filepath = Path(args.file)
    
    if not filepath.exists():
        print(f"Error: File not found: {filepath}")
        return 1
    
    if filepath.suffix.lower() not in [".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"]:
        print(f"Error: Unsupported file type: {filepath.suffix}")
        print("Supported formats: JPG, JPEG, PNG, BMP, GIF, WEBP")
        return 1
    
    try:
        image = Image.open(filepath)
        ascii_art, colors = image_to_ascii(
            image, args.width, args.height, args.ascii_set, args.invert,
            args.color, args.brightness, args.contrast, args.maintain_aspect
        )
        
        if args.output:
            # If output is an image format, create and save as image
            if args.output.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                output_image = create_image_from_ascii(ascii_art, colors, args.font_size)
                save_to_file(output_image, args.output, is_image=True)
            else:
                # Save as text file
                save_to_file(ascii_art, args.output, is_image=False)
        else:
            # Print to terminal with ANSI colors if enabled
            if args.color:
                for char, color in zip(ascii_art, colors):
                    if char == '\n':
                        print(char, end='')
                    else:
                        r, g, b = color
                        print(f"\033[38;2;{r};{g};{b}m{char}\033[0m", end='')
            else:
                print(ascii_art)
            
    except Exception as e:
        print(f"Error processing image: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())