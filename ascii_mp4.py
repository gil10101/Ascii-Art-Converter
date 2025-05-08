#!/usr/bin/env python3
"""
ASCII Art Video/Image Converter
A tool to convert images and videos to ASCII art with multiple rendering options.
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from pathlib import Path
from typing import List, Optional, Tuple, Union
import re

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
    "numbers": "9876543210 ",
    "braille": "⠿⠯⠟⠻⠽⠾⠷⠮⠭⠝⠞⠵⠹⠪⠫⠺⠼⠧⠏⠉⠙⠋⠓⠕⠗⠛⠒⠜⠄⠨⠔⠢⠖⠦⠇⠸⠰⠡⠃⠊⠂⠑⠱⠆⠀",
}

# Define Chafa symbol sets with descriptions (for reference)
CHAFA_SYMBOL_SETS = [
    "all", "ascii", "ascii-simple", "block", "border", "box", "braille", 
    "diagonal", "half", "latin", "legacy", "math", "none", "quad", 
    "sextant", "space", "stipple", "wide"
]

# Color support
class ColorSupport(Enum):
    NONE = 0
    ANSI_16 = 1
    ANSI_256 = 2
    TRUECOLOR = 3

# Terminal detection and capabilities
def detect_terminal_capabilities() -> Tuple[int, int, ColorSupport]:
    """Detect terminal width, height, and color support level."""
    # Get terminal size
    width, height = shutil.get_terminal_size((80, 24))
    
    # Detect color support
    color_support = ColorSupport.NONE
    if os.environ.get("COLORTERM") in ("truecolor", "24bit"):
        color_support = ColorSupport.TRUECOLOR
    elif os.environ.get("TERM") and "256" in os.environ.get("TERM", ""):
        color_support = ColorSupport.ANSI_256
    elif os.environ.get("TERM") and os.environ.get("TERM") not in ("dumb", ""):
        color_support = ColorSupport.ANSI_16
        
    return width, height, color_support

def is_tool_available(name: str) -> bool:
    """Check if an external tool is available."""
    return shutil.which(name) is not None

def resize_image(
    image: Image.Image, width: int, height: int, maintain_aspect: bool = True
) -> Image.Image:
    """Resize an image to the specified dimensions."""
    if maintain_aspect:
        # Calculate the exact aspect ratio of the input image
        aspect_ratio = image.height / image.width
        
        # Calculate potential dimensions based on width and height constraints
        new_width_from_height = int(height / aspect_ratio)
        new_height_from_width = int(width * aspect_ratio)
        
        # Choose dimensions that don't exceed either target
        if new_height_from_width <= height:
            # Width is the limiting factor
            new_width = width
            new_height = new_height_from_width
        else:
            # Height is the limiting factor
            new_width = new_width_from_height
            new_height = height
        
        # Update width and height with our calculated values
        width = new_width
        height = new_height
    
    return image.resize((width, height), Image.LANCZOS).convert("L")

def apply_brightness_contrast(image: Image.Image, brightness: float, contrast: float) -> Image.Image:
    """Apply brightness and contrast adjustments to an image."""
    if brightness == 0 and contrast == 0:
        return image
    
    # Apply brightness
    if brightness != 0:
        factor = 255.0 * (brightness / 100.0)
        offset_matrix = [factor] * 256
        image = image.point(lambda x: max(0, min(255, x + offset_matrix[x])))
    
    # Apply contrast
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
) -> str:
    """Convert an image to ASCII art."""
    # Get character set
    chars = ASCII_SETS.get(ascii_set, ASCII_SETS["standard"])
    
    # Process image
    image = resize_image(image, width, height, maintain_aspect)
    image = apply_brightness_contrast(image, brightness, contrast)
    
    # Invert if requested
    if invert:
        chars = chars[::-1]
    
    # Get grayscale pixels
    pixels = list(image.getdata())
    
    if not color:
        # Simple ASCII conversion
        char_count = len(chars) - 1
        ascii_str = ""
        for i, pixel in enumerate(pixels):
            ascii_str += chars[min(int(pixel * char_count / 255), char_count)]
            if (i + 1) % width == 0:
                ascii_str += '\n'
        
        return ascii_str
    else:
        # Get original color image
        color_image = image.convert("RGB")
        color_pixels = list(color_image.getdata())
        
        # Color ASCII conversion
        char_count = len(chars) - 1
        ascii_str = ""
        
        for i, (pixel, color_pixel) in enumerate(zip(pixels, color_pixels)):
            r, g, b = color_pixel
            char = chars[min(int(pixel * char_count / 255), char_count)]
            
            # Use ANSI color escape codes
            ascii_str += f"\033[38;2;{r};{g};{b}m{char}\033[0m"
            
            if (i + 1) % width == 0:
                ascii_str += '\n'
        
        return ascii_str

def convert_with_chafa(
    image_path: Union[str, Path], 
    width: int, 
    height: int, 
    chafa_args: str = "",
    color: bool = True
) -> str:
    """Convert image to ASCII art using chafa."""
    cmd = ["chafa"]
    
    # Set default color arguments if not explicitly provided
    if not color and "--colors=" not in chafa_args and "--fg-only" not in chafa_args:
        cmd.extend(["--colors=none", "--fg-only"])
    
    # Check if braille symbols are requested
    using_braille = "braille" in chafa_args
    
    # Add user args if provided (these will override our defaults)
    if chafa_args:
        # Split by spaces, but preserve quoted arguments
        import shlex
        cmd.extend(shlex.split(chafa_args))
    
    # Add size if not specified in chafa_args
    if not any(arg.startswith("--size=") for arg in cmd):
        cmd.append(f"--size={width}x{height}")
    
    # Ensure braille symbols are correctly used if requested
    if using_braille and not any(arg.startswith("--symbols=") or arg == "--symbols" for arg in cmd):
        cmd.append("--symbols=braille")
    
    # Add image path
    cmd.append(str(image_path))
    
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, text=True, check=False)
        if using_braille:
            print("Using braille symbols for Chafa output")
        return result.stdout
    except Exception as e:
        print(f"Error running chafa: {e}")
        return f"[Error converting image with chafa: {e}]"

def extract_frames(
    video_path: Union[str, Path], 
    fps: int, 
    tmp_dir: Union[str, Path],
    width: Optional[int] = None,
    height: Optional[int] = None
) -> List[Path]:
    """Extract frames from a video."""
    scale_filter = ""
    if width and height:
        scale_filter = f",scale={width}:{height}:flags=lanczos"
    
    cmd = [
        "ffmpeg", "-i", str(video_path),
        "-vf", f"fps={fps}{scale_filter}",
        "-q:v", "2",  # High quality
        f"{tmp_dir}/%05d.png",
        "-loglevel", "quiet"
    ]
    
    subprocess.run(cmd, check=False)
    return sorted(Path(tmp_dir).glob("*.png"))

def process_frame(
    frame_path: Path, 
    use_chafa: bool, 
    width: int, 
    height: int, 
    chafa_args: str,
    ascii_set: str,
    invert: bool,
    color: bool,
    brightness: float,
    contrast: float,
    maintain_aspect: bool
) -> str:
    """Process a single frame to ASCII."""
    if use_chafa:
        return convert_with_chafa(str(frame_path), width, height, chafa_args, color)
    else:
        image = Image.open(frame_path)
        return image_to_ascii(
            image, width, height, ascii_set, invert, color, brightness, contrast, maintain_aspect
        )

def play_ascii_frames(frames: List[str], delay: float, loop: bool = False):
    """Display ASCII frames with the specified delay."""
    try:
        iterations = 0
        while True:
            iterations += 1
            for frame in frames:
                # Clear screen and move cursor to top-left
                print("\033[H\033[J", end="")
                # Print frame
                print(frame, end="", flush=True)
                # Wait
                time.sleep(delay)
                
            if not loop:
                break
                
            # Add delay between loops
            if iterations > 0:
                time.sleep(0.5)
                
    except KeyboardInterrupt:
        print("\n\033[0m[Playback interrupted]")

class ProgressBar:
    """Simple progress bar for showing conversion progress."""
    def __init__(self, total: int, prefix: str = "Progress", length: int = 50):
        self.total = total
        self.prefix = prefix
        self.length = length
        self.current = 0
        self._lock = threading.Lock()
        
    def update(self, increment: int = 1):
        """Increment progress bar."""
        with self._lock:
            self.current += increment
            percent = self.current / self.total * 100
            filled_length = int(self.length * self.current // self.total)
            bar = '█' * filled_length + '-' * (self.length - filled_length)
            sys.stdout.write(f'\r{self.prefix}: |{bar}| {percent:.1f}% ({self.current}/{self.total})')
            sys.stdout.flush()
            
            if self.current >= self.total:
                sys.stdout.write('\n')
                
    def finish(self):
        """Complete progress bar."""
        with self._lock:
            sys.stdout.write('\n')
            sys.stdout.flush()

def save_to_file(content: str, output_file: Union[str, Path]):
    """Save ASCII art to a file."""
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"Output saved to {output_file}")

def list_chafa_symbol_sets():
    """Print available Chafa symbol sets."""
    print("\nAvailable Chafa Symbol Sets:")
    for name in CHAFA_SYMBOL_SETS:
        print(f"  {name}")
    print("\nSets can be combined with '+', e.g., '--symbols block+braille'")
    print("Use with -c option, e.g., -c \"--symbols braille --fg-only\"")

def ansi_to_plain_text(ansi_text: str) -> str:
    """Strip ANSI color codes from text."""
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    return ansi_escape.sub('', ansi_text)

def ascii_frame_to_image(
    ascii_text: str, 
    font_size: int = 14, 
    bg_color: Tuple[int, int, int] = (0, 0, 0),
    text_color: Tuple[int, int, int] = (255, 255, 255)
) -> Image.Image:
    """
    Convert ASCII text frame to an image.
    This handles both plain ASCII and ANSI color-coded ASCII, including Unicode braille characters.
    """
    # Strip ANSI escape codes to get plain text for measuring
    plain_text = ansi_to_plain_text(ascii_text)
    lines = plain_text.split('\n')
    
    # Find a font that supports braille characters
    font = None
    fonts_to_try = [
        "fonts/unifont_jp-16.0.03.otf",
        "fonts/BrailleCc0-DOeDd.ttf",  # Prioritize braille font
        "DejaVuSansMono.ttf",
        "FreeMono.ttf", 
        "NotoSansMono-Regular.ttf",
        "UbuntuMono-Regular.ttf",
        "LiberationMono-Regular.ttf",
        # Add system fonts paths
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/TTF/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/freefont/FreeMono.ttf",
        "/System/Library/Fonts/Monaco.ttf",  # macOS
        "C:\\Windows\\Fonts\\consola.ttf",   # Windows
    ]
    
    for font_name in fonts_to_try:
        try:
            font = ImageFont.truetype(font_name, font_size)
            # Test if this font supports braille
            if "⠿" in plain_text:
                # If we found braille in the text, make sure our font can render it
                test_img = Image.new('RGB', (30, 30), (0, 0, 0))
                test_draw = ImageDraw.Draw(test_img)
                test_draw.text((5, 5), "⠿", fill=(255, 255, 255), font=font)
                # Additional check could be added here if needed
            break
        except (IOError, OSError):
            continue
    
    if font is None:
        # Fallback to default if no suitable font found
        font = ImageFont.load_default()
        font_size = 10  # Default font is usually smaller
        print("Warning: Could not load a font with braille support. Output quality may be reduced.")
    
    # Get text dimensions
    # Find the max line length but ignore trailing spaces that might be causing the black bar
    max_line_length = max(len(ansi_to_plain_text(line.rstrip())) for line in ascii_text.split('\n'))
    
    # Calculate character dimensions to match desired output size
    # For braille characters, we need a more square aspect ratio
    char_width = int(font_size * 0.6)  # Increased from 0.4 to 0.6 for more square cells
    char_height = int(font_size * 0.7)  # Reduced from 0.8 to 0.7 for more square cells
    
    # Calculate image dimensions based on actual text content
    img_width = max_line_length * char_width
    img_height = len(lines) * char_height
    
    # Ensure dimensions are even numbers for video encoding
    img_width = (img_width + 1) & ~1  # Round up to next even number
    img_height = (img_height + 1) & ~1  # Round up to next even number
    
    # Create image with background color
    image = Image.new('RGB', (img_width, img_height), bg_color)
    draw = ImageDraw.Draw(image)
    
    # Parse ANSI colored text and draw with appropriate colors
    y_pos = 0
    for line in ascii_text.split('\n'):
        # Strip trailing spaces from the line before rendering
        line = line.rstrip()
        x_pos = 0
        
        # Simple ANSI color parsing - this handles basic color codes
        segments = re.split(r'(\x1B\[[0-9;]*m)', line)
        current_color = text_color
        
        for segment in segments:
            if segment.startswith('\x1B['):
                # Parse color code
                if '38;2;' in segment:  # RGB true color
                    try:
                        # Extract RGB values from code like \x1B[38;2;R;G;Bm
                        match = re.search(r'38;2;(\d+);(\d+);(\d+)', segment)
                        if match:
                            r, g, b = map(int, match.groups())
                            current_color = (r, g, b)
                    except Exception:
                        current_color = text_color
                elif segment.endswith('0m'):  # Reset
                    current_color = text_color
            else:
                # Draw text segment with current color
                if segment:
                    # Use textlength for proper character width measurement
                    # This is especially important for special Unicode characters
                    draw.text((x_pos, y_pos), segment, fill=current_color, font=font)
                    # Get the actual rendered width
                    try:
                        # For newer Pillow versions
                        text_width = draw.textlength(segment, font=font)
                    except AttributeError:
                        # Fallback for older Pillow versions
                        text_width = font.getsize(segment)[0]
                    x_pos += text_width
        
        y_pos += char_height
    
    return image

def save_video_from_ascii_frames(
    ascii_frames: List[str],
    output_path: str,
    framerate: int = 15,
    font_size: int = 14,
    bg_color: Tuple[int, int, int] = (0, 0, 0),
    video_codec: str = "libx264",
    crf: int = 18  # Lower is better quality
) -> bool:
    """
    Convert ASCII frames to a video file.
    
    Args:
        ascii_frames: List of ASCII art frames
        output_path: Path to save the output video
        framerate: Frames per second
        font_size: Font size for text rendering
        bg_color: Background color (RGB tuple)
        video_codec: FFmpeg video codec
        crf: Constant Rate Factor for quality (lower is better)
        
    Returns:
        True on success, False on failure
    """
    if not ascii_frames:
        print("Error: No ASCII frames to save")
        return False
        
    try:
        # Check if we have braille characters in our ASCII frames
        has_braille = any("⠀" in frame or "⠁" in frame or "⠿" in frame for frame in ascii_frames)
        if has_braille and font_size < 20:
            print("Warning: Font size may be too small for braille characters. Consider using --font-size 20 or higher for better readability.")
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_dir_path = Path(tmp_dir)
            
            print(f"Converting ASCII frames to images...")
            progress = ProgressBar(len(ascii_frames), "Creating video frames")
            
            # Convert ASCII frames to images
            frame_paths = []
            for i, frame in enumerate(ascii_frames):
                img = ascii_frame_to_image(frame, font_size, bg_color)
                frame_path = temp_dir_path / f"frame_{i:05d}.png"
                img.save(frame_path)
                frame_paths.append(frame_path)
                progress.update()
            
            # Generate video from images using FFmpeg
            ffmpeg_cmd = [
                "ffmpeg",
                "-y",  # Overwrite output file if it exists
                "-framerate", str(framerate),
                "-i", f"{tmp_dir}/frame_%05d.png",
                "-c:v", video_codec,
                "-pix_fmt", "yuv420p",  # For compatibility
                "-crf", str(crf),
                "-preset", "medium",
                output_path
            ]
            
            print(f"Encoding video with FFmpeg...")
            result = subprocess.run(ffmpeg_cmd, check=False, capture_output=True)
            
            if result.returncode != 0:
                print(f"Error encoding video: {result.stderr.decode()}")
                return False
                
            print(f"Video saved to: {output_path}")
            return True
            
    except Exception as e:
        print(f"Error saving video: {str(e)}")
        return False

def calculate_font_size_for_dimensions(width, height, ascii_width, ascii_height):
    """Calculate appropriate font size to achieve desired output dimensions."""
    # Estimate how many characters we need for the given width and height
    # Each character takes about 0.6*font_size pixels wide and 0.7*font_size pixels high
    # So: width = ascii_width * 0.6 * font_size, height = ascii_height * 0.7 * font_size
    
    font_size_for_width = width / (ascii_width * 0.6)
    font_size_for_height = height / (ascii_height * 0.7)
    
    # Use the smaller of the two to ensure it fits
    font_size = min(font_size_for_width, font_size_for_height)
    
    # Round to nearest integer
    return max(4, round(font_size))  # Ensure minimum font size of 4

def main():
    """Main function for the ASCII art converter."""
    
    # Detect terminal capabilities
    term_width, term_height, color_support = detect_terminal_capabilities()
    
    # Create argument parser
    parser = argparse.ArgumentParser(
        description="Convert videos and images to ASCII art",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Input/output arguments
    parser.add_argument("-f", "--file", required=True, help="Input video or image file")
    parser.add_argument("-o", "--output", help="Output file (for single images)")
    parser.add_argument("--info", action="store_true", help="Display file info before processing")
    
    # Dimension arguments
    parser.add_argument("--width", type=int, default=term_width, 
                      help="ASCII output width (for videos, defaults to original width if not specified)")
    parser.add_argument("--height", type=int, default=term_height-2, 
                      help="ASCII output height (for videos, defaults to original height if not specified)")
    parser.add_argument("--scale-factor", type=float, default=4.0,
                      help="Scale down video dimensions by this factor (e.g., 4.0 = 1/4 of original size)")
    parser.add_argument("--maintain-aspect", action="store_true", default=True,
                      help="Maintain aspect ratio of the original image")
    
    # Video-specific arguments
    parser.add_argument("--framerate", type=int, default=15, 
                      help="Video frame rate for conversion")
    parser.add_argument("--loop", action="store_true", help="Loop video playback")
    
    # Video output options
    parser.add_argument("--save-video", help="Save ASCII video to file (requires FFmpeg)")
    parser.add_argument("--font-size", type=int, default=14,
                      help="Font size for rendered video output")
    parser.add_argument("--full-resolution", action="store_true", 
                      help="Force full resolution output matching the original video")
    parser.add_argument("--bg-color", default="0,0,0",
                      help="Background color for video (R,G,B format)")
    parser.add_argument("--video-codec", default="libx264",
                      help="Video codec for output (FFmpeg format)")
    parser.add_argument("--video-quality", type=int, default=18,
                      help="Video quality (CRF, 0-51, lower is better)")
    
    # Appearance arguments
    parser.add_argument("--ascii-set", choices=list(ASCII_SETS.keys()), default="standard",
                      help="ASCII character set to use (for native processing)")
    parser.add_argument("--list-symbols", action="store_true",
                      help="List available Chafa symbol sets")
    parser.add_argument("--invert", action="store_true", 
                      help="Invert brightness in ASCII output (native mode only)")
    parser.add_argument("--brightness", type=float, default=0,
                      help="Adjust brightness (-100 to 100) (native mode only)")
    parser.add_argument("--contrast", type=float, default=0,
                      help="Adjust contrast (-100 to 100) (native mode only)")
    parser.add_argument("--braille-set", action="store_true",
                      help="Add braille characters to ASCII character set (native mode only)")
    
    # Color options
    parser.add_argument("--color", action="store_true", 
                      help="Enable color output (if supported)")
    
    # External tools
    parser.add_argument("--force-native", action="store_true",
                      help="Force native processing instead of external tools")
    parser.add_argument("-c", "--chafa-args", default="", 
                      help="Extra arguments for chafa (if installed)")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Check if just listing symbols
    if args.list_symbols:
        list_chafa_symbol_sets()
        return 0
    
    # Validate input file
    filepath = Path(args.file)
    if not filepath.exists():
        print(f"Error: File not found: {filepath}")
        return 1
        
    # Display file info if requested
    if args.info:
        file_size = filepath.stat().st_size / (1024 * 1024)  # Size in MB
        print(f"\nFile Information:")
        print(f"  Path: {filepath.absolute()}")
        print(f"  Size: {file_size:.2f} MB")
        print(f"  Type: {filepath.suffix.lower()[1:].upper()}")
        
        # Get dimensions for images
        if filepath.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"]:
            try:
                with Image.open(filepath) as img:
                    print(f"  Dimensions: {img.width} x {img.height}")
                    print(f"  Mode: {img.mode}")
            except Exception as e:
                print(f"  Image info error: {e}")
        
        # Get video info if ffmpeg is available
        elif filepath.suffix.lower() in [".mp4", ".mov", ".avi", ".mkv", ".webm"] and is_tool_available("ffprobe"):
            try:
                cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0", 
                       "-show_entries", "stream=width,height,duration,r_frame_rate", 
                       "-of", "default=noprint_wrappers=1", str(filepath)]
                result = subprocess.run(cmd, capture_output=True, text=True)
                print(f"  Video info: \n    {result.stdout.replace('=', ': ').replace('\\n', '\\n    ')}")
            except Exception as e:
                print(f"  Video info error: {e}")
        
        print("")
    
    # Parse background color
    try:
        bg_color = tuple(map(int, args.bg_color.split(',')))
        if len(bg_color) != 3 or not all(0 <= c <= 255 for c in bg_color):
            print("Warning: Invalid background color format. Using black.")
            bg_color = (0, 0, 0)
    except ValueError:
        print("Warning: Invalid background color format. Using black.")
        bg_color = (0, 0, 0)
    
    # Check if color is supported
    if args.color and color_support == ColorSupport.NONE:
        print("Warning: Terminal does not support color. Disabling color output.")
        args.color = False
        
    # Add braille characters to ASCII set if requested
    if args.braille_set and args.ascii_set == "standard":
        print("Using braille character set for native rendering")
        args.ascii_set = "braille"
    
    # Check for external tools
    use_chafa = not args.force_native and is_tool_available("chafa")
    has_ffmpeg = is_tool_available("ffmpeg")
    
    # If using chafa with braille symbols, ensure the correct arguments are passed
    if use_chafa and "--symbols" not in args.chafa_args and "braille" in args.chafa_args:
        args.chafa_args = f"--symbols braille {args.chafa_args}"
    
    # Check if video saving is requested but FFmpeg is not available
    if args.save_video and not has_ffmpeg:
        print("Error: FFmpeg is required for saving video but not found.")
        print("Please install FFmpeg: https://ffmpeg.org/download.html")
        return 1
    
    # Process based on file type
    file_extension = filepath.suffix.lower()
    
    # Image processing
    if file_extension in [".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"]:
        if use_chafa:
            ascii_art = convert_with_chafa(
                str(filepath), args.width, args.height, args.chafa_args, args.color
            )
        else:
            image = Image.open(filepath)
            ascii_art = image_to_ascii(
                image, args.width, args.height, args.ascii_set, args.invert, 
                args.color, args.brightness, args.contrast, args.maintain_aspect
            )
        
        # Output
        if args.output:
            save_to_file(ascii_art, args.output)
        else:
            print(ascii_art)
            
        # Save as video frame if requested
        if args.save_video:
            print("Converting single image to video frame...")
            save_video_from_ascii_frames(
                [ascii_art], args.save_video, args.framerate,
                args.font_size, bg_color, args.video_codec, args.video_quality
            )
            
    # Video processing
    elif file_extension in [".mp4", ".mov", ".avi", ".mkv", ".webm"]:
        if not has_ffmpeg:
            print("Error: FFmpeg is required for video processing but not found.")
            print("Please install FFmpeg: https://ffmpeg.org/download.html")
            return 1
        
        # Get original video dimensions
        try:
            cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0", 
                  "-show_entries", "stream=width,height", 
                  "-of", "default=noprint_wrappers=1", str(filepath)]
            result = subprocess.run(cmd, capture_output=True, text=True, check=False)
            output = result.stdout.strip()
            
            # Extract original dimensions
            width_match = re.search(r'width=(\d+)', output) 
            height_match = re.search(r'height=(\d+)', output)
            
            if width_match and height_match:
                original_width = int(width_match.group(1))
                original_height = int(height_match.group(1))
                
                # If width or height is not manually specified, use the original dimensions
                if args.width == term_width:
                    args.width = int(original_width / args.scale_factor)
                if args.height == term_height-2:
                    args.height = int(original_height / args.scale_factor)
                
                print(f"Original dimensions: {original_width}x{original_height}")
                if args.scale_factor != 1.0:
                    print(f"Scaling down by factor of {args.scale_factor:.1f} to {args.width}x{args.height}")
                
                # Warn if dimensions are very large for terminal display
                if args.width > term_width * 3 or args.height > term_height * 3:
                    print(f"Warning: Using large dimensions ({args.width}x{args.height}) which may not fit in terminal.")
                    print(f"For terminal display only, consider specifying smaller dimensions with --width and --height.")
                
                # If saving video, calculate appropriate font size for 1:1 pixel mapping
                if args.save_video and (args.font_size == 14 or args.full_resolution):  # 14 is the default
                    args.font_size = calculate_font_size_for_dimensions(
                        original_width, original_height, args.width, args.height
                    )
                    print(f"Auto-calculated font size: {args.font_size} for 1:1 pixel mapping")
                
                print(f"Using dimensions: {args.width}x{args.height}")
        except Exception as e:
            print(f"Warning: Could not get original video dimensions: {e}")
        
        print(f"Converting video to ASCII art at {args.framerate} FPS...")
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Extract frames
            frame_paths = extract_frames(
                str(filepath), args.framerate, tmp_dir, 
                args.width if not use_chafa else None,
                args.height if not use_chafa else None
            )
            
            if not frame_paths:
                print("Error: No frames extracted from video")
                return 1
                
            total_frames = len(frame_paths)
            progress = ProgressBar(total_frames, "Converting frames")
            ascii_frames = []
            
            # Process frames with parallel execution
            with ThreadPoolExecutor(max_workers=min(os.cpu_count() or 4, 8)) as executor:
                futures = []
                
                for frame_path in frame_paths:
                    future = executor.submit(
                        process_frame, 
                        frame_path, use_chafa, args.width, args.height, 
                        args.chafa_args, args.ascii_set, args.invert, args.color,
                        args.brightness, args.contrast, args.maintain_aspect
                    )
                    futures.append(future)
                
                # Collect results in order
                for future in futures:
                    ascii_frames.append(future.result())
                    progress.update()
            
            # Save as video if requested
            if args.save_video:
                print(f"Saving ASCII video to {args.save_video}...")
                save_video_from_ascii_frames(
                    ascii_frames, args.save_video, args.framerate,
                    args.font_size, bg_color, args.video_codec, args.video_quality
                )
            
            # Play frames in terminal
            if not args.save_video or (args.save_video and args.loop):
                play_ascii_frames(ascii_frames, delay=1.0/args.framerate, loop=args.loop)
            
    else:
        print(f"Error: Unsupported file type: {file_extension}")
        print("Supported formats: JPG, PNG, BMP, GIF, WEBP, MP4, MOV, AVI, MKV, WEBM")
        return 1
        
    return 0

if __name__ == "__main__":
    sys.exit(main())