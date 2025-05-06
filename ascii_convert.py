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

try:
    from PIL import Image
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
        # Calculate height based on aspect ratio
        aspect_ratio = image.height / image.width
        # Each character in terminal takes about twice as much vertical as horizontal space
        corrected_aspect = aspect_ratio * 0.5
        new_height = int(width * corrected_aspect)
        
        # Make sure height doesn't exceed target
        if height and new_height > height:
            new_height = height
            width = int(new_height / corrected_aspect)
    
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
    
    # Add user args if provided (these will override our defaults)
    if chafa_args:
        cmd.extend(chafa_args.strip().split())
    
    # Add size if not specified in chafa_args
    if not any(arg.startswith("--size=") for arg in cmd):
        cmd.append(f"--size={width}x{height}")
    
    # Add image path
    cmd.append(str(image_path))
    
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, text=True, check=False)
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
        scale_filter = f",scale={width}:{height}:force_original_aspect_ratio=decrease"
    
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

def main():
    """Main function for the ASCII art converter."""
    # Show banner
    print("""
    ╔════════════════════════════════════════════╗
    ║          ASCII ART VIDEO CONVERTER         ║
    ╚════════════════════════════════════════════╝
    """)
    
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
                      help="ASCII output width")
    parser.add_argument("--height", type=int, default=term_height-2, 
                      help="ASCII output height")
    parser.add_argument("--maintain-aspect", action="store_true", default=True,
                      help="Maintain aspect ratio of the original image")
    
    # Video-specific arguments
    parser.add_argument("--framerate", type=int, default=15, 
                      help="Video frame rate for conversion")
    parser.add_argument("--loop", action="store_true", help="Loop video playback")
    
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
    
    # Check if color is supported
    if args.color and color_support == ColorSupport.NONE:
        print("Warning: Terminal does not support color. Disabling color output.")
        args.color = False
    
    # Check for external tools
    use_chafa = not args.force_native and is_tool_available("chafa")
    has_ffmpeg = is_tool_available("ffmpeg")
    
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
            
    # Video processing
    elif file_extension in [".mp4", ".mov", ".avi", ".mkv", ".webm"]:
        if not has_ffmpeg:
            print("Error: FFmpeg is required for video processing but not found.")
            print("Please install FFmpeg: https://ffmpeg.org/download.html")
            return 1
        
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
            
            # Play frames
            play_ascii_frames(ascii_frames, delay=1.0/args.framerate, loop=args.loop)
            
    else:
        print(f"Error: Unsupported file type: {file_extension}")
        print("Supported formats: JPG, PNG, BMP, GIF, WEBP, MP4, MOV, AVI, MKV, WEBM")
        return 1
        
    return 0

if __name__ == "__main__":
    sys.exit(main())