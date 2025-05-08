from PIL import Image, ImageDraw, ImageFont
import sys

def test_braille():
    try:
        # Try to find a suitable font
        fonts = ["fonts/BrailleCc0-DOeDd.ttf", "DejaVuSansMono.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 
                 "NotoSansMono-Regular.ttf", "/usr/share/fonts/truetype/noto/NotoSansMono-Regular.ttf"]
        
        font = None
        for f in fonts:
            try:
                font = ImageFont.truetype(f, 24)
                print(f"Using font: {f}")
                break
            except:
                continue
                
        if not font:
            font = ImageFont.load_default()
            print("Using default font")
            
        # Create test image
        img = Image.new('RGB', (500, 100), (0, 0, 0))
        draw = ImageDraw.Draw(img)
        draw.text((10, 10), "Braille test: ⠿⠯⠟⠻⠽⠾⠷", fill=(255, 255, 255), font=font)
        
        # Save for checking
        img.save("braille_test.png")
        print("Saved test image to braille_test.png")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_braille()