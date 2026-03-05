from PIL import Image
import sys

def resize_image(input_path, output_path, max_width=1024):
    try:
        img = Image.open(input_path)
        w, h = img.size
        if w > max_width:
            ratio = max_width / float(w)
            new_h = int(h * ratio)
            img = img.resize((max_width, new_h), Image.LANCZOS)
            print(f"Resized {input_path} from {w}x{h} to {max_width}x{new_h}")
        else:
            print(f"Image {input_path} already small enough ({w}x{h})")
        
        img.save(output_path)
        print(f"Saved to {output_path}")

    except Exception as e:
        print(f"Error resizing: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 2:
        resize_image(sys.argv[1], sys.argv[2])
    else:
        resize_image("img5.jpg", "img5_resized.jpg")
