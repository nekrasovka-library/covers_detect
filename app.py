from typing import Optional
import os
import shutil
from scan.scan import find_book
import numpy as np
from PIL import Image

DETAILED = False
input_dir = './workdir/data/input'
output_dir = './workdir/data/processed'
error_dir = './workdir/data/errors'
history_dir = './workdir/data/history'
MODEL_TYPE_1 = 'unet'
MODEL_TYPE_2 = 'sam'

def find_book_final(img: Image.Image, detailed) -> Optional[Image.Image]:
    res = find_book(img, model_type = MODEL_TYPE_1, detailed = detailed)
    
    if res[0] is None:
        res = find_book(img, model_type = MODEL_TYPE_2, detailed = detailed)
    return res[0]

def save_image_from_array(array: np.ndarray, path: str):
    # Convert numpy array to PIL Image
    image = Image.fromarray(array)
    image.save(path)
    
def ensure_directories(*dirs):
    for directory in dirs:
        if not os.path.exists(directory):
            os.makedirs(directory)
    
def process_images(input_dir: str, output_dir: str, error_dir: str, history_dir: str, detailed: bool = DETAILED):
    ensure_directories(output_dir, error_dir, history_dir)
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.jpg', '.png')):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            error_path = os.path.join(error_dir, filename)
            history_path = os.path.join(history_dir, filename)
            
            with Image.open(input_path) as img:
                result = find_book_final(img, detailed=detailed)
                if result is not None:
                    save_image_from_array(result, output_path)
                    shutil.move(input_path, history_path)
                    print(f"Successfully processed {filename}")
                else:
                    shutil.move(input_path, error_path)
                    print(f"Error processing {filename}, saved to errors")



if __name__ == "__main__":
    process_images(input_dir, output_dir, error_dir, history_dir, detailed=DETAILED)










