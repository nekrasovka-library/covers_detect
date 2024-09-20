import PIL
from PIL import Image, ImageEnhance, ImageOps
import cv2
import numpy as np
import os
# import pymatting

from scan.coordinates import original_rectangles
from scan.сomponents import find_areas
# from scanner.scan.segmentation import ImageProcessor #ModelBG,
from scan.quadrilateral_utils import extract_rectangle
# from .metadata import fix_orientation
from scan.dledge import EdgeDetection
from scan.counters import perspective_transform, find_cnts, process_contour, edges_book, calculate_percentage_above_threshold, find_book_canny
from  scan.matting import BackgroundRemover
from scan.csam import apply_sam_algorithm
from scan.sam import Sam


# Global variables for models
# SAM = None
# BackgroundRemover = None

CHECKPOINT_PATH = '/workdir/weights/mobile_sam.pt'
SAM = Sam(checkpoint_path = CHECKPOINT_PATH)

BackgroundRemover = BackgroundRemover()

# modelbg = ModelBG()


# imageprocessor = ImageProcessor(1024)
ENHANCE_FACTORS = [1, 1.5, 2]
ENHANCE_FACTORS = [1]


PREPROCESS_PARAMS = {'enhance_params': {'factor': 1.0},
                     'clah_params': {'clipLimit': 1, 'tileGridSize': (20, 20)},
      
      'canny_params': {'threshold1': 113, 'threshold2': 198},
      'gaus_blur_params': {'ksize': (7, 7), 'sigmaX': 0},
      'morph_params': {'kernel_size': (15, 15),
   'iterations': 2,
   'erosion_iterations': 1}}

        
        
        

TRESHOLD_CONFIDENCE = 5 
def find_book(img: Image, model_type='unet', thresholds=[TRESHOLD_CONFIDENCE, 1], sam: Sam = SAM, detailed=False):
    """
    Find book using different models based on the specified model type and detail level.

    Args:
        img (PIL.Image): Input image.
        model_type (str): Model type to use ('sam', 'unit', 'classic_cv').
        thresholds (list): List of thresholds to try.
        sam (Sam): SAM model instance.
        detailed (bool): If True, return additional information; otherwise, return only the book.

    Returns:
        Tuple: (book, preprocessed_img, info, area_segmented) if detailed is True,
               or (book,) if detailed is False.
    """
    details = {}
    if model_type == 'unet':
        print(f'attempt: {model_type}')
        image = img.copy()
        output = BackgroundRemover.remove_background(image)
        area_segmented = calculate_percentage_above_threshold(output)
        gray_image = cv2.cvtColor(np.array(output), cv2.COLOR_RGBA2GRAY)
    
        # Use threshold-based detection
        for threshold in thresholds:
            preprocessed_img = cv2.threshold(gray_image, threshold, 255, cv2.THRESH_BINARY)[1]
            cnt, info = find_cnts(preprocessed_img)
            if cnt is not None and len(cnt) != 0:
                book = perspective_transform(np.array(img), cnt)
                print(f'It"s successfully extracted by the first algorithm')
                if detailed:
                    details['preprocessed_img'] = preprocessed_img
                    details['info'] = info
                    details['area_segmented'] = area_segmented
                    return book, details
                else:
                    return book,

    elif model_type == 'sam' and sam:
        print(f'attempt: {model_type}')
        book, postprocessed_img, scaled_subset, input_label = apply_sam_algorithm(sam, img)
        if book is not None:
            if detailed:
                details['postprocessed_img'] = postprocessed_img
                details['scaled_subset'] = scaled_subset
                details['input_label'] = input_label
                return book, details
            else:
                return book,

    elif model_type == 'canny':
        print(f'attempt: {model_type}')
        book, mean_angle = find_book_canny(img, preprocess_params = preprocess_params)
        if book is not None:
            if detailed:
                details['postprocessed_img'] = postprocessed_img
                details['scaled_subset'] = scaled_subset
                details['input_label'] = input_label
                return book, details
            else:
                return book,

    print('Book extraction failed')
    return (None,) if not detailed else (None, details)




# TRESHOLD_CONFIDENCE = 5        
# def find_book_rem(img: Image, thresholds=[TRESHOLD_CONFIDENCE, 1], sam=SAM) -> np.array:
#     """
#     Find book using different thresholds and return the result for the first successful threshold.

#     Args:
#         img (PIL.Image): Input image.
#         thresholds (list): List of thresholds to try.

#     Returns:
#         Tuple: (book, preprocessed_img, info, area_segmented) for the first successful threshold, or (None, None, None, area_segmented) if none of the thresholds produce a result.
#     """

#     image = img.copy()
#     output = BackgroundRemover.remove_background(image)
#     area_segmented = calculate_percentage_above_threshold(output)
#     gray_image = cv2.cvtColor(np.array(output), cv2.COLOR_RGBA2GRAY)
#     # print_memory_info("After Background Removal")


#     for threshold in thresholds:
#         preprocessed_img = cv2.threshold(gray_image, threshold, 255, cv2.THRESH_BINARY)[1]
#         cnt, info = find_cnts(preprocessed_img)
#         if cnt is not None and len(cnt) != 0:
#             book = perspective_transform(np.array(img), cnt)
#             print(f'It"s successfully extracted by the first algorithm')
#             return book, preprocessed_img, info, area_segmented

#     if sam:
#         book, postprocessed_img, scaled_subset, input_label = apply_sam_algorithm(sam, img)
#         if book is not None:
#             return book, postprocessed_img, info, area_segmented
#     print('Book extraction failed')
#     return None, None, None, None
#     # return None, preprocessed_img, None, area_segmented


