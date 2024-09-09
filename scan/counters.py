import cv2
# import imutils
from skimage.filters import threshold_local
from matplotlib import pyplot as plt
import numpy as np
import PIL
from PIL import Image, ImageEnhance
from .segmentation import ImageProcessor#ModelBG,
# from PIL import Image, ExifTags
# from PIL import Image, ImageEnhance
from typing import List, Tuple
import io
from .coordinates import calculate_original_coordinates
from .corRect import QuadrilateralChecker
# from scan.parallelogram import ParallelogramFinder
from  .matting import BackgroundRemover
from .quadrilateral_utils import extract_rectangle
# from metadata.metadata import fix_orientation
"""sometimes if the photo is made by mobile phone we need to get information about orientation photo from metadata"""

from PIL import ImageOps, Image
BackgroundRemover = BackgroundRemover()
def enhance_image(im: PIL.Image, factor = 1) -> PIL.Image:
    enhancer = ImageEnhance.Contrast(im)
    enhanced_im = enhancer.enhance(factor)
    return enhanced_im

TRESHOLD_CONFIDENCE = 5

# def fix_orientation(image_path):
#     with Image.open(image_path) as img:
#         try:
#             exif = img._getexif()
#             orientation = exif.get(ExifTags.ORIENTATION)


#             if orientation in exif:
#                 if exif[orientation] == 3:
#                     img = img.transpose(Image.ROTATE_180)
#                 elif exif[orientation] == 6:
#                     img = img.transpose(Image.ROTATE_270)
#                 elif exif[orientation] == 8:
#                     img = img.transpose(Image.ROTATE_90)

#         except (AttributeError, KeyError, IndexError):
#             # metadata doesn't exist
#             pass

#         # save image in buffer
#         buffer = io.BytesIO()
#         img.save(buffer, format='JPEG')

#     # copy image from buffer
#     buffer.seek(0, 0)
#     img = Image.open(buffer).copy()

#     return img
    
"""reoder returns the following: first coordinate will be that of the top left corner, the second will be that of the top right corner, the third will be of the bottom right corner, and the fourth coordinate will be that of the bottom left corner."""
def reoder(pts):
    rect = np.zeros((4, 2), dtype = "float32")
    pts = np.float32(pts.reshape((4, 2)))
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    # print('rect', rect, type(rect[0]))
    return rect

def auto_canny(image, sigma=0.33):#it's not the best everytime
    # compute the median of the single channel pixel intensities
    v = np.median(image)
    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper, L2gradient=True)
    # return the edged image
    return edged

def resize_image(path):
    image = Image.open(path)
    height, width = image.size
    # print(width, height)
    scale_percent = (1024 / min(width, height))
    width_size = scale_percent*width
    height_size = scale_percent*height
    image = image.resize((int(height_size), int(width_size)), Image.LANCZOS)
    return image

ENHANCE_PARAMS = {'factor': 1.0}
CLAH_PARAMS = {'clipLimit': 4.0, 'tileGridSize': (8, 8)}
CANNY_PARAMS = {'threshold1': 0, 'threshold2': 200, 'L2gradient': True}
GAUS_BLUR_PARAMS = {'ksize': (5, 5), 'sigmaX': 0}
MORPHOLOGICAL_PARAMS = {'kernel_size': (5, 5), 'iterations': 2, 'erosion_iterations': 1}
PREPROCESS_PARAMS = {
                    'clah_params': CLAH_PARAMS,
                    'canny_params': CANNY_PARAMS,
                    'gaus_blur_params': GAUS_BLUR_PARAMS,
                    'morph_params': MORPHOLOGICAL_PARAMS,
                    'enhance_params': ENHANCE_PARAMS
                    }

def preprocess(img: PIL.Image,
               enhance_params = None,
               clah_params = None, canny_params = None,
               gaus_blur_params = None, morph_params = None,
              ):
    if isinstance(img, np.ndarray):
        # Convert NumPy array to PIL Image
        img = Image.fromarray(img)
    image = img.copy()
    
    # kernel = np.ones((5,5),np.uint8)
    # img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations= 3)
    if enhance_params:
        image = enhance_image(image, **enhance_params)#increase contrast
    image = np.array(image)  

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if clah_params:
        enhancer = cv2.createCLAHE(**clah_params)
        gray_image = enhancer.apply(gray_image)
    
    if gaus_blur_params:
        blurred_image = cv2.GaussianBlur(gray_image, **gaus_blur_params)
    else:
        blurred_image = gray_image
    if canny_params:
        edged_img = cv2.Canny(blurred_image, **canny_params)#1
    else:
        edged_img = auto_canny(blurred_image) 
    
    # Apply morphological operations (dilation and erosion)
    if morph_params:
        kernel = np.ones(morph_params['kernel_size'], np.uint8)
        img_dilation = cv2.dilate(edged_img, kernel, iterations=morph_params['iterations'])
        img_erosion = cv2.erode(img_dilation, kernel, iterations=morph_params['erosion_iterations'])
    else:
        img_erosion = edged_img

    return img_erosion

def is_rectangles_intersect(rect1, rect2):# check if rectangels have intersection
    #get coordinate lists of x and y coordinates for every image
    r1_x = [rect1[i][0] for i in range(4)]
    r1_y = [rect1[i][1] for i in range(4)]
    r2_x = [rect2[i][0] for i in range(4)]
    r2_y = [rect2[i][1] for i in range(4)]

    if (max(r1_x) < min(r2_x) or max(r2_x) < min(r1_x) or
        max(r1_y) < min(r2_y) or max(r2_y) < min(r1_y)):
        return False
    else:
        return True

def new_rect_intersection(rects, new_rect):#check if new rectangle has intersection with others
    for rect in rects:
        if is_rectangles_intersect(rect, new_rect):
            return True
    return False

def is_valid_rect(rect, epsilon=1e-6):
    # Check aspect ratio of rectangle
    w, h = np.linalg.norm(rect[0] - rect[1]), np.linalg.norm(rect[1] - rect[2])
    aspect_ratio = max(w, h) / (min(w, h) + epsilon)
    return aspect_ratio <= 4



def find_cnts(img: PIL.Image, multiple = False, extract_rect = True) -> PIL.Image:#multiple = False means if we search one book on image of many(True)
    """
    if multiple is True, there will be extracted all quadrilaterals
    if extract_rect is True, there will be extracted trectangle from quadrilateral
    """
    cnts, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # cnts, _ = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # cnts, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    image_area = img.shape[0] * img.shape[1]
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:15]
    # print(cnts)
    sel_countours=[]
    y = []
    doc = None#points of book
        
    for c in cnts:
        peri = cv2.arcLength(c, True)#the length of perimeter, True means the contour must be closed
        approx = cv2.approxPolyDP(c, 0.05 * peri, True)#0.05
        if len(approx) == 4:
            doc = reoder(approx)
            rect = QuadrilateralChecker(doc, image_area)
            # print(rect.check_angles(), rect.is_valid_rect(), rect.check_square(image_area))
            if rect.check_angles() and rect.is_valid_rect() and rect.check_square():
                if extract_rect:
                    doc = extract_rectangle(
                        {
                        "angles2points": rect.get_angles2points(),
                        "points": doc,
                        
                           }
                    )
                    rect = QuadrilateralChecker(doc, image_area)
                    
                    
                    
                    
                    
                if not multiple:#if we search only one book, because cnts is sorted by area, so we may search the first area with 4 lines
                    # angles = rect.get_angles()
                    # average_angle = abs(90 - (np.sum(angles)/4))
                    get_angles2points = rect.get_angles2points()
                    angles = [el[1] for el in get_angles2points]
                    average_angle = abs(90 - (np.sum(angles)/4))

                    info = {
                        # "angles": angles,
                        "average_angle": average_angle,
                        "angles2points": get_angles2points,
                        "points": doc,
                        "angles": angles
                        
                           }

                    return doc, info
                else:
                    if not new_rect_intersection(y, doc):#the rectangle doesn't have intersection with others, that's already detected
                        y.append(doc)   
    return y, None


def calculate_percentage_above_threshold(image, threshold = TRESHOLD_CONFIDENCE):
    """
    Calculate the percentage of pixels in the alpha channel (4th channel) above a given threshold.

    Parameters:
    - image: PIL.Image
    - threshold: The threshold value.

    Returns:
    - The percentage of pixels above the threshold.
    """
    # Convert PIL image to NumPy array
    np_image = np.array(image)

    # Ensure the image has an alpha channel
    if np_image.shape[-1] != 4:
        raise ValueError("Image must have an alpha channel (4th channel) for threshold calculation.")

    # Count pixels above threshold in the alpha channel
    count_pixels_above_threshold = np.sum(np_image[:, :, -1] > threshold)

    total_pixels = np_image.shape[0] * np_image.shape[1]
    percentage_pixels_above_threshold = (count_pixels_above_threshold / total_pixels) * 100

    return percentage_pixels_above_threshold
    
def edges_book(mask):
    """
    first coordinate will be that of the top left corner,
    the second will be that of the top right corner,
    the third will be of the bottom right corner,
    and the fourth coordinate will be that of the bottom left corner.
    """
    
    rect = np.zeros((4, 2), dtype = "float32")
    mask_pixels = np.array(np.where(mask != 0))
    np_sum = mask_pixels.sum(axis = 0)

    rect[0] = mask_pixels[:, np.argmin(np_sum)]

    rect[2] = mask_pixels[:, np.argmax(np_sum)]
    diff = -np.diff(mask_pixels, axis = 0)

    rect[1] = mask_pixels[:, np.argmin(diff)]
    rect[3] = mask_pixels[:, np.argmax(diff)]

    return rect

def perspective_transform(
    image: np.array,
    points: List[Tuple[int, int]]#4 points of box for the book(it's a result of scanner)
) -> np.array:
    
    rect = reoder(points)#reoder points to get them in the fixed order
    # rect = points
    # (tl, tr, bl, br) = points
    (bl, br, tr, tl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    # maxWidth = max(int(widthA), int(widthB))
    maxWidth = max(round(widthA), round(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    # maxHeight = max(int(heightA), int(heightB))
    maxHeight = max(round(heightA), round(heightB))
    
    new_coor = np.array([
      [0, 0],
      [maxWidth, 0],
      [maxWidth, maxHeight],
      [0, maxHeight]], dtype="float32")
    
    matrix = cv2.getPerspectiveTransform(rect, new_coor)
    wraped_img = cv2.warpPerspective(image, matrix, (maxWidth, maxHeight))
    
    return wraped_img

def process_contour(cnt, preprocessed_img, original_img):
    # Convert contour points to a list of tuples with integer coordinates
    cnt_list = list(map(lambda x: (int(x[0]), int(x[1])), cnt))

    # Calculate original coordinates
    cnts_final = calculate_original_coordinates(preprocessed_img, original_img, cnt_list)

    # Convert the result to NumPy arrays of float32
    cnts_final = np.array(list(map(lambda x: np.array(x, dtype=np.float32), cnts_final)))

    return cnts_final

def find_book_canny(img: PIL.Image, preprocess_params = PREPROCESS_PARAMS) -> np.array:
    image = img.copy()
    preprocessed_img = preprocess(image, **preprocess_params)
    cnt, info = find_cnts(preprocessed_img)
    # return np.array(img), cnt, angles
    if cnt is None or len(cnt) == 0:
        return None, None
    # cnts_final = process_contour(cnt, preprocessed_img, img)
    book = perspective_transform(np.array(img), cnt)
    return book, info

# def find_book_rem(img: PIL.Image, treshold = TRESHOLD_CONFIDENCE) -> np.array: #using remgb
#     image = img.copy()
#     output = BackgroundRemover.remove_background(image)

#     #calculate the area of segmented image to find out cases where there's no anything on the image
#     area_segmented = calculate_percentage_above_threshold(output)
#     gray_image = cv2.cvtColor(np.array(output), cv2.COLOR_RGBA2GRAY)
#     # gray_image = cv2.GaussianBlur(gray_image, **{'ksize': (13, 13), 'sigmaX': 0})

#     preprocessed_img = cv2.threshold(gray_image, treshold, 255, cv2.THRESH_BINARY)[1]
#     cnt, info = find_cnts(preprocessed_img)
#     if cnt is None or len(cnt) == 0:
#         return None, None, None, area_segmented
#         # return None, preprocessed_img
#     # cnts_final = process_contour(cnt, preprocessed_img, img)
#     book = perspective_transform(np.array(img), cnt)
#     return book, preprocessed_img, info, area_segmented

def find_book_rem(img: PIL.Image, thresholds=[TRESHOLD_CONFIDENCE, 1]) -> np.array: #using remgb
    """
    Find book using different thresholds and return the result for the first successful threshold.

    Args:
        img (PIL.Image): Input image.
        thresholds (list): List of thresholds to try.

    Returns:
        Tuple: (book, preprocessed_img, info, area_segmented) for the first successful threshold, or (None, None, None, area_segmented) if none of the thresholds produce a result.
    """
    image = img.copy()
    output = BackgroundRemover.remove_background(image)

    # Calculate the area of segmented image to find out cases where there's no anything on the image
    area_segmented = calculate_percentage_above_threshold(output)
    gray_image = cv2.cvtColor(np.array(output), cv2.COLOR_RGBA2GRAY)


    for threshold in thresholds:
        preprocessed_img = cv2.threshold(gray_image, threshold, 255, cv2.THRESH_BINARY)[1]
        cnt, info = find_cnts(preprocessed_img)

        if cnt is not None and len(cnt) != 0:
            book = perspective_transform(np.array(img), cnt)
            return book, preprocessed_img, info, area_segmented
    return None, None, None, area_segmented


def dl_find_book(img: PIL.Image, model, factor = 1) -> np.array:
    
    image = img.copy()
    image = enhance_image(image, factor = factor)#increase contrast
    
    imageprocessor = ImageProcessor(2048)
    image = np.array(imageprocessor(image))
    # image = np.array(image)
    
    # preprocessed_img = model(image, model)
    preprocessed_img = model.detect_edges(image)
    # imgray = cv2.cvtColor(preprocessed_img, cv2.COLOR_BGR2GRAY)
    # blur = cv2.GaussianBlur(preprocessed_img, (5,5), 0)
    # thresh = cv2.threshold(blur, 120, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    ret, thresh = cv2.threshold(preprocessed_img, 90, 255, 0)
    
    kernel = np.ones((3, 3), np.uint8)
    # print(thresh.shape)
    img_dilation = cv2.dilate(thresh, kernel, iterations=2)
    img_erosion = cv2.erode(img_dilation, kernel, iterations=1)
    
    cnt, c = find_cnts(thresh)
    if cnt is not None and len(cnt) > 0:
        area = cv2.contourArea(cnt)
    else:
        area = 0
        # print(0)
        return None
    # , preprocessed_img, thresh, img_erosion

    image_area = preprocessed_img.shape[0] * preprocessed_img.shape[1]
    # print(area, preprocessed_img.shape)
    # print(area, image_area)
    # print('area * 2.5 < image_area', area * 2.5 < image_area)
    # print(cnt.any())
    # if not cnt.any() or area * 2.5 < image_area:
    
    if area * 10 < image_area or area > image_area * 0.95:
        # print(area, image_area)
        # print('1')
        return None
    # , preprocessed_img, thresh, img_erosion, cnt, c
    
    # print(area/image_area)
    # area = cv2.contourArea(cnt)
    
    # cnts_final = process_contour(cnt, preprocessed_img, img)
    book = perspective_transform(np.array(img), cnts_final)
    
    # book = perspective_transform(image, cnts_final)
    # book = perspective_transform(image, np.array(list(map(lambda x: np.array(x, dtype=np.float32), cnt_list))))
    # print(area)
    # if area < 70000:
    #     return None, cnt, preprocessed_img, image, img, c, area
    # return book, cnt, preprocessed_img, image, img, c, area
    return book