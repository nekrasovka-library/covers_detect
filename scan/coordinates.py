from typing import List, Tuple
import PIL.Image
import numpy as np

"""
calculate the coordinates for original_img
"""
def calculate_original_coordinates(
    processed_img: PIL.Image or np.ndarray,
    original_img: PIL.Image or np.ndarray,
    processed_points: List[Tuple[int, int]]
) -> List[Tuple[int, int]]:
    
    if isinstance(processed_img, np.ndarray):
        processed_img = PIL.Image.fromarray(processed_img)
        
    if isinstance(original_img, np.ndarray):
        original_img = PIL.Image.fromarray(original_img)
    
    # calculate ratio between the sizes of images
    rate_w = original_img.width / processed_img.width
    rate_h = original_img.height / processed_img.height
    
    original_points = []
    for point in processed_points:
        # define the coordinates on the original photo
        original_x = int(point[0] * rate_w)
        original_y = int(point[1] * rate_h)
        original_points.append((original_x, original_y))
        
    return original_points

'''function to cut rectangle with the fixed points'''
def cut_box(
    image: PIL.Image,
    points: List[Tuple[int, int]]#left top and right bottom points
) -> PIL.Image:
    assert len(points) == 2, "The list should contain exactly 2 tuples:left top and right bottom points"
 
    # Size of the image in pixels (size of original image)
    # pts_0 = list(map(lambda x: [(x[0], x[1]), (x[0] + x[2], x[1] + x[3])], pts))
    left, top = points[0][0], points[0][1]
    right, bottom = points[1][0], points[1][1]
 
    # Setting the points for cropped image
    # Cropped image of above dimension
    # (It will not change original image)
    points = []
    copy = image.copy()
    im = copy.crop((left, top, right, bottom))
    return im
    

def original_rectangles(
    original_img: PIL.Image,
    im: PIL.Image, #seg image,
    rects: List[Tuple[int, int, int, int]] #left top coordinates x, y and width and height
) -> List[Tuple[int, int, int, int]]:#return the list of books' rectangles on the original photo
    
    pts = list(map(lambda x: [(x[0], x[1]), (x[0] + x[2], x[1] + x[3])], rects))#left top and right bottom coords x, y
    pts_org = [calculate_original_coordinates(im, original_img, el) for el in pts]
    rect_images = [cut_box(original_img, el) for el in pts_org]
    
    return rect_images
