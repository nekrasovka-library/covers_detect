import cv2
import numpy as np


def get_component_book(mask: np.array, max_ratio: int = 0.8, min_ratio: int = 0.1):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8)*255, connectivity=8)
    max_ids_square = np.argsort(stats[:, -1])
    max_ids_square = np.flip(max_ids_square)
    height_mask, width_mask = mask.shape[0], mask.shape[1]
    square_mask = height_mask * width_mask
    """
    the component with background has a big width and height,
    so we can find it by this feature, if width and height have great values
    (some books can be big too, be accurate)
    """
    min_h, max_h = 0.1, 0.99
    min_w, max_w = 0.1, 0.95
    for i in max_ids_square:
        area = stats[i, cv2.CC_STAT_AREA]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        ratio_h = h/height_mask
        ratio_w = w/width_mask
        ratio = area/square_mask
        if min_ratio < ratio < max_ratio and min_h < ratio_h < max_h and min_w < ratio_w < max_w:
            # print(area, i)
            book_component = (labels == i).astype("uint8") * 255
            
            # book_component = np.transpose(book_component)
            return book_component, centroids[i]
    return None, None


###it's not used
"""
Function find_areas takes as input image in format RGBA,
where the 4th coordinate is equal 255 if the the model
defined the point belongs one of the books and 0 otherwise.
The function returns rectangles/boxes for books with padding
and also we check if the rectangle is not too small
and check that rations height/width and width/height are normal
"""

def find_areas(image, threshold_area=100, max_aspect_ratio=4, padding=5):
    # create mask, where the 4th channel is a value True/False(255 is True and 0 is False)
    mask = image[:, :, 3] >= 50
    
    # Applying threshold
    # threshold = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    #apply algorithm connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)

    # find areas with books 
    book_regions = []
    for i in range(1, num_labels): # skip background label 0
        left = stats[i, cv2.CC_STAT_LEFT]
        top = stats[i, cv2.CC_STAT_TOP]
        width = stats[i, cv2.CC_STAT_WIDTH]
        height = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        
        # ensure the rectangle with padding is not out of image bounds 
        left_pad = max(left - padding, 0)
        top_pad = max(top - padding, 0)
        right_pad = min(left + width + padding, image.shape[1])
        bottom_pad = min(top + height + padding, image.shape[0])
        width_pad = right_pad - left_pad
        height_pad = bottom_pad - top_pad
    
        # check the area and aspect ratio
        if area > threshold_area and \
                width > 0 and height > 0 and \
                width/height <= max_aspect_ratio and height/width <= max_aspect_ratio:
            book_regions.append((left_pad, top_pad, width_pad, height_pad))            

    return book_regions