import cv2
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
import os
from scan.сomponents import get_component_book
from scan.counters import perspective_transform, find_cnts, process_contour, edges_book

"""
Clusters SAM 
apply_sam_algorithm is realisation of segmentation the book by clustering the embeddings patches from SAM after that get the mask of the book by founded previously points with SAM again.
"""
POINTS_12 = [(7, 5), (5, 7), (10, 5), (3, 4), (4, 3)]
WEIGHTS_P = [0.8, 0.75, 0.85, 0.7]
def apply_sam_algorithm(sam, image):
    EMB_DIM = 256
    resized_image = np.array(image)
    #get embeddings by SAM
    print(1)
    output_np = get_embeddings(sam = sam, image = resized_image)
    print(2)
    embs = output_np.reshape(EMB_DIM, -1).transpose()
    
    #apply KMEANS to cluster embeddings and get true labels by info that patches near the edges of image are almost all are correspond to the table 
    segmented_image = perform_clustering(embs, output_np)

    mask = get_mask(segmented_image, resized_image)
    # return mask, embs, output_np, segmented_image
    
    #use clustering alghoritm from open cv to find the component with the book, it has the biiggest square except the the table. But with defining the step higher true_labels we will not wind the component that correspond to the table
    component, component_centroid = get_component_book(mask)
    scaled_subset = None
    if component is not None:
        scaled_subset, postprocessed_img, input_label, cnt = process_component(sam, component, component_centroid, image)
        if scaled_subset is not None:
            return process_image_for_book_extraction(scaled_subset, postprocessed_img, image, input_label, cnt)
    return None, None, None, None

def get_embeddings(sam, image):
    sam.predictor.set_image(image)
    embs = sam.model_res(image)
    output_np = embs.squeeze(0).detach().cpu().numpy()
    h, w = embs_resize((64, 64), (image.shape[1], image.shape[0]))
    output_np = output_np[:, :h, :w]
    return output_np

def perform_clustering(embeddings, output_np):
    kmeans = KMeans(n_clusters=2, random_state=0, n_init="auto").fit(embeddings)
    cluster_labels = kmeans.labels_
    cluster_labels = cluster_labels.reshape((output_np.shape[1], output_np.shape[2]))
    
    cluster_labels = true_labels(cluster_labels)
        
    colors = np.array([[0, 0, 0], [255, 255, 255]])
    segmented_image = colors[cluster_labels]
    # kmeans = KMeans(n_clusters=2, random_state=0, n_init="auto").fit(embeddings)
    return segmented_image

def get_segmented_image(cluster_labels):
    colors = np.array([[0, 0, 0], [255, 255, 255]])
    return colors[true_labels(cluster_labels)]

def process_component(sam, component, component_centroid, img,
                      points_12 = POINTS_12, weights_p = WEIGHTS_P):
    component_edges = edges_book(component)
    for i, (n1, n2) in enumerate(points_12):
        for weight in weights_p:
            subset, input_label  = subset_points_2(component_edges, component_centroid, n1, n2, weight=weight)
            scaled_subset = subset[:, ::-1]
            postprocessed_img, best_score = sam.process_image(scaled_subset, input_label)
            cnt, info = find_cnts(postprocessed_img)
            cnt = process_contour(cnt, postprocessed_img, img)
            if cnt is not None and len(cnt) != 0:
                print(f'It"s successfully extracted on the {i}"th param and weight {weight}: {(n1, n2)}')
                return scaled_subset, postprocessed_img, input_label, cnt
    return None, None, None, None

def process_image_for_book_extraction(scaled_subset, postprocessed_img, image, input_label, cnt):
    book = perspective_transform(np.array(image), cnt)
    return book, postprocessed_img, scaled_subset, input_label



def scale_points(subset_points: np.array,  original_shape: tuple, shape_image_encoder: tuple = (1024, 1024)):
    subset_height, subset_width = shape_image_encoder
    original_height, original_width = original_shape
    
    scale_x = original_width / subset_width
    scale_y = original_height / subset_height
    
    original_points = []
    for x, y in subset_points:
        # print(x, y)
        original_x = int(x * scale_x)
        original_y = int(y * scale_y)
        # print(original_x, original_y)
        
        original_points.append((original_x, original_y))
    
    return np.array(original_points)

def get_mask(segmented_image, original_image):
    # Compute size ratio
    segmented_image_shape = segmented_image.shape
    original_image_shape = original_image.shape
    size_ratio = (original_image_shape[0] / segmented_image_shape[0], original_image_shape[1] / segmented_image_shape[1])

    mask = np.zeros_like(original_image[:,:,0], dtype=bool)
    # mask = np.zeros_like(original_image, dtype=bool)
    for i in range(segmented_image_shape[0]):
        for j in range(segmented_image_shape[1]):
            if segmented_image[i, j, 0] == 255:  # Check if pixel value is 255
                x_start = i * size_ratio[0]
                y_start = j * size_ratio[1]
                mask[int(x_start):int(x_start+size_ratio[0]), int(y_start):int(y_start+size_ratio[1])] = True 
                
    return mask

def get_points_between(p1, p2, n: int = 5, include: bool = True):
    """
    p1 and p2 are points with coordinates x and y
    n : int, optional
        Number of points to generate between p1 and p2 (default is 5).
    include : bool, optional
        Whether to include the endpoints p1 and p2 (default is True).
    
    Returns:
    np.linspace n points between p1 and p2 
    
    """
    if n < 2:
        raise ValueError("Number of points (n) must be at least 2.")
    x_points = np.linspace(p1[0], p2[0], n)
    y_points = np.linspace(p1[1], p2[1], n)
    if not include:
        x_points = x_points[1: -1]
        y_points = y_points[1: -1]
    return np.column_stack((x_points, y_points))

"""
Below there are 2 functions subset_points and subset_points_2 to get subset of points, that correspond the book
"""
def subset_points(component_edges, component_centroid):
    """
    get subset points from the rectangle of the book
    """
    
    def points_contour_rect(component_edges):
        """
        component_edges is an np array of 4 points representing the corners of the rectangle of the book.
        get subset points between the corners of the rectangle of the book.
        """
        points = np.vstack(
                    (
                    get_points_between(component_edges[1], component_edges[2]),
                    get_points_between(component_edges[0], component_edges[1]),
                    get_points_between(component_edges[2], component_edges[3]),
                    get_points_between(component_edges[0], component_edges[3])
                    )
                   )
        return points

    points = points_contour_rect(component_edges)
    center = [np.mean(component_edges[:, 0]), np.mean(component_edges[:, 1])]
    # final_points = points.copy()
    final_points = np.array([])
    # print(center, component_centroid)
    for point in points:
        
        # print(point)
        inner_points = get_points_between(point, center, n = 5, include = False)
        # print(inner_points)
        
        final_points = np.vstack((final_points, inner_points)) if final_points.size else inner_points
        
    input_label = np.array([1] * len(final_points))#1 means foreground
    return final_points, input_label

def subset_points_2(component_edges, component_centroid, n_points_1:int = 7, n_points_2: int = 5, weight: float = 0.8):
    """
    get subset points from the rectangle of the book
    """
    
    def get_point_on_line(point1, point2, weight=0.8):
        point1 = np.array(point1)
        point2 = np.array(point2)
        point_w = point1 * weight + point2 * (1 - weight)
        return point_w

    def points_edges_subrect(component_edges, weight = 0.8):
        """
        component_edges is an np array of 4 points representing the corners of the rectangle of the book.
        get subset points between the corners of the subrectangle of the book.
        """
        center = [np.mean(component_edges[:, 0]), np.mean(component_edges[:, 1])]
        # points = get_point_on_line()
        # print(component_edges[0], center)
        
        points = np.vstack(
                    (
                    get_point_on_line(component_edges[0], center, weight=weight),
                    get_point_on_line(component_edges[1], center, weight=weight),
                    get_point_on_line(component_edges[2], center, weight=weight),
                    get_point_on_line(component_edges[3], center, weight=weight)
                    )
                   )
        return points
    
    points = points_edges_subrect(component_edges, weight = weight)
    # n_points_1 = 10
    # n_points_2 = 5
    points_1 = get_points_between(points[0], points[3], n = n_points_1, include = True)
    points_2 = get_points_between(points[1], points[2], n = n_points_1, include = True)
    final_points = np.array([])

    for p1, p2 in zip(points_1, points_2):
        inner_points = get_points_between(p1, p2, n = n_points_2, include = True)
        
        final_points = np.vstack((final_points, inner_points)) if final_points.size else inner_points
    input_label = np.array([1] * len(final_points))#1 means foreground
    return final_points, input_label



def true_labels(cluster_labels:np.array):
    """
    By the information of edges of image we can define which ebmeddings are foreground(book) and thar are backround(table)
    
    Inverts the labels in the cluster_labels array if the mean percentage 
    of label 1 along the image edges is greater than 50%.

    Args:
    - cluster_labels (np.array): Array containing clustered labels.

    Returns:
    - np.array: Array with inverted labels if mean percentage of label 1 
      along image edges is greater than 50%, otherwise returns the original
      cluster_labels array.
    """
    
    def calculate_edge_label_percentages(cluster_labels:np.array):
        """
        Calculate the mean percentage of label 1 along the top, bottom, left, and right
        edges of the image.

        Args:
        - cluster_labels (np.array): Array containing clustered labels.

        Returns:
        - float: Mean percentage of label 1 along image edges.
        """
        height, width = cluster_labels.shape

        # Define edge regions
        top_edge = cluster_labels[0, :]
        bottom_edge = cluster_labels[height-1, :]
        left_edge = cluster_labels[:, 0]
        right_edge = cluster_labels[:, width-1]

        # Calculate percentages for each edge
        top_percentage = np.mean(top_edge == 1) * 100
        bottom_percentage = np.mean(bottom_edge == 1) * 100
        left_percentage = np.mean(left_edge == 1) * 100
        right_percentage = np.mean(right_edge == 1) * 100
        # print(top_percentage, bottom_percentage, left_percentage, right_percentage)

        # Calculate mean percentage across all edges
        mean_percentage = np.mean([top_percentage, bottom_percentage, left_percentage, right_percentage])
        return mean_percentage

    mean_percentage = calculate_edge_label_percentages(cluster_labels)
    if mean_percentage > 50:
        cluster_labels = np.where(cluster_labels == 1, 0, 1)
    return cluster_labels


def embs_resize(embedding_size=(64, 64), original_size=(6000, 4000)):
    """
    Resize embeddings based on original and target image sizes.
    
    Parameters:
        embedding_size (tuple): The size of each embedding (default is (64, 64)).
        original_size (tuple): The size of the original image (default is (6000, 4000)).
    
    Returns:
        tuple: The new size of the embeddings after resizing.
    """

    scale = max(original_size)
    new_width = embedding_size[0] * original_size[0] // scale
    new_height = embedding_size[1] * original_size[1] // scale
    
    return new_height, new_width

