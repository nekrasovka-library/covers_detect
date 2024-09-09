import numpy as np
import cv2

class QuadrilateralChecker():
    # Constants
    ASPECT_RATIO_THRESHOLD = 2.5
    ANGLE_THRESHOLD = 2
    SQUARE_RATIO_1 = 15
    SQUARE_RATIO_2 = 0.7
    
    def __init__(self, points, image_square):
        """Initialize a rectangle with its corner points.

        Args:
            points (list of numpy.ndarray): A list containing exactly 4 corner points of the rectangle.
                The points should be provided in a consistent order(fucntion reoder from counters makes this)

        Raises:
            ValueError: If the input list 'points' does not contain exactly 4 points.
        """
        if len(points) != 4:
            raise ValueError("A rectangle must have exactly 4 points")
        self.points = points
        self.image_square = image_square

    def calculate_angle(self, p1, p2, p3, epsilon=1e-6):
        """Calculate the angle in degrees between three points.

        Args:
            p1 (numpy.ndarray): The first point.
            p2 (numpy.ndarray): The vertex point.
            p3 (numpy.ndarray): The third point.
            epsilon (float, optional): A small value added to the denominator to avoid division by zero.

        Returns:
            float: The angle in degrees between the vectors formed by p1-p2 and p3-p2.
        """
        p12 = p1 - p2
        p23 = p3 - p2
        cosine_angle = np.dot(p12, p23) / (np.linalg.norm(p12) * np.linalg.norm(p23) + epsilon)
        angle_rad = np.arccos(cosine_angle)
        angle_deg = np.degrees(angle_rad)
        return angle_deg
    
    def check_angles(self, threshold = None, list_angles = False):
        """Check if all angles between neighboring lines are approximately 90 degrees.

        Args:
            threshold (float, optional): The maximum allowable deviation from 90 degrees in degrees.
                If not provided, the default threshold from the class attribute ANGLE_THRESHOLD is used.

        Returns:
            bool: True if all angles are approximately 90 degrees, False otherwise.
        """
        if threshold is None:
            threshold = self.ANGLE_THRESHOLD
        for i in range(4):
            angle = self.calculate_angle(
                self.points[i],
                self.points[(i + 1) % 4],
                self.points[(i + 2) % 4]
            )
            # print(self.points[i], self.points[(i + 1) % 4], self.points[(i + 2) % 4])
            # print(angle)
            # print(abs(90 - angle))
            if abs(90 - angle) > threshold:
                return False
        return True
    
    def get_angles2points(self):
        """Calculate the average angle between neighboring lines in the polygon.

        Returns:
            float: The average angle in degrees.
        """
        angles_points = []
        for i in range(4):
            angle = self.calculate_angle(
                self.points[i],
                self.points[(i + 1) % 4],
                self.points[(i + 2) % 4]
            )
            angles_points.append([[self.points[i],
                self.points[(i + 1) % 4],
                self.points[(i + 2) % 4]], angle])
        return angles_points
        # average_angle = total_angle / 4
        # if not list_angles:
        #     return abs(90 - average_angle)
        # return angles
    
    def check_square(self, ratio_1 = None, ratio_2 = None):
        """Check if the rectangle represents a book based on area ratios compared to the area of the entire image.

        Args:
            image_square (float): The area of the entire image.
            ratio_1 (float, optional): The minimum required ratio of the rectangle's area to the image's area.
                If not provided, the default ratio from the class attribute SQUARE_RATIO_1 is used (10 times bigger).
            ratio_2 (float, optional): The maximum required ratio of the rectangle's area to the image's area.
                If not provided, the default ratio from the class attribute SQUARE_RATIO_2 is used (95% of the image).

        Returns:
            bool: True if the rectangle is considered a book, False otherwise.
        """
        if ratio_1 is None:
            ratio_1 = self.SQUARE_RATIO_1
        if ratio_2 is None:
            ratio_2 = self.SQUARE_RATIO_2
            
        square = cv2.contourArea(self.points)
        print('SQUARE', square/self.image_square)
        # Check if the rectangle's area ratio to the image's area is greater than or equal to the specified ratio.
        if square * ratio_1 < self.image_square or square > self.image_square * ratio_2: 
            return False
        return True
    
    def is_valid_rect(self, threshold=None, epsilon=1e-6):
        """Check if the points form a valid rectangle based on aspect ratio.

        The aspect ratio of a rectangle is defined as the ratio of its longer side 
        to its shorter side. If the aspect ratio is much greater than the specified threshold,
        the rectangle is considered invalid. The epsilon parameter is used to avoid 
        division by zero errors when the sides are close in length.

        Args:
            threshold (float): The maximum allowed aspect ratio. If not provided, 
                the default threshold from the class attribute ASPECT_RATIO_THRESHOLD is used.
            epsilon (float): A small value added to the denominator to avoid division by zero.

        Returns:
            bool: True if the points form a valid rectangle, False otherwise.
        """
        
        if threshold is None:
            threshold = self.ASPECT_RATIO_THRESHOLD
        
        w, h = np.linalg.norm(self.points[0] - self.points[1]), np.linalg.norm(self.points[1] - self.points[2])
        aspect_ratio = max(w, h) / (min(w, h) + epsilon)
        if aspect_ratio <= threshold:
            return True
        return False
    
# class ParallelogramFinder(Rectangle):
#     def __init__(self, points, angles):
#         super().__init__(points)
#         self.angles = angles
#         self.points2angles = [[i, angles[i], [points[i], points[(i + 1) % 4], points[(i + 2) % 4]]] for i in range(4)]
#         self.error_point = self._get_error_point()

#     def _get_error_point(self):
#         """
#         Finds the point that does not correspond to the rectangle
#         """
#         error_point = []
#         for points in range(4):
#             point1 = self.points2angles[points]
#             point2 = self.points2angles[(points + 3) % 4]
#             angle1 = 90 - point1[1]
#             angle2 = 90 - point2[1]
#             if (angle1 > 1 and angle2 < -1) or (angle1 < -1 and angle2 > 1):
#                 error_point.append(point1)
#         if len(error_point) == 1:
#             return error_point[0][2][0]
#         return None

#     def get_parallelogram(self):
#         p1, p2, p3, p4 = self.points
#         if np.array_equal(p1, self.error_point):
#             p23 = p3 - p2
#             point = p4 - p23
#             new_points = [point, p2, p3, p4]
#         elif np.array_equal(p2, self.error_point):
#             p41 = p1 - p4
#             point = p3 - p41
#             new_points = [p1, point, p3, p4]
#         elif np.array_equal(p3, self.error_point):
#             p12 = p2 - p1
#             point = p4 - p12
#             new_points = [p1, p2, point, p4]
#         elif np.array_equal(p4, self.error_point):
#             p12 = p1 - p2
#             point = p3 - p12
#             new_points = [p1, p2, p3, point]
#         else:
#             # Handle the case where no error point is found
#             new_points = self.points

#         # Check if the new points form a valid rectangle
#         if self.is_valid_rect() and self.check_angles() and self.check_square():
#             return np.array(new_points)
#         else:
#             return None

    
    