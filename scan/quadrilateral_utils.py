import numpy as np

def reoder(pts):
    """
    reoder returns the following:
    first coordinate will be that of the top left corner,
    the second will be that of the top right corner,
    the third will be of the bottom right corner,
    and the fourth coordinate will be that of the bottom left corner.
    """
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

def find_max_element_index(data):
    angles_radian = [sublist[-1] for sublist in data]
    
    angles = [abs(90 - el) for el in angles_radian]
    
    min_angle = min(angles)
    min_index = angles.index(min_angle)
    return data[min_index]

def find_fourth_point(A, B, C):
    """
    Calculates the fourth point of a rectangle given three points.
    
    Parameters:
    A -- The main point of the rectangle.
    B -- A neighboring point to A.
    C -- Another neighboring point to A.
    
    Returns:
    D -- The fourth point of the rectangle.
    """
    AB = B - A
    D = C + AB
    return D

def project_point_onto_line(A, B, C, perpendicular = True):
    """
    Projects a point C onto a line defined by points A and B.
    
    Parameters:
    A -- The first point on the line.
    B -- The second point on the line.
    C -- The point to be projected.
    perpendicular -- Whether to project perpendicularly (True) or directly onto the line (False).
    
    Returns:
    projected_point -- The projected point of C onto the line.
    """
    # print(A, B, C)
    AB = B - A
    AC = C - A
    if perpendicular:
        AD = np.array([AB[1], -AB[0]], dtype=np.float32)  # AD is the line perpendicular to AB
    else:
        AD = AB  # AD is the same as AB for non-perpendicular projection
        

    length_AB = np.linalg.norm(AB)
    length_AD = np.linalg.norm(AD)

    # projection AC on AD
    projection = (np.dot(AC, AD) / (length_AD))

    # Calculation of the projected point's coordinates
    projected_point = A + projection * ( AD / length_AD)

    return projected_point

def find_integer_rect(points):
    rect = reoder(points)
    (bl, br, tr, tl) = rect
    
    width = np.linalg.norm(br - bl)
    height = np.linalg.norm(tl - bl)

    v1 = br - bl
    br = bl + int(width) * (v1 / width)
    
    v2 = tl - bl
    tl = bl + int(height) * (v2 / height)
    tr = tl + (br - bl)
    
    rect = np.array([tl, tr, bl, br], dtype = "float32")
    return rect

def extract_rectangle(data):
    best_points = find_max_element_index(data['angles2points'])
    # print('data points', data['points'])
    
    best_points_list = [(el[0], el[1]) for el in best_points[0]]
    points_list = [(el[0], el[1]) for el in data['points']]
    last_point = np.array([el for el in points_list if el not in best_points_list])

    # print('best_points', best_points)
    # print('last_point', last_point)
    A = best_points[0][1]
    B = best_points[0][0]
    projected_point_D_AB = project_point_onto_line(A, B, last_point, perpendicular = False)
    dist_AB, dist_D_AB = np.linalg.norm(A - B), np.linalg.norm(A - projected_point_D_AB)
    # print('A B', A, B)
    # print('AB', dist_AB, dist_D_AB, B, projected_point_D_AB)
    if dist_D_AB < dist_AB:
        B = projected_point_D_AB
        
    
    projected_point_1_1 = project_point_onto_line(A, B, best_points[0][2])
    projected_point_1_2 = project_point_onto_line(A, best_points[0][0], last_point)
    # print('projected_point', projected_point_1_1, projected_point_1_2)
    dist_1_1, dist_1_2 = np.linalg.norm(A - projected_point_1_1), np.linalg.norm(A - projected_point_1_2)
    if dist_1_1 < dist_1_2:
        C = projected_point_1_1
    else:
        C = projected_point_1_2
    # print('dist_1_1, dist_1_2', dist_1_1, dist_1_2)
        
    D = find_fourth_point(A, B, C)
    final_points = np.array([A, B, C, D], dtype = "float32")
    final_points = find_integer_rect(reoder(final_points))
    points = reoder(final_points)
    # points = reoder(np.round(np.array(final_points), decimals=0))
    # print('final', points)
    return points
    
    
    