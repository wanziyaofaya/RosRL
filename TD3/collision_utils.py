def point_in_poly(x, y, poly):
    """
    判断点(x, y)是否在多边形poly内
    poly: [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]，按顺序排列
    返回True表示在多边形内，False表示在外
    """
    n = len(poly)
    inside = False
    px, py = x, y
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[(i+1)%n]
        if ((yi > py) != (yj > py)):
            intersect_x = (xj - xi) * (py - yi) / (yj - yi + 1e-10) + xi
            if px < intersect_x:
                inside = not inside
    return inside

def check_collision(x, y):
    obstacles_rect = [
    (-3.43, 1.76, 2.59, 0.21),  # 11*
    (-2.28, 0.35, 0.21, 2.84),   # 13*
    (2.06, 2.93, 1.34, 0.21),  # 15*
    (2.06, 2.11, 0.21, 1.04),  # 16*
    (2.09, 2.11, 1.34, 0.21),  # 17*
    (3.22, 2.11, 0.21, 1.04), # 18*

    (-3.45, -3.37, 2.15, 0.21), # 26* 标准矩形
    (1.48, -2.59, 1.84, 0.24), # 28*
    (3.10, -2.57, 0.24, 2.59), # 29*
    (-5.52, 5.33, 11.02, 0.17), # 6*
    (-5.52, -5.52, 0.17, 11.02), # 7*
    (-5.52, -5.52, 11.02, 0.17), # 8*
    (5.33, -5.52, 0.17, 11.02), # 9*

    (-4.68, 4.36, 0.30, 0.30), # fire_hydrant*
    (4.30, -3.77, 0.99, 0.10), # back*
    (5.19, -4.16, 0.11, 0.49),  # left_side*
    (4.29, -4.16, 0.11, 0.49), # right_side*

    # (4.31, -4.16, 0.97, 0.49), # bottom
    # (4.31, -4.16, 0.97, 0.49), # top
    # (4.31, -4.16, 0.97, 0.49), # low_shelf
    # (4.31, -4.16, 0.97, 0.49), # high_shelf

    # (-5.11, -0.96, 1.57, 0.87), # surface
    (-3.72, -0.21, 0.13, 0.13), # front_left_leg*
    (-3.72, -0.97, 0.13, 0.13), # front_right_leg*
    (-5.07, -0.97, 0.13, 0.13), # back_right_leg*
    (-5.07, -0.21, 0.13, 0.13), # back_left_leg*

    (3.68, 0.74, 0.59, 0.49), # cardboard_box_0*
    (-0.28, -4.20, 0.59, 0.49), # cardboard_box_1*
    (-5.17, 2.74, 0.59, 0.49), # cardboard_box_2*
    (-0.30, 3.76, 0.59, 0.49), # cardboard_box_3*
    ]

    obstacles_poly = [
        ((-3.29, -3.41),(-2.28, -1.65),(-2.51, -1.58),(-3.42, -3.34)),  # 24
        ((-2.59, -1.66),(-1.49, -3.41),(-1.28, -3.23),(-2.38, -1.50)),  # 25
    ]

    # 判断矩形障碍物
    for ox, oy, length, width in obstacles_rect:
        if ox <= x <= ox + length and oy <= y <= oy + width:
            return False
    # 判断斜矩形障碍物
    for poly in obstacles_poly:
        if point_in_poly(x, y, poly):
            return False
    return True
