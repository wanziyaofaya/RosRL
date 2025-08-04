import math

# 解析 obstacle_2d_info.txt，存储所有障碍物信息
def load_obstacles(filename):
    obstacles = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if 'type:box' in line:
                # box type:box center:(x, y) size:(w, h) yaw:yaw corners:[(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
                name = line.split(' type:')[0]
                center = eval(line.split('center:')[1].split(')')[0] + ')')
                size = eval(line.split('size:')[1].split(')')[0] + ')')
                yaw = float(line.split('yaw:')[1].split()[0].replace(',', ''))
                corners = eval(line.split('corners:')[1])
                obstacles.append({'type': 'box', 'name': name, 'center': center, 'size': size, 'yaw': yaw, 'corners': corners})
            elif 'type:cylinder' in line:
                # table type:cylinder center:(x, y) radius:r yaw:yaw
                name = line.split(' type:')[0]
                center = eval(line.split('center:')[1].split(')')[0] + ')')
                radius = float(line.split('radius:')[1].split()[0].replace(',', ''))
                yaw = float(line.split('yaw:')[1].split()[0].replace(',', ''))
                obstacles.append({'type': 'cylinder', 'name': name, 'center': center, 'radius': radius, 'yaw': yaw})
            elif 'type:mesh' in line:
                # fire_hydrant type:mesh center:(x, y) yaw:yaw uri:... scale:(...) corners:...
                name = line.split(' type:')[0]
                center = eval(line.split('center:')[1].split(')')[0] + ')')
                yaw = float(line.split('yaw:')[1].split()[0].replace(',', ''))
                corners_str = line.split('corners:')[1].strip()
                if corners_str == 'None':
                    corners = None
                else:
                    corners = eval(corners_str)
                obstacles.append({'type': 'mesh', 'name': name, 'center': center, 'yaw': yaw, 'corners': corners})
            # 你可以根据需要添加 sphere 类型的解析
    return obstacles

# 点是否在多边形内（射线法）
def point_in_polygon(point, polygon):
    x, y = point
    n = len(polygon)
    inside = False
    px1, py1 = polygon[0]
    for i in range(n+1):
        px2, py2 = polygon[i % n]
        if min(py1, py2) < y <= max(py1, py2) and x <= max(px1, px2):
            if py1 != py2:
                xinters = (y - py1) * (px2 - px1) / (py2 - py1 + 1e-10) + px1
            if px1 == px2 or x <= xinters:
                inside = not inside
        px1, py1 = px2, py2
    return inside

# 点是否在圆内
def point_in_circle(point, center, radius):
    x, y = point
    cx, cy = center
    return (x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2

# 判断点是否与任一障碍物碰撞
def is_point_in_obstacle(point, obstacles):
    for obs in obstacles:
        if obs['type'] == 'box' and obs['corners']:
            if point_in_polygon(point, obs['corners']):
                return True
        elif obs['type'] == 'cylinder':
            if point_in_circle(point, obs['center'], obs['radius']):
                return True
        elif obs['type'] == 'mesh' and obs['corners']:
            if point_in_polygon(point, obs['corners']):
                return True
    return False

