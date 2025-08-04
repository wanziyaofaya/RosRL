import xml.etree.ElementTree as ET
import math
import os

# 解析SDF文件，提取所有障碍物的2D占地范围
# 支持box、mesh等类型

def parse_pose(pose_str):
    vals = [float(x) for x in pose_str.strip().split()]
    if len(vals) == 6:
        x, y, z, roll, pitch, yaw = vals
    else:
        x, y, z = vals[:3]
        roll = pitch = yaw = 0.0
    return x, y, z, roll, pitch, yaw

def rotate_point(x, y, theta):
    xr = x * math.cos(theta) - y * math.sin(theta)
    yr = x * math.sin(theta) + y * math.cos(theta)
    return xr, yr

def get_box_2d_corners(center, size, yaw):
    cx, cy = center
    sx, sy = size
    # 以中心为原点，计算四个角点
    corners = [
        (cx - sx/2, cy - sy/2),
        (cx - sx/2, cy + sy/2),
        (cx + sx/2, cy + sy/2),
        (cx + sx/2, cy - sy/2),
    ]
    # 旋转
    rotated = [rotate_point(x-cx, y-cy, yaw) for x, y in corners]
    rotated = [(x+cx, y+cy) for x, y in rotated]
    return rotated

def get_mesh_2d_bbox(mesh_path, scale=(1.0, 1.0, 1.0)):
    """
    解析dae/stl文件，返回mesh的2D包围盒四角点（未旋转，原点为中心）
    只支持dae格式，且只取x/y
    """
    if not os.path.exists(mesh_path):
        return None
    try:
        tree = ET.parse(mesh_path)
        root = tree.getroot()
        ns = ''
        if root.tag.startswith('{'):
            ns = root.tag.split('}')[0] + '}'
        # 兼容不同DAE写法，遍历所有float_array，找id包含positions的
        float_arrays = root.findall(f'.//{ns}float_array')
        pos_array = None
        for arr in float_arrays:
            if 'positions' in arr.attrib.get('id', ''):
                pos_array = arr
                break
        if pos_array is None:
            return None
        vals = [float(x) for x in pos_array.text.strip().split()]
        points = [(vals[i]*scale[0], vals[i+1]*scale[1]) for i in range(0, len(vals), 3)]
        xs = [x for x, y in points]
        ys = [y for x, y in points]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        # 以中心为原点
        cx = (min_x + max_x) / 2
        cy = (min_y + max_y) / 2
        sx = max_x - min_x
        sy = max_y - min_y
        corners = [
            (cx - sx/2, cy - sy/2),
            (cx - sx/2, cy + sy/2),
            (cx + sx/2, cy + sy/2),
            (cx + sx/2, cy - sy/2),
        ]
        return corners, (cx, cy)
    except Exception as e:
        print(f"mesh解析失败: {mesh_path}, {e}")
        return None

def extract_obstacles(sdf_path, save_path):
    tree = ET.parse(sdf_path)
    root = tree.getroot()
    ns = ''
    if root.tag.startswith('{'):
        ns = root.tag.split('}')[0] + '}'
    obstacles = []
    for model in root.findall(f'.//{ns}model'):
        model_name = model.attrib.get('name', '')
        for link in model.findall(f'{ns}link'):
            link_pose = link.find(f'{ns}pose')
            if link_pose is not None:
                lx, ly, lz, lroll, lpitch, lyaw = parse_pose(link_pose.text)
            else:
                lx = ly = lz = lroll = lpitch = lyaw = 0.0
            for collision in link.findall(f'{ns}collision'):
                col_pose = collision.find(f'{ns}pose')
                if col_pose is not None:
                    cx, cy, cz, croll, cpitch, cyaw = parse_pose(col_pose.text)
                else:
                    cx = cy = cz = croll = cpitch = cyaw = 0.0
                pose_x = lx + cx
                pose_y = ly + cy
                pose_yaw = lyaw + cyaw
                geom = collision.find(f'{ns}geometry')
                if geom is not None:
                    # box
                    box = geom.find(f'{ns}box')
                    if box is not None:
                        size = box.find(f'{ns}size').text.strip().split()
                        sx, sy = float(size[0]), float(size[1])
                        corners = get_box_2d_corners((pose_x, pose_y), (sx, sy), pose_yaw)
                        obstacles.append({
                            'model': model_name,
                            'type': 'box',
                            'center': (pose_x, pose_y),
                            'size': (sx, sy),
                            'yaw': pose_yaw,
                            'corners': corners
                        })
                    # cylinder
                    cylinder = geom.find(f'{ns}cylinder')
                    if cylinder is not None:
                        radius = float(cylinder.find(f'{ns}radius').text.strip())
                        length = float(cylinder.find(f'{ns}length').text.strip())
                        obstacles.append({
                            'model': model_name,
                            'type': 'cylinder',
                            'center': (pose_x, pose_y),
                            'radius': radius,
                            'yaw': pose_yaw
                        })
                    # sphere
                    sphere = geom.find(f'{ns}sphere')
                    if sphere is not None:
                        radius = float(sphere.find(f'{ns}radius').text.strip())
                        obstacles.append({
                            'model': model_name,
                            'type': 'sphere',
                            'center': (pose_x, pose_y),
                            'radius': radius
                        })
                    # mesh
                    mesh = geom.find(f'{ns}mesh')
                    if mesh is not None:
                        uri = mesh.find(f'{ns}uri').text.strip()
                        scale_elem = mesh.find(f'{ns}scale')
                        if scale_elem is not None:
                            scale = tuple(float(x) for x in scale_elem.text.strip().split())
                        else:
                            scale = (1.0, 1.0, 1.0)
                        # 解析mesh文件，获得2D包围盒
                        mesh_path = uri
                        if uri.startswith('model://'):
                            # 只处理本地相对路径
                            mesh_path = uri.replace('model://', '../catkin_ws/src/multi_robot_scenario/meshes/')
                        bbox = get_mesh_2d_bbox(mesh_path, scale)
                        if bbox:
                            corners, mesh_center = bbox
                            # 旋转+平移
                            rotated = [rotate_point(x, y, pose_yaw) for x, y in corners]
                            rotated = [(x+pose_x, y+pose_y) for x, y in rotated]
                        else:
                            rotated = None
                        obstacles.append({
                            'model': model_name,
                            'type': 'mesh',
                            'center': (pose_x, pose_y),
                            'yaw': pose_yaw,
                            'uri': uri,
                            'scale': scale,
                            'corners': rotated
                        })
    # 保存为txt
    with open(save_path, 'w') as f:
        for obs in obstacles:
            if obs['type'] == 'box':
                f.write(f"{obs['model']} type:box center:{obs['center']} size:{obs['size']} yaw:{obs['yaw']} corners:{obs['corners']}\n")
            elif obs['type'] == 'cylinder':
                f.write(f"{obs['model']} type:cylinder center:{obs['center']} radius:{obs['radius']} yaw:{obs['yaw']}\n")
            elif obs['type'] == 'sphere':
                f.write(f"{obs['model']} type:sphere center:{obs['center']} radius:{obs['radius']}\n")
            elif obs['type'] == 'mesh':
                f.write(f"{obs['model']} type:mesh center:{obs['center']} yaw:{obs['yaw']} uri:{obs['uri']} scale:{obs['scale']} corners:{obs['corners']}\n")
    print(f"提取完成，障碍物数量: {len(obstacles)}，已保存到 {save_path}")

if __name__ == '__main__':
    sdf_path = '../catkin_ws/src/multi_robot_scenario/launch/TD3.world'
    save_path = 'obstacle_2d_info.txt'
    extract_obstacles(sdf_path, save_path)
