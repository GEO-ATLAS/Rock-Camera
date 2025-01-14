import numpy as np
from matplotlib.colors import hsv_to_rgb
def create_obb_from_diagonal_and_width(p1, p2, width):
    # 计算OBB的中心点
    obb_center = (p1 + p2) / 2.0

    # 计算OBB的半边长
    obb_half_lengths = np.linalg.norm(p2 - p1) / 2.0  # 对角线长度的一半

    # 计算OBB的方向（对角线的方向）
    obb_direction = (p2 - p1) / np.linalg.norm(p2 - p1)

    # 计算OBB的宽度
    obb_width = width / 2.0

    # 返回OBB的中心点、半边长、方向和宽度
    return obb_center, obb_half_lengths, obb_direction, obb_width

def calculate_obb_corners(obb_center, obb_half_lengths, obb_direction, obb_width):
    # 计算OBB的主轴
    obb_axis1 = obb_direction
    obb_axis2 = np.array([-obb_direction[1], obb_direction[0]])  # 垂直于主轴的向量

    # 计算OBB的四个顶点
    obb_vertex1 = obb_center + obb_axis1 * obb_half_lengths + obb_axis2 * obb_width
    obb_vertex2 = obb_center + obb_axis1 * obb_half_lengths - obb_axis2 * obb_width
    obb_vertex3 = obb_center - obb_axis1 * obb_half_lengths - obb_axis2 * obb_width
    obb_vertex4 = obb_center - obb_axis1 * obb_half_lengths + obb_axis2 * obb_width

    # 以顺时针排序四个顶点
    obb_corners_clockwise = np.array([obb_vertex1, obb_vertex2, obb_vertex3, obb_vertex4])

    return obb_corners_clockwise

def normalDegree(rad):
    if rad <0:
        return rad+np.pi
    return rad
def angle_to_color(rad):
    hue = rad / np.deg2rad(220)  # Normalize angle to [0, 1]
    saturation = 1.0
    value = 1.0
    hsv_color = np.array([hue, saturation, value])
    rgb_color = hsv_to_rgb(hsv_color)
    scaled_color = tuple(int(val * 255) for val in rgb_color)
    #使用BGR=，cv中使用BGR
    return (scaled_color[2],scaled_color[1],scaled_color[0])