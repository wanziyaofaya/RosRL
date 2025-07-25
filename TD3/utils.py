"""
@file: node.py
@breif: 2-dimension node data stucture
@author: Yang Haodong, Wu Maojia
@update: 2024.3.15
"""
from abc import abstractmethod, ABC
import math
class Node(object):
    """
    Class for searching nodes.

    Parameters:
        current (tuple): current coordinate
        parent (tuple): coordinate of parent node
        g (float): path cost
        h (float): heuristic cost

    Examples:
        >>> from env import Node
        >>> node1 = Node((1, 0), (2, 3), 1, 2)
        >>> node2 = Node((1, 0), (2, 5), 2, 8)
        >>> node3 = Node((2, 0), (1, 6), 3, 1)
        ...
        >>> node1 + node2
        >>> Node((2, 0), (2, 3), 3, 2)
        ...
        >>> node1 == node2
        >>> True
        ...
        >>> node1 != node3
        >>> True
    """
    def __init__(self, current: tuple, parent: tuple = None, g: float = 0, h: float = 0) -> None:
        self.current = current
        self.parent = parent
        self.g = g
        self.h = h
    
    def __add__(self, motion):
        if isinstance(motion, Node):
            return Node((self.x + motion.x, self.y + motion.y), self.parent, self.g + motion.g, self.h)
        elif isinstance(motion, tuple) and len(motion) == 3:
            # motion格式: (dx, dy, cost)
            dx, dy, cost = motion
            return Node((self.x + dx, self.y + dy), self.parent, self.g + cost, self.h)
        else:
            raise ValueError("Invalid motion format")

    def __eq__(self, node) -> bool:
        if not isinstance(node, Node):
            return False
        return self.current == node.current
    
    def __ne__(self, node) -> bool:
        return not self.__eq__(node)

    def __lt__(self, node) -> bool:
        assert isinstance(node, Node)
        return self.g + self.h < node.g + node.h or \
                (self.g + self.h == node.g + node.h and self.h < node.h)

    def __hash__(self) -> int:
        return hash(self.current)

    def __str__(self) -> str:
        return "Node({}, {}, {}, {})".format(self.current, self.parent, self.g, self.h)

    def __repr__(self) -> str:
        return self.__str__()
    
    @property
    def x(self) -> float:
        return self.current[0]
    
    @property
    def y(self) -> float:
        return self.current[1]

    @property
    def px(self) -> float:
        if self.parent:
            return self.parent[0]
        else:
            return None

    @property
    def py(self) -> float:
        if self.parent:
            return self.parent[1]
        else:
            return None

class Env:
    """Environment class placeholder."""
    def __init__(self):
        pass        

class Planner(ABC):
    def __init__(self, start: tuple, goal: tuple, env: Env) -> None:
        # plannig start and goal
        self.start = Node(start, start, 0, 0)
        self.goal = Node(goal, goal, 0, 0)
        # environment
        self.env = env
        # graph handler

    def dist(self, node1: Node, node2: Node) -> float:
        return math.hypot(node2.x - node1.x, node2.y - node1.y)
    
    def angle(self, node1: Node, node2: Node) -> float:
        return math.atan2(node2.y - node1.y, node2.x - node1.x)

    @abstractmethod
    def plan(self):
        '''
        Interface for planning.
        '''
        pass

    @abstractmethod
    def run(self):
        '''
        Interface for running both plannig and animation.
        '''
        pass

class Grid:
    """Grid environment for path planning."""
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.grid = [[0 for _ in range(height)] for _ in range(width)]
        self.obstacles = set()
        
        # 8-connected motion model
        self.motions = [
            (1, 0, 1.0),      # 右
            (-1, 0, 1.0),     # 左
            (0, 1, 1.0),      # 上
            (0, -1, 1.0),     # 下
            (1, 1, 1.414),    # 右上
            (-1, 1, 1.414),   # 左上
            (1, -1, 1.414),   # 右下
            (-1, -1, 1.414),  # 左下
        ]
        
    def add_obstacle(self, x, y):
        """Add obstacle at grid position (x, y)"""
        if 0 <= x < self.width and 0 <= y < self.height:
            self.grid[x][y] = 1
            self.obstacles.add((x, y))
    
    def is_obstacle(self, x, y):
        """Check if position (x, y) is an obstacle"""
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return True
        return self.grid[x][y] == 1

class Map:
    """Placeholder class for Map."""
    def __init__(self, x_range, y_range, obs_rect=None, obs_circ=None, boundary=None):
        self.x_range = x_range
        self.y_range = y_range
        self.obs_rect = obs_rect or []
        self.obs_circ = obs_circ or []
        self.boundary = boundary or [(x_range[0], y_range[0], x_range[1] - x_range[0], y_range[1] - y_range[0])]