import heapq
from graph_search import GraphSearcher
from utils import Env, Grid, Node


class AStar(GraphSearcher):
    def run(self):
        pass

    def __init__(self, start: tuple, goal: tuple, env: Grid, heuristic_type: str = "euclidean", weight: float = 1.1) -> None:
        super().__init__(start, goal, env, heuristic_type)
        self.weight = weight  # 启发式权重，大于1会使路径更倾向于直线
        self.length = len(env.grid)  # 网格地图的长度
        self.width = len(env.grid[0])  # 网格地图的宽度
        self.environment = env.grid  # 添加环境网格数据

    def __str__(self) -> str:
        return "A*"

    def plan(self) -> tuple:
        """
        A*路径规划主函数，返回总代价、路径点列表（终点到起点）、扩展节点列表。
        """
        # 检查起点和终点有效性
        if not self.isValid(self.start.current):
            print(f"起点无效: {self.start.current}")
            return float('inf'), [], []
        if not self.isValid(self.goal.current):
            print(f"终点无效: {self.goal.current}")
            return float('inf'), [], []

        OPEN = []
        heapq.heappush(OPEN, self.start)
        CLOSED = dict()
        g_score = {self.start.current: 0}


        goal_threshold = 2.0  # 允许的到达目标距离
        while OPEN:
            node = heapq.heappop(OPEN)
            # 跳过已扩展过的节点
            if node.current in CLOSED:
                continue
            CLOSED[node.current] = node

            # 到达目标（允许一定距离误差）
            if self.dist(node, self.goal) < goal_threshold:
                # print("CLOSED:", CLOSED)
                cost, path = self.extractPath(CLOSED, node)
                # 路径顺序为终点到起点
                return cost, path, list(CLOSED.values())

            # 扩展邻居
            for node_n in self.getNeighbor(node):
                tentative_g = node.g + self.dist(node, node_n)
                if node_n.current in g_score and tentative_g >= g_score[node_n.current]:
                    continue
                g_score[node_n.current] = tentative_g
                node_n.g = tentative_g
                node_n.parent = node.current
                node_n.h = self.weight * self.h(node_n, self.goal)
                heapq.heappush(OPEN, node_n)

        # 没找到路径
        print("A*未找到可行路径")
        return float('inf'), [], list(CLOSED.values())

    def getNeighbor(self, node: Node) -> list:
        neighbors = []
        
        # 只定义8个方向（上下左右+对角线）
        motions = [
            (-1, -1, 1.414), (-1, 0, 1), (-1, 1, 1.414),
            (0, -1, 1),               (0, 1, 1),
            (1, -1, 1.414),  (1, 0, 1), (1, 1, 1.414)
        ]
        
        x, y = node.current  # 获取当前节点的坐标
        
        for dx, dy, cost in motions:
            new_x = x + dx
            new_y = y + dy
            new_pos = (new_x, new_y)
            # 检查新节点是否有效，且与当前节点连线无碰撞
            if self.isValid(new_pos) and self.isPathClear((x, y), new_pos):
                new_node = Node(new_pos, node.current, node.g + cost, 0)
                neighbors.append(new_node)
        
        return neighbors

    def extractPath(self, closed_list: dict, end_node=None) -> tuple:
        # 获取原始路径
        cost = 0
        if end_node is None:
            node = closed_list[self.goal.current]
        else:
            node = end_node
        path = [node.current]
        # 收集原始路径点（从终点到起点）
        while node.current != self.start.current:
            if node.parent not in closed_list:
                print("Warning: Path extraction failed - parent node not found")
                return float('inf'), []
            node_parent = closed_list[node.parent]
            cost += self.dist(node, node_parent)
            node = node_parent
            path.append(node.current)
        # 路径简化（去冗余）
        pruned_path = self.prune_path(path[::-1])  # 反转为起点到终点，去冗余
        return cost, pruned_path[::-1]  # 再反转回终点到起点

    def prune_path(self, path):
        """
        路径简化：去除冗余节点，输入为起点到终点顺序，输出为简化后的路径（起点到终点顺序）
        """
        if len(path) <= 2:
            return path[:]
        pruned = [path[0]]
        i = 0
        while i < len(path) - 1:
            j = i + 1
            # 找到最远可以直连的点
            while j < len(path) and self.isPathClear(path[i], path[j]):
                j += 1
            # j-1是最后一个可直连的点
            pruned.append(path[j-1])
            i = j - 1
        return pruned
        

    def isValid(self, point: tuple) -> bool:
        x, y = point
        # 检查是否在网格范围内
        if not (0 <= x < self.length and 0 <= y < self.width):
            return False
        # 检查是否为障碍物
        if self.environment[int(round(x))][int(round(y))] == 1:
            return False
        return True
        
    def isPathClear(self, start_point, end_point) -> bool:

        """
        支持浮点坐标的路径可行性检测，采用DDA采样法。
        """
        x1, y1 = start_point
        x2, y2 = end_point
        dx = x2 - x1
        dy = y2 - y1
        steps = int(max(abs(dx), abs(dy)) * 30) 

        if steps == 0:
            grid_x, grid_y = int(round(x1)), int(round(y1))
            return self.isValid((grid_x, grid_y))

        x_inc = dx / steps
        y_inc = dy / steps

        x, y = x1, y1
        for _ in range(steps + 1):
            grid_x, grid_y = int(round(x)), int(round(y))
            if not self.isValid((grid_x, grid_y)):
                return False
            x += x_inc
            y += y_inc
        return True