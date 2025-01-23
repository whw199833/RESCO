from collections import defaultdict
class JunctionManager:
    def __init__(self, sumo, degree):
        self.sumo = sumo
        self.degree = degree  # 共享范围
        self.junctions = self._init_junctions()  # 初始化所有路口信息
        self.shared_junctions = self._init_shared_junctions()  # 计算共享范围内的路口

    def _init_junctions(self):
        """初始化所有路口信息，并获取每个路口的相邻路口"""
        junctions = {}
        for junction_id in self.sumo.junction.getIDList():
            position = self.sumo.junction.getPosition(junction_id)
            adjacent_junctions = self._get_adjacent_junctions(junction_id)
            junctions[junction_id] = {
                'junction_id': junction_id,
                'position': position,
                'adjacent': adjacent_junctions,  # 相邻路口
                'signals': []  # 属于该路口的信号灯
            }
        return junctions

    def _get_adjacent_junctions(self, junction_id):
        """获取与指定路口直接相邻的其他路口"""
        adjacent_junctions = set()
        all_edges = self.sumo.edge.getIDList()
        for edge in all_edges:
            from_junction = self.sumo.edge.getFromJunction(edge)
            to_junction = self.sumo.edge.getToJunction(edge)
            if from_junction == junction_id:
                adjacent_junctions.add(to_junction)
            elif to_junction == junction_id:
                adjacent_junctions.add(from_junction)
        return list(adjacent_junctions)

    def _init_shared_junctions(self):
        """计算每个路口在指定度范围内的其他路口"""
        shared_junctions = defaultdict(list)
        for junction_id in self.junctions:
            shared_junctions[junction_id] = self._bfs_shared_junctions(junction_id, self.degree)
        return shared_junctions

    def _bfs_shared_junctions(self, start_junction, degree):
        """使用 BFS 计算指定度范围内的路口"""
        visited = set()
        queue = [(start_junction, 0)]  # (路口, 当前度)
        shared = []

        while queue:
            current_junction, current_degree = queue.pop(0)
            if current_junction in visited:
                continue
            visited.add(current_junction)

            if current_degree > 0:  # 跳过起点
                shared.append(current_junction)

            if current_degree < degree:
                for neighbor in self.junctions[current_junction]['adjacent']:
                    queue.append((neighbor, current_degree + 1))

        return shared

    def get_shared_junctions(self, junction_id):
        """获取与指定路口在共享范围内的其他路口"""
        return self.shared_junctions.get(junction_id, [])

    def get_junction_by_id(self, junction_id):
        """根据路口 ID 获取路口信息"""
        for junction in self.junctions:
            if junction['junction_id'] == junction_id:
                return junction
        raise ValueError(f"Junction {junction_id} not found.")

    def get_junction_by_agent(self, agent_idx):
        """根据智能体索引获取对应的路口"""
        for junction in self.junctions:
            if agent_idx in junction['signals']:
                return junction
        raise ValueError(f"Agent {agent_idx} does not belong to any junction.")