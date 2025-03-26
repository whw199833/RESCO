from collections import defaultdict
from resco_benchmark.config.map_config import map_configs


class JunctionManager:
    def __init__(self, sumo, config_name):
        self.sumo = sumo
        self.junctions = self._init_junctions()  # 初始化所有路口信息
        self.name = config_name
        self._degree_cache = {}

    def _init_junctions(self):
        junctions = {}
        # 获取所有合法路口ID（官方注册的路口）
        valid_junc_ids = set(self.sumo.junction.getIDList())

        # 收集所有通过边发现的路口（可能包含非标准路口）
        edge_junc_ids = set()
        for edge in self.sumo.edge.getIDList():
            edge_junc_ids.add(self.sumo.edge.getFromJunction(edge))
            edge_junc_ids.add(self.sumo.edge.getToJunction(edge))

        # 合并所有可能的ID，但过滤非法路口
        all_junc_ids = valid_junc_ids.union(edge_junc_ids)

        for junc_id in all_junc_ids:
            # 改进后的存在性检查
            is_valid = junc_id in valid_junc_ids
            position = self.sumo.junction.getPosition(junc_id) if is_valid else (0, 0)

            junctions[junc_id] = {
                'junction_id': junc_id,
                'position': position,
                'adjacent': self._get_adjacent_junctions(junc_id) if is_valid else [],
                'signals': [],
                'is_valid': is_valid  # 新增有效性标识
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

    def _bfs_shared_junctions(self, start_junction, degree):
        # 缓存优化
        if (start_junction, degree) in self._degree_cache:
            return self._degree_cache[(start_junction, degree)]

        visited = set()
        queue = [(start_junction, 0)]
        shared = []

        while queue:
            current, current_degree = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)

            if current_degree > 0:
                shared.append(current)

            if current_degree < degree:
                # 预加载邻接关系
                if current not in self.junctions:
                    continue
                neighbors = self.junctions[current]['adjacent']
                queue.extend((n, current_degree + 1) for n in neighbors)

        # 缓存结果
        self._degree_cache[(start_junction, degree)] = shared
        return shared

    def get_shared_junctions(self, junction_id, degree):
        """动态计算指定degree范围内的共享路口"""
        return self._bfs_shared_junctions(junction_id, degree)
