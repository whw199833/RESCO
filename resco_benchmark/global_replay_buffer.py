from collections import defaultdict
from pfrl import replay_buffer
class GlobalReplayBuffer():
    def __init__(self, junction_manager):
        self.buffer = []  # 全局缓冲区
        self.index_map = defaultdict(list)  # 路口索引映射
        self.junction_manager = junction_manager  # 路口信息管理器

    def append(self, transition, intersection_id):
        """将经验添加到全局缓冲区，并更新索引"""
        self.buffer.append(transition)
        self.index_map[intersection_id].append(len(self.buffer) - 1)

    def sample(self, intersection_id, degree):
        """从全局缓冲区中采样属于指定路口及其共享范围内的经验"""
        shared_junctions = self.junction_manager.get_shared_junctions(intersection_id)
        indices = []
        for junction in shared_junctions:
            indices.extend(self.index_map[junction])
        return [self.buffer[i] for i in indices]