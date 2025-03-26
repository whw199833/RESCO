import numpy as np
from collections import defaultdict
import torch


class GlobalReplayBuffer:
    def __init__(self, junction_manager):
        self._alpha = 0.6
        self.junction_manager = junction_manager
        self.capacity = int(5e5)  # 缩小容量适配短期训练
        self.alpha = 0.9  # 更激进的衰减系数
        self.age = torch.zeros(self.capacity, dtype=torch.short, device='cuda')  # 使用short节省显存
        self.junc_exp_map = defaultdict(list)
        self.global_step = 0

        # GPU缓存优化
        self._gpu_priorities = torch.zeros(self.capacity, device='cuda')
        self._gpu_valid_mask = torch.zeros(self.capacity, dtype=torch.bool, device='cuda')
        self._gpu_exp_types = torch.zeros(self.capacity, dtype=torch.long, device='cuda')
        self.junc_exp_map_gpu = defaultdict(lambda: torch.empty(0, dtype=torch.long, device='cuda'))

        # 主存储结构
        self._storage = []
        self._priorities = np.full(self.capacity, 1e-8, dtype=np.float32)
        self._index = 0
        self._max_priority = 1.0

        self._exp_type_map = {
            "meta_degree": 0,
            "meta_actions": 1,
            "controller": 2
        }
        self.weights = {'controller': 1.0, 'meta_degree': 0.6, 'meta_actions': 0.3}  # 新增权重

    def step(self):
        """应每个环境步调用一次"""
        self.global_step += 1
        # 每10步执行年龄衰减
        if self.global_step % 10 == 0:
            decay_mask = (self.age > 30)  # 更激进阈值
            self._gpu_priorities[decay_mask] *= self.alpha
            self.age[decay_mask] = 0

    def __len__(self):
        return len(self._storage)

    def _update_gpu_cache(self):
        """同步CPU到GPU的关键数据"""
        if len(self._storage) == 0:
            return

        # 增量更新优先级和类型
        start_idx = max(0, len(self._storage) - 10000)
        end_idx = len(self._storage)

        cpu_priorities = np.nan_to_num(
            self._priorities[start_idx:end_idx],
            nan=1e-8,
            posinf=1e8,
            neginf=1e-8
        )
        self._gpu_priorities[start_idx:end_idx] = torch.from_numpy(cpu_priorities).to('cuda')

        exp_types = [self._exp_type_map[e.get('exp_type', 'controller')]
                     for e in self._storage[start_idx:end_idx]]
        self._gpu_exp_types[start_idx:end_idx] = torch.tensor(exp_types, device='cuda')

        # 同步路口索引映射（全量更新）
        for jid in self.junc_exp_map:
            cpu_indices = np.array(self.junc_exp_map[jid], dtype=np.int64)
            self.junc_exp_map_gpu[jid] = torch.from_numpy(cpu_indices).to('cuda')

    def _apply_priority_decay(self):
        """每100步触发的衰减操作"""
        decay_mask = (self.age > 50)  # 更短的衰减年龄阈值
        self._gpu_priorities[decay_mask] *= self.alpha
        self.age[decay_mask] = 0  # 重置年龄

    def append(self, junction_id, transition, priority):
        idx = self._index % self.capacity

        # 动态衰减（每100步）
        if self._index % 100 == 0:
            self._apply_priority_decay()

        # 更新存储和年龄
        if len(self._storage) < self.capacity:
            self._storage.append(transition)
            self.age[idx] = 0  # 新增经验年龄初始化
        else:
            old_junc_id = self._storage[idx].get('junction_id')
            if old_junc_id in self.junc_exp_map:
                self.junc_exp_map[old_junc_id].remove(idx)
            self._storage[idx] = transition
            self.age[idx] = 0  # 覆盖时重置年龄

        self.junc_exp_map[junction_id].append(idx)
        self._priorities[idx] = priority
        self._max_priority = max(self._max_priority, priority)

        # 减少同步频率
        if self._index % 256== 0:  # 每10步同步
            self._update_gpu_cache()

        self._index += 1

    def sample_by_degree(self, junction_id, degree, batch_size, beta=0.4, exp_type="controller"):
        # 获取邻接路口（带缓存）
        if (junction_id, degree) not in self.junction_manager._degree_cache:
            self.junction_manager._degree_cache[(junction_id, degree)] = (
                self.junction_manager.get_shared_junctions(junction_id, degree)
            )
        neighbor_juncs = self.junction_manager._degree_cache[(junction_id, degree)]

        # 合并GPU索引
        candidate_indices = torch.cat([
            self.junc_exp_map_gpu[jid]
            for jid in neighbor_juncs + [junction_id]
        ], dim=0).unique()

        if len(candidate_indices) == 0:
            return [], [], []
        type_code = self._exp_type_map[exp_type]
        valid_mask = self._gpu_exp_types[candidate_indices] == type_code
        candidate_indices = candidate_indices[valid_mask]
        # GPU概率计算
        priorities = self._gpu_priorities[candidate_indices]
        uniform_probs = torch.ones_like(priorities) / len(candidate_indices)
        probs = 0 * torch.softmax(priorities, dim=0) + 1 * uniform_probs

        # 采样
        sample_size = min(batch_size, len(candidate_indices))
        selected_idx = torch.multinomial(probs, sample_size, replacement=False)
        selected_indices = candidate_indices[selected_idx].cpu().numpy()

        # 更新年龄计数器
        self.age[candidate_indices[selected_idx]] = 0  # 重置被采样经验的年龄

        return (
            [self._storage[i] for i in selected_indices],
            [],  # 简化短期训练的权重计算
            selected_indices.tolist()
        )

    def sample_all(self, batch_size, beta=0.4, exp_type=None):
        # 类型筛选
        if exp_type is not None:
            type_code = self._exp_type_map[exp_type]
            valid_mask = self._gpu_exp_types[:len(self._storage)] == type_code
            valid_indices = torch.where(valid_mask)[0]
        else:
            valid_indices = torch.arange(len(self._storage), device='cuda')

        if len(valid_indices) == 0:
            return [], [], []

        # 动态衰减处理（采样时自动应用）
        age_mask = self.age[valid_indices] > 30  # 超过100步未采样
        self._gpu_priorities[valid_indices[age_mask]] *= 0.85  # 应用衰减

        # 混合采样策略（优先级+均匀）
        priorities = self._gpu_priorities[valid_indices]

        # 优先级部分（60%）
        priority_probs = torch.softmax(priorities * 0.5, dim=0)

        # 均匀部分（40%）
        uniform_probs = torch.ones_like(priorities) / len(valid_indices)

        # 组合概率
        probs = 0 * priority_probs + 1 * uniform_probs

        # 采样
        sample_size = min(batch_size, len(valid_indices))
        selected_idx = torch.multinomial(probs, sample_size, replacement=False)
        selected_indices = valid_indices[selected_idx].cpu().numpy()

        # 更新年龄计数器（重置被采样经验的年龄）
        self.age[valid_indices[selected_idx]] = 0

        # 权重计算（带温度系数）
        weights = (len(valid_indices) * probs[selected_idx]) ** (-beta)
        weights /= weights.max() + 1e-8

        return (
            [self._storage[i] for i in selected_indices],
            weights.cpu().numpy(),
            selected_indices.tolist()
        )

    def update_priorities(self, indices, new_priorities, agent_type):
        indices_tensor = torch.tensor(indices, dtype=torch.long, device='cuda')
        new_priorities = torch.tensor(new_priorities, device='cuda') * self.weights[agent_type]

        # 原子最大操作+年龄感知衰减
        current_p = self._gpu_priorities[indices_tensor]
        blended_p = torch.maximum(current_p, new_priorities)
        final_p = torch.where(
            self.age[indices_tensor] > 30,  # 年龄超过阈值时衰减
            blended_p * self.alpha,
            blended_p
        )
        self._gpu_priorities[indices_tensor] = final_p
        self._max_priority = max(self._max_priority, final_p.max().item())

        # 异步回写CPU
        if self._index % 100 == 0:
            self._priorities = self._gpu_priorities.cpu().numpy()