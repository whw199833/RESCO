from collections import defaultdict
from typing import Any, Sequence
import numpy as np

import torch
import torch.nn as nn
from torch.optim import AdamW

from resco_benchmark.dqn_old3 import DQN, MetaDQN
from pfrl.q_functions import DiscreteActionValueHead

from resco_benchmark.agents.agent import IndependentAgent, Agent
from resco_benchmark.global_replay_buffer import GlobalReplayBuffer
import torch.nn.functional as F


class ActionFusion(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim * 2, 64),  # 输入维度扩展为双倍
            nn.ReLU(),
            nn.Linear(64, 1),  # 压缩到单一注意力权重
            nn.Sigmoid()  # 输出[0,1]区间
        )

    def forward(self, sug_Q, Q):
        combined = torch.cat([sug_Q, Q], dim=-1)  # 拼接两种Q值
        attn_weights = self.attention(combined)  # 生成动态权重
        return attn_weights * sug_Q + (1 - attn_weights) * Q


class CrossAttention(nn.Module):
    def __init__(self, dim=128):
        super().__init__()
        self.dim = dim
        self.num_heads = 4
        self.head_dim = dim // 4

        # 查询、键、值投影
        self.q_proj = nn.Linear(dim, dim)

        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)

        # 门控机制
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Linear(dim, 1),
            nn.Sigmoid()
        )

        # 初始化参数
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, shared_feature, feature):
        # 移除多余的view操作
        batch_size = shared_feature.size(0)
        dim = shared_feature.size(-1)

        # 投影操作保持二维
        q = self.q_proj(shared_feature)  # [batch, dim]
        k = self.k_proj(feature)  # [batch, dim]
        v = self.v_proj(feature)  # [batch, dim]

        # 调整多头维度
        q = q.view(batch_size, self.num_heads, self.head_dim).transpose(0, 1)  # [heads, batch, head_dim]
        k = k.view(batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        v = v.view(batch_size, self.num_heads, self.head_dim).transpose(0, 1)

        # 计算注意力分数
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)  # [heads, batch, batch]
        attn_weights = F.softmax(attn_scores, dim=-1)

        # 值融合
        fused = torch.matmul(attn_weights, v)  # [heads, batch, head_dim]
        fused = fused.transpose(0, 1).contiguous().view(batch_size, dim)  # [batch, dim]
        # 值融合

        gate = self.gate(torch.cat([shared_feature, fused], dim=1))
        # 残差连接
        return gate * fused + (1 - gate) * shared_feature


class BufferGroupManager:
    def __init__(self, junction_manager):
        self.junction_manager = junction_manager
        self.group_buffers = {}  # {(obs_shape, act_space): GlobalReplayBuffer}
        self.agent_groups = {}  # {agent_id: (obs_shape, act_space)}

    def get_group_key(self, obs_space, act_space):
        """生成唯一分组标识"""
        obs_key = tuple(obs_space) if isinstance(obs_space, (list, tuple)) else (obs_space,)
        return obs_key, act_space

    def register_agent(self, agent_id, obs_space, act_space):
        """注册智能体到对应分组"""
        group_key = self.get_group_key(obs_space, act_space)
        self.agent_groups[agent_id] = group_key

        # 为新的分组创建buffer
        if group_key not in self.group_buffers:
            self.group_buffers[group_key] = GlobalReplayBuffer(self.junction_manager)
        return group_key

    def get_buffer(self, agent_id):
        """获取智能体所属分组的buffer"""
        group_key = self.agent_groups[agent_id]
        return self.group_buffers[group_key]


class IDQN(IndependentAgent):
    def __init__(self, config, obs_act, map_name, thread_number, junction_manager=None):

        super().__init__(config, obs_act, map_name, thread_number)

        self.junction_manager = junction_manager
        self.buffer_group_manager = BufferGroupManager(junction_manager)
        group_counter = defaultdict(list)
        for key in obs_act:
            obs_space = obs_act[key][0]
            act_space = obs_act[key][1]

            group_key = self.buffer_group_manager.register_agent(key, obs_space, act_space)
            group_counter[group_key].append(key)

        # 打印分组信息
        print("Agent Grouping Results:")
        for i, (group_key, agents) in enumerate(group_counter.items()):
            print(f"Group {i + 1} (obs: {group_key[0]}, act: {group_key[1]}):")
            print(f"  Agents: {agents}")

        def conv2d_size_out(size, kernel_size=2, stride=1):
            return (size - (kernel_size - 1) - 1) // stride + 1

        for key in obs_act:
            obs_space = obs_act[key][0]
            act_space = obs_act[key][1]
            self.CrossAttention = CrossAttention()
            self.group_buffer = self.buffer_group_manager.get_buffer(key)
            h = conv2d_size_out(obs_space[1])
            w = conv2d_size_out(obs_space[2])
            self.model_1 = nn.Sequential(
                nn.Conv2d(obs_space[0], 64, kernel_size=(2, 2)),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(h * w * 64, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, act_space),
                DiscreteActionValueHead()
            )
            shared_layers = nn.Sequential(
                nn.Conv2d(obs_space[0], 64, kernel_size=(2, 2)),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(h * w * 64, 128),

            )

            # 显式定义双头结构
            degree_head = nn.Sequential(
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 4),  # 输出degree选项
                DiscreteActionValueHead()
            )

            action_head = nn.Sequential(
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, act_space),  # 与低层动作空间对齐
                DiscreteActionValueHead()
            )
            self.feat_adapter = nn.Linear(64, 128).to(self.device)
            # 组合成model_2
            self.model_2 = nn.ModuleList([
                shared_layers,
                self.CrossAttention,
                degree_head,
                action_head
            ])
            self.fusion = ActionFusion(input_dim=act_space).to(self.device)
            self.agents[key] = DQNAgent(config, act_space, self.model_1, self.model_2, self.fusion, self.feat_adapter,
                                        num_agents=0,
                                        junction_manager=self.junction_manager, key=key,
                                        global_buffer=self.group_buffer)
            if self.config['load']:
                print('LOADING SAVED MODEL FOR EVALUATION')
                self.agents[key].load(self.config['log_dir'] + 'agent_' + key + '.pt')
                self.agents[key].agent.training = False


class DQNAgent(Agent):
    def __init__(self, config, act_space, model_1, model_2, fusion, feat_adapter, num_agents=0, junction_manager=None,
                 key=None,
                 global_buffer=None):
        super().__init__()
        self.current_degree = None
        self.junction_manager = junction_manager
        self.global_buffer = global_buffer
        self.num_agents = num_agents
        self.feat_adapter = feat_adapter
        self.key = key
        self.global_buffer = global_buffer
        self.ctrl_model = model_1
        self.meta_model = model_2
        self.fusion = fusion
        self.optimizer_1 = torch.optim.Adam([
            {'params': self.ctrl_model.parameters(), 'lr': 1e-3}])
        self.shared_optim = torch.optim.Adam([
            {'params': self.meta_model[0].parameters()},  # 共享卷积层
            {'params': self.meta_model[1].parameters()},  # CrossAttention
            {'params': self.feat_adapter.parameters()},  # 新增适配器参数
            {'params': self.fusion.parameters()}  # 确保融合层参数被包含
        ], lr=1e-3, weight_decay=1e-3)
        self.optimizer_degree = torch.optim.Adam([  # 使用新型优化器
            {'params': self.meta_model[2].parameters(), 'lr': 1e-3}
        ])  # 启用梯度中心化

        self.optimizer_action = torch.optim.Adam([
            {'params': self.meta_model[3].parameters(), 'lr': 1e-3}
        ])
        self.explorer = DualHeadEpsilonGreedy(
            degree_space_size=4,
            action_space_size=act_space,
            start_epsilon=1.0,  # 更高初始探索率
            end_epsilon=0.05,  # 更低最终探索率
            decay_steps=5000,  # 更缓慢的衰减
            head_scale=0.5  # 差异化探索强度
        )

        self.agent = MetaDQN(
            self.meta_model, self.optimizer_degree, self.optimizer_action, self.shared_optim, self.feat_adapter,
            self.global_buffer,
            config['GAMMA'],
            explorer=self.explorer,
            gpu=self.device.index, minibatch_size=config['BATCH_SIZE'],
            replay_start_size=1000, phi=lambda x: np.asarray(x, dtype=np.float32),
            target_update_interval=config['TARGET_UPDATE'], key=self.key,
        )
        self.controller = DQN(
            self.ctrl_model, self.fusion, self.optimizer_1, self.global_buffer, config['GAMMA'], explorer=self.explorer,
            gpu=self.device.index, minibatch_size=config['BATCH_SIZE'],
            replay_start_size=1000, phi=lambda x: np.asarray(x, dtype=np.float32),
            target_update_interval=config['TARGET_UPDATE'], key=self.key, degree=3, meta_controller=self.agent,
        )
        self.agent.link_controller(self.controller)
        all_params = set(id(p) for p in self.meta_model.parameters())
        optim_params = set()
        for group in self.shared_optim.param_groups:
            optim_params.update(id(p) for p in group['params'])
        for group in self.optimizer_degree.param_groups:
            optim_params.update(id(p) for p in group['params'])
        for group in self.optimizer_action.param_groups:
            optim_params.update(id(p) for p in group['params'])

        missing_params = all_params - optim_params
        if missing_params:
            print(f"警告: {len(missing_params)}个参数未加入优化器!")

    def meta_forward(self, x):
        shared = self.meta_model[0](x)
        Attention = self.meta_model[1](shared)
        degree_q = self.meta_model[2](Attention)
        action_q = self.meta_model[3](Attention)
        return degree_q, action_q

    def ctrl_forward(self, x):
        return self.ctrl_model(x)

    def act(self, observation, valid_acts=None, reverse_valid=None):
        feature = self.controller.get_intermediate_feature(observation)
        self.current_degree, q_suggestion = self.agent.batch_act([observation], feature)

        return self.controller.batch_act([observation], q_suggestion)[0]

    def observe(self, observation, reward, done, info, degree=None):
        self.agent.observe(observation, reward, done, False)
        self.controller.batch_observe([observation], [reward], [done], [False], degree=self.current_degree)

    def save(self, path):
        torch.save({
            'model_state_dict_1': self.ctrl_model.state_dict(),
            'optimizer_state_dict_1': self.optimizer_1.state_dict(),
            'model_state_dict_2': self.meta_model.state_dict(),
            'optimizer_state_dict_2': self.optimizer_2.state_dict(),
        }, path + '.pt')

    def load(self, path):
        self.ctrl_model.load_state_dict(torch.load(path)['model_state_dict_1'])
        self.optimizer_1.load_state_dict(torch.load(path)['optimizer_state_dict_1'])
        self.meta_model.load_state_dict(torch.load(path)['model_state_dict_2'])
        self.optimizer_2.load_state_dict(torch.load(path)['optimizer_state_dict_2'])


class DualHeadEpsilonGreedy:
    def __init__(self, degree_space_size=5, action_space_size=8,
                 start_epsilon=0.9, end_epsilon=0.1,
                 decay_steps=10000, head_scale=0.7):
        # 各头的动作空间维度
        self.head_sizes = {
            "degree": degree_space_size,
            "action": action_space_size
        }

        # ε参数配置
        self.start_epsilon = start_epsilon
        self.end_epsilon = end_epsilon
        self.decay_steps = decay_steps
        self._epsilon = start_epsilon

        # 动态缩放因子（防止两个头同时探索）
        self.head_scale = head_scale  # 0.5~1.0之间

    def compute_epsilon(self, t):
        """带指数衰减的动态ε计算"""
        self._epsilon = self.end_epsilon + (
                self.start_epsilon - self.end_epsilon
        ) * np.exp(-1. * t / self.decay_steps)
        return self._epsilon

    def select_action(self, t, greedy_action_func, action_value=None, head_type="degree"):
        epsilon = self.compute_epsilon(t) * self.head_scale
        action_size = self.head_sizes[head_type]

        if np.random.random() < epsilon:
            if action_value is not None:
                # 修复维度问题
                q_values = action_value.q_values.detach().cpu().numpy().squeeze()  # 移除多余的batch维度
                assert q_values.size == action_size, f"Q值维度{q_values.shape}与动作空间{action_size}不匹配"

                prob = numpy_softmax(-q_values, axis=-1)
                prob = prob / prob.sum()  # 确保概率和为1
                return np.random.choice(action_size, p=prob)
            else:
                return np.random.randint(0, action_size)
        else:
            return greedy_action_func()


def numpy_softmax(x, axis=-1):
    """数值稳定的numpy softmax实现"""
    x = x - np.max(x, axis=axis, keepdims=True)  # 防止数值溢出
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
