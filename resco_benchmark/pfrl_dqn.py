
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn
from pfrl import explorers
from pfrl.agents import DQN
from pfrl.explorer import Explorer
from pfrl.utils.contexts import evaluating
from resco_benchmark.agents.agent import IndependentAgent, Agent

from global_replay_buffer import GlobalReplayBuffer
from junction_manager import JunctionManager


class MetaController(nn.Module):
    """高层决策模型：选择采样范围度数 n"""
    def __init__(self, input_dim, output_dim):
        super(MetaController, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.fc(x)


class Controller(nn.Module):
    """低层决策模型：选择交通灯控制动作"""
    def __init__(self, input_dim, action_dim):
        super(Controller, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

    def forward(self, x):
        return self.fc(x)

class IDQN(IndependentAgent):
    def __init__(self, config, obs_act, map_name, thread_number):
        super().__init__(config, obs_act, map_name, thread_number)

        # 初始化路口信息
        # 初始化路口信息管理器
        self.junction_manager = JunctionManager(self.sumo, config.get('degree', 1))

        for key in obs_act:
            obs_space = obs_act[key][0]
            act_space = obs_act[key][1]

            def conv2d_size_out(size, kernel_size=2, stride=1):
                return (size - (kernel_size - 1) - 1) // stride + 1

            h = conv2d_size_out(obs_space[1])
            w = conv2d_size_out(obs_space[2])

            # 双头网络：一个头输出度，另一个头输出交通灯控制动作
            class DualHeadModel(nn.Module):
                def __init__(self):
                    super(DualHeadModel, self).__init__()
                    self.shared_layers = nn.Sequential(
                        nn.Conv2d(obs_space[0], 64, kernel_size=(2, 2)),
                        nn.ReLU(),
                        nn.Flatten(),
                        nn.Linear(h * w * 64, 64),
                        nn.ReLU(),
                        nn.Linear(64, 64),
                        nn.ReLU(),
                    )
                    self.degree_head = nn.Linear(64, 5)  # 输出度（1, 2, 3, 4,5）
                    self.action_head = nn.Linear(64, act_space)  # 输出交通灯控制动作

                def forward(self, x):
                    x = self.shared_layers(x)
                    degree = self.degree_head(x)  # 度输出
                    action = self.action_head(x)  # 交通灯控制动作输出
                    return degree, action

            model = DualHeadModel()

            self.agents[key] = DQNAgent(config, act_space, model, num_agents=len(obs_act),
                                        junction_manager=self.junction_manager, degree=config.get('degree', 1))

            if self.config['load']:
                print('LOADING SAVED MODEL FOR EVALUATION')
                self.agents[key].load(self.config['log_dir'] + 'agent_' + key + '.pt')
                self.agents[key].agent.training = False



class DQNAgent(Agent):
    def __init__(self, config, act_space, model,degree=1,  num_agents=0, junction_manager=None):
        super().__init__()

        self.model = model
        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 初始化全局缓冲区
        self.global_buffer = GlobalReplayBuffer(junction_manager)

        # 初始化探索策略
        if num_agents > 0:
            explorer = SharedEpsGreedy(
                config['EPS_START'],
                config['EPS_END'],
                num_agents * config['steps'],
                lambda: np.random.randint(act_space),
            )
        else:
            explorer = explorers.LinearDecayEpsilonGreedy(
                config['EPS_START'],
                config['EPS_END'],
                config['steps'],
                lambda: np.random.randint(act_space),
            )

        if num_agents > 0:
            print('USING SHAREDDQN')
            self.agent = SharedDQN(
                self.model, self.optimizer, config['GAMMA'], explorer,
                gpu=self.device.index, minibatch_size=config['BATCH_SIZE'],
                replay_start_size=config['BATCH_SIZE'], phi=lambda x: np.asarray(x, dtype=np.float32),
                target_update_interval=config['TARGET_UPDATE'] * num_agents,
                update_interval=num_agents, global_buffer=self.global_buffer,
                junction_manager=junction_manager, degree=degree
            )
        else:
            self.agent = DQN(
                self.model, self.optimizer, self.global_buffer, config['GAMMA'], explorer,
                gpu=self.device.index, minibatch_size=config['BATCH_SIZE'],
                replay_start_size=config['BATCH_SIZE'], phi=lambda x: np.asarray(x, dtype=np.float32),
                target_update_interval=config['TARGET_UPDATE']
            )

    def act(self, observation, valid_acts=None, reverse_valid=None):
        if isinstance(self.agent, SharedDQN):
            return self.agent.act(observation, valid_acts=valid_acts, reverse_valid=reverse_valid)
        else:
            return self.agent.act(observation)

    def observe(self, observation, reward, done, info):
        if isinstance(self.agent, SharedDQN):
            self.agent.observe(observation, reward, done, info)
        else:
            self.agent.observe(observation, reward, done, False)

    def save(self, path):
        print(f"Saving model and optimizer to: {path}.pt")
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path + '.pt')

    def load(self, path):
        self.model.load_state_dict(torch.load(path)['model_state_dict'])
        self.optimizer.load_state_dict(torch.load(path)['optimizer_state_dict'])

def select_action_epsilon_greedily(epsilon, random_action_func, greedy_action_func):
    # 根据 epsilon-greedy 策略选择动作
    if np.random.rand() < epsilon:
        return random_action_func(), False  # 以 epsilon 概率选择随机动作
    else:
        return greedy_action_func(), True  # 以 1-epsilon 概率选择贪婪动作
class SharedDQN(DQN):
    def __init__(self, q_function: torch.nn.Module, optimizer: torch.optim.Optimizer,
                 gamma: float, explorer: Explorer, gpu, minibatch_size, replay_start_size, phi,
                 target_update_interval, update_interval, global_buffer: GlobalReplayBuffer,
                 junction_manager: JunctionManager, degree):
        super().__init__(q_function, optimizer, global_buffer, gamma, explorer,
                         gpu=gpu, minibatch_size=minibatch_size, replay_start_size=replay_start_size, phi=phi,
                         target_update_interval=target_update_interval, update_interval=update_interval)
        self.global_buffer = global_buffer  # 全局缓冲区
        self.junction_manager = junction_manager  # 路口信息管理器
        self.degree = degree  # 共享范围的度

        # 初始化高层决策模型
        self.meta_controller = MetaController(input_dim=q_function.input_dim, output_dim=5)  # 输出度数 1 到 5
        self.meta_optimizer = torch.optim.Adam(self.meta_controller.parameters())

        # 初始化低层决策模型
        self.controller = Controller(input_dim=q_function.input_dim, action_dim=q_function.output_dim)
        self.controller_optimizer = torch.optim.Adam(self.controller.parameters())
        num_agents = len(self.junction_manager.junctions)
        # 初始化当前路口 ID 列表
        self.current_junction_id = [None] * num_agents
    def act(self, obs: Any, valid_acts=None, reverse_valid=None) -> Any:
        """选择动作"""
        return self.batch_act([obs], valid_acts=[valid_acts], reverse_valid=[reverse_valid])[0]

    def observe(self, obs: Sequence[Any], reward: Sequence[float], done: Sequence[bool], reset: Sequence[bool]) -> None:
        """观察环境反馈"""
        self.batch_observe(obs, reward, done, reset)

    def batch_act(self, batch_obs: Sequence[Any], valid_acts=None, reverse_valid=None) -> Sequence[Any]:
        """批量选择动作"""
        if valid_acts is None:
            return super(SharedDQN, self).batch_act(batch_obs)

        with torch.no_grad(), evaluating(self.model):
            # 获取当前智能体的路口 ID
            for i in range(len(batch_obs)):
                junction_info = self.junction_manager.get_junction_by_agent(i)
                self.current_junction_id[i] = junction_info['junction_id']

            # 高层决策：选择度数 n
            degree_logits = self.meta_controller(torch.FloatTensor(np.stack(batch_obs)))
            degrees = torch.argmax(degree_logits, dim=-1).cpu().numpy() + 1  # 度数范围 1 到 5

            # 从全局缓冲区中采样经验
            sampled_experiences = []
            for i, obs in enumerate(batch_obs):
                sampled_exp = self.global_buffer.sample(self.current_junction_id[i], degrees[i])
                if sampled_exp:
                    sampled_experiences.extend(sampled_exp)

            # 低层决策：选择有效动作
            if sampled_experiences:
                # 使用采样到的经验训练低层决策模型
                self._train_controller(sampled_experiences)

            # 评估模型并获取动作值
            batch_av = self._evaluate_model_and_update_recurrent_states(batch_obs)
            batch_qvals = batch_av.params[0].detach().cpu().numpy()

            # 选择贪婪动作
            batch_argmax = []
            for i in range(len(batch_obs)):
                batch_item = batch_qvals[i]
                max_val, max_idx = None, None
                for idx in valid_acts[i]:
                    batch_item_qval = batch_item[idx]
                    if max_val is None or batch_item_qval > max_val:
                        max_val = batch_item_qval
                        max_idx = idx
                batch_argmax.append(max_idx)
            batch_argmax = np.asarray(batch_argmax)

        # 选择动作（训练模式使用探索策略，评估模式使用贪婪动作）
        if self.training:
            batch_action = []
            for i in range(len(batch_obs)):
                av = batch_av[i: i + 1]
                greed = batch_argmax[i]
                act, greedy = self.explorer.select_action(
                    self.t,
                    lambda: greed,
                    action_value=av,
                    num_acts=len(valid_acts[i])
                )
                if not greedy:
                    act = reverse_valid[i][act]
                batch_action.append(act)

            # 更新上一观察和动作
            self.batch_last_obs = list(batch_obs)
            self.batch_last_action = list(batch_action)
        else:
            batch_action = batch_argmax

        # 将动作映射到有效动作空间
        valid_batch_action = []
        for i in range(len(batch_action)):
            valid_batch_action.append(valid_acts[i][batch_action[i]])

        return valid_batch_action

    def batch_observe(self, batch_obs: Sequence[Any], batch_reward: Sequence[float],
                      batch_done: Sequence[bool], batch_reset: Sequence[bool]) -> None:
        """批量观察环境反馈"""
        for i in range(len(batch_obs)):
            self.t += 1
            self._cumulative_steps += 1

            # 获取当前智能体所属的路口
            current_junction = self.junction_manager.get_junction_by_agent(i)
            # 获取共享范围内的路口
            shared_junctions = self.junction_manager.get_shared_junctions(current_junction['junction_id'])

            # 如果上一个观察存在，则构建经验
            if self.batch_last_obs[i] is not None:
                assert self.batch_last_action[i] is not None
                # 构建经验
                transition = {
                    "state": self.batch_last_obs[i],
                    "action": self.batch_last_action[i],
                    "reward": batch_reward[i],
                    "next_state": batch_obs[i],
                    "next_action": None,
                    "is_state_terminal": batch_done[i],
                }

                # 将经验添加到全局缓冲区，并更新索引
                self.global_buffer.append(transition, current_junction['junction_id'])
                for junction in shared_junctions:
                    self.global_buffer.append(transition, junction)

                # 如果环境被重置或任务完成，则清空当前路口的上一观察和动作
                if batch_reset[i] or batch_done[i]:
                    self.batch_last_obs[i] = None
                    self.batch_last_action[i] = None
                    self.global_buffer.stop_current_episode(current_junction['junction_id'])

            # 更新当前路口的上一观察和动作
            self.batch_last_obs[i] = batch_obs[i]


        # 更新回放缓冲区
        self.replay_updater.update_if_necessary(self.t)

    def _train_meta_controller(self, transition):
        """训练高层决策模型"""
        state = torch.FloatTensor(transition["state"])
        next_state = torch.FloatTensor(transition["next_state"])
        reward = torch.FloatTensor([transition["reward"]])

        # 预测当前状态的度数
        degree_logits = self.meta_controller(state)
        degree = torch.argmax(degree_logits).item() + 1

        # 计算目标值
        with torch.no_grad():
            next_degree_logits = self.meta_controller(next_state)
            next_degree = torch.argmax(next_degree_logits).item() + 1
            target = reward + self.gamma * next_degree

        # 计算损失
        loss = nn.MSELoss()(degree_logits[degree - 1], target)

        # 更新模型
        self.meta_optimizer.zero_grad()
        loss.backward()
        self.meta_optimizer.step()

    def _train_controller(self, experiences):
        """训练低层决策模型"""
        states = torch.FloatTensor([exp["state"] for exp in experiences])
        actions = torch.LongTensor([exp["action"] for exp in experiences])
        rewards = torch.FloatTensor([exp["reward"] for exp in experiences])
        next_states = torch.FloatTensor([exp["next_state"] for exp in experiences])

        # 预测 Q 值
        q_values = self.controller(states)
        q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # 计算目标值
        with torch.no_grad():
            next_q_values = self.controller(next_states)
            next_q_value = next_q_values.max(1)[0]
            target = rewards + self.gamma * next_q_value

        # 计算损失
        loss = nn.MSELoss()(q_value, target)

        # 更新模型
        self.controller_optimizer.zero_grad()
        loss.backward()
        self.controller_optimizer.step()

class SharedEpsGreedy(explorers.LinearDecayEpsilonGreedy):
    def select_action(self, t, greedy_action_func, action_value=None, num_acts=None):
        # 计算当前的 epsilon 值
        self.epsilon = self.compute_epsilon(t)

        # 如果没有指定动作数量，使用默认的随机动作函数
        if num_acts is None:
            fn = self.random_action_func
        else:
            # 否则，生成一个随机动作
            fn = lambda: np.random.randint(num_acts)

        # 使用 epsilon-greedy 策略选择动作
        a, greedy = select_action_epsilon_greedily(
            self.epsilon, fn, greedy_action_func
        )

        # 记录日志
        greedy_str = "greedy" if greedy else "non-greedy"
        self.logger.debug("t:%s a:%s %s", t, a, greedy_str)

        # 返回动作
        if num_acts is None:
            return a
        else:
            return a, greedy