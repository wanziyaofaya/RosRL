import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import inf
from torch.utils.tensorboard import SummaryWriter

from replay_buffer import ReplayBuffer
from velodyne_env_rrt import GazeboEnv

# 评估函数，用于测试训练好的网络性能
def evaluate(network, epoch, eval_episodes=10): # 要评估的TD3网络，当前训练轮次，评估回合数，默认10回合
    avg_reward = 0.0 # 平均奖励
    col = 0 # 碰撞次数计数器
    for _ in range(eval_episodes):
        count = 0 # 当前回合步数
        state = env.reset() # 重置环境
        done = False
        while not done and count < 501: # 最多执行501步
            action = network.get_action(np.array(state)) # 获取动作
            a_in = [(action[0] + 1) / 2, action[1]] # 线速度映射到[0,1]，角速度保持[-1,1]
            state, reward, done, _ = env.step(a_in) # 执行动作
            avg_reward += reward # 累加奖励
            count += 1
            if reward < -90:  # 记录碰撞
                col += 1
    avg_reward /= eval_episodes # 计算平均奖励
    avg_col = col / eval_episodes  # 计算平均碰撞次数
    print("..............................................")
    print(
        "Average Reward over %i Evaluation Episodes, Epoch %i: %f, %f"
        % (eval_episodes, epoch, avg_reward, avg_col)
    )
    print("..............................................")
    return avg_reward


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()

        self.layer_1 = nn.Linear(state_dim, 800) # 第一层：state_dim → 800
        self.layer_2 = nn.Linear(800, 600) # 第二层：800 → 600
        self.layer_3 = nn.Linear(600, action_dim) # 第三层：600 → action_dim
        self.tanh = nn.Tanh() # 激活函数

    def forward(self, s): # 前向传播函数
        # 输入状态s，经过三层全连接网络和Tanh激活函数
        # 第一层：输入状态s，输出经过ReLU激活的结果
        s = F.relu(self.layer_1(s))
        # 第二层：输入上一层输出，输出经过ReLU激活的结果
        s = F.relu(self.layer_2(s))
        # 第三层：输入上一层输出，输出经过Tanh激活的动作a
        a = self.tanh(self.layer_3(s))
        return a


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        self.layer_1 = nn.Linear(state_dim, 800)
        self.layer_2_s = nn.Linear(800, 600)
        self.layer_2_a = nn.Linear(action_dim, 600)
        self.layer_3 = nn.Linear(600, 1)

        self.layer_4 = nn.Linear(state_dim, 800)
        self.layer_5_s = nn.Linear(800, 600)
        self.layer_5_a = nn.Linear(action_dim, 600)
        self.layer_6 = nn.Linear(600, 1)

    def forward(self, s, a):
        s1 = F.relu(self.layer_1(s))
        self.layer_2_s(s1)
        self.layer_2_a(a)
        s11 = torch.mm(s1, self.layer_2_s.weight.data.t())
        s12 = torch.mm(a, self.layer_2_a.weight.data.t())
        s1 = F.relu(s11 + s12 + self.layer_2_a.bias.data)
        q1 = self.layer_3(s1)

        s2 = F.relu(self.layer_4(s))
        self.layer_5_s(s2)
        self.layer_5_a(a)
        s21 = torch.mm(s2, self.layer_5_s.weight.data.t())
        s22 = torch.mm(a, self.layer_5_a.weight.data.t())
        s2 = F.relu(s21 + s22 + self.layer_5_a.bias.data)
        q2 = self.layer_6(s2)
        return q1, q2


# TD3 network
class TD3(object):
    def __init__(self, state_dim, action_dim, max_action):
        # Initialize the Actor network
        self.actor = Actor(state_dim, action_dim).to(device)
        self.actor_target = Actor(state_dim, action_dim).to(device) # 目标网络用于稳定训练过程，避免训练波动
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters())

        # Initialize the Critic networks
        # Critic 输入是 (state, action)，输出是 Q 值
        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters())

        self.max_action = max_action

        # 用于记录 TensorBoard 日志（如损失和 Q 值），以及训练步数计数器
        self.writer = SummaryWriter()
        self.iter_count = 0

    def get_action(self, state):
        # Function to get the action from the actor
        state = torch.Tensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten() # 获取 actor 输出的动作，转为 numpy 格式返回

    # training cycle
    # 主训练函数，使用从 replay buffer 中采样的经验进行训练
    def train(
        self,
        replay_buffer,
        iterations,
        batch_size=100,
        discount=1, # 奖励折扣因子（通常为 0.99）
        tau=0.005, # 软更新比例
        policy_noise=0.2,  # discount=0.99，添加到 target 动作上的噪声幅度
        noise_clip=0.5, # 噪声裁剪范围
        policy_freq=2, # policy 更新频率（TD3 特有）
    ):
        
        # 初始化记录平均 Q 值、最大 Q 值和平均损失
        av_Q = 0
        max_Q = -inf
        av_loss = 0


        for it in range(iterations):

            (
                batch_states,
                batch_actions,
                batch_rewards,
                batch_dones,
                batch_next_states,
            ) = replay_buffer.sample_batch(batch_size)
            state = torch.Tensor(batch_states).to(device)
            next_state = torch.Tensor(batch_next_states).to(device)
            action = torch.Tensor(batch_actions).to(device)
            reward = torch.Tensor(batch_rewards).to(device)
            done = torch.Tensor(batch_dones).to(device)


            next_action = self.actor_target(next_state) # 使用目标actor网络预测下一个状态的动作


            # 给 next_action 添加 clipped 高斯噪声，以增加策略的鲁棒性（TD3 的关键点）
            noise = torch.Tensor(batch_actions).data.normal_(0, policy_noise).to(device)
            noise = noise.clamp(-noise_clip, noise_clip)
            next_action = (next_action + noise).clamp(-self.max_action, self.max_action)

            # 用目标 critic 网络分别计算 Q1 和 Q2
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)


            target_Q = torch.min(target_Q1, target_Q2) # 取两个critic的最小值作为目标 Q（TD3 另一个关键点，避免 Q 值过高估计）
            av_Q += torch.mean(target_Q)
            max_Q = max(max_Q, torch.max(target_Q))

            target_Q = reward + ((1 - done) * discount * target_Q).detach() # 计算 TD 目标，注意使用 detach() 防止梯度传播

            current_Q1, current_Q2 = self.critic(state, action) # 用当前 critic 网络计算 Q 值

            # 两个 Q 值分别与目标 Q 做 MSE 计算并加和
            loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

            # 梯度归零，反向传播并更新 critic 网络
            self.critic_optimizer.zero_grad()
            loss.backward()
            self.critic_optimizer.step()


            if it % policy_freq == 0: # 每隔 policy_freq 次才更新 actor 网络（TD3 的第三个关键点）
                # 计算当前策略的 Q 值，用来最大化（等价于最小化负 Q）
                actor_grad, _ = self.critic(state, self.actor(state))
                actor_grad = -actor_grad.mean()

                # 更新 actor 网络
                self.actor_optimizer.zero_grad()
                actor_grad.backward()
                self.actor_optimizer.step()

                for param, target_param in zip(
                    self.actor.parameters(), self.actor_target.parameters()
                ):
                    target_param.data.copy_(
                        tau * param.data + (1 - tau) * target_param.data
                    ) # 使用 soft update 更新 actor_target 网络的参数

                for param, target_param in zip(
                    self.critic.parameters(), self.critic_target.parameters()
                ):
                    target_param.data.copy_(
                        tau * param.data + (1 - tau) * target_param.data
                    ) # 同样地更新 critic_target 网络

            av_loss += loss # 累加损失

        # 记录平均损失、Q 值到 TensorBoard
        self.iter_count += 1 
        self.writer.add_scalar("loss", av_loss / iterations, self.iter_count)
        self.writer.add_scalar("Av. Q", av_Q / iterations, self.iter_count)
        self.writer.add_scalar("Max. Q", max_Q, self.iter_count)

    def save(self, filename, directory): # 保存 actor 和 critic 网络的参数到文件中
        torch.save(self.actor.state_dict(), "%s/%s_actor.pth" % (directory, filename))
        torch.save(self.critic.state_dict(), "%s/%s_critic.pth" % (directory, filename))

    def load(self, filename, directory): # 从文件中加载 actor 和 critic 网络的参数
        self.actor.load_state_dict(
            torch.load("%s/%s_actor.pth" % (directory, filename))
        )
        self.critic.load_state_dict(
            torch.load("%s/%s_critic.pth" % (directory, filename))
        )


# Set the parameters for the implementation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 自动检测是否有 GPU 可用，有就使用 CUDA，否则用 CPU
seed = 0  # Random seed number
eval_freq = 5e3  # After how many steps to perform the evaluation
max_ep = 500  # maximum number of steps per episode
eval_ep = 10  # number of episodes for evaluation
max_timesteps = 5e6  # Maximum number of steps to perform
expl_noise = 1  # Initial exploration noise starting value in range [expl_min ... 1]
expl_decay_steps = (
    500000  # Number of steps over which the initial exploration noise will decay over
)
expl_min = 0.1  # Exploration noise after the decay in range [0...expl_noise]
batch_size = 40  # Size of the mini-batch
discount = 0.99999  # Discount factor to calculate the discounted future reward (should be close to 1)
tau = 0.005  # Soft target update variable (should be close to 0)
policy_noise = 0.2  # Added noise for exploration
noise_clip = 0.5  # Maximum clamping values of the noise
policy_freq = 2  # Frequency of Actor network updates
buffer_size = 1e6  # Maximum size of the buffer
file_name = "TD3_velodyne"  # name of the file to store the policy
save_model = True  # Weather to save the model or not
load_model = False  # Weather to load a stored model
random_near_obstacle = True  # To take random actions near obstacles or not

# Create the network storage folders
if not os.path.exists("./results"):
    os.makedirs("./results")
if save_model and not os.path.exists("./pytorch_models"):
    os.makedirs("./pytorch_models")

# Create the training environment
environment_dim = 20
robot_dim = 4
env = GazeboEnv("multi_robot_scenario.launch", environment_dim)
time.sleep(5)
torch.manual_seed(seed)
np.random.seed(seed)
state_dim = environment_dim + robot_dim
action_dim = 2
max_action = 1

# Create the network
network = TD3(state_dim, action_dim, max_action)
# Create a replay buffer
replay_buffer = ReplayBuffer(buffer_size, seed)
if load_model:
    try:
        network.load(file_name, "./pytorch_models")
    except:
        print(
            "Could not load the stored model parameters, initializing training with random parameters"
        )

# Create evaluation data store
evaluations = []

timestep = 0
timesteps_since_eval = 0
episode_num = 0
done = True
epoch = 1

count_rand_actions = 0
random_action = []

# Begin the training loop
while timestep < max_timesteps:

    # On termination of episode
    if done:
        if timestep != 0:
            network.train(
                replay_buffer,
                episode_timesteps,
                batch_size,
                discount,
                tau,
                policy_noise,
                noise_clip,
                policy_freq,
            )

        if timesteps_since_eval >= eval_freq:
            print("Validating")
            timesteps_since_eval %= eval_freq
            evaluations.append(
                evaluate(network=network, epoch=epoch, eval_episodes=eval_ep)
            )
            network.save(file_name, directory="./pytorch_models")
            np.save("./results/%s" % (file_name), evaluations)
            epoch += 1

        state = env.reset()
        done = False

        episode_reward = 0
        episode_timesteps = 0
        episode_num += 1

    # add some exploration noise
    if expl_noise > expl_min:
        expl_noise = expl_noise - ((1 - expl_min) / expl_decay_steps)

    action = network.get_action(np.array(state))
    action = (action + np.random.normal(0, expl_noise, size=action_dim)).clip(
        -max_action, max_action
    )

    # If the robot is facing an obstacle, randomly force it to take a consistent random action.
    # This is done to increase exploration in situations near obstacles.
    # Training can also be performed without it
    if random_near_obstacle:
        if (
            np.random.uniform(0, 1) > 0.85
            and min(state[4:-8]) < 0.6
            and count_rand_actions < 1
        ):
            count_rand_actions = np.random.randint(8, 15)
            random_action = np.random.uniform(-1, 1, 2)

        if count_rand_actions > 0:
            count_rand_actions -= 1
            action = random_action
            action[0] = -1

    # Update action to fall in range [0,1] for linear velocity and [-1,1] for angular velocity
    a_in = [(action[0] + 1) / 2, action[1]]
    next_state, reward, done, target = env.step(a_in)
    done_bool = 0 if episode_timesteps + 1 == max_ep else int(done)
    done = 1 if episode_timesteps + 1 == max_ep else int(done)
    episode_reward += reward

    # Save the tuple in replay buffer
    replay_buffer.add(state, action, reward, done_bool, next_state)

    # Update the counters
    state = next_state
    episode_timesteps += 1
    timestep += 1
    timesteps_since_eval += 1

# After the training is done, evaluate the network and save it
evaluations.append(evaluate(network=network, epoch=epoch, eval_episodes=eval_ep))
if save_model:
    network.save("%s" % file_name, directory="./models")
np.save("./results/%s" % file_name, evaluations)
