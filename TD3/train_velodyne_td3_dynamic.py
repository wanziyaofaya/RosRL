import os
import numpy as np
import torch
from train_velodyne_td3 import TD3
from velodyne_env_dynamic import DynamicObstacleGazeboEnv

def main():
    # 设置随机种子以确保可重复性
    seed = 0
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # 环境参数
    state_dim = 36 + 4  # 32 激光数据 + 4 机器人状态
    action_dim = 2      # 线速度和角速度
    max_action = 1      # 动作范围 [-1, 1]
    
    # 创建动态环境
    env = DynamicObstacleGazeboEnv("multi_robot_scenario.launch", 32)
    
    # 初始化TD3智能体
    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
    }
    policy = TD3(**kwargs)
    
    # 如果存在预训练模型，加载它
    model_path = "pytorch_models"
    if os.path.exists(os.path.join(model_path, "TD3_velodyne_actor.pth")):
        policy.load(model_path)
        print("Loaded pre-trained model")
    
    # 训练参数
    max_timesteps = 1e6
    batch_size = 256
    
    # 开始训练
    total_timesteps = 0
    episode_num = 0
    done = True
    
    while total_timesteps < max_timesteps:
        if done:
            print(f"Episode {episode_num} completed after {total_timesteps} timesteps")
            episode_num += 1
            state = env.reset()
            done = False
        
        # 选择动作
        if total_timesteps < 10000:  # 初始探索阶段
            action = np.random.uniform(-max_action, max_action, size=action_dim)
        else:
            action = policy.select_action(np.array(state))
            
        # 执行动作
        next_state, reward, done = env.step(action)
        
        # 存储转换
        policy.store_transition(state, action, reward, next_state, done)
        
        state = next_state
        total_timesteps += 1
        
        # 训练智能体
        if total_timesteps > batch_size:
            policy.train(batch_size)
            
        # 定期保存模型
        if total_timesteps % 5000 == 0:
            policy.save(model_path)
            
if __name__ == "__main__":
    main()
