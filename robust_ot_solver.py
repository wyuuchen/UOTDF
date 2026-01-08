import numpy as np
import torch
import gym
import d4rl
import argparse
import os
import random
import copy
from pathlib import Path
import yaml
import h5py

import algo.utils as utils
from envs.common import call_env

import ott
import scipy as sp

# import ot
import jax.numpy as jnp
import numpy as np
import jax
import gc

def solve_robust_ot(
    src_data, 
    tar_data, 
    cost_type='cosine',
    epsilon=0.05,      # 熵正则化系数
    lambda_src=5.0,    # 源域筛选力度 (越小过滤越狠)
    lambda_tar=0.1     # 目标域乐观程度 (越小越宽容)
):
    src_B = src_data.shape[0]
    tgt_B = tar_data.shape[0]
    src_embs = jnp.array(src_data.reshape(src_B, -1), dtype=jnp.float16)  # (batch_size1 + batch_size2, dim)
    tgt_embs = jnp.array(tar_data.reshape(tgt_B, -1), dtype=jnp.float16)  # (batch_size1 + batch_size2, dim)
    if cost_type == 'euclidean':
        cost_fn = ott.geometry.costs.Euclidean()
    elif cost_type == 'cosine':
        cost_fn = ott.geometry.costs.Cosine()
    else:
        raise NotImplementedError

    scale_cost = 'max_cost'
    geom = ott.geometry.pointcloud.PointCloud(
        src_embs, 
        tgt_embs, 
        cost_fn=cost_fn, 
        epsilon=epsilon, 
        scale_cost=scale_cost)

    # 3. 计算松弛系数 tau (核心数学推导的落地)
    # tau = lambda / (lambda + epsilon)
    # tau -> 1.0 (Hard constraint), tau -> 0.0 (No constraint)
    tau_a = lambda_src / (lambda_src + epsilon)
    tau_b = lambda_tar / (lambda_tar + epsilon)

    # 4. 定义双边非平衡问题 (Double Unbalanced Problem)
    prob = ott.problems.linear.linear_problem.LinearProblem(
        geom,
        tau_a=tau_a, # 源域松弛：允许丢弃数据
        tau_b=tau_b  # 目标域松弛：允许分布偏移
    )

    # 5. 求解 Sinkhorn
    # 使用 jittable solver
    solver = ott.solvers.linear.sinkhorn.Sinkhorn(
        threshold=1e-5, 
        max_iterations=100
    )
    sinkhorn_output = solver(prob)
    # 6. 获取源域样本的有效权重 (Marginals)
    # marginals[0] 即为论文中的 w_i = sum_j pi_ij
    # 如果样本被过滤，这个值会趋近于 0
    # 注意：我们不需要显式计算庞大的 coupling_matrix，直接取边缘分布即可
    source_weights = sinkhorn_output.marginal(1)
    # 将 NaN 替换为 0 (防止数值不稳定)
    source_weights = jnp.nan_to_num(source_weights)
    return source_weights

def filter_dataset(src_replay_buffer, tar_replay_buffer, args):
    src_num = src_replay_buffer.state.shape[0]
    # 拼接 (s, a, s')
    srcdata = np.hstack([src_replay_buffer.state, src_replay_buffer.action, src_replay_buffer.next_state])
    tar_num = tar_replay_buffer.state.shape[0]
    tardata = np.hstack([tar_replay_buffer.state, tar_replay_buffer.action, tar_replay_buffer.next_state])
    # 1. 拼接两部分数据以计算统一的统计量 (保证处于同一刻度空间)
    all_data = np.vstack([srcdata, tardata])
    
    # 2. 计算均值和标准差 (加上 1e-6 防止除以 0)
    mean = np.mean(all_data, axis=0)
    std = np.std(all_data, axis=0) + 1e-6
    
    # 3. 应用变换
    srcdata_norm = (srcdata - mean) / std
    tardata_norm = (tardata - mean) / std
    # =======================================

    if 'ant' in args.env.lower():
        print("Applying Feature Masking for Ant: Using only Pose (dim 0-13)")
        # 仅截取前 13 维用于 OT 计算
        # 注意：这里只影响距离计算，不影响 RL 训练时 replay buffer 里存的数据
        src_for_ot = srcdata_norm[:, :13]
        tar_for_ot = tardata_norm[:, :13]
    else:
        # 其他环境暂时保持全维度，或者你也想尝试只取前半部分
        # src_for_ot = srcdata_norm
        # tar_for_ot = tardata_norm
        
        # 激进策略：对于所有 MuJoCo 任务，前半部分通常都是 Pose，后半部分是 Vel
        # 你可以尝试统一只用前半部分
        dim = srcdata_norm.shape[1]
        half_dim = dim // 2
        print(f"Applying Generic Feature Masking: Using first {half_dim} dims")
        src_for_ot = srcdata_norm[:, :half_dim]
        tar_for_ot = tardata_norm[:, :half_dim]

    weights_result = []
    
    # 使用 functools.partial 固定静态参数，以便 JIT 编译
    from functools import partial
    
    # 这里的 solve_robust_ot 就是上面定义的新函数
    batch_solve = jax.jit(partial(
        solve_robust_ot, 
        cost_type=args.metric,
        epsilon=args.epsilon,
        lambda_src=args.lambda_src,
        lambda_tar=args.lambda_tar
    ))
    
    # 分批处理源数据 (防止显存溢出)
    batch_size = 10000 
    iter_time = int(np.ceil(src_num / batch_size))
    
    for i in range(iter_time):
        start_idx = batch_size * i
        end_idx = min(batch_size * (i + 1), src_num)
        
        # 1. 使用标准化后的源数据
        # src_batch = srcdata_norm[start_idx:end_idx]  # 使用 _norm
        # weights = batch_solve(src_batch, tardata_norm) # 使用 _norm
        src_batch = src_for_ot[start_idx:end_idx]

        weights_jax = batch_solve(src_batch, tar_for_ot)
        
        # Convert to CPU immediately to free GPU memory for this batch
        part_res = jax.device_get(weights_jax).tolist()
        weights_result.extend(part_res)
        
        # Explicitly delete the JAX array reference for this batch
        del weights_jax

        # weights = batch_solve(src_batch, tar_for_ot)
        
        # # 转回 CPU numpy
        # part_res = jax.device_get(weights).tolist()
        # weights_result.extend(part_res)
        
        if (i + 1) % 5 == 0:
            print(f'Processed {end_idx} / {src_num} transitions...')
            
    weights_result = np.array(weights_result)
    # 归一化处理 (可选，视后续 RL 算法需求而定，建议保留原始相对大小)
    # weights_result = weights_result / (np.max(weights_result) + 1e-8)
    print("Cleaning up JAX memory...")
    del batch_solve
    del src_for_ot
    del tar_for_ot
    
    # Clear JAX internal backend cache
    jax.clear_backends()
    
    # Force Python Garbage Collection
    gc.collect()
    return weights_result

def compute_and_save_weights(src_replay_buffer, tar_replay_buffer, save_path, args):
    """
    封装函数：计算权重并保存为 HDF5
    """
    import h5py
    
    # 1. 计算权重 (调用你已有的 filter_dataset)
    # 注意：filter_dataset 内部需要 args.epsilon, args.lambda_src 等参数
    weights = filter_dataset(src_replay_buffer, tar_replay_buffer, args)
    
    # 2. 确保目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    replay_dataset = dict(cost=weights) # 保持 key='cost' 以兼容下游
    
    with h5py.File(save_path, 'w') as hfile:
        for k in replay_dataset:
            hfile.create_dataset(k, data=replay_dataset[k], compression='gzip')
            
        # 可选：保存元数据方便追溯
        hfile.attrs['epsilon'] = args.epsilon
        hfile.attrs['lambda_src'] = args.lambda_src
        hfile.attrs['lambda_tar'] = args.lambda_tar
    return weights

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     # ... 原有参数保持不变 ...
#     parser.add_argument("--dir", default="./costlogs")
#     parser.add_argument("--policy", default="OTDF", help='policy to use') # 这里的名字其实无所谓，只是读取config
#     parser.add_argument("--env", default="halfcheetah")
#     parser.add_argument("--seed", default=0, type=int)            
#     parser.add_argument("--metric", default='cosine', type=str)
#     parser.add_argument('--srctype', default='medium', type=str)
#     parser.add_argument("--tartype", default='medium', type=str)
    
#     # === 新增参数 ===
#     parser.add_argument("--epsilon", default=0.05, type=float, help="Entropy regularization")
#     parser.add_argument("--lambda_src", default=5.0, type=float, help="Source filtering strength (smaller = harder filter)")
#     parser.add_argument("--lambda_tar", default=0.1, type=float, help="Target optimism strength (smaller = more optimistic)")
    
#     args = parser.parse_args()

#     with open(f"{str(Path(__file__).parent.absolute())}/config/{args.policy.lower()}/{args.env.replace('-', '_')}.yaml", 'r', encoding='utf-8') as f:
#         config = yaml.safe_load(f)

#     print("------------------------------------------------------------")
#     print("Policy: {}, Env: {}, Seed: {}".format(args.policy, args.env, args.seed))
#     print("------------------------------------------------------------")
    
#     # outdir = args.dir + '/' + args.env + '-srcdatatype-' + args.srctype + '-tardatatype-' + args.tartype

#     # if not os.path.exists(args.dir):
#     #     os.makedirs(args.dir)
    
#     if '_' in args.env:
#         args.env = args.env.replace('_', '-')
    
#     # train env
#     src_env_name = args.env.split('-')[0] + '-' + args.srctype + '-v2'
#     src_env = gym.make(src_env_name)
#     src_env.seed(args.seed)
#     # test env
#     tar_env = call_env(config['tar_env_config'])
#     tar_env.seed(args.seed)
#     # eval env
#     src_eval_env = copy.deepcopy(src_env)
#     src_eval_env.seed(args.seed + 100)
#     tar_eval_env = copy.deepcopy(tar_env)
#     tar_eval_env.seed(args.seed + 100)

#     # seed all
#     src_env.action_space.seed(args.seed)
#     tar_env.action_space.seed(args.seed)
#     torch.manual_seed(args.seed)
#     np.random.seed(args.seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False
#     torch.cuda.manual_seed_all(args.seed)
#     random.seed(args.seed)

#     state_dim = src_env.observation_space.shape[0]
#     action_dim = src_env.action_space.shape[0] 
#     max_action = float(src_env.action_space.high[0])
#     min_action = -max_action
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    

#     config.update({
#         'state_dim': state_dim,
#         'action_dim': action_dim,
#         'max_action': max_action,
#     })


#     src_replay_buffer = utils.OTReplayBuffer(state_dim, action_dim, device)
#     tar_replay_buffer = utils.ReplayBuffer(state_dim, action_dim, device)

#     # load offline datasets
#     src_dataset = d4rl.qlearning_dataset(src_env)
#     tar_dataset = utils.call_tar_dataset(args.env, args.tartype)

#     src_replay_buffer.convert_D4RL(src_dataset)
#     tar_replay_buffer.convert_D4RL(tar_dataset)

#     # ... ReplayBuffer 加载代码保持不变 ...
    
#     print("Computing Robust Weights...")
#     # 调用新的 filter_dataset
#     # 注意：这里返回的是 weights (保留的质量)，而不是 cost (距离)
#     # 在下游 RL 训练中，应该直接用这个 weight 乘以 loss
#     weights = filter_dataset(src_replay_buffer, tar_replay_buffer, args)

#     print('Computation done. Saving...')
    
#     # 保存结果
#     # 建议将文件名加上参数后缀，方便对比实验
#     out_name = f"{args.env}-{args.srctype}-{args.tartype}-lsrc{args.lambda_src}-ltar{args.lambda_tar}"
#     outdir = os.path.join(args.dir, out_name)
#     # 保存为 hdf5
#     # 注意：这里 key 依然叫 'cost' 是为了兼容下游代码读取，
#     # 但实际上存的是权重(weight)。需要在 RL 训练代码里知晓这一点：
#     # 如果是 OTDF (Cost)，原本逻辑可能是 exp(-cost)
#     # 现在直接是 Weight，逻辑应改为直接使用 weight
#     replay_dataset = dict(
#         cost = weights, 
#     )

#     with h5py.File(outdir + ".hdf5", 'w') as hfile:
#         for k in replay_dataset:
#             hfile.create_dataset(k, data=replay_dataset[k], compression='gzip')
    
#     print(f"Saved to {outdir}.hdf5")