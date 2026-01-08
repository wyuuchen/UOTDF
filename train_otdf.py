import numpy as np
import torch
import gym
import argparse
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import random
import math
import time
import copy
from pathlib import Path
import yaml

import setproctitle
from algo.OTDF import OTDF
import algo.utils as utils
from envs.env_utils import call_terminal_func
from envs.common import call_env
from tensorboardX import SummaryWriter
from algo.get_normalized_score import get_normalized_score
import d4rl
from evaluate import evaluate_visualization
import robust_ot_solver
import gc

def eval_policy(policy, env, eval_episodes=10, eval_cnt=None):
    eval_env = env

    avg_reward = 0.
    for episode_idx in range(eval_episodes):
        state, done = eval_env.reset(), False
        while not done:
            action = policy.select_action(np.array(state))
            next_state, reward, done, _ = eval_env.step(action)

            avg_reward += reward
            state = next_state
    avg_reward /= eval_episodes

    print("[{}] Evaluation over {} episodes: {}".format(eval_cnt, eval_episodes, avg_reward))

    return avg_reward


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="./logs")
    parser.add_argument("--policy", default="OTDF")
    parser.add_argument("--env", default="halfcheetah")
    parser.add_argument("--seed", default=0, type=int)            
    parser.add_argument("--save-model", action="store_true")        # Save model and optimizer parameters
    parser.add_argument('--srctype', default='medium', type=str)
    parser.add_argument("--tartype", default='medium', type=str)
    parser.add_argument("--steps", default=1e6, type=int)
    parser.add_argument("--weight", action="store_true")
    parser.add_argument("--proportion", default=0.8, type=float)
    parser.add_argument("--noreg", action="store_true")
    parser.add_argument("--reg_weight", default=0.5, type=float)
    
    # Parameter used to calculate optimal transport
    parser.add_argument("--epsilon", default=0.01, type=float, help="Entropy regularization")
    parser.add_argument("--metric", default='euclidean', type=str)     # metric used in optimal transport
    parser.add_argument("--lambda_src", default=0.05, type=float, help="Source filtering strength")
    parser.add_argument("--lambda_tar", default=0.5, type=float, help="Target optimism strength")

    # Parameter used to filter low quality datasets
    parser.add_argument("--filter_threshold", default=1.0, type=float, help= "ratio used to filter dataset")
    
    
    
    args = parser.parse_args()

    with open(f"{str(Path(__file__).parent.absolute())}/config/{args.env.replace('-', '_')}.yaml", 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    print("------------------------------------------------------------")
    print("Env: {}, Seed: {}".format( args.env, args.seed))
    print("------------------------------------------------------------")
    
    base_exp_name = args.env + '-srcdatatype-' + args.srctype + '-tardatatype-' + args.tartype
    hyper_param_str = f"eps-{args.epsilon}|src-{args.lambda_src}|tar-{args.lambda_tar}|filter-{args.filter_threshold}"
    setproctitle.setproctitle(f"{base_exp_name}")
    outdir = os.path.join(args.dir,  base_exp_name, hyper_param_str, f"r{args.seed}")
    writer = SummaryWriter('{}/tb'.format(outdir))
    if args.save_model and not os.path.exists("{}/models".format(outdir)):
        os.makedirs("{}/models".format(outdir))
    
    if '_' in args.env:
        args.env = args.env.replace('_', '-')
    
    # train env
    src_env_name = args.env.split('-')[0] + '-' + args.srctype + '-v2'
    src_env = gym.make(src_env_name)
    src_env.seed(args.seed)
    # test env
    tar_env = call_env(config['tar_env_config'])
    tar_env.seed(args.seed)
    # eval env
    src_eval_env = gym.make(src_env_name)
    src_eval_env.seed(args.seed + 100)
    tar_eval_env = call_env(config['tar_env_config'])
    tar_eval_env.seed(args.seed + 100)

    # seed all
    src_env.action_space.seed(args.seed)
    tar_env.action_space.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)

    state_dim = src_env.observation_space.shape[0]
    action_dim = src_env.action_space.shape[0] 
    max_action = float(src_env.action_space.high[0])
    min_action = -max_action
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    config['metric'] = args.metric

    weight = True if args.weight else False
    noreg = True if args.noreg else False

    config.update({
        'state_dim': state_dim,
        'action_dim': action_dim,
        'max_action': max_action,
        'weight': weight,
        'proportion': float(args.proportion),
        'noreg': noreg,
        'reg_weight': args.reg_weight,
        'filter_threshold': args.filter_threshold
    })

    policy = OTDF(config, device)
    
    ## write logs to record training parameters
    with open(os.path.join(outdir, 'log.txt'), 'w') as f:
        f.write('\n Env: {}, seed: {}'.format( args.env, args.seed))
        for item in config.items():
            f.write('\n {}'.format(item))

    src_replay_buffer = utils.OTReplayBuffer(state_dim, action_dim, device)
    tar_replay_buffer = utils.ReplayBuffer(state_dim, action_dim, device)

    # load offline datasets
    src_dataset = d4rl.qlearning_dataset(src_env)
    tar_dataset = utils.call_tar_dataset(args.env, args.tartype)

    src_replay_buffer.convert_D4RL(src_dataset)
    tar_replay_buffer.convert_D4RL(tar_dataset)

    weight_filename = "robust_weights.hdf5"
    weight_file_path = os.path.join(outdir, weight_filename)
    
    print(f"Target weight file path: {weight_file_path}")

    # 2. 检查或计算权重
    if os.path.exists(weight_file_path) :
        print("Found existing weights in log dir. Loading directly...")
        robust_weights = utils.load_robust_weights(weight_file_path)
    else:
        print("Weights not found in log dir. Calculating now...")
        
        # 计算并保存到 outdir
        robust_weights = robust_ot_solver.compute_and_save_weights(
            src_replay_buffer, 
            tar_replay_buffer, 
            weight_file_path, 
            args
        )
        
    # 3. 计算完成后，立即生成评估图表 (存入 outdir)
    # 这样每次跑实验，你都能在文件夹里直接看到 correlation 图
    # try:
    #     evaluate_visualization(
    #         env_name=args.env, 
    #         srctype=args.srctype, 
    #         tartype=args.tartype, 
    #         weight_file=weight_file_path, 
    #         save_dir=outdir
    #     )
    # except Exception as e:
    #     print(f"[Warning] Visualization failed: {e}")
    #     print("Skipping visualization and continuing training...")
    # print("Clearing GPU memory before RL training...")
    # 1. Delete large objects used for OT that are no longer needed
    if 'robust_weights' in locals():
        # If robust_weights is just a numpy array, it's on CPU, which is fine.
        # But if you had any JAX objects lingering, delete them.
        pass 

    # 2. Force Python GC to kill any hanging JAX references
    gc.collect()
    
    # 3. Empty PyTorch Cache (in case JAX/PyTorch interactions left fragments)
    torch.cuda.empty_cache()
    # 4. 将权重注入 Buffer

    global_mean = robust_weights.mean() + 1e-12
    global_max = robust_weights.max()
    
    print(f"Global Weights Stats -> Mean: {global_mean:.2e}, Max: {global_max:.2e}")
    # 方案 A: 均值归一化 (保持原来的 scale 意义)
    # 这样处理后，buffer 里存的就是 importance，均值为 1.0
    normalized_weights = robust_weights / global_mean

    src_replay_buffer.set_weights(normalized_weights)
    print(f"Robust weights ready. Mean: {normalized_weights.mean():.4e}")
    eval_cnt = 0

    # whether to pretrain VAE
    if not noreg:
        policy.train_vae(tar_replay_buffer, config['batch_size'], writer)
    
    eval_src_return = eval_policy(policy, src_eval_env, eval_cnt=eval_cnt)
    eval_tar_return = eval_policy(policy, tar_eval_env, eval_cnt=eval_cnt)
    norm_tar_return = get_normalized_score(args.env, eval_tar_return)
    eval_cnt += 1

    for t in range(int(args.steps)):
        policy.train(src_replay_buffer, tar_replay_buffer, config['batch_size'], writer)

        if (t + 1) % config['eval_freq'] == 0:
            src_eval_return = eval_policy(policy, src_eval_env, eval_cnt=eval_cnt)
            tar_eval_return = eval_policy(policy, tar_eval_env, eval_cnt=eval_cnt)
            norm_tar_return = get_normalized_score(args.env, tar_eval_return)
            writer.add_scalar('test/source return', src_eval_return, global_step = t+1)
            writer.add_scalar('test/target return', tar_eval_return, global_step = t+1)
            writer.add_scalar('test/target_normalized_score', norm_tar_return, global_step=t+1)
            print("[{}] Normalized Score of Target Domain is {}".format(eval_cnt, norm_tar_return))
            print("*"*30)
            eval_cnt += 1

            if args.save_model:
                policy.save('{}/models/model'.format(outdir))
    writer.close()