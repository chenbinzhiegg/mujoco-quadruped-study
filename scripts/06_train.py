"""
阶段 6：用 PPO 训练四足行走

运行（先激活环境）：
    conda activate mujoco_env
    python scripts/06_train.py --steps 1000000

说明：
    为什么用 stable-baselines3？
    因为它把 PPO 的全部细节都封装好了，你只需要关心"状态、动作、奖励"三件事。
    这和阶段 2~5 你手写 PD / 步态 / 反馈是同一套思路，只是"控制器"从人写变成网络学。

训练完会得到 models/ 下的 .zip，用 07_eval.py 看效果（记得加同样的 --residual）。
"""

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor

from envs.quadruped_env import QuadrupedEnv

import torch

# 小网络 + 单进程环境时，多线程反而互相抢 CPU，锁成 1 个线程更快
torch.set_num_threads(1)

ROOT = pathlib.Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1_000_000, help="总训练步数")
    parser.add_argument("--envs", type=int, default=16, help="并行环境数量")
    parser.add_argument("--speed", type=float, default=0.4, help="目标前进速度 m/s")
    parser.add_argument("--episode", type=float, default=10.0, help="单回合时长 s")
    parser.add_argument("--vec", choices=["dummy", "subproc"], default="subproc",
                        help="subproc 用多进程并行，快很多；dummy 方便调试")
    parser.add_argument("--residual", action="store_true",
                        help="以阶段 5 的传统控制器为基线，RL 只学修正量（推荐）")
    parser.add_argument("--out", type=str, default=None,
                        help="默认按模式存成 models/ppo_quad.zip 或 models/ppo_quad_residual.zip")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.out is None:
        name = "ppo_quad_residual" if args.residual else "ppo_quad"
        args.out = str(ROOT / "models" / name)

    model_path = pathlib.Path(args.out)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    log_dir = ROOT / "logs" / model_path.stem
    log_dir.mkdir(parents=True, exist_ok=True)

    def make_env():
        return QuadrupedEnv(
            target_speed=args.speed,
            episode_seconds=args.episode,
            residual=args.residual,
        )

    env_fns = [make_env for _ in range(args.envs)]
    env = SubprocVecEnv(env_fns) if args.vec == "subproc" else DummyVecEnv(env_fns)
    env = VecMonitor(env, str(log_dir))

    model = PPO(
        "MlpPolicy",
        env,
        n_steps=512,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.005,
        learning_rate=3e-4,
        policy_kwargs=dict(net_arch=[256, 128], log_std_init=-0.5),
        verbose=1,
        seed=args.seed,
        device="cpu",          # 这个规模用 CPU 反而更快（GPU 的收益要上万维观测）
    )

    checkpoint = CheckpointCallback(
        save_freq=max(1, 100_000 // args.envs),
        save_path=str(model_path.parent),
        name_prefix=model_path.stem,
    )

    print("=" * 60)
    print(f"开始训练: {args.steps} 步, {args.envs} 个并行环境, 目标速度 {args.speed} m/s")
    print(f"日志目录: {log_dir}")
    print("=" * 60)

    model.learn(total_timesteps=args.steps, callback=checkpoint)
    model.save(str(model_path))
    env.close()

    print()
    print(f"训练完成，模型保存到: {model_path}.zip")
    print(f"查看曲线: tensorboard --logdir {log_dir}")


if __name__ == "__main__":
    main()
