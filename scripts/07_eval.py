"""
阶段 6（续）：看训练成果

运行：
    python scripts/07_eval.py                       # 无窗口，打印指标
    python scripts/07_eval.py --render              # 打开窗口看它走
    python scripts/07_eval.py --speed 0.6           # 换个目标速度考它

输出指标：
    平均前进速度、平均躯干高度、姿态 RMS、每回合存活时间。
    和阶段 5 的传统控制器对比，就是这份项目书最后一节的"实验报告"素材。
"""

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np

from stable_baselines3 import PPO

from envs.quadruped_env import QuadrupedEnv
from quad_utils import STAND_Z

ROOT = pathlib.Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None,
                        help="默认按模式自动选择 models/ppo_quad_residual.zip 或 ppo_quad.zip")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--speed", type=float, default=0.4)
    parser.add_argument("--residual", action="store_true", help="对应训练时的 residual 模式")
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    if args.model is None:
        name = "ppo_quad_residual.zip" if args.residual else "ppo_quad.zip"
        args.model = str(ROOT / "models" / name)

    if not pathlib.Path(args.model).exists():
        raise SystemExit(f"找不到模型 {args.model}\n先运行：python scripts/06_train.py --steps 300000 --residual")

    env = QuadrupedEnv(
        target_speed=args.speed,
        randomize=False,
        residual=args.residual,
        render_mode="human" if args.render else None,
    )
    model = PPO.load(args.model, env=env, device="cpu")

    speeds, heights, rolls, pitches, lengths = [], [], [], [], []

    for ep in range(args.episodes):
        obs, _ = env.reset()
        done = False
        n = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            speeds.append(info["vx"])
            heights.append(info["height"])
            rolls.append(info["roll"])
            pitches.append(info["pitch"])
            n += 1
            if args.render:
                env.render()
        lengths.append(n * env.dt)
        print(f"回合 {ep + 1}: 存活 {lengths[-1]:.1f}s")

    env.close()

    print()
    print("=" * 46)
    print(f"目标速度        : {args.speed:.2f} m/s")
    print(f"平均前进速度    : {np.mean(speeds):+.3f} m/s")
    print(f"平均躯干高度    : {np.mean(heights):.3f} m（站立 {STAND_Z:.3f}）")
    print(f"roll / pitch RMS: {np.degrees(np.sqrt(np.mean(np.square(rolls)))):.1f}° / "
          f"{np.degrees(np.sqrt(np.mean(np.square(pitches)))):.1f}°")
    print(f"平均存活时间    : {np.mean(lengths):.2f} s")
    print("=" * 46)


if __name__ == "__main__":
    main()
