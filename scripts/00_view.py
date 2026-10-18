"""
阶段 0：把机器人"打开看看"

运行：
    conda activate mujoco_env
    python scripts/00_view.py

看到什么：
    一个 12 自由度的四足机器人站在地面上。
    按住鼠标左键拖动 = 旋转视角；右键拖动 = 平移；滚轮 = 缩放。
"""

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import mujoco

from quad_utils import JOINTS, STAND, launch_viewer, load, sleep_forever


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true", help="不开窗口，只打印信息")
    args = parser.parse_args()

    model, data = load()

    print("=" * 62)
    print("模型加载成功")
    print("=" * 62)
    print(f"刚体数量 (nbody)    : {model.nbody}")
    print(f"关节坐标维度 (nq)   : {model.nq}   (7 个浮基 + 12 个关节)")
    print(f"速度维度   (nv)     : {model.nv}   (6 个浮基 + 12 个关节)")
    print(f"执行器数量 (nu)     : {model.nu}")
    print(f"总质量              : {model.body_mass.sum():.2f} kg")
    print(f"物理步长            : {model.opt.timestep * 1000:.1f} ms")
    print()
    print("12 个关节（顺序 = 执行器顺序）：")
    for i, name in enumerate(JOINTS):
        print(f"  [{i:2d}] {name:14s} 站立角 {STAND[i]:+.2f} rad")
    print()
    print(f"初始躯干高度        : {data.qpos[2]:.4f} m")
    print("=" * 62)

    if args.headless:
        print("无窗口模式结束。")
        return

    with launch_viewer(model, data) as viewer:
        if viewer is not None:
            print("窗口已打开，关闭窗口即可退出。")
        sleep_forever(viewer)


if __name__ == "__main__":
    main()
