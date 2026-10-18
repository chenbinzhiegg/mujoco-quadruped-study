# 四足机器人 MuJoCo 学习项目

完整路线、理论说明和排期见 **[项目书.md](项目书.md)**。

## 环境

```bash
conda activate mujoco_env
pip install -r requirements.txt
```

## 命令速查

| 阶段 | 命令 | 你会看到 |
|---|---|---|
| 0 | `python scripts/00_view.py` | 打开模型窗口 |
| 1 | `python scripts/01_inspect.py` | 刚体/关节/执行器/传感器对照表 |
| 2 | `python scripts/02_stand.py` | 站立 → 蹲下 → 起立 |
| 2 | `python scripts/02_stand.py --kp 20` | PD 变软，被重力压下去 |
| 3 | `python scripts/03_leg_ik.py` | 一条腿在空中画圆 |
| 4 | `python scripts/04_trot.py` | 开环 trot，走几步后摔倒 |
| 4 | `python scripts/04_trot.py --gait walk` | 四拍慢走 |
| 5 | `python scripts/05_balance.py` | 姿态闭环，连续走 20 秒 |
| 5 | `python scripts/05_balance.py --no-balance` | 对照组：摔倒 |
| 6 | `python scripts/06_train.py --steps 250000 --residual --speed 0.5` | 残差 RL 训练（推荐） |
| 6 | `python scripts/06_train.py --steps 800000` | 从零 RL 训练 |
| 6 | `python scripts/07_eval.py --residual --speed 0.5 --episodes 5` | 评估残差策略 |
| 6 | `python scripts/07_eval.py --episodes 5` | 评估从零策略 |
| 6 | `python scripts/07_eval.py --residual --speed 0.5 --render` | 开窗口看它走 |

加 `--headless`（00~05）可以在没有图形界面时只看数据。

## 文件

- `quadruped_v2.xml`：主线模型（12 自由度，可转向）
- `quadruped.xml`：最早的 8 自由度模型（已修掉非法的 `rcgba` 属性，保留对照；只能前后走）
- `quad_utils.py`：加载、逆运动学、步态生成、姿态读取、仿真循环
- `envs/quadruped_env.py`：Gymnasium 环境
- `scripts/`：按阶段编号的脚本
