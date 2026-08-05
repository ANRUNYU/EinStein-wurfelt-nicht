# 爱因斯坦棋（EinStein）

这是一个基于 5×5 棋盘的爱因斯坦棋规则仿真程序。

## 规则实现
- 红方起始区：左上 3-2-1 三角形区域
- 蓝方起始区：右下 3-2-1 三角形区域
- 每方 6 个棋子，编号 1-6
- 红方可走方向：右、下、右下
- 蓝方可走方向：左、上、左上
- 掷骰子决定棋子编号；若该棋子已出局，则可以走与该数字最接近的在盘棋子
- 吃掉目标格存在的对方棋子
- 先到达对方出发区角点或将对方全部吃掉获胜

## 运行
在终端中执行：

```bash
python einstein.py
```

程序支持随机开局或手动输入开局摆放。

## AI 模式与训练
- `einstein_ai.py` 实现了无 GUI 的 `GameState`、固定 18 动作空间、启发式/MLP/GNN/Ensemble AI、MCTS 和模型加载。
- 默认 AI 使用启发式价值网络。安装 `torch` 后，可切换到 `mlp`、`gnn` 或 `ensemble`，并加载训练好的模型。
- 统一模型路径：
  - MLP：`einstein_value_model_mlp.pt`
  - GNN：`einstein_value_model_gnn.pt`
  - Ensemble：自动配对 MLP/GNN，缺少部分模型时降级运行。

### 安装依赖
```bash
python -m pip install -r requirements.txt
```

### 标准训练流程
详见 `AI_TRAINING_README.md`。

```bash
python train_selfplay.py --games 1000 --output data/selfplay.jsonl

python train_value_model.py --data data/selfplay.jsonl --epochs 20 --batch-size 128 --output einstein_value_model_mlp.pt

python evaluate_ai.py --ai-a mlp --model-a einstein_value_model_mlp.pt --ai-b heuristic --games 200
```

旧的 `train_value_network.py` 仍可使用，用于边自博弈边训练 MLP/GNN。
