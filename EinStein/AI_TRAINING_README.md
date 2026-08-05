# 爱因斯坦棋 AI 训练说明

本文档统一说明项目中的 AI 训练记录、模型保存路径、模型加载逻辑和常用命令。

## 默认模型路径

项目统一使用以下模型文件名：

- MLP：`einstein_value_model_mlp.pt`
- GNN：`einstein_value_model_gnn.pt`
- Ensemble：同时读取上面两个文件；如果只加载到其中一个，会自动降级为“单模型 + 启发式”混合评估。

旧版 MLP 文件 `einstein_value_model.pt` 仍可尝试加载，但建议后续统一保存为 `einstein_value_model_mlp.pt`。

## 训练思路

AI 由三部分组成：

- `heuristic`：无需模型文件，使用手写局面评估。
- `mlp`：使用 `MLPValueNetwork` 评估当前行动方胜率，输出范围为 `[-1, 1]`。
- `gnn`：保留现有 GNN 价值网络路径。
- `ensemble`：综合 MLP、GNN 和启发式；缺少部分模型时自动降级。

训练流程推荐分三步：

1. 自我对弈生成 JSONL 数据。
2. 用 JSONL 训练 MLP 价值网络。
3. 用自动对战评估新旧 AI。

## 为什么训练数据包含骰子

爱因斯坦棋每回合由骰子决定可行动棋子。相同棋盘在不同骰子下的可行动作和价值不同，所以训练样本必须包含骰子点数。`GameState.encode(die=...)` 会把骰子 one-hot 编入特征。

## 固定 18 动作空间

每方最多 6 个棋子，每个棋子有 3 个方向：

`6 个棋子 × 3 个方向 = 18`

模型或训练数据用 `action_index` 表示动作，非法动作由 `legal_action_mask(die)` 过滤。GUI 外部接口仍保持 `(label, target)`，不会影响现有玩家操作和 AI 落子记录。

## 当前玩家视角归一化

默认编码使用当前行动方视角：

- 当前行动方始终被视为向右下角前进。
- 蓝方行动时会做 180 度旋转并交换双方身份。

这样 MLP 不必分别学习“红方向右下”和“蓝方向左上”两套模式，训练更稳定。

## 生成自我对弈数据

```bash
python train_selfplay.py --games 1000 --output data/selfplay.jsonl
```

常用参数：

```bash
python train_selfplay.py --games 2000 --ai-kind ensemble --model-path einstein_value_model_mlp.pt --max-moves 200 --seed 42
```

如果 `data/` 不存在，脚本会自动创建。

## 训练 MLP 模型

```bash
python train_value_model.py --data data/selfplay.jsonl --epochs 20 --batch-size 128 --output einstein_value_model_mlp.pt
```

训练会保存统一档案格式，包含：

- `format`
- `value_kind`
- `feature_size`
- `state_dict`
- `best_state_dict`
- `history`
- `metadata`
- `replay_buffer`

如果没有安装 `torch`，脚本会给出安装依赖提示。

## 评估 AI

```bash
python evaluate_ai.py --ai-a mlp --model-a einstein_value_model_mlp.pt --ai-b heuristic --games 200
```

输出包括：

- AI A 胜率
- AI B 胜率
- 平均步数
- 红方胜率
- 蓝方胜率
- 到达终点获胜次数
- 吃光对方获胜次数

评估时双方会轮流执红/蓝，降低颜色偏差。

## 兼容旧训练脚本

`train_value_network.py` 仍保留，用于边自博弈边训练 MLP/GNN。默认输出也已改为统一路径：

```bash
python train_value_network.py --value-kind mlp --episodes 200 --epochs 30
python train_value_network.py --value-kind gnn --episodes 200 --epochs 30
```

后续训练可使用旧模型评估：

```bash
python train_value_network.py --value-kind mlp --episodes 200 --epochs 30 --eval-against-old
```

## GUI 中加载模型

1. 运行：

```bash
python einstein.py
```

2. 在 “AI 模型” 中选择 `MLP`、`GNN` 或 `Ensemble`。
3. 模型路径默认会随类型切换：
   - MLP：`einstein_value_model_mlp.pt`
   - GNN：`einstein_value_model_gnn.pt`
   - Ensemble：输入 MLP 路径即可自动配对 GNN 路径。
4. 点击“加载模型”。

模型维度不匹配时，GUI 会显示明确错误，需要使用当前代码重新训练模型。

## 可选优化方向

- 继续训练 GNN，让 Ensemble 更稳定。
- 增加策略网络，直接学习 18 维动作概率。
- 在自博弈中混合随机开局、启发式和模型策略，减少数据偏差。
- 将评估结果写入 CSV，方便长期跟踪模型版本。
