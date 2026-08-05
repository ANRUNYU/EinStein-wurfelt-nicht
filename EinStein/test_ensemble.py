#!/usr/bin/env python3
"""
测试 Ensemble AI 的简单脚本。
假设已经训练了 MLP 和 GNN 模型。
"""

import os
from einstein_ai import HybridAI, GameState, default_model_path

def test_ensemble():
    # 假设 MLP/GNN 模型使用项目统一默认命名。
    mlp_path = default_model_path("mlp")
    gnn_path = default_model_path("gnn")

    if not os.path.exists(mlp_path):
        print(f"MLP 模型文件不存在: {mlp_path}")
        return
    if not os.path.exists(gnn_path):
        print(f"GNN 模型文件不存在: {gnn_path}")
        return

    # 创建 Ensemble AI
    ai = HybridAI(value_kind="ensemble", simulations=80, workers=2, mlp_path=mlp_path, gnn_path=gnn_path)

    # 测试预测
    state = GameState()
    value = ai.value_network.predict(state)
    print(f"Ensemble 预测初始状态价值: {value:.4f}")

    # 测试选择动作
    move = ai.choose_move(state, 1)  # 掷骰子 1
    print(f"Ensemble 选择动作: {move}")

if __name__ == "__main__":
    test_ensemble()
