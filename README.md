一、软件成长历程
1．软件的起源及成长过程
本项目以5×5爱因斯坦棋为研究对象，目标是实现兼具规则完整性、图形化交互、棋谱记录和可持续训练能力的人机博弈程序。爱因斯坦棋由骰子随机性与确定性走子共同构成：红、蓝双方各有编号1至6的六枚棋子，红方由左上角向右下角推进，蓝方由右下角向左上角推进；掷骰结果决定可行动棋子，若对应棋子已经被吃，则选择编号距离骰子点数最近的仍可行动棋子。任一方到达对方目标角，或吃光对方棋子，即取得胜利。
项目开发初期首先完成棋盘表示、随机或指定开局、掷骰选子、三方向移动、吃子、胜负判定和双方轮流行动等基础规则。主程序使用EinsteinGame维护GUI对局状态，棋盘以坐标到“阵营、编号”的映射保存，能够直接支持棋子定位、合法目标计算和局面恢复。随后加入玩家对玩家、玩家对AI、自由行走、悔棋、先手设置、状态提示等交互功能，并补充棋谱面板，对开局顺序、每次骰子、行动者、起止坐标、吃子、跳过回合和最终胜者进行连续记录。
在智能决策方面，项目先以启发式局面评价建立稳定基线，再形成HybridAI统一入口，并接入蒙特卡洛树搜索、MLP价值网络、GNN价值网络和Ensemble混合评估。GUI始终通过GameState.from_einstein(game)生成独立局面，再调用choose_move(state, die)获得“棋子编号、目标坐标”，因此模型选择和搜索实现可以演进，而不会改变玩家操作流程或棋盘规则。
为使AI能够通过数据迭代改进，项目进一步完善无GUI的GameState训练环境，加入局面克隆、合法动作枚举、固定动作索引、动作掩码、状态编码、动作执行和终局原因记录；同时提供自我对弈数据生成、MLP训练、新旧AI评估、统一模型归档与GUI加载接口。当前系统已形成“GUI交互—规则状态—混合AI—搜索决策—自我对弈—价值网络训练—自动对战评估—模型回载”的闭环。该闭环具备持续优化能力，但模型质量仍取决于自我对弈规模、探索多样性、训练收敛情况以及新旧模型对战结果，不能仅凭训练轮数保证提升。

2．软件的开发语言和调试环境
程序主要使用Python 3开发，图形界面采用标准库Tkinter；基础规则、启发式AI和GUI不依赖深度学习框架即可运行。神经网络训练与推理使用NumPy和PyTorch，项目依赖文件要求torch 2.0及以上版本。训练数据采用UTF-8编码的JSON Lines格式，模型采用PyTorch的.pt归档格式，当前开发与测试环境为Windows。
图形程序入口为einstein.py；训练和评估脚本均可在命令行独立运行，不导入Tkinter，也不使用硬编码绝对路径。程序会自动创建data等输出目录，并为自我对弈和评估提供随机种子及最大回合数参数，便于复现实验和防止异常对局无限运行。

3．软件模块与文件组织
einstein.py：主程序与GUI入口，包含EinsteinGame、EinsteinGUI、棋盘显示、玩家操作、AI回合、悔棋、自由行走、模型加载和棋谱记录。
einstein_ai.py：AI与无GUI训练核心，包含GameState、状态编码、启发式/MLP/GNN/Ensemble价值网络、ParallelMCTS、HybridAI、模型保存加载和兼容训练器。
train_selfplay.py：批量自我对弈并生成逐步JSONL训练样本。
train_value_model.py：读取JSONL样本训练MLP价值网络，打印训练集和验证集损失并保存统一模型归档。
train_value_network.py：保留边自我对弈边训练的兼容流程，可使用历史经验、评估旧模型并设置接受阈值。
evaluate_ai.py：让两类AI轮流执红、执蓝自动对战，统计胜率、平均步数、颜色胜率和终局原因。
模型与文档：MLP默认文件为einstein_value_model_mlp.pt，GNN默认文件为einstein_value_model_gnn.pt；AI_TRAINING_README.md记录训练流程与模型规范。

二、软件技术要点或创新性工作
本软件面向5×5爱因斯坦棋，整体由规则状态、图形界面、价值评估、蒙特卡洛树搜索、自我对弈、模型训练和自动评估等模块组成。设计重点是保证棋类规则与AI训练解耦：同一套GameState既能由GUI当前对局转换得到，也能在命令行中独立创建、复制、执行动作和判断终局。
1．规则一致的无GUI状态环境
GameState使用字典保存25个棋盘位置上的棋子归属与编号，并通过START_POSITIONS、GOAL_CORNERS和MOVE_OFFSETS统一描述开局区域、目标角和双方三个前进方向。clone()与copy()复制棋盘、行动方、胜者及终局原因，训练搜索过程不会修改GUI原局面；from_einstein()则负责从主程序的EinsteinGame复制当前棋局。
legal_actions(die)先根据骰子点数筛选可行动编号，再枚举相应棋子的三个方向，过滤越界和本方棋子占用位置。apply_action(action)返回新状态并处理起点清除、目标吃子、目标角获胜、吃光获胜和行动方切换。若没有合法动作，状态执行跳过回合。winner_reason区分goal与capture_all，既便于GUI提示，也便于评估脚本统计不同胜法。
2．含骰子信息的统一局面编码
神经网络输入维度为339，组成如下：25个棋位×13个棋盘通道=325维，其中红1至红6、蓝1至蓝6各占六个通道，另有一个占用通道；再加入当前行动方2维one-hot、骰子点数6维one-hot、双方剩余棋子数2维，以及双方平均与最短目标距离4维。骰子是爱因斯坦棋状态的一部分，相同棋盘在不同骰子点数下具有不同合法动作，因此自我对弈样本必须保存骰子并参与编码。
默认采用当前行动方视角归一化。蓝方行动时，棋盘旋转180度并交换双方身份，使当前行动方在棋盘通道中始终表现为向右下角推进，从而减少网络分别学习两套镜像模式的负担。为兼容已有327维旧MLP模型，加载器会根据第一层权重自动识别输入维度；旧模型推理时使用legacy编码，新训练流程则只接受当前339维样本。
3．固定18维动作空间与合法动作掩码
每方最多六枚棋子，每枚棋子最多三个方向，因此动作索引固定为6×3=18维。Action由label、direction和target组成，action_to_index()使用“(label-1)×3+direction”映射到0至17，index_to_action()完成逆变换。legal_action_mask(die)在合法索引处置1，其余置0，可用于训练策略网络或过滤模型输出。对GUI的外部接口仍返回(label, target)，从而保持原有点击、AI走子和棋谱记录逻辑不变。
4．多层次价值评估与自动降级
启发式评估：综合双方剩余棋子数、到目标角的推进距离、棋子编号价值、被对方攻击风险、对方接近终点威胁以及直接到达目标角等因素，最后经tanh归一化到[-1,1]。该评价不仅观察单步距离，还考虑材料、进攻和防守。
MLP价值网络：输入339维状态特征，经两个128单元全连接层和ReLU激活后，由单输出层及tanh给出当前行动方的局面价值。网络使用自我对弈终局结果作为监督信号，以均方误差损失训练。
GNN价值网络：将25个棋位视为图节点，每个节点输入13维棋盘特征；节点嵌入后进行两轮邻域平均与特征融合，再以全局平均池化输出局面价值。当前GNN路径主要利用棋盘拓扑特征，骰子和全局统计特征尚未直接进入图卷积，是后续可继续增强的方向。
Ensemble混合评估：默认融合MLP、GNN和启发式分值；当两个学习模型分歧较大时，提高启发式权重。当某个模型文件缺失或加载失败时，系统使用剩余模型与启发式继续运行；若学习模型均不可用，则退化为纯启发式，避免GUI因单个模型故障退出。
5．价值网络与随机模拟结合的蒙特卡洛树搜索
ParallelMCTS的节点记录访问次数N、累计价值W、平均价值Q、先验概率P、父子关系、对应动作和当前节点骰子。选择阶段采用Q+c_puct×P×sqrt(N_parent)/(1+N_child)平衡利用与探索；扩展阶段依据当前骰子产生全部合法子节点，无动作时产生跳过回合节点。
叶节点评估同时使用价值网络预测和有限深度随机模拟结果，并进行二次组合；回传时沿路径逐层累计价值并在双方轮次间交替取反，以表达零和对抗。搜索可由多个线程分别运行子树模拟，最后按根节点动作访问次数聚合选择着法。若模拟次数设为0，则直接对所有合法一步动作进行价值评估，适合快速生成数据和小规模测试。
6．自我对弈数据生成与价值学习
train_selfplay.py在无GUI环境中随机摆放双方编号并随机或按规则设置先手，每回合随机掷骰，由指定的heuristic、mlp、gnn或ensemble AI选择动作。每一步保存state、die、legal_mask、action_index、player、winner和value_target。终局后，以该步行动方为视角回填监督目标：最终获胜记+1，失败记-1，达到最大步数仍未终局记0。默认max_moves为200，seed参数用于复现实验。
train_value_model.py读取JSONL数据，过滤特征维度不匹配样本，随机划分训练集和验证集，使用Adam优化器和MSELoss训练MLP，并逐轮打印train loss与validation loss。标准命令为：python train_value_model.py --data data/selfplay.jsonl --epochs 20 --batch-size 128 --output einstein_value_model_mlp.pt。若输出位置已有兼容模型，脚本默认继续训练；使用--fresh可从新模型开始。
兼容脚本train_value_network.py保留边自我对弈边训练的流程，并支持历史经验回放、与启发式对战、与旧模型对战、接受胜率阈值和训练历史容量控制。因此项目具备持续迭代能力：生成新对局数据、训练候选模型、与旧模型或基线对战、达到标准后保存并在GUI回载。标准离线流程仍由命令行显式触发，便于保留训练可控性和实验记录。
7．统一的模型归档、版本与加载机制
模型文件统一命名为einstein_value_model_mlp.pt和einstein_value_model_gnn.pt，旧版einstein_value_model.pt作为MLP兼容路径。归档格式标识为einstein-ai-model-v2，保存format、value_kind、feature_size、state_dict、best_state_dict、history、replay_buffer和metadata，可记录训练脚本、样本量、损失、时间戳及历史版本。加载时根据权重形状恢复MLP输入维度和隐藏层大小，结构不匹配时给出当前编码维度及重新训练提示。
GUI选择MLP、GNN或Ensemble后，可输入模型路径并点击“加载模型”。MLP和GNN使用各自默认文件；Ensemble可由MLP路径自动推导配套GNN路径，并在状态栏显示实际加载组件。未安装torch、文件不存在、模型结构不兼容或部分组件失败时，程序均提供明确提示，而不会破坏玩家对玩家和启发式AI功能。
8．公平的自动对战评估
evaluate_ai.py让AI A与AI B进行批量自动对战，并逐局交换红蓝身份，以降低先手和颜色偏差。评估输出包括AI A胜率、AI B胜率、和局率、平均步数、红方胜率、蓝方胜率、到达目标角获胜次数和吃光对方获胜次数。通过固定随机种子、相同最大回合数和相同搜索预算，可以对候选模型、旧模型和启发式基线进行可重复比较。
9．图形化交互与棋谱记录
EinsteinGUI保留玩家对玩家、玩家对AI、玩家执红或执蓝、红蓝随机或指定摆放、随机先手、自由行走、掷骰、合法位置高亮、悔棋和胜负提示等功能。AI回合先从EinsteinGame复制GameState，再调用统一choose_move接口，因此训练环境和GUI规则保持一致。悔棋以完整快照恢复棋盘、轮次、骰子、候选棋子、胜者和棋谱状态；玩家对AI模式可连续回退到玩家可重新决策的位置。
棋谱功能在新局开始时记录双方初始摆放和先手，之后按回合记录玩家或AI、骰子点数、棋子编号、起点、终点、吃子信息、无棋可走的跳过回合和最终胜者。记录面板与实际走子使用同一调用链，AI加载模型后仍会正常记谱；复制功能可将当前棋谱复制到剪贴板，便于赛后复盘和问题定位。
10．稳定性验证与后续优化
项目主程序和AI、训练、评估脚本已通过python -m py_compile语法检查；无torch环境下可运行GUI规则和启发式自我对弈，安装torch后可加载现有MLP/GNN模型并完成小规模训练。训练脚本对缺失依赖、数据不存在、输入维度不兼容和输出目录不存在等情况均进行处理。
后续可从四方面继续提升：第一，增加直接输出18维动作概率的策略网络，用搜索访问分布训练策略；第二，让GNN同时接收骰子、行动方、棋子数量和距离等全局特征；第三，建立自动模型晋级机制，只有候选模型在交换颜色评估中稳定超过旧模型时才替换正式模型；第四，扩大并分层保存训练数据和评估记录，加入开局覆盖、难局采样、模型版本号和CSV实验日志。

三、软件参考文献
[1] Kocsis L, Szepesvári C. Bandit Based Monte-Carlo Planning. Machine Learning: ECML 2006, LNCS 4212, Springer, 2006: 282-293.
[2] Silver D, Huang A, Maddison C J, et al. Mastering the Game of Go with Deep Neural Networks and Tree Search. Nature, 2016, 529: 484-489.
[3] Kipf T N, Welling M. Semi-Supervised Classification with Graph Convolutional Networks. Proceedings of ICLR, 2017.
[4] Paszke A, Gross S, Massa F, et al. PyTorch: An Imperative Style, High-Performance Deep Learning Library. Advances in Neural Information Processing Systems, 2019, 32.
[5] Python Software Foundation. Python 3 Documentation: tkinter - Python interface to Tcl/Tk. https://docs.python.org/3/library/tkinter.html.

