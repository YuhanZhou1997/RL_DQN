# RL_DQN
输入：memory: ((s,a,s',r))
     n_actions: 动作数量。 每个动作用数字表示，从0开始
     n_features: 每个状态s的特征数量
     epoch: 训练轮数
     learning_rate：神经网络学习率
     replace_target_iter： 代理每试探多少下，更新神经网络target_net参数
     reward_decay：奖励衰减系数
     batch_size：每次从memory中采样多少个进行训练
函数：.choose_action（） 输入状态代理选择动作
     .learn() 进行训练
     .plot_cost() 画出每轮训练误差
     

由于使用回顾型数据，agent无法与环境进行直接交互
因此直接存储完所有的one-step-transition [s,a,s',r]进行训练
