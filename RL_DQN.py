# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 10:01:09 2020

@author: Administrator
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 16:41:37 2020

@author: Administrator
"""

# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import tensorflow as tf
tf.reset_default_graph() 
np.random.seed(1)
tf.set_random_seed(1)
 
 
# Deep Q Network off-policy
'''
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
'''

class DeepQNetwork:
    def __init__(
            self,
            n_actions,
            n_features,
            memory,    #记忆库
            epoch=1000,
            learning_rate=0.01,
            replace_target_iter=100,
            reward_decay=0.9,
            batch_size=32,
    ):
        self.n_actions = n_actions 
        self.n_features = n_features
        self.memory = memory
        self.epoch=epoch
        self.lr = learning_rate
        self.replace_target_iter = replace_target_iter
        self.gamma = reward_decay
        
        self.batch_size = batch_size   #神经网络学习 

        self._build_net()

        self.memory_size = len(memory)
        self.learn_step_counter=0
        self.sess = tf.Session()
 
        self.sess.run(tf.global_variables_initializer())
        self.cost_his = []
 
    def _build_net(self):
        # ------------------ build evaluate_net ------------------
        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')  
        # input
        self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name='Q_target')  # for calculating loss
        with tf.variable_scope('eval_net'):
            # c_names(collections_names) are the collections to store variables
            c_names, n_l1, w_initializer, b_initializer = \
                ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES], 64,\
                tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)  # config of layers
 
            # first layer. collections is used later when assign to target net
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.s, w1) + b1)
 
            # second layer. collections is used later when assign to target net
            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                self.q_eval = tf.nn.relu(tf.matmul(l1, w2) + b2)
                
#            with tf.variable_scope('l3'):
#                w3 = tf.get_variable('w3', [n_l2, self.n_actions], initializer=w_initializer, collections=c_names)
#                b3 = tf.get_variable('b3', [1, self.n_actions], initializer=b_initializer, collections=c_names)
#                self.q_eval = tf.matmul(l2, w3) + b3    #[batch_size,self.n_action]
 
        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
        with tf.variable_scope('train'):
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)
 
        # ------------------ build target_net ------------------
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')    # input
        with tf.variable_scope('target_net'):
            # c_names(collections_names) are the collections to store variables
            c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]
 
            # first layer. collections is used later when assign to target net
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.s_, w1) + b1)
 
            # second layer. collections is used later when assign to target net
            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                self.q_next = tf.nn.relu(tf.matmul(l1, w2) + b2)
                
#            with tf.variable_scope('l3'):
#                w3 = tf.get_variable('w3', [n_l2, self.n_actions], initializer=w_initializer, collections=c_names)
#                b3 = tf.get_variable('b3', [1, self.n_actions], initializer=b_initializer, collections=c_names)
#                self.q_next = tf.matmul(l2, w3) + b3
 

 
    def choose_action(self, observation):
        # to have batch dimension when feed into tf placeholder
        observation = observation[np.newaxis, :]   #shape=(1,n_features)
        actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
        action = np.argmax(actions_value)#未加axis＝,返回一个索引数值

        return action
    
    def _replace_target_params(self):
        t_params = tf.get_collection('target_net_params')
        e_params = tf.get_collection('eval_net_params')
        self.sess.run([tf.assign(t, e) for t, e in zip(t_params, e_params)])
 

 
    def learn(self):
        # check to replace target parameters
        if self.learn_step_counter%self.replace_target_iter==0:
            self._replace_target_params()
 
        # sample batch memory from all memory
        for i in range(0,self.epoch):
            
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        
            batch_memory = self.memory[sample_index, :]
 
    
             #通过神经网络得到q_next, q_eval 
            q_next, q_eval = self.sess.run(
                    [self.q_next, self.q_eval],
                    feed_dict={
                            #[s, a, s', r]
                            self.s_: batch_memory[:, (self.n_features+1):(2*self.n_features+1)],  # fixed params
                            self.s: batch_memory[:, :(self.n_features)],  # newest params
                    })
 
            # change q_target w.r.t q_eval's action
            q_target = q_eval.copy()
 
            batch_index = np.arange(self.batch_size, dtype=np.int32)
            eval_act_index = batch_memory[:, self.n_features].astype(int)  #action   astype(int) 转换数组的数据类型
            reward = batch_memory[:, -1]                  #reward                       
            
            #更新q_target参数
            q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next, axis=1)
 
            # train eval network
            _, self.cost = self.sess.run([self._train_op, self.loss],
                                     feed_dict={self.s: batch_memory[:, :self.n_features],
                                                self.q_target: q_target})
            self.cost_his.append(self.cost)
            print('第'+str(i)+'轮:  loss:'+str(self.cost))
            # increasing epsilon
            self.learn_step_counter += 1
 
    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()

 


#from RL_brain_DQN import DeepQNetwork
import tensorflow as tf
tf.reset_default_graph()

#F:[[s,a,s',r]]

#F为一个矩阵，包含了所有[s,a,s',r]转移状态
F=np.array([[]])



n_actions=6
n_features=32
batchsize=int(len(F)/10)
RL = DeepQNetwork(n_actions, n_features,
                  memory=F,
                  epoch=50,
                  learning_rate=0.05,
                  replace_target_iter=1, #由于为回顾型数据，无法与环境交互，直接设为1
                  reward_decay=0.9,
                  batch_size=batchsize)
RL.learn()
RL.plot_cost()
