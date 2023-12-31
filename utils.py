import torch
import torch.nn.functional as F
import collections
import random
import numpy as np
import threading

shared_lock = threading.Lock()

def onehot_from_logits(logits, eps=0.01):
    ''' 生成最优动作的独热 (one-hot) 形式 '''
    argmax_acs = (logits == logits.max(1, keepdim=True)[0]).float()
    # 生成随机动作,转换成独热形式
    rand_acs = torch.autograd.Variable(torch.eye(logits.shape[1])[[
        np.random.choice(range(logits.shape[1]), size=logits.shape[0])
    ]], requires_grad=False).to(logits.device)
    # 通过epsilon-贪婪算法来选择用哪个动作
    return torch.stack([
        argmax_acs[i] if r > eps else rand_acs[i]
        for i, r in enumerate(torch.rand(logits.shape[0]))
    ])

def onehot_logits(logits, eps=0.01):
    ''' 生成最优动作的独热 (one-hot) 形式 '''
    argmax_acs = (logits == logits.max(1, keepdim=True)[0]).float()
    # 生成随机动作,转换成独热形式
    return argmax_acs

def k_onehot_from_logits(logits, eps=0.01):
    ''' 生成最优动作的独热 (one-hot) 形式 '''
    argmax_acs = (logits == logits.max(1, keepdim=True)[0]).float()
    logits = logits - argmax_acs
    # 生成随机动作,转换成独热形式
    rand_acs = torch.autograd.Variable(torch.eye(logits.shape[1])[[
        np.random.choice(range(logits.shape[1]), size=logits.shape[0])
    ]], requires_grad=False).to(logits.device)
    # 通过epsilon-贪婪算法来选择用哪个动作
    return torch.stack([
        argmax_acs[i] if r > eps else rand_acs[i]
        for i, r in enumerate(torch.rand(logits.shape[0]))
    ]), logits

def ktop_from_logits(logits, k):
    ''' 生成最优动作的K热编码 (k-hot encoding) 形式 '''
    logits = np.array(logits)
    logits = torch.tensor(logits)
    top_k_acs = torch.topk(logits, k, dim=1)[1]
    khot_acs = torch.zeros_like(logits)
    khot_acs.scatter_(1, top_k_acs, 1)
    return khot_acs.float()

def onetop_logits(logits):
    ''' 生成最优动作的K热编码 (k-hot encoding) 形式 '''
    logits = np.array(logits)
    logits = torch.tensor(logits)
    top_k_acs = torch.topk(logits, 1, dim=1)[1]
    khot_acs = torch.zeros_like(logits)
    khot_acs.scatter_(1, top_k_acs, 1)
    return khot_acs.float()

def onetop_from_logits(logits, eps=0.01):
    ''' 生成最优动作的K热编码 (k-hot encoding) 形式 '''
    logits = np.array(logits)
    logits = torch.tensor(logits)
    top_k_acs = torch.topk(logits, 1, dim=1)[1]
    khot_acs = torch.zeros_like(logits)
    khot_acs.scatter_(1, top_k_acs, 1)
    # 生成随机动作,转换成独热形式
    rand_acs = torch.autograd.Variable(torch.eye(logits.shape[1])[[
        np.random.choice(range(logits.shape[1]), size=logits.shape[0])
    ]], requires_grad=False).to(logits.device)
    # 通过epsilon-贪婪算法来选择用哪个动作
    return torch.stack([
        khot_acs[i] if r > eps else rand_acs[i]
        for i, r in enumerate(torch.rand(logits.shape[0]))
    ])
    # return khot_acs.float()

def khot_from_logits(logits, k, eps=0.01):
    ''' 生成最优动作的K热编码 (k-hot encoding) 形式 '''
    new_logits = logits
    action, new_logits = k_onehot_from_logits(new_logits, eps)
    for i in range(k - 1):
        new_action, new_logits = k_onehot_from_logits(new_logits, eps)
        action = action + new_action
    return action


    # ''' 生成最优动作的K热编码 (k-hot encoding) 形式 '''
    # top_k_acs = torch.topk(logits, k, dim=1)[1]
    # khot_acs = torch.zeros_like(logits)
    # khot_acs.scatter_(1, top_k_acs, 1)

    # # 通过epsilon-贪婪算法来选择用哪个动作
    # rand_acs = torch.autograd.Variable(torch.zeros_like(logits))
    # rand_indices = np.random.choice(range(logits.shape[1]), size=(logits.shape[0], k), replace=False)
    # for i in range(logits.shape[0]):
    #     rand_acs[i][rand_indices[i]] = 1

    # return torch.stack([
    #     khot_acs[i] if r > eps else rand_acs[i]
    #     for i, r in enumerate(torch.rand(logits.shape[0]))
    # ])


def sample_gumbel(shape, eps=1e-20, tens_type=torch.FloatTensor):
    """从Gumbel(0,1)分布中采样"""
    U = torch.autograd.Variable(tens_type(*shape).uniform_(),
                                requires_grad=False)
    return -torch.log(-torch.log(U + eps) + eps)

def gumbel_softmax_sample(logits, temperature):
    """ 从Gumbel-Softmax分布中采样"""
    y = logits + sample_gumbel(logits.shape, tens_type=type(logits.data)).to(
        logits.device)
    return F.softmax(y / temperature, dim=1)

def gumbel_softmax(logits, temperature=1.0, eps=0.1):
    """从Gumbel-Softmax分布中采样,并进行离散化"""
    y = gumbel_softmax_sample(logits, temperature)
    y_hard = onehot_from_logits(y, eps)
    y = (y_hard.to(logits.device) - y).detach() + y
    # 返回一个y_hard的独热量,但是它的梯度是y,我们既能够得到一个与环境交互的离散动作,又可以
    # 正确地反传梯度
    return y 

class ReplayBuffer:
    # 经验回放池
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        # 添加经验
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done

    def size(self):
        return len(self.buffer)
