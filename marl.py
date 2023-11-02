import server
import client
import utils
import logging
import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch import no_grad, eq
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from Model.Resnet8 import ResNet_cifar

# 决策网络
class TwoLayerNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size,hidden_size)
        self.fc3 = nn.Linear(hidden_size,output_size)
    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        return self.fc3(out)
    
class DDPG:
    # DDPG Agent
    def __init__(self, ddpg_id, state_dim, action_dim, critic_input_dim, hidden_dim,
                actor_lr, critic_lr, device, eps=0.1, num_online_clients=1):
        self.ddpg_id = ddpg_id
        self.actor = TwoLayerNet(state_dim, hidden_dim, action_dim).to(device)
        self.actor_target = TwoLayerNet(state_dim, hidden_dim, action_dim).to(device)

        self.critic = TwoLayerNet(critic_input_dim, hidden_dim, 1).to(device)
        self.critic_target = TwoLayerNet(critic_input_dim, hidden_dim, 1).to(device)

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.eps = eps
        self.num_online_clients = num_online_clients

    def take_action(self, state, done, explore=False):
        action = self.actor(state)
        if explore:
            action = utils.gumbel_softmax(action, self.eps)
        else:
            if not done:
                action = utils.khot_from_logits(action, self.num_online_clients)
            else:
                # all zero
                action = torch.zeros_like(action)
        return action.detach().cpu().numpy()[0]
    
    def soft_update(self, net, target_net, tau):
        for target_param, param in zip(target_net.parameters(), net.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
    
    def save_model(self, mark):
        name = mark+ '_ddpg_' + str(self.ddpg_id)
        torch.save(self.actor.state_dict(), name + '_actor.pth')
        torch.save(self.actor_target.state_dict(), name + '_actor_target.pth')
        torch.save(self.critic.state_dict(), name + '_critic.pth')
        torch.save(self.critic_target.state_dict(), name + '_critic_target.pth')

    def load_model(self, model):
        self.actor.load_state_dict(copy.deepcopy(model['actor']))
        self.actor_target.load_state_dict(copy.deepcopy(model['actor_target']))
        self.critic.load_state_dict(copy.deepcopy(model['critic']))
        self.critic_target.load_state_dict(copy.deepcopy(model['critic_target']))


class MADDPG:
    def __init__(self, agents_num, device, actor_lr, critic_lr, hidden_dim, state_dims, action_dims, 
                 critic_input_dim, gamma, tau, eps=0.1, num_online_clients=1):
        self.agents = []
        self.agents_num = agents_num
        self.eps = eps
        for i in range(self.agents_num):
            self.agents.append(DDPG(i, state_dims[i], action_dims[i], critic_input_dim, hidden_dim,
                                    actor_lr, critic_lr, device, eps, num_online_clients))
        self.gamma = gamma
        self.tau = tau
        self.device = device
        self.critic_criterion = nn.MSELoss()

    @property
    def policies(self):
        return [agent.actor for agent in self.agents]
    
    @property
    def target_policies(self):
        return [agent.actor_target for agent in self.agents]
    
    def take_action(self, states, done, explore):
        states = [
            torch.tensor([states[i]], dtype=torch.float, device=self.device)
            for i in range(self.agents_num)
        ]
        return [
            agent.take_action(state, done_i, explore)
            for agent, state, done_i in zip(self.agents, states, done)
        ]
    
    def update(self, samples, agent_id):
        obs, act, rew, next_obs, done = samples
        cur_agent = self.agents[agent_id]

        cur_agent.critic_optimizer.zero_grad()
        all_target_act = [
            utils.onehot_from_logits(pi(_next_obs))
            for pi, _next_obs in zip(self.target_policies, next_obs)
        ]
        target_critic_input = torch.cat((*next_obs, *all_target_act), dim=1)
        target_critic_value = rew[agent_id].view(-1, 1) + self.gamma * cur_agent.critic_target(
                                target_critic_input) * (1 - done[agent_id].view(-1, 1))
        critic_input = torch.cat((*obs, *act), dim=1)
        critic_value = cur_agent.critic(critic_input)
        critic_loss = self.critic_criterion(critic_value, target_critic_value.detach())
        critic_loss.backward()
        cur_agent.critic_optimizer.step()

        cur_agent.actor_optimizer.zero_grad()
        cur_actor_out = cur_agent.actor(obs[agent_id])
        cur_act_vf_in = utils.gumbel_softmax(cur_actor_out, self.eps)
        all_actor_acs = []
        for i, (pi, _obs) in enumerate(zip(self.policies, obs)):
            if i == agent_id:
                all_actor_acs.append(cur_act_vf_in)
            else:
                all_actor_acs.append(utils.onehot_from_logits(pi(_obs)))
        vf_in = torch.cat((*obs, *all_actor_acs), dim=1)
        actor_loss = -cur_agent.critic(vf_in).mean()
        actor_loss += (cur_actor_out**2).mean() * 1e-3
        actor_loss.backward()
        cur_agent.actor_optimizer.step()

    def update_all_targets(self):
        for agent in self.agents:
            agent.soft_update(agent.actor, agent.actor_target, self.tau)
            agent.soft_update(agent.critic, agent.critic_target, self.tau)

    def save_model(self, mark):
        for agent in self.agents:
            agent.save_model(mark)

    def load_model(self, model_path):
        for i in range(self.agents_num):
            model_dict = {}
            actor_model_path = model_path + str(i) + '_actor.pth'
            actor_target_model_path = model_path + str(i) + '_actor_target.pth'
            critic_model_path = model_path + str(i) + '_critic.pth'
            critic_target_model_path = model_path + str(i) + '_critic_target.pth'
            logging.info('load model from {}'.format(actor_model_path))
            model_dict['actor'] = torch.load(actor_model_path)
            model_dict['actor_target'] = torch.load(actor_target_model_path)
            model_dict['critic'] = torch.load(critic_model_path)
            model_dict['critic_target'] = torch.load(critic_target_model_path)
            self.agents[i].load_model(model_dict)






        

