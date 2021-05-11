import argparse
import pickle
from collections import namedtuple
from itertools import count
import random
import os, time
import numpy as np
import sys
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal, Categorical
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter

import sys
sys.path.append('d:/jidi/ppo/snakes')
from env.chooseenv import make


height=6
width=8

class Actor(nn.Module):
    def __init__(self,observation_space,action_space):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(observation_space, 128)
        self.action_head = nn.Linear(128, action_space)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x=self.action_head(x)
        action_prob = F.softmax(x, dim=1)
        return action_prob


class Critic(nn.Module):
    def __init__(self,observation_space):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(observation_space, 128)
        self.state_value = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        value = self.state_value(x)
        return value


class PPO():


    def __init__(self,observation_space,action_space,args):
        super(PPO, self).__init__()
        self.clip_param = args.clip_param
        self.max_grad_norm = args.max_grad_norm
        self.ppo_update_time = args.ppo_update_time
        self.buffer_capacity = args.buffer_capacity
        self.batch_size = args.batch_size
        self.gamma = args.gamma
        self.eps = args.eps
        self.entropy_coef = args.entropy_coef

        self.actor_net = Actor(observation_space,action_space)
        self.critic_net = Critic(observation_space)
        self.buffer = []
        self.counter = 0
        self.training_step = 0
        self.writer = SummaryWriter('runs')

        self.lossvalue_norm=True
        self.loss_coeff_value=0.5

        self.actor_optimizer = optim.Adam(self.actor_net.parameters(), 1e-3)
        self.critic_net_optimizer = optim.Adam(self.critic_net.parameters(), 3e-3)
        if not os.path.exists('runs'):
            os.makedirs('runs')
            os.makedirs('runs')

    def select_action(self, state,train):
        state = torch.from_numpy(state).float().unsqueeze(0)
        with torch.no_grad():
            action_prob = self.actor_net(state)
            if train==True:
                if random.random() > self.eps:
                    action_prob = action_prob
                else:
                    action_prob = torch.tensor(np.random.uniform(low=0, high=1, size=(1,4)))
                self.eps*=0.99999
                self.eps=max(self.eps,0.01)
            else:
                action_prob = action_prob
        c = Categorical(action_prob)
        action = c.sample()
        #todo:the sum of probs here not equals 1
        # can let prob=c.probs[action.item()] to solve this
        return action.item(), action_prob[:, action.item()].item()

    def save_param(self):
        torch.save(self.actor_net.state_dict(), 'actor_net.pth')
        torch.save(self.critic_net.state_dict(), 'critic_net.pth')
        print('save completely')

    def load(self):
        self.actor_net.load_state_dict(torch.load( 'actor_net.pth'))
        self.critic_net.load_state_dict(torch.load('critic_net.pth'))
        print('load completely')

    def store_transition(self, transition):
        self.buffer.append(transition)
        self.counter += 1

    def update(self, i_ep):
        state = torch.tensor([t.state for t in self.buffer], dtype=torch.float)
        action = torch.tensor([t.action for t in self.buffer], dtype=torch.long).view(-1, 1)
        reward = [t.reward for t in self.buffer]
        # update: don't need next_state
        # reward = torch.tensor([t.reward for t in self.buffer], dtype=torch.float).view(-1, 1)
        # next_state = torch.tensor([t.next_state for t in self.buffer], dtype=torch.float)
        old_action_log_prob = torch.tensor([t.a_log_prob for t in self.buffer], dtype=torch.float).view(-1, 1)

        R = 0
        Gt = []
        for r in reward[::-1]:
            R = r + self.gamma * R
            Gt.insert(0, R)
        Gt = torch.tensor(Gt, dtype=torch.float)
        # print("The agent is updateing....")
        for i in range(self.ppo_update_time):
            #
            self.entropy_coef*=0.9999

            for index in BatchSampler(SubsetRandomSampler(range(len(self.buffer))), self.batch_size, False):
                if self.training_step % 1000 == 0:
                    print('I_ep {} ，train {} times'.format(i_ep, self.training_step))
                # with torch.no_grad():
                Gt_index = Gt[index].view(-1, 1)
                V = self.critic_net(state[index])
                delta = Gt_index - V
                advantage = delta.detach()
                # epoch iteration, PPO core!!!
                #根据action,选出其在actor网络训练得到对应的 action_prob
                action_prob = self.actor_net(state[index]).gather(1, action[index])  # new policy
                c = Categorical(action_prob)
                entropy_loss=c.entropy().mean()
                    #  entropy = -(action_prob*torch.log(old_action_log_prob+1.e-10)+ \
                    #  (1.0-action_prob)*torch.log(1.0-old_action_log_prob+1.e-10))

                a=torch.tensor(1e-5,requires_grad=True)
                ratio = action_prob / torch.maximum(old_action_log_prob[index],a.repeat(action_prob.shape,1))
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * advantage

                loss_surr=-(torch.min(surr1, surr2)).mean()
                if self.lossvalue_norm:
                    return_std=Gt[index].std()
                    loss_value=torch.mean((self.critic_net(state[index])-Gt[index]).pow(2))/return_std
                else:
                    loss_value = torch.mean((self.critic_net(state[index]) - Gt[index]).pow(2))

                # update actor network
                action_loss = loss_surr+self.loss_coeff_value*loss_value-entropy_loss*max(self.entropy_coef,0.2)
                self.writer.add_scalar('train_loss/action_loss', action_loss, global_step=self.training_step)

                self.actor_optimizer.zero_grad()
                action_loss.backward()
                nn.utils.clip_grad_norm_(self.actor_net.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

                # update critic network
                value_loss = F.mse_loss(Gt_index, V)
                self.writer.add_scalar('train_loss/value_loss', value_loss, global_step=self.training_step)
                self.critic_net_optimizer.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.critic_net.parameters(), self.max_grad_norm)
                self.critic_net_optimizer.step()
                self.training_step += 1
            #print('i', i, 'entropy_coef', self.entropy_coef)  # MAX->MIN desent

        del self.buffer[:]  # clear experience

def get_observation(state,info,observation_space):
    state = np.array(state)
    state=np.squeeze(state,2)
    observations=np.zeros(observation_space)

    snakes_position=np.array(info['snakes_position'])
    beans_position=np.array(info['beans_position']).flatten()

    # head position of the controlled snake
    observations[:2]=snakes_position[0][0]

        # 3*3 surroundings around the snake's head
    head_x = snakes_position[0][0][1]
    head_y = snakes_position[0][0][0]
    head_surrounding=get_surrounding(state,head_x,head_y)

    # get beans' position
    observations[2:12]=beans_position[:]

    #get other snakes' heads
    observations[12:14] = snakes_position[1][0][:]

    #get surroundings' situation
    observations[14:18]=head_surrounding[:]

    return observations

def get_surrounding(state,x,y):

    a1=state[(y+1)%height][x]
    a2=state[(y-1)%height][x]
    a3=state[y][(x-1)%width]
    a4=state[y][(x+1)%width]
    surrounding=[a1,a2,a3,a4]
    return surrounding

def joint_action_func(action, state, info, opponent_policy):
    joint_action=np.zeros(1*2)
    if opponent_policy=='random':
        other_snake=random.randint(0,3)
    else:
        other_snake=greedy_snake(state,info)
    joint_action[:1]=action
    joint_action[1:]=other_snake
    return joint_action

def greedy_snake(state,info):
    pass

def main(args):
    # Parameters
    game='snakes_1v1'
    global env
    env=make(game,conf=None)

    gamma =args.gamma
    #render = True

    seed = args.seed

    # wrapped

    observation_space = 18
    #print('observation_space',observation_space)

    action_space = env.get_action_dim()
    #print('action_space',action_space)
    opponent_policy=args.opponent_policy
    torch.manual_seed(seed)

    Agent = globals()[str(args.algo)]
    agent= Agent(observation_space,action_space,args)
    Transition = namedtuple('Transition', ['state', 'action', 'a_log_prob', 'reward'])

    if args.mode=='train':
        # logger = get_log()
        # logger.info("info message")

        #agent.load()

        for i_epoch in range(args.max_episode):
            state,info = env.reset()
            observation=get_observation(state,info,observation_space)
            episode_reward=np.zeros(2)
            #if render: env.render()

            for t in count():
                # todo:select action with noise
                action, action_prob = agent.select_action(observation,train=True)
                joint_action = joint_action_func(action, state, info,opponent_policy)
                next_state, reward, done, _,info = env.step(env.encode(joint_action))
                snakes_position = np.array(info['snakes_position'], dtype=object)
                beans_position = np.array(info['beans_position'], dtype=object)
                snake_heads = [snake[0] for snake in snakes_position]
                episode_reward+=reward
                step_reward=np.zeros(1)

                if reward[0]>0:
                    step_reward[0]+=20
                else:
                    self_head = np.array(snake_heads[0])
                    dists = [np.sqrt(np.sum(np.square(other_head - self_head))) for other_head in beans_position]
                    step_reward[0] -= min(dists)
                    if reward[0] < 0:
                        step_reward[0] -= 10

                if done:
                    if np.sum(episode_reward[:1])>np.sum(episode_reward[1:2]):
                        #print('reward_before',reward)
                        step_reward[0]+=50
                        #print('reward_before',reward)
                    elif np.sum(episode_reward[1:2])>np.sum(episode_reward[:1]):
                        #reward=abs(state[0])
                        step_reward[0]-=25
                    else:
                        step_reward[0]+=10

                else:
                    if np.sum(episode_reward[:1])> np.sum(episode_reward[1:2]):
                        # print('reward_before',reward)
                        step_reward[0] += 10
                        # print('reward_before',reward)
                    else:
                        step_reward[0] -= 5



                trans = Transition(observation, action, action_prob, step_reward)
                #print('reward',reward)
                agent.store_transition(trans)

                state = next_state
                next_observation=get_observation(state,info,observation_space)
                observation=next_observation


                if done:

                    if len(agent.buffer) >= agent.batch_size:
                        agent.update(i_epoch)
                    agent.writer.add_scalar('Steptime/steptime', t, global_step=i_epoch)
               # agent.writer.add_scalar('episode/return', sum(r * gamma ** t for t, r in enumerate(rewards)), i_epoch)
                    agent.writer.add_scalar('train/return', episode_reward[0], i_epoch)
                    agent.writer.add_scalar('train/opponent_return', episode_reward[1], i_epoch)

                    print('episode',i_epoch,'return', episode_reward[0],'eps',agent.eps)
                    print('episode',i_epoch,'opponent_return',episode_reward[1])

                    break

        agent.save_param()

    else:
        win_times=0
        agent.load()
        for i_epoch in range(args.eval_times):
            state,info=env.reset()
            Gt_0=0
            Gt_1=0
            observation=get_observation(state,info,observation_space)
            for i in count():
                action, action_prob = agent.select_action(observation, train=False)
                joint_action = joint_action_func(action, state, info, opponent_policy)
                next_state, reward, done, _, info = env.step(env.encode(joint_action))
                Gt_0+=reward[0]
                Gt_1+=reward[1]
                state=next_state
                observation=get_observation(next_state,info,observation_space)
                if done:
                    if Gt_0>Gt_1:
                        win_times+=1

                    print('episode',i_epoch,'Gt_0',Gt_0)
                    print('episode',i_epoch,'opponent_return',Gt_1)
                    agent.writer.add_scalar('evaluate/return', Gt_0, i_epoch)
                    agent.writer.add_scalar('evaluate/opponent_return', Gt_1, i_epoch)
                    break
        print('win_times is :',win_times)
        print('----------------------------------------------------------------------------------------')





if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # parser.add_argument('--scenario', default="classic_CartPole-v0", type=str)

    parser.add_argument('--max_episode', default=100000,type=int)
    parser.add_argument('--algo', default="PPO", type=str, help="dqn/PPO/a2c")
    parser.add_argument('--buffer_capacity', default=int(1e5), type=int)
    parser.add_argument('--seed', default=777, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--clip_param', default=0.2, type=int)
    parser.add_argument('--max_grad_norm', default=0.5, type=int)
    parser.add_argument('--ppo_update_time', default=10, type=int)
    parser.add_argument('--gamma', default=0.99)
    parser.add_argument('--entropy_coef', default=0.5)
    parser.add_argument('--eps', default=1)
    parser.add_argument('--opponent_policy', default='random',help='random/greedy')
    parser.add_argument('--mode', default="train", type=str, help="train/evaluate")
    parser.add_argument('--eval_times', default=1000, type=int)

    args = parser.parse_args()
    main(args)
    print("end")
