import torch
from torch import optim
import numpy as np
from model import ActorCritic
import gymnasium as gym
from plotting import plot_episodes
import torch.multiprocessing as mp
from torch.nn import functional as F
'''
Problem: Keep pole within 12 degrees standing on a cart
State: Vector of cart position, cart velo, pole angle, pole velo
Reward: +1 every step pole angle NOT more than 12 degrees from center and cart position is in the window
'''

def run_episode(worker_env, worker_model, N_steps=10):
    raw_state = np.array(worker_env.env.state)
    state = torch.from_numpy(raw_state).float()
    values, logprobs, rewards = [],[],[]
    done = False
    j=0
    G=torch.Tensor([0])                       
    while j < N_steps and not done:    
        j+=1
        policy, value = worker_model(state)
        values.append(value)
        logits = policy.view(-1)
        action_dist = torch.distributions.Categorical(logits=logits)
        action = action_dist.sample()
        logprob_ = policy.view(-1)[action]
        logprobs.append(logprob_)
        state_, _, done, info = worker_env.step(action.detach().numpy())
        state = torch.from_numpy(state_).float()
        if done:
            reward = -10
            worker_env.reset()
        else:                                 
            reward = 1.0
            G = value.detach()
        rewards.append(reward)
    return values, logprobs, rewards, G

def update_params(worker_opt, values, logprobs, rewards, G, clc=0.1, gamma=0.95):
    # reverse order of all tensors
    rewards = torch.Tensor(rewards).flip(dims=(0,)).view(-1)          
    logprobs = torch.stack(logprobs).flip(dims=(0,)).view(-1)
    values = torch.stack(values).flip(dims=(0,)).view(-1)
    Returns = []
    ret_ = G
    # for each reward compute the return value and append it to returns
    for r in range(rewards.shape[0]):                                 
        ret_ = rewards[r] + gamma * ret_
        Returns.append(ret_)
    Returns = torch.stack(Returns).view(-1)
    Returns = F.normalize(Returns,dim=0)
    actor_loss = -1*logprobs * (Returns - values.detach())  
    # critic learns to predict the return          
    critic_loss = torch.pow(values - Returns,2)       
    # sum actor and critic loss to get an overall loss                
    loss = actor_loss.sum() + clc*critic_loss.sum()                   
    loss.backward()
    worker_opt.step()
    return actor_loss, critic_loss, len(rewards)

def worker(t, worker_model, counter, params):
    worker_env = gym.make("CartPole-v1")
    worker_env.reset()
    # each process runs its own isolated env but shares model
    worker_opt = optim.Adam(lr=1e-4,params=worker_model.parameters())
    worker_opt.zero_grad()
    for i in range(params['epochs']):
        worker_opt.zero_grad()
        # run game
        values, logprobs, rewards, g = run_episode(worker_env,worker_model)
        # update params
        actor_loss,critic_loss,eplen = update_params(worker_opt,values,logprobs,rewards,g)      
        # update global counter
        counter.value = counter.value + 1      

def main():
    # NN serves as Policy Network and Reward value for State
    # input -> state vector
    # output -> probability distribution over possible actions, and expected average reward for State
    MasterNode = ActorCritic()
    # allows params of model to be shared across processes
    MasterNode.share_memory()
    # store instantiated processes
    processes = []
    params = {
        'epochs': 1000,
        'n_workers': 7,
    }
    # global counter with type int
    counter = mp.Value('i', 0)
    # for each worker
    for i in range(params['n_workers']):
        # start worker process
        p = mp.Process(target=worker, args=(i, MasterNode, counter, params))
        p.start() 
        processes.append(p)
    for p in processes:
        # each process waits for it to finisih before returning to main process                    
        p.join()
    for p in processes:
        # make sure each process is terminated
        p.terminate()
    
    print(counter.value,processes[1].exitcode)

main()