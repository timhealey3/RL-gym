import torch
import numpy as np
from model import Model
import gymnasium as gym
from plotting import plot_episodes
'''
Problem: Keep pole within 12 degrees standing on a cart
State: Vector of cart position, cart velo, pole angle, pole velo
Reward: +1 every step pole angle NOT more than 12 degrees from center and cart position is in the window
'''

# multiply probability by discounted returnn
# use probability to back prop 
def discount_rewards(rewards, gamma=0.99):
    lenr = len(rewards)
    # compute exponentially decaying rewards to discourage actions that lead directly to failure
    disc_return = torch.pow(gamma, torch.arange(lenr).float()) * rewards
    # normalize rewards to be within 0, 1
    disc_return /= disc_return.max()
    return disc_return

# Policy gradient loss function
# Args: 
# preds -> 1D tensor of action probabilities for the actions taken at timestamp t
# r -> 1D tensor of discounted rewards for timestamp t
# Returns:
# tensor -> scalar loss value
def loss_fn(preds, r):
    return -1 * torch.sum(r * torch.log(preds))

def training(model, env, learning_rate, optimizer, MAX_DUR, MAX_EPISODES, gamma):
    # REINFORCE training loop
    # The score is how many steps it took to lose, because the more steps, the longer it took to lose
    score = []
    for episodes in range(MAX_EPISODES):
        # reset vars
        curr_state, info = env.reset()
        done = False
        transitions = []
        # run a game
        for t in range(MAX_DUR):
            # all action probabilities
            act_prob = model(torch.from_numpy(curr_state).float()) 
            # sample from probabilites and pick action 0 or 1 
            action = np.random.choice(np.array([0,1]), p=act_prob.data.numpy())
            prev_state = curr_state
            # environment "responds" by producing new state and reward
            curr_state, reward, done, truncate, info = env.step(action)
            # add state, action, and timestamp to transitions 
            transitions.append((prev_state, action, t+1))
            # break if lost
            if done:
                break
        
        ep_len = len(transitions)
        # add to score the amount of steps
        score.append(ep_len)
        # all rewards in one tensor 
        reward_batch = torch.Tensor([r for (s,a,r) in transitions]).flip(dims=(0,))
        # computes discounted versions of the rewards
        disc_rewards = discount_rewards(reward_batch)
        state_batch = torch.Tensor([s for (s,a,r) in transitions])
        action_batch = torch.Tensor([a for (s,a,r) in transitions])
        # Recomputes the action probabilities for all the states in the episode
        pred_batch = model(state_batch)
        # Subsets the action-probabilities associated with the actions that were actually taken
        prob_batch = pred_batch.gather(dim=1,index=action_batch.long().view(-1,1)).squeeze()
        loss = loss_fn(prob_batch, disc_rewards)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return score

def main():
    # NN serves as Policy Network
    # input -> state vector
    # output -> probability distribution over possible actions
    model = Model()
    learning_rate = 0.0009
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # create gym environement for CarPole problem
    env = gym.make('CartPole-v1')
    MAX_DUR = 200
    MAX_EPISODES = 500
    gamma = 0.99
    # train
    score = training(model, env, learning_rate, optimizer, MAX_DUR, MAX_EPISODES, gamma)
    # plot
    plot_episodes(score)

main()