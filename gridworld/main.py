from Gridworld import Gridworld
import numpy as np
import torch
import random
from collections import deque
import copy
from model import Model

''' 
World info:
State: tensor representing positions of all objects on grid
Reward: -10 dying, -1 for non winning move, +10 for win
 '''

def action_set(choice):
    actions = {
        0: 'u',
        1: 'd',
        2: 'l',
        3: 'r',
    }
    return actions.get(choice)

def test_model(model, mode='static', display=True):
    i = 0
    test_game = Gridworld(mode=mode)
    state_ = test_game.board.render_np().reshape(1,64) + np.random.rand(1,64)/10.0
    state = torch.from_numpy(state_).float()
    if display:
        print("Initial State:")
        print(test_game.display())
    status = 1
    while(status == 1):
        qval = model(state)
        qval_ = qval.data.numpy()
        action_ = np.argmax(qval_)
        action = action_set(action_)
        if display:
            print('Move #: %s; Taking action: %s' % (i, action))
        test_game.makeMove(action)
        state_ = test_game.board.render_np().reshape(1,64) + np.random.rand(1,64)/10.0
        state = torch.from_numpy(state_).float()
        if display:
            print(test_game.display())
        reward = test_game.reward()
        if reward != -1:
            if reward > 0:
                status = 2
                if display:
                    print("Game won! Reward: %s" % (reward,))
            else:
                status = 0
                if display:
                    print("Game LOST. Reward: %s" % (reward,))
        i += 1
        if (i > 15):
            if display:
                print("Game lost; too many moves.")
            break
    win = True if status == 2 else False
    return win

def train_model(model, model2, epochs, sync_freq, mem_size, batch_size, replay, max_moves, gamma, loss_fn, optimizer, epsilon):
    losses = []
    move = 0
    j = 0
    for i in range(epochs):
        # initialize game
        game = Gridworld(size=4, mode='random')
        # add small amount of noise to board b/c we use ReLU which makes 0s non differntiable also helps with overfitting
        state_ = game.board.render_np().reshape(1,64) + np.random.rand(1,64)/10.0
        state1 = torch.from_numpy(state_).float()
        game_running = True
        move = 0
        # while game is running
        while (game_running):
            move += 1
            j += 1
            # get prediction Q value from model with input of state
            qVal = model(state1)
            qVal_ = qVal.data.numpy()
            # if epislon greed exploration
            if (random.random() < epsilon):
                action_ = np.random.randint(0, 4)
            # else pick high reward option
            else:
                action_ = np.argmax(qVal_)
            # convert to eligible action 
            action = action_set(action_)
            # make action on game board
            game.makeMove(action)
            # create board with noise for the new state
            state2_ = game.board.render_np().reshape(1,64) + np.random.rand(1,64)/10.0
            state2 = torch.from_numpy(state2_).float()
            # get the actual reward for the new state
            reward = game.reward()
            done = True if reward > 0 else False
            # experience data for this move
            exp =  (state1, action_, reward, state2, done)
            replay.append(exp)
            # update board state
            state1 = state2
            # if theres enough data, do batched expereince learning
            if len(replay) > batch_size:
                # grab subset of data
                minibatch = random.sample(replay, batch_size)
                # unpack experience replday data
                state1_batch = torch.cat([s1 for (s1,a,r,s2,d) in minibatch])
                action_batch = torch.Tensor([a for (s1,a,r,s2,d) in minibatch])
                reward_batch = torch.Tensor([r for (s1,a,r,s2,d) in minibatch])
                state2_batch = torch.cat([s2 for (s1,a,r,s2,d) in minibatch])
                done_batch = torch.Tensor([d for (s1,a,r,s2,d) in minibatch])
                # generate q values for batch of state 1
                Q1 = model(state1_batch)
                with torch.no_grad():
                    # generate q values for batch of state 2
                    Q2 = model2(state2_batch)
                # calculate target q values with discount (gamma), masked for terminal states
                Y = reward_batch + gamma * ((1 - done_batch) * torch.max(Q2,dim=1)[0])
                # extract predicted Q values for actions from state 1
                X = Q1.gather(dim=1,index=action_batch.long().unsqueeze(dim=1)).squeeze()
                # loss function of the difference between the state 1 Q values from Q model and state 2 q values from Q target model
                loss = loss_fn(X, Y.detach())
                # zero out gradients
                optimizer.zero_grad()
                # back prop
                loss.backward()
                losses.append(loss.item())
                # update parameters
                optimizer.step()
                if j % sync_freq == 0:
                    model2.load_state_dict(model.state_dict())
            if reward != -1 or move > max_moves: 
                game_running = False
                move = 0
        if epsilon > 0.1:
            epsilon -= (1/epochs)


def main():
    # create model and target model
    model = Model()
    model2 = copy.deepcopy(model)
    model2.load_state_dict(model.state_dict())
    # init parameters
    lr = 1e-3
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # discount factor
    gamma = 0.9
    epsilon = 1.0
    epochs = 5000
    # how often model and target model sync params
    sync_freq = 500
    # set up experience replay - Dequeue (FIFO) of 1000 size
    mem_size = 1000
    batch_size = 200
    replay = deque(maxlen=mem_size)
    max_moves = 50
    train_model(model, model2, epochs, sync_freq, mem_size, batch_size, replay, max_moves, gamma, loss_fn, optimizer, epsilon)
    # test model
    max_games = 1000
    wins = 0
    for i in range(max_games):
        win = test_model(model, mode='random', display=False)
        if win:
            wins += 1
    win_perc = float(wins) / float(max_games)
    print("Games played: {0}, # of wins: {1}".format(max_games,wins))
    print("Win percentage: {}".format(win_perc))

main()