'''' Trains an agent with stochastic Policy Gradients on Pong'''
from asyncio import create_subprocess_exec
import numpy as np
import pickle
import gym


#hyperparameters
H = 200 # number of layers
batch_size = 10 # how many episodes for a param update
learning_rate = 1e-4
gamma = 0.99 #discount factor
decay_rate = 0.99 # decay rate for RMSProp leaky sum of grad^2                                                                                                
resume = True #True if you want to resume from checkpoint
render = False

#model initilization
D = 80*80
if resume:
    with open('save.p', 'rb') as f:
    # load using pickle de-serializer
        model = pickle.load(f)
# pickle.load('save.p', 'rb') - original load 
else:
    model = {}
    model['W1'] = np.random.randn(H,D) / np.sqrt(D) # Xavier initilization
    model['W2'] = np.random.randn(H) / np.sqrt(H)

grad_buffer = {k: np.zeros_like(v) for k,v in model.items()} #update buffers that add up gradients
rmsprop_cache = {k: np.zeros_like(v) for k,v in model.items()} #rmsprop memory

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x)) #sigmoid 'squashing' function [0.1]

def prepro(I):
    '''prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector'''
    I = I[35:195] #crop
    I = I[::2,::2,0] #downsample by factor of 2
    I[I == 144] = 0 #erase background type 1
    I[I == 109] = 0 #erase background type 2
    I[I != 0] = 1  #everything else (paddles, ball) just set to 1

def discount_rewards(r):
    '''take 1D float array of rewards and compute discounted reward'''
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
        if r[t] !=0: running_add = 0 #reset sum
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r

def policy_forward(x):
    h = np.dot(model['W1'], x)
    h[h<0] = 0 #ReLu 
    logp = np.dot(model['W2'], h)
    p = sigmoid(logp)
    return p, h #return probability


def policy_backward(eph, epdlogp):
    '''backward pass. (eph is array of intermediate hidden states)'''
    dW2 = np.dot(eph.T, epdlogp).ravel()
    dh = np.outer(epdlogp, model['W2'])
    dh[eph <= 0] = 0
    dW1 = np.dot(dh.T, epx)
    return {'W1':dW1, 'W2':dW2}

env = gym.make('ALE/Pong-v5')
observation = env.reset()
prev_x = None #used in computing the difference frame
xs, hs, dlogps, drs = [],[],[],[]
running_reward = None
reward_sum = 0
episode_number = 0
while True:
    if render: env.render()

    #preprocess the observation, set input to network to be difference image
    cur_x = prepro(observation)
    x = cur_x - prev_x if prev_x is not None else np.zeros(D)
    prev_x = cur_x

    #forward the policy network and sample an action from the returned probability
    aprob, h = policy_forward(x)
    action = 2 if np.random.uniform() < aprob else 3 # roll the dice

    #record various intermediates (needed later for backprop)
    xs.append(x) #observation
    hs.append(h) #hidden state
    y = 1 if action == 2 else 0 # a fake label
    dlogps.append(y - aprob) #grad that encourages the action that was taken

    #step the environment and get new measurements
    observation, reward, done, info = env.step(action)
    reward_sum += reward

    drs.append(reward)

    if done:
        episode_number += 1

        #stack together
        epx = np.vstack(xs)
        eph = np.vstack(hs)
        epdlogp = np.vstack(dlogps)
        epr = np.vstack(drs)
        xs, hs, dlogps, drs = [],[],[],[] #reset array memory

        #compute the discounted reward backwards through time
        discounted_epr = discount_rewards(epr)
        #standardize the rewards
        discounted_epr -= np.mean(discounted_epr)
        discounted_epr /= np.std(discounted_epr)

        epdlogp *= discounted_epr #modulate the gradient with advantage
        grad = policy_backward(eph, epdlogp)
        for k in model: grad_buffer[k] += grad[k] #accumulate grad

        #perform rmspropr parameter update every batch_size episodes
        if episode_number % batch_size == 0:
            for k,v in model.items():
                g = grad_buffer[k] #gradient
                rmsprop_cache[k] = decay_rate * rmsprop_cache[k] + (1 - decay_rate) * g**2
                model[k] += learning_rate * g / (np.sqrt(rmsprop_cache[k]) + 1e-5)
                grad_buffer[k] = np.zeros_like(v) #reset batch gradient buffer

        #boring book-keeping
        running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
        print ('resetting env. episode reward total was %f. running mean: %f' % (reward_sum, running_reward))
        if episode_number % 100 == 0: pickle.dump(model, open('save.p', 'wb'))
        reward_sum = 0
        observation = env.reset() # reset env
        prev_x = None

    if reward != 0: # Pong has either +1 or -1 reward exactly when game ends.
        print ('ep %d: game finished, reward: %f' % (episode_number, reward) + ('' if reward == -1 else ' !!!!!!!!'))
        






