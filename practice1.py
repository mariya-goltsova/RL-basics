import gym
import gym_maze
import numpy as np
import random
import time

env = gym.make('maze-sample-5x5-v0')
#env.reset()
action_n = 4
state_n = 25 # total number of states

class RandomAgent():
    def __init__(self, action_n):
        self.action_n = action_n
        
    def get_action(self, state):
        action = np.random.randint(self.action_n)
        return action
    
def get_state(obs):
    return int(np.sqrt(state_n) * obs[0] + obs[1])

def get_trajectory(env, agent, max_len = 1000, visualise = False):
    trajectory = {'states': [], 'actions' : [], 'rewards' : []}
    
    obs = env.reset()
    state = get_state(obs)

    #create traectory
    for _ in range(max_len):
        trajectory['states'].append(state)
        
        action = agent.get_action(state)
        trajectory['actions'].append(action)
        
        obs, reward, done, _ = env.step(action)
        trajectory['rewards'].append(reward)
        
        state = get_state(obs)
        
        if visualise:
            time.sleep(.1)
            env.render()
            
        if done:
            break
    
    return trajectory

#
class CrossEntropyAgent():
    def __init__(self, state_n, action_n):
        self.state_n = state_n
        self.action_n = action_n
        #матрица стратегия. сначала задается равномерно. сумма верятностей д-ий в строке = 1. по строкам состояния, по столбцам - д-я
        self.model = np.ones((self.state_n, self.action_n)) / self.action_n
    
    #получ. на вх сост-е, а выдает д-е
    def get_action(self, state):
        action = np.random.choice(np.arange(self.action_n), p = self.model[state])
        return int(action)
    
    def fit(self, elite_tr):
        new_model = np.zeros((self.state_n, self.action_n))
        for t in elite_tr:
            for state, action in zip(t['states'], t['actions']):
                new_model[state][action] += 1 
        for state in range(self.state_n):
            if np.sum(new_model[state]) > 0:
                new_model[state] /= np.sum(new_model[state]) 
            else:
                new_model[state] = self.model[state].copy()
        self.model = new_model
        return None
            
total_reward = 0

agent = CrossEntropyAgent(state_n, action_n)
trajectory_n = 50
iteration_n = 100
q_param = .9

for it in range(iteration_n):
    
    #policy evaluation
    trajectories = [get_trajectory(env, agent) for _ in range(trajectory_n)]
    total_reward = [np.sum(i['rewards']) for i in trajectories]
    print('iter', it, 'mean total reward', np.mean(total_reward))
    
    #policy improvement
    quantile = np.quantile(total_reward, q_param)
    elite_tr = []
    for tr in trajectories:
        r = np.sum(tr['rewards'])
        if r > quantile:
            elite_tr.append(tr)
            
    agent.fit(elite_tr)


tr = get_trajectory(env, agent, visualise = True)
print(tr)

