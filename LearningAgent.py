import numpy as np
import matplotlib.pyplot as plt
import random

	
# all possible actions
up = 0
down = 1
left = 2
right = 3
actions = [up, down, left, right]

# grid-world
height = 4
width = 12

# initial state action value
stateActionValue = np.zeros((height, width, 4))
startState = [3, 0]
goalState = [3, 11]

# reward for each action in each state
actionReward = np.zeros((height, width, 4))
actionReward[:, :, :] = -1
actionReward[2, 1:11, down] = -100	
actionReward[3, 0, right] = -100

# state corresponding to each action
actionState = []
for i in range(0, height):
    actionState.append([])
    for j in range(0, width):
        state = dict()
        state[up] = [max(i - 1, 0), j]
        state[left] = [i, max(j - 1, 0)]
        state[right] = [i, min(j + 1, width - 1)]
        if i == 2 and 1 <= j <= 10:
            state[down] = startState
        else:
            state[down] = [min(i + 1, height - 1), j]
        actionState[-1].append(state)
actionState[3][0][right] = startState

#Choose Random action
def flipCoin(p):
    r = random.random()
    return r < p

# Compute the action to take in the current state.  With probability self.epsilon, 
# take a random action and take the best policy action otherwise
def chooseRandomAction(state, stateActionValue):
    if flipCoin(epsilon):
        return random.choice(actions)
    else:
        return np.argmax(stateActionValue[state[0], state[1], :])

epsilon = 0.1 
alpha = 0.4
gamma = 1 
		
# Sarsa
def sarsaAlgo(stateActionValue):
    currentState = startState
    currentAction = chooseRandomAction(currentState, stateActionValue)
    rewards = 0
    while currentState != goalState:
        newState = actionState[currentState[0]][currentState[1]][currentAction]
        newAction = chooseRandomAction(newState, stateActionValue)
        reward = actionReward[currentState[0], currentState[1], currentAction]
        rewards += reward
        valueTarget = stateActionValue[newState[0], newState[1], newAction]
        valueTarget *= gamma
        stateActionValue[currentState[0], currentState[1], currentAction] += alpha * (reward +
            valueTarget - stateActionValue[currentState[0], currentState[1], currentAction])
        currentState = newState
        currentAction = newAction
    return rewards

# Q-Learning
def qLearningAlgo(stateActionValue):
    currentState = startState
    rewards = 0
    while currentState != goalState:
        currentAction = chooseRandomAction(currentState, stateActionValue)
        reward = actionReward[currentState[0], currentState[1], currentAction]
        rewards += reward
        newState = actionState[currentState[0]][currentState[1]][currentAction]
        stateActionValue[currentState[0], currentState[1], currentAction] += alpha * (
            reward + gamma * np.max(stateActionValue[newState[0], newState[1], :]) -
            stateActionValue[currentState[0], currentState[1], currentAction])
        currentState = newState
    return rewards

def DrawLearningCurve():
    average = 10
    episodes = 500
    runs = 15
    rewardsSarsa = np.zeros(episodes)
    rewardsQLearning = np.zeros(episodes)
    for run in range(0, runs):
        stateActionValueSarsa = np.copy(stateActionValue)
        stateActionValueQLearning = np.copy(stateActionValue)
        for i in range(0, episodes):
            rewardsSarsa[i] += max(sarsaAlgo(stateActionValueSarsa), -100)
            rewardsQLearning[i] += max(qLearningAlgo(stateActionValueQLearning), -100)

    # averaging over independent runs
    rewardsSarsa /= runs
    rewardsQLearning /= runs

    # averaging over successive episodes
    afterSmoothSarsa = rewardsSarsa
    afterSmoothQLearning = rewardsQLearning
    for i in range(average, episodes):
        afterSmoothSarsa[i] = np.mean(rewardsSarsa[i - average: i + 1])
        afterSmoothQLearning[i] = np.mean(rewardsQLearning[i - average: i + 1])

  
    plt.figure("Learning Curve")
    plt.plot(afterSmoothSarsa, label='Sarsa')
    plt.plot(afterSmoothQLearning, label='Q-Learning')
    plt.xlabel('Episodes')
    plt.ylabel('Sum of rewards during episode')
    plt.legend()

DrawLearningCurve()
plt.show()
