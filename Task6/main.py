import minigrid
import gymnasium as gym
import numpy as np
import random
from collections import defaultdict

np.random.seed(33)

class QLearningAgent:
    def __init__(
        self,
        n_actions,
        alpha=0.1,      # learning rate
        gamma=0.99,     # discount
        epsilon=0.1     # exploration
    ):
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.Q = defaultdict(lambda: np.zeros(n_actions))

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        return np.argmax(self.Q[state])

    def update(self, state, action, reward, next_state, done):
        best_next = np.max(self.Q[next_state])
        target = reward if done else reward + self.gamma * best_next
        self.Q[state][action] += self.alpha * (target - self.Q[state][action])
        
def extract_state(env):
    base_env = env.unwrapped   
    x, y = base_env.agent_pos
    direction = base_env.agent_dir
    return (x, y, direction)


def train(
    episodes=1501,
    max_steps=300,
    alpha=0.1,
    gamma=0.99,
    epsilon=0.1
):
    env = gym.make(
        "MiniGrid-FourRooms-v0",
        max_steps=max_steps,
        goal_pos=(2, 2),
        agent_pos=(7, 7)
    )

    agent = QLearningAgent(
        n_actions=env.action_space.n,
        alpha=alpha,
        gamma=gamma,
        epsilon=epsilon
    )

    rewards_history = []

    for episode in range(episodes):
        obs, _ = env.reset()
        state = extract_state(env)
        total_reward = 0


        for step in range(max_steps):
            action = agent.select_action(state)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state = extract_state(env)
            agent.update(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            if done:
                break
        rewards_history.append(total_reward)
        if episode > 500:
            agent.epsilon = max(0.05, agent.epsilon * 0.995)


        if (episode + 1) % 100 == 0:
            avg = np.mean(rewards_history[-500:])
            print(f"Epizod {episode+1}, średnia nagroda (100): {avg:.3f}")

    env.close()
    return agent, rewards_history



if __name__ == "__main__":
    agent, rewards = train(
        episodes=1501,
        alpha=0.2,
        gamma=0.99,
        epsilon=0.4
)
    print('gamma =' + '0.99' +  ' + ' + 'alpha =' + '0.2' +  ' + ' 'epsilon = ' + '0.4')
    
    agent, rewards = train(
        episodes=1501,
        alpha=0.4,
        gamma=0.99,
        epsilon=0.4
)
    print('Gamma =' + '0.99' +  ' + ' + 'Alpha =' + '0.4' +  ' + ' 'epsilon = ' + '0.4')
    agent, rewards = train(
        episodes=1501,
        alpha=0.4,
        gamma=0.99,
        epsilon=0.8
)
    print('Gamma =' + '0.99' +  ' + ' + 'Alpha =' + '0.4' +  ' + ' 'epsilon = ' + '0.8')
    agent, rewards = train(
        episodes=1501,
        alpha=0.2,
        gamma=0.99,
        epsilon=0.4
)
    print('Gamma =' + '0.98' +  ' + ' + 'Alpha =' + '0.4' +  ' + ' 'epsilon = ' + '0.8')
    agent, rewards = train(
        episodes=1501,
        alpha=0.2,
        gamma=0.98,
        epsilon=0.4
)
    print('Gamma =' + '0.98' +  ' + ' + 'Alpha =' + '0.2' +  ' + ' 'epsilon = ' + '0.4')

