import collections
import gym
import envs
import numpy as np
import plaidml.keras
plaidml.keras.install_backend()

import tensorflow as tf
import tensorflow_probability as tfp
import tqdm
import keras
from sklearn.preprocessing import StandardScaler

from matplotlib import pyplot as plt
from tensorflow.keras import layers
from typing import Any, List, Sequence, Tuple


from droneAgent_AC2 import *


# Create the environment
tf.get_logger().setLevel('ERROR')
#####https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/reinforcement_learning/actor_critic.ipynb#scrollTo=qbIMMkfmRHyC
#####https://medium.com/@asteinbach/actor-critic-using-deep-rl-continuous-mountain-car-in-tensorflow-4c1fb2110f7c

# Small epsilon value for stabilizing division operations
eps = np.finfo(np.float32).eps.item()


class Agent2():

    def __init__(self):
        self.runningRewards = 0

    def main(self):
        env = gym.make('droneGym-v0')

        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005, clipvalue=1)
        huber_loss = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)
        num_actions = env.action_space.n  # 2
        num_hidden_units = 128
        model = ActorCritic(num_actions, num_hidden_units)
        scaler = setUpScaler(env)
        model.load_weights(r'C:\Users\Stephen\PycharmProjects\Thesis2\tempStuff\checkpointStarting')

        # set up plotter, hope it's dynamic
        plt.ion()
        fig ,[ax1 ,ax2]  = plt.subplots(2 ,1 ,sharex = True)
        h1, = ax1.plot([] ,[], 'b.' ,label = 'Reward')
        h2, = ax2.plot([] ,[], 'y.' ,label = 'Loss')
        ax1.legend()
        ax2.legend()
        plt.show()

        max_episodes = 10000
        max_steps_per_episode = 1000

        # Cartpole-v0 is considered solved if average reward is >= 195 over 100
        # consecutive trials
        reward_threshold = 3800
        running_reward = 0
        running_loss = 0

        # Discount factor for future rewards
        gamma = 0.99


        with tqdm.trange(max_episodes) as t:
            while self.runningRewards > -5:
                for i in t:
                    initial_state = tf.constant(env.reset(), dtype=tf.float32)
                    episode_reward, lent, loss = train_step(initial_state, model, optimizer, gamma, max_steps_per_episode, scaler, env)
                    episode_reward = int(episode_reward)

                    running_reward = episode_reward*0.01 + running_reward*.99
                    running_loss = loss*0.01 + running_loss*.99
                    self.runningRewards = running_reward

                    t.set_description(f'Episode {i}')
                    t.set_postfix(
                        episode_reward=episode_reward, running_reward=running_reward, loss = loss, running_loss = running_loss, time = lent)

                    h1.set_xdata(np.append(h1.get_xdata(), i))
                    h1.set_ydata(np.append(h1.get_ydata(), running_reward))
                    h2.set_xdata(np.append(h2.get_xdata(), i))
                    h2.set_ydata(np.append(h2.get_ydata(), running_loss))
                    ax1.relim()
                    ax1.autoscale_view()
                    ax2.relim()
                    ax2.autoscale_view()
                    # ax.plot(range(0,i+1), running_reward,'b.', label = 'Reward')
                    # ax.plot(range(0,i+1), running_loss,'y.', label = 'Loss')
                    # ax.legend()
                    plt.draw()
                    fig.canvas.flush_events()

                    # Show average episode reward every 10 episodes
                    if i % 10 == 0:
                        pass # print(f'Episode {i}: average reward: {avg_reward}')

                    if running_reward > reward_threshold:
                        break

                    if episode_reward > 1500 or lent > 800:
                        env.render(epNum=i)

                    if self.runningRewards<-5:
                        break

        print(f'\nSolved at episode {i}: average reward: {running_reward:.2f}!')


if __name__ == "__main__":

    tt = Agent2()

    while 1:

        try:
            tt.runningRewards = 0
            tt.main()
        except Exception as e:
            print(e)
            continue

    print('tt')

