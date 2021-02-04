import gym
import envs
import numpy as np
import keras
from keras import layers
import matplotlib.pyplot as plt
import tensorflow as tf

GAMMA = .99
LEARNING_RATE = .01
EPISODES_TO_TRAIN = 4
max_steps_per_episode = 20000
problem = 'droneGym-v0'
env = gym.make(problem)
num_hidden = 300

stateSize = env.observation_space.shape
actionSize = env.action_space.shape

lower_bound = 0
upper_bound = 100

def get_actor():
    inputs = layers.Input(shape=(stateSize))
    common = layers.Dense(num_hidden, activation="relu")(inputs)
    common = layers.Dense(num_hidden * 2, activation="relu")(common)
    action = layers.Dense(actionSize, activation="sigmoid")(common)

    model = keras.Model(inputs, action)

    return model, model.trainable_weights, inputs

def get_critic():


    inputs1 = layers.Input(shape=[stateSize])
    inputs2 = layers.Input(shape=[actionSize], name='action2')
    w1 = layers.Dense(num_hidden, activation="relu")(inputs1)
    a1 = layers.Dense(num_hidden*2, activation="linear")(inputs2)
    h1 = layers.Dense(num_hidden*2, activation="linear")(w1)
    h2 = layers.merge.add([h1, a1])
    h3 = layers.Dense(num_hidden*2, activation='relu')(h2)
    critic = layers.Dense(1, activation='linear')(h3)

    model = layers.Model(input = [inputs1,inputs2], outputs = critic)
    adam = keras.optimizers.Adam(lr = .01)
    model.compile(loss = 'mse', optimizer = adam)

    return model, inputs1, inputs2

# def target_train(model):
#     actor_weights = model.get_weights()
#     actor_target_weights =

if __name__ == "__main__":

    actor,_,_ = get_actor()
    currState = env.reset()


    for j in range(max_steps_per_episode):
        a_t =






 actor.predict(currState)
        ob, r_t, done, info = env.step(a_t[0])