import pandas as pd
import gym
import pandas as pd
import numpy as np
import envs
import tensorflow as tf
import keras
from keras import layers

gamma = 0.99
env = gym.make('droneGym-v0')
eps = np.finfo(np.float32).eps.item()
max_steps_per_episode = 20000


num_inputs = env.x.shape
num_actions = 1
num_hidden = 128

# inputs = layers.Input(shape=(num_inputs))
# common = layers.Dense(num_hidden, activation="relu")(inputs)
# common = layers.Dense(num_hidden*2, activation="relu")(common)
# action = layers.Dense(num_actions, activation="sigmoid")(common)
# critic = layers.Dense(1)(common)
#
# inputs1 = layers.Input(shape=[12])
# inputs2 = layers.Input(shape=[4], name = 'action2')
# w1 = layers.Dense(128, activation="relu")(inputs1)
# a1 = layers.Dense(256, activation="linear")(inputs2)
# h1 = layers.Dense(256, activation="linear")(w1)
# h2 = layers.merge.add([h1,a1])
# h3 = layers.Dense(256, activation='relu')(h2)
# critic = layers.Dense(1, activation='linear')(h1)
#
#
# actorModel = keras.Model(inputs=inputs, outputs=action)
# criticModel = keras.Model(inputs = [inputs1, inputs2], outputs = critic)
#
# optimizer = keras.optimizers.Adam(learning_rate=0.01)
# criticModel.compile(loss = 'mse', optimizer= optimizer)

# num_inputs = 4
# num_actions = 2
# num_hidden = 128

inputs = layers.Input(shape=num_inputs)
common = layers.Dense(num_hidden, activation="relu")(inputs)
common = layers.Dropout(.1)(inputs)
common = layers.Dense(num_hidden*2, activation="relu")(common)
action = layers.Dense(num_actions, activation="sigmoid")(common)
critic = layers.Dense(1)(common)

model = keras.Model(inputs=inputs, outputs=[action, critic])





optimizer = keras.optimizers.Adam(learning_rate=0.01)
huber_loss = keras.losses.Huber()
action_probs_history = []
critic_value_history = []
rewards_history = []
running_reward = 0
episode_count = 0

while True:  # Run until solved
    state = env.reset()
    episode_reward = 0
    with tf.GradientTape() as tape:
        for timestep in range(1, max_steps_per_episode):
            # env.render(); Adding this line would show the attempts
            # of the agent in a pop up window.

            #preproccess the state
            state = np.abs(state)
            if np.sum(np.isnan(state)):
                print("states gone NaN somehow")

            state = tf.convert_to_tensor(state)
            state = tf.expand_dims(state, 0)

            # Predict action probabilities and estimated future rewards
            # from environment state
            action_probs, critic_value = model(state)
            if np.sum(np.isnan(action_probs)):
                print('Actions have gone NaN somehow')
            #criticVal = model([state, action_probs])
            critic_value_history.append(critic_value[0, 0])

            # Sample action from action probability distribution
            if np.random.random() > .6:
                action = [env.action_space.sample()]

            action = 200 * (action_probs - .5)  # np.random.choice(num_actions, p=np.squeeze(action_probs))
            action_probs_history.append(tf.math.log(
                np.max(action_probs)))  # action_probs_history.append(tf.math.log(action_probs[0, action]))


            # Apply the sampled action in our environment
            state, reward, done, _ = env.step(action)
            rewards_history.append(reward)
            episode_reward += reward

            if done:
                episode_reward = episode_reward/timestep
                break

        # Update running reward to check condition for solving
        running_reward = 0.05 * episode_reward + (1 - 0.05) * running_reward

        # Calculate expected value from rewards
        # - At each timestep what was the total reward received after that timestep
        # - Rewards in the past are discounted by multiplying them with gamma
        # - These are the labels for our critic
        returns = []
        discounted_sum = 0
        for r in rewards_history[::-1]:
            discounted_sum = r + gamma * discounted_sum
            returns.insert(0, discounted_sum)

        # Normalize
        returns = np.array(returns)
        returns = (returns - np.mean(returns)) / (np.std(returns) + eps)
        returns = returns.tolist()

        # Calculating loss values to update our network
        history = zip(action_probs_history, critic_value_history, returns)
        actor_losses = []
        critic_losses = []
        for log_prob, value, ret in history:
            # At this point in history, the critic estimated that we would get a
            # total reward = `value` in the future. We took an action with log probability
            # of `log_prob` and ended up recieving a total reward = `ret`.
            # The actor must be updated so that it predicts an action that leads to
            # high rewards (compared to critic's estimate) with high probability.
            diff = ret - value
            actor_losses.append(-log_prob * diff)  # actor loss

            # The critic must be updated so that it predicts a better estimate of
            # the future rewards.
            critic_losses.append(
                huber_loss(tf.expand_dims(value, 0), tf.expand_dims(ret, 0))
            )

        # Backpropagation
        loss_value = sum(actor_losses) + sum(critic_losses)
        grads = tape.gradient(loss_value, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        # Clear the loss and reward history
        action_probs_history.clear()
        critic_value_history.clear()
        rewards_history.clear()

    # Log details
    episode_count += 1
    if episode_count % 10 == 0:
        template = "running reward: {:.2f} at episode {}"
        print(template.format(running_reward, episode_count))

    if episode_count % 500 == 0:
        env.render(epNum = episode_count)

    if running_reward > .9:  # Condition to consider the task solved
        print("Solved at episode {}!".format(episode_count))
        break