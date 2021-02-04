import collections
import gym
import envs
import numpy as np
import tensorflow as tf
import tqdm
import keras

from matplotlib import pyplot as plt
from tensorflow.keras import layers
from typing import Any, List, Sequence, Tuple


# Create the environment
env = gym.make('droneGym-v0')
#####https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/reinforcement_learning/actor_critic.ipynb#scrollTo=qbIMMkfmRHyC

# Small epsilon value for stabilizing division operations
eps = np.finfo(np.float32).eps.item()

class ActorCritic(tf.keras.Model):
  """Combined actor-critic network."""

  def __init__(self,num_actions: int,num_hidden_units: int):
    """Initialize."""
    super().__init__()

    # self.common = layers.Dense(num_hidden_units, activation="relu")
    # self.common = layers.Dense(num_hidden_units, activation="relu")
    # self.actor = layers.Dense(num_actions, activation = "sigmoid")
    # self.critic = layers.Dense(1)

    inputs = layers.Input(shape=15)
    common = layers.Dense(num_hidden_units, activation="tanh")(inputs)
    common = layers.Dropout(.2)(inputs)
    common = layers.Dense(num_hidden_units * 2, activation="relu")(common)
    action = layers.Dense(num_actions, activation="sigmoid")(common)
    critic = layers.Dense(1)(common)

    self.model = keras.Model(inputs=inputs, outputs=[action, critic])

  # def call(self, inputs: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
  #   x = self.common(inputs)
  #   return self.actor(x), self.critic(x)
  def call(self, inputs: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    act, cri = self.model(inputs)
    return act, cri
    # return self.model



def env_step(action: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Returns state, reward and done flag given an action."""
  state, reward, done, _ = env.step(action)
  return (state.astype(np.float32),
          np.array(reward, np.int32),
          np.array(done, np.int32))


def tf_env_step(action: tf.Tensor) -> List[tf.Tensor]:
  return tf.numpy_function(env_step, [action],
                           [tf.float32, tf.int32, tf.int32])

def run_episode(
    initial_state: tf.Tensor,
    model: tf.keras.Model,
    max_steps: int) -> List[tf.Tensor]:
  """Runs a single episode to collect training data."""

  action_probs = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
  values = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
  rewards = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)

  initial_state_shape = initial_state.shape
  state = initial_state

  for t in tf.range(max_steps):
    # Convert state into a batched tensor (batch size = 1)
    state = tf.expand_dims(state, 0)
    state = np.abs(state)
    state = tf.convert_to_tensor(state)
    if np.sum(np.isnan(state)):
      print("states gone NaN somehow")

    prevState = state

    # Run the model and to get action probabilities and critic value
    action_logits_t, value = model(state)
    if action_logits_t[0][0] <0 or action_logits_t[0][0] >1 or np.isnan(action_logits_t[0][0]):
      print('tt')

    # Sample next action from the action probability distribution
    # action = tf.random.categorical(action_logits_t, 1)[0, 0]
    action_probs_t = tf.nn.softmax(action_logits_t)
    action = 200 * (action_logits_t - .5)
    if np.random.random() > .8:
      action = tf.constant([[env.action_space.sample()]])

    # Store critic values
    values = values.write(t, tf.squeeze(value))

    # Store log probability of the action chosen
    # action_probs = action_probs.write(t, action_probs_t[0, action])
    action_probs = action_probs.write(t, action_logits_t[0,0])


    # Apply action to the environment to get next state and reward
    state, reward, done = tf_env_step(action)
    state.set_shape(initial_state_shape)

    # Store reward
    rewards = rewards.write(t, reward)

    if tf.cast(done, tf.bool):
      break

  action_probs = action_probs.stack()
  values = values.stack()
  rewards = rewards.stack()

  return action_probs, values, rewards


def get_expected_return(
    rewards: tf.Tensor,
    gamma: float,
    standardize: bool = True) -> tf.Tensor:
  """Compute expected returns per timestep."""

  n = tf.shape(rewards)[0]
  returns = tf.TensorArray(dtype=tf.float32, size=n)

  # Start from the end of `rewards` and accumulate reward sums
  # into the `returns` array
  rewards = tf.cast(rewards[::-1], dtype=tf.float32)
  discounted_sum = tf.constant(0.0)
  discounted_sum_shape = discounted_sum.shape
  for i in tf.range(n):
    reward = rewards[i]
    discounted_sum = reward + gamma * discounted_sum
    discounted_sum.set_shape(discounted_sum_shape)
    returns = returns.write(i, discounted_sum)
  returns = returns.stack()[::-1]

  if standardize:
    returns = ((returns - tf.math.reduce_mean(returns)) /
               (tf.math.reduce_std(returns) + eps))

  return returns


def compute_loss(
    action_probs: tf.Tensor,
    values: tf.Tensor,
    returns: tf.Tensor) -> tf.Tensor:
  """Computes the combined actor-critic loss."""

  advantage = returns - values

  action_log_probs = tf.math.log(action_probs)
  actor_loss = -tf.math.reduce_sum(action_log_probs * advantage)

  critic_loss = huber_loss(returns,values)
  critic_loss_temp = tf.math.reduce_mean(np.power(advantage,2))

  return actor_loss + critic_loss


# @tf.function
def train_step(
    initial_state: tf.Tensor,
    model: tf.keras.Model,
    optimizer: tf.keras.optimizers.Optimizer,
    gamma: float,
    max_steps_per_episode: int) -> tf.Tensor:
  """Runs a model training step."""

  with tf.GradientTape() as tape:

    # Run the model for one episode to collect training data
    action_probs, values, rewards = run_episode(
        initial_state, model, max_steps_per_episode)

    # Calculate expected returns
    returns = get_expected_return(rewards, gamma)
    #########TEMPORARY MAYBE
    # returns = tf.cast(rewards, dtype = tf.float32)

    # Convert training data to appropriate TF tensor shapes
    action_probs, values, returns = [
        tf.expand_dims(x, 1) for x in [action_probs, values, returns]]

    # Calculating loss values to update our network
    loss = compute_loss(action_probs, values, returns)

    if np.isnan(loss):
      print("Loss is NAN")

  # Compute the gradients from the loss
  grads = tape.gradient(loss, model.trainable_variables)

  # Apply the gradients to the model's parameters
  optimizer.apply_gradients(zip(grads, model.trainable_variables))

  episode_reward = tf.math.reduce_sum(rewards)

  return episode_reward, len(rewards), loss.numpy()



if __name__ == "__main__":

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005, clipvalue=1)
    huber_loss = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)
    num_actions = env.action_space.n  # 2
    num_hidden_units = 128
    model = ActorCritic(num_actions, num_hidden_units)
    model.load_weights(r'C:\Users\Stephen\PycharmProjects\Thesis2\tempStuff\checkpointStarting')

    max_episodes = 10000
    max_steps_per_episode = 1000

    # Cartpole-v0 is considered solved if average reward is >= 195 over 100
    # consecutive trials
    reward_threshold = 1000
    running_reward = 0
    running_loss = 0

    # Discount factor for future rewards
    gamma = 0.99

    with tqdm.trange(max_episodes) as t:
      for i in t:
        initial_state = tf.constant(env.reset(), dtype=tf.float32)
        episode_reward, lent, loss = train_step(initial_state, model, optimizer, gamma, max_steps_per_episode)
        episode_reward = int(episode_reward)

        running_reward = episode_reward*0.01 + running_reward*.99
        running_loss = loss*0.01 + running_loss*.99

        t.set_description(f'Episode {i}')
        t.set_postfix(
            episode_reward=episode_reward, running_reward=running_reward, loss = loss, running_loss = running_loss)

        # Show average episode reward every 10 episodes
        if i % 10 == 0:
          pass # print(f'Episode {i}: average reward: {avg_reward}')

        if running_reward > reward_threshold:
            break

        if episode_reward > 500 or lent > 800:
          env.render(epNum=i)

    print(f'\nSolved at episode {i}: average reward: {running_reward:.2f}!')