import numpy as np
import envs
import gym
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
from matplotlib import pyplot as plt
import tensorflow as tf
from stable_baselines.common.policies import ActorCriticPolicy, register_policy, nature_cnn, mlp_extractor
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import A2C

# Custom MLP policy of three layers of size 128 each for the actor and 2 layers of 32 for the critic,
# with a nature_cnn feature extractor
class CustomPolicy(ActorCriticPolicy):
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, **kwargs):
        super(CustomPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=reuse, scale=True)

        with tf.compat.v1.variable_scope("model", reuse=reuse):
            activ = tf.nn.sigmoid

            pi_latent2, vf_latent2 = mlp_extractor(self.processed_obs,net_arch = [128, dict(vf=[156, 156], pi=[128])], act_fun = tf.nn.relu, **kwargs)
            actionSpace = tf.compat.v1.layers.dense(pi_latent2, ac_space.n, activation= 'sigmoid', name = 'pf')
            value_fn = tf.compat.v1.layers.dense(vf_latent2, 1, name='vf')
            vf_latent = vf_latent2

            # pi_h = extracted_features
            # for i, layer_size in enumerate([128, 128, 128]):
            #     pi_h = activ(tf.compat.v1.layers.dense(pi_h, layer_size, name='pi_fc' + str(i)))
            # pi_latent = pi_h
            #
            # vf_h = extracted_features
            # for i, layer_size in enumerate([32, 32]):
            #     vf_h = activ(tf.compat.v1.layers.dense(vf_h, layer_size, name='vf_fc' + str(i)))
            # value_fn = tf.compat.v1.layers.dense(vf_h, 1, name='vf')
            # vf_latent = vf_h

            self._proba_distribution, self._policy, self.q_value = \
                self.pdtype.proba_distribution_from_latent(actionSpace, vf_latent, init_scale=0.01)

        self._value_fn = value_fn
        self._setup_init()

    def step(self, obs, state=None, mask=None, deterministic=False):
        if deterministic:
            action, value, neglogp = self.sess.run([self.deterministic_action, self.value_flat, self.neglogp],
                                                   {self.obs_ph: obs})
        else:
            action, value, neglogp = self.sess.run([self.action, self.value_flat, self.neglogp],
                                                   {self.obs_ph: obs})
        return action, value, self.initial_state, neglogp

    def proba_step(self, obs, state=None, mask=None):
        return self.sess.run(self.policy_proba, {self.obs_ph: obs})

    def value(self, obs, state=None, mask=None):
        return self.sess.run(self.value_flat, {self.obs_ph: obs})



def lrGenerator(t):
    tt = -1*(t-1) * 10000
    lr = .0025*np.cos(tt/2000*np.pi) + .0025

    return lr

# eggs = (.025*np.cos(t/2000*np.pi) + .025 for t in range(1,10002))

env = gym.make('droneGym-v0')

# model = PPO2(MlpPolicy, env, verbose=0, learning_rate= .00005, n_steps = 10000, nminibatches=1)
# model = model.load('testOfVariedAngles.zip')
model = PPO2(CustomPolicy, env, verbose = 0,n_steps = 3000, nminibatches=1)
model.learning_rate = lrGenerator
model.env = DummyVecEnv([lambda: env])
model.learn(total_timesteps=100000000)

# set up plotter, hope it's dynamic
plt.ion()
fig, [ax1, ax2] = plt.subplots(2, 1, sharex=True)
h1, = ax1.plot([], [], 'b.', label='Reward')
h2, = ax2.plot([], [], 'y.', label='Loss')
ax1.legend()
ax2.legend()
plt.show()

obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)

    h1.set_xdata(np.append(h1.get_xdata(), i))
    h1.set_ydata(np.append(h1.get_ydata(), rewards))
    # h2.set_xdata(np.append(h2.get_xdata(), i))
    # h2.set_ydata(np.append(h2.get_ydata(), running_loss))
    ax1.relim()
    ax1.autoscale_view()
    ax2.relim()
    ax2.autoscale_view()
    plt.draw()
    fig.canvas.flush_events()

env.close()



print('tt')