import tensorflow as tf   #the as basically means as
import gym
import os
import numpy as np
import matplotlib.pyplot as plt   #for the graphs and stuff

#%matplotlib inline

# TODO: Load an environment
env = gym.make("CartPole-v1") #the environment

# TODO: Print observation and action spaces
print(env.observation_space)  #this is Box. How many states it can be in
print(env.action_space)       #this is Discrete, how many actions it can do

# TODO Make a random agent
#does no learning here
games_to_play = 10

for i in range(games_to_play):
    # Reset the environment
    obs = env.reset()
    episode_rewards = 0
    done = False

    while not done:
        # Render the environment so we can watch
        env.render()

        # Choose a random action
        action = env.action_space.sample()

        # Take a step in the environment with the chosen action
        obs, reward, done, info = env.step(action)  #env.step(action) is a function that returns 4 (different) values: obs, reward, done, info
        episode_rewards += reward

    # Print episode total rewards when done
    print(episode_rewards)

# Close the environment



# TODO Build the policy gradient neural network
class Agent:
    def __init__(self, num_actions, state_size):
        initializer = tf.contrib.layers.xavier_initializer()   #gives random weightage to neurons in the beginning
        self.input_layer = tf.placeholder(dtype=tf.float32, shape=[None, state_size])  #what is this?

        hidden_layer = tf.layers.dense(self.input_layer, 16, activation=tf.nn.relu, kernel_initializer=initializer) #the number is the amount of nodes
        hidden_layer_2 = tf.layers.dense(hidden_layer, 8, activation=tf.nn.relu, kernel_initializer=initializer)
        hidden_layer_3 = tf.layers.dense(hidden_layer_2, 8, activation=tf.nn.relu, kernel_initializer=initializer)

        # Output of neural net
        out = tf.layers.dense(hidden_layer_3, num_actions, activation=None)

        self.outputs = tf.nn.softmax(out)  #softmax is the certainty of the output neuron - probability of a given action being taken
        self.choice = tf.argmax(self.outputs, axis=1)   #returns index of highest value probability in the output
        #training procedure
        self.rewards = tf.placeholder(shape=[None, ], dtype=tf.float32)
        self.actions = tf.placeholder(shape=[None, ], dtype=tf.int32)

        one_hot_actions = tf.one_hot(self.actions, num_actions)
        #one hot actions is when there are a lot of 0s and one 1. Everything else is 0% chance and 1 is 100%
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=out, labels=one_hot_actions)
        #cross entropy is when it creates a combined matrix (a list of numbers) of nodes and edges

        self.loss = tf.reduce_mean(cross_entropy * self.rewards)

        self.gradients = tf.gradients(self.loss, tf.trainable_variables())

        # Create a placeholder list for gradients
        self.gradients_to_apply = []
        for index, variable in enumerate(tf.trainable_variables()):
            gradient_placeholder = tf.placeholder(tf.float32)
            self.gradients_to_apply.append(gradient_placeholder)

        # Create the operation to update gradients with the gradients placeholder.
        optimizer = tf.train.AdamOptimizer(learning_rate=1e-2)
        self.update_gradients = optimizer.apply_gradients(zip(self.gradients_to_apply, tf.trainable_variables()))
        self.update_gradients = optimizer.apply_gradients(zip(self.gradients_to_apply, tf.trainable_variables()))


# TODO Create the discounted and normalized rewards function
discount_rate = 0.95


def discount_and_normalize_rewards(rewards):
    discounted_rewards = np.zeros_like(rewards)
    total_rewards = 0

    for i in reversed(range(len(rewards))):
        total_rewards = total_rewards * discount_rate + rewards[i]
        discounted_rewards[i] = total_rewards

    discounted_rewards -= np.mean(discounted_rewards)
    discounted_rewards /= np.std(discounted_rewards)

    return discounted_rewards


# TODO Create the training loop
tf.reset_default_graph()

# Modify these to match shape of actions and states in your environment
num_actions = 2     #corresponds to discrete - number of actions it can do
state_size = 4     #box - number of states it can be observed in
#num_actions was 2 (left or right) and state_size was 4 for cartpole (left, right, left falling, right falling)

path = "./cartpole-pg/"

max_steps_per_episode = 10000
training_episodes = 1000
episode_batch_size = 5

agent = Agent(num_actions, state_size)

init = tf.global_variables_initializer()

saver = tf.train.Saver(max_to_keep=2)

if not os.path.exists(path):
    os.makedirs(path)

with tf.Session() as sess:
    sess.run(init)

    total_episode_rewards = []

    # Create a buffer of 0'd gradients
    gradient_buffer = sess.run(tf.trainable_variables())
    for index, gradient in enumerate(gradient_buffer):
        gradient_buffer[index] = gradient * 0

    for episode in range(training_episodes):

        state = env.reset()

        episode_history = []
        episode_rewards = 0

        for step in range(max_steps_per_episode):

            if episode % 100 == 0:
                env.render()

            # Get weights for each action
            action_probabilities = sess.run(agent.outputs, feed_dict={agent.input_layer: [state]})
            action_choice = np.random.choice(range(num_actions), p=action_probabilities[0])

            state_next, reward, done, _ = env.step(action_choice)  #action_choice is a list for pendulum
            episode_history.append([state, action_choice, reward, state_next])
            state = state_next

            episode_rewards += reward

            if done or step + 1 == max_steps_per_episode:
                total_episode_rewards.append(episode_rewards)
                episode_history = np.array(episode_history)
                episode_history[:, 2] = discount_and_normalize_rewards(episode_history[:, 2])

                ep_gradients = sess.run(agent.gradients, feed_dict={agent.input_layer: np.vstack(episode_history[:, 0]),
                                                                    agent.actions: episode_history[:, 1],
                                                                    agent.rewards: episode_history[:, 2]})
                # add the gradients to the grad buffer:
                for index, gradient in enumerate(ep_gradients):
                    gradient_buffer[index] += gradient

                break

        if episode % episode_batch_size == 0:

            feed_dict_gradients = dict(zip(agent.gradients_to_apply, gradient_buffer))

            sess.run(agent.update_gradients, feed_dict=feed_dict_gradients)

            for index, gradient in enumerate(gradient_buffer):
                gradient_buffer[index] = gradient * 0

        if episode % 100 == 0:
            saver.save(sess, path + "pg-checkpoint", episode)
            print("Average reward / 100 eps: " + str(np.mean(total_episode_rewards[-100:])))

# TODO Create the testing loop
testing_episodes = 5


with tf.Session() as sess:
    checkpoint = tf.train.get_checkpoint_state(path)
    saver.restore(sess, checkpoint.model_checkpoint_path)

    for episode in range(testing_episodes):
        state = env.reset()
        episode_rewards = 0
        for step in range(max_steps_per_episode):
            env.render()

            # Get Action
            action_argmax = sess.run(agent.choice, feed_dict={agent.input_layer: [state]})
            action_choice = action_argmax[0]

            state_next, reward, done, _ = env.step(action_choice)
            state = state_next

            episode_rewards += reward

            if done or step + 1 == max_steps_per_episode:
                print("Total rewards of episode " + str(episode) + ": " + str(episode_rewards))
                break

# Run to close the environment
env.close()