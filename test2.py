import tensorflow as tf
import gym
import Box2D
import os
import numpy as np
import matplotlib.pyplot as plt

#%matplotlib inline


# WAYS TO MAKE A PROGRAM LEARN:
# # - CHANGE LEARNING RATE
# # - ADD LAYERS
# # - ADD NODES


# TODO: Load an environment
env = gym.make("LunarLander-v2")

# TODO: Print observation and action spaces
print(env.observation_space)
print(env.action_space)

# # TODO Make a random agent
games_to_play = 10
print("RANDOM AGENT")
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
        obs, reward, done, info = env.step(action)
        episode_rewards += reward

    # Print episode total rewards when done
    print(episode_rewards)

# Close the environment



# TODO Build the policy gradient neural network
class Agent:
    def __init__(self, num_actions, state_size):
        initializer = tf.contrib.layers.xavier_initializer()
        self.input_layer = tf.placeholder(dtype=tf.float32, shape=[None, state_size])

        hidden_layer = tf.layers.dense(self.input_layer, 8, activation=tf.nn.relu, kernel_initializer=initializer)
        hidden_layer_2 = tf.layers.dense(hidden_layer, 8, activation=tf.nn.relu, kernel_initializer=initializer)
        hidden_layer_3 = tf.layers.dense(hidden_layer_2, 8, activation=tf.nn.relu, kernel_initializer=initializer)
        hidden_layer_4 = tf.layers.dense(hidden_layer_3, 8, activation=tf.nn.relu, kernel_initializer=initializer)
        hidden_layer_5 = tf.layers.dense(hidden_layer_4, 8, activation=tf.nn.relu, kernel_initializer=initializer)
        hidden_layer_6 = tf.layers.dense(hidden_layer_5, 8, activation=tf.nn.relu, kernel_initializer=initializer)
        hidden_layer_7 = tf.layers.dense(hidden_layer_6, 8, activation=tf.nn.relu, kernel_initializer=initializer)
        hidden_layer_8 = tf.layers.dense(hidden_layer_7, 8, activation=tf.nn.relu, kernel_initializer=initializer)
        # hidden_layer_9 = tf.layers.dense(hidden_layer_8, 14, activation=tf.nn.relu, kernel_initializer=initializer)
        # hidden_layer_10 = tf.layers.dense(hidden_layer_9, 14, activation=tf.nn.relu, kernel_initializer=initializer)
        # hidden_layer_11 = tf.layers.dense(hidden_layer_10, 14, activation=tf.nn.relu, kernel_initializer=initializer)
        # hidden_layer_12 = tf.layers.dense(hidden_layer_11, 14, activation=tf.nn.relu, kernel_initializer=initializer)
        # hidden_layer_13 = tf.layers.dense(hidden_layer_12, 14, activation=tf.nn.relu, kernel_initializer=initializer)
        # hidden_layer_14 = tf.layers.dense(hidden_layer_13, 14, activation=tf.nn.relu, kernel_initializer=initializer)
        # hidden_layer_15 = tf.layers.dense(hidden_layer_14, 14, activation=tf.nn.relu, kernel_initializer=initializer)
        # hidden_layer_16 = tf.layers.dense(hidden_layer_15, 14, activation=tf.nn.relu, kernel_initializer=initializer)

        # Output of neural net
        out = tf.layers.dense(hidden_layer_8, num_actions, activation=None) #the output comes from the last layer (duh)

        self.outputs = tf.nn.softmax(out)
        self.choice = tf.argmax(self.outputs, axis=1)

        self.rewards = tf.placeholder(shape=[None, ], dtype=tf.float32)
        self.actions = tf.placeholder(shape=[None, ], dtype=tf.int32)

        one_hot_actions = tf.one_hot(self.actions, num_actions)
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=out, labels=one_hot_actions)

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
        # self.update_gradients = optimizer.apply_gradients(zip(self.gradients_to_apply, tf.trainable_variables()))


# TODO Create the discounted and normalized rewards function
discount_rate = 0.95  #0.95


def discount_and_normalize_rewards(rewards):
    discounted_rewards = np.zeros_like(rewards)
    total_rewards = 0

    for i in reversed(range(len(rewards))):
        total_rewards = total_rewards * discount_rate + rewards[i]
        discounted_rewards[i] = total_rewards

    discounted_rewards -= np.mean(discounted_rewards)
    discounted_rewards /= np.std(discounted_rewards) + 1

    return discounted_rewards


# TODO Create the training loop
tf.reset_default_graph()

# Modify these to match shape of actions and states in your environment
num_actions = 4     #corresponds to discrete - number of actions it can do
state_size = 8     #box - number of states it can be observed in
#num_actions was 2 (left or right) and state_size was 4 for cartpole (left, right, left falling, right falling)

path = "./LunarLander-pg/"

max_steps_per_episode = 500
training_episodes = 1000
episode_batch_size = 5

agent = Agent(num_actions, state_size)

init = tf.global_variables_initializer()

saver = tf.train.Saver(max_to_keep=2)

if not os.path.exists(path):
    os.makedirs(path)
print("TRAINING LOOP")
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
            # print("here1")
            if episode % 100 == 0:
                env.render()

            # Get weights for each action
            action_probabilities = sess.run(agent.outputs, feed_dict={agent.input_layer: [state]})
            # actions = []

            action_choice = np.random.choice(range(num_actions), p=action_probabilities[0])
            # action_choice_1 = np.random.choice(range(num_actions), p=action_probabilities[0])
            # action_choice_2 = np.random.choice(range(num_actions), p=action_probabilities[0])
            # action_choice_3 = np.random.choice(range(num_actions), p=action_probabilities[0])

            # actions.append(action_choice)
            # actions.append(action_choice_1)
            # actions.append(action_choice_2)
            # actions.append(action_choice_3)

            #print(actions)

            state_next, reward, done, _ = env.step(action_choice)  #action_choice is a list for pendulum
            episode_history.append([state, action_choice, reward, state_next])
            state = state_next

            # print("here2")
            episode_rewards += reward

            if done or step + 1 == max_steps_per_episode:
                # print("here3")

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
print("TESTING LOOP")

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
            # action_choice_1 = action_argmax[0]  #index should be 0 for all of them
            # action_choice_2 = action_argmax[0]
            # action_choice_3 = action_argmax[0]
            #
            # actions_2 = []
            # actions_2.append(action_choice)
            # actions_2.append(action_choice_1)
            # actions_2.append(action_choice_2)
            # actions_2.append(action_choice_3)

            state_next, reward, done, _ = env.step(action_choice)
            state = state_next

            episode_rewards += reward

            if done or step + 1 == max_steps_per_episode:
                print("Total rewards of episode " + str(episode) + ": " + str(episode_rewards))
                break

# Run to close the environment
env.close()

#TODO minor problem - machine is staying dumb