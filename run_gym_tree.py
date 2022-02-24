import numpy as np, sys, gym
from matplotlib import animation
import matplotlib.pyplot as plt
from genepro.util import tree_from_prefix_repr

# gist to save gif from https://gist.github.com/botforge/64cbb71780e6208172bbf03cd9293553
def save_frames_as_gif(frames, path='./', filename='evolved_cartpole.gif'):
  plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)
  patch = plt.imshow(frames[0])
  plt.axis('off')
  def animate(i):
      patch.set_data(frames[i])
  anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
  anim.save(path + filename, writer='imagemagick', fps=60)

# same environemnt and fitness function of the notebook
env_name = "CartPole-v1"
env = gym.make(env_name)

# frames to save a gif
frames = list()

def fitness_function(tree, num_episodes=5, episode_duration=500, render=False, ignore_done=False):
  rewards = list()
  durations = list()
  for _ in range(num_episodes):
    # get initial state
    observation = env.reset()
    # we do not have an action at time -1, let's set it randomly
    action = env.action_space.sample()
    for t in range(episode_duration):
      if render:
        frames.append(env.render(mode="rgb_array"))
      # build up the input sample for GP
      input_sample = np.concatenate((observation, [action])).reshape((1,-1))
      # get output (squeezing because it is encapsulated in an array)
      output = tree.get_output(input_sample).astype(float).squeeze()
      action = 0 if output < .5 else 1
      observation, reward, done, _ = env.step(action)
      rewards.append(reward)
      if done and not ignore_done:
        break
    # keep track of how long this lasted
    durations.append(t)

  # compute and return fitness
  fitness = np.sum(rewards)
  return fitness


# read-in the tree and evaluate it with rendering
prefix_tree_repr = sys.argv[1]
tree = tree_from_prefix_repr(prefix_tree_repr)

fitness_function(tree, num_episodes=1, episode_duration=500, render=True, ignore_done=False)
env.close()
save_frames_as_gif(frames)