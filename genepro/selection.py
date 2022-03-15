import numpy as np
from copy import deepcopy

def tournament_selection(contestants : list, num_to_select : int, tournament_size : int=4) -> list:
  """
  Performs tournament selection on the contestants until the given number of selected contestants is reached;
  note that `len(contestants)` needs to be a multiple of `tournament_size` and similarly for `num_to_select`

  Parameters
  ----------
  contestants : list
    list of Node containing trees that undergo the selection
  num_survivors : int
    how many should be selected
  tournament_size : int, optional
    the size (window) of tournament selection (default is 4)

  Returns
  -------
  list
    list containing (copies of) the trees that were selected
  """
  selected = list()

  n = len(contestants)
  num_selected_per_parse = n // tournament_size
  num_parses = num_to_select // num_selected_per_parse

  # assert quantities are compatible
  assert n / tournament_size == num_selected_per_parse, "Number of contestants {} is not a multiple of tournament size {}".format(n,tournament_size)
  assert num_to_select / num_selected_per_parse == num_parses
 
  for _ in range(num_parses):
    # shuffle
    np.random.shuffle(contestants)
    fitnesses = np.array([t.fitness for t in contestants])

    winning_indices = np.argmax(fitnesses.reshape((-1, tournament_size)), axis=1)
    winning_indices += np.arange(0, n, tournament_size)

    selected += [deepcopy(contestants[i]) for i in winning_indices]

  return selected
 