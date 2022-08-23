import numpy as np
from numpy.random import random as randu
from numpy.random import normal as randn
from numpy.random import choice as randc
from numpy.random import shuffle
from copy import deepcopy

from genepro.node import Node
from genepro.node_impl import Constant


def generate_random_tree(internal_nodes : list, leaf_nodes : list, max_depth : int, curr_depth : int=0):
  """
  Recursive method to generate a random tree containing the given types of nodes and up to the given maximum depth

  Parameters
  ----------
  internal_nodes : list
    list of nodes with arity > 0 (i.e., sub-functions or operations)
  leaf_nodes : list
    list of nodes with arity==0 (also called terminals, e.g., features and constants)
  max_depth : int
    maximum depth of the tree (recall that the root node has depth 0)
  curr_depth : int
    the current depth of the tree under construction, it is set by default to 0 so that calls to `generate_random_tree` need not specify it

  Returns
  -------
  Node
    the root node of the generated tree
  """

  # heuristic to generate a semi-normal centered on relatively large trees
  prob_leaf = 0.01 + (curr_depth / max_depth)**3

  if curr_depth == max_depth or randu() < prob_leaf:
    n = deepcopy(randc(leaf_nodes))
  else:
    n = deepcopy(randc(internal_nodes))

  for _ in range(n.arity):
    c = generate_random_tree(internal_nodes, leaf_nodes, max_depth, curr_depth+1)
    n.insert_child(c)

  return n

def subtree_crossover(tree : Node, donor : Node, unif_depth : int=True) -> Node:
  """
  Performs subtree crossover and returns the resulting offspring

  Parameters
  ----------
  tree : Node
    the tree that participates and is modified by crossover
  donor : Node
    the second tree that participates in crossover, it provides candidate subtrees
  unif_depth : bool, optional
    whether uniform random depth sampling is used to pick the root of the subtrees to swap (default is True)

  Returns
  -------
  Node
    the tree after crossover (warning: replace the original tree with the returned one to avoid undefined behavior)
  """
  # pick a subtree to replace
  n = __sample_node(tree, unif_depth)
  m = deepcopy(__sample_node(donor, unif_depth))

  # remove ref to parent of m
  m.parent = None
  # swap
  p = n.parent
  if p:
    i = p.detach_child(n)
    p.insert_child(m,i)
  else:
    tree = m
  return tree

def node_level_crossover(tree : Node, donor : Node, same_depth : bool=False, prob_swap : float=0.1) -> Node:
  """
  Performs crossover at the level of single nodes

  Parameters
  ----------
  tree : Node
    the tree that participates and is modified by crossover
  donor : Node
    the second tree for crossover, which provides candidate nodes
  same_depth : bool, optional
    whether node-level swaps should occur only between nodes at the same depth level (default is False)
  prob_swap : float, optional
    the probability of swapping a node in tree with one in donor (default is 0.1)

  Returns
  -------
  Node
    the tree after crossover 
  """
  nodes = tree.get_subtree()
  donor_nodes = donor.get_subtree()

  donor_node_arity = dict()
  donor_node_arity_n_depth = dict()
  for n in donor_nodes:
    arity = n.arity
    if arity not in donor_node_arity:
      donor_node_arity[arity] = [n]
    else:
      donor_node_arity[arity].append(n)
    # also record depths if same_depth==True
    if same_depth:
      depth = n.get_depth()
      ar_n_dep = (arity, depth)
      if ar_n_dep not in donor_node_arity_n_depth:
        donor_node_arity_n_depth[ar_n_dep] = [n]
      else:
        donor_node_arity_n_depth[ar_n_dep].append(n)

  for n in nodes:
    if randu() < prob_swap:
      # find compatible donor
      arity = n.arity
      if same_depth:
        depth = n.get_depth()
        compatible_nodes = donor_node_arity_n_depth[(arity,depth)] if (arity,depth) in donor_node_arity_n_depth else None
      else:
        compatible_nodes = donor_node_arity[arity] if arity in donor_node_arity else None
      # if no compatible nodes, skip
      if compatible_nodes is None or len(compatible_nodes) == 0:
        continue
      # swap
      m = deepcopy(randc(compatible_nodes))
      m.parent = None
      m._children = list()
      p = n.parent
      if p:
        i = p.detach_child(n)
        p.insert_child(m,i)
      else:
        tree = m
      for c in n._children:
        m.insert_child(c)
  
  return tree


def subtree_mutation(tree : Node, internal_nodes : list, leaf_nodes : list, 
  unif_depth : bool=True, max_depth : int=4, prob_leaf : float=0.25) -> Node:
  """
  Performs subtree mutation and returns the resulting offspring

  Parameters
  ----------
  tree : Node
    the tree that participates and is modified by crossover
  internal_nodes : list
    list of possible internal nodes to generate the mutated branch
  leaf_nodes : list
    list of possible leaf nodes to generate the mutated branch
  unif_depth : bool, optional
    whether uniform random depth sampling is used to pick the root of the subtree to mutate (default is True)
  max_depth : int, optional
    the maximal depth of the mutated branch (default is 4)
  prob_leaf : float, optional
    the probability of sampling a leaf when generating the mutated branch (default is 0.25)
  Returns
  -------
  Node
    the tree after mutation (warning: replace the original tree with the returned one to avoid undefined behavior)
  """
  # pick a subtree to replace
  n = __sample_node(tree, unif_depth)
  # generate a random branch
  branch = generate_random_tree(internal_nodes, leaf_nodes, max_depth, prob_leaf)
  # swap
  p = n.parent
  if p:
    i = p.detach_child(n)
    p.insert_child(branch,i)
  else:
    tree = branch
  return tree

def coeff_mutation(tree : Node, prob_coeff_mut : float= 0.25, temp : float=0.25) -> Node:
  """
  Applies random coefficient mutations to constant nodes 

  Parameters
  ----------
  tree : Node
    the tree to which coefficient mutations are applied
  prob_coeff_mut : float, optional
    the probability with which coefficients are mutated (default is 0.25)
  temp : float, optional
    "temperature" that indicates the strength of coefficient mutation, it is relative to the current value (i.e., v' = v + temp*abs(v)*N(0,1))

  Returns
  -------
  Node
    the tree after coefficient mutation (it is the same as the tree in input)
  """
  coeffs = [n for n in tree.get_subtree() if type(n) == Constant]
  for c in coeffs:
    # decide wheter it should be applied
    if randu() < prob_coeff_mut:
      v = c.get_value()
      # update the value by +- temp relative to current value
      new_v = v + temp*np.abs(v)*randn()
      c.set_value(new_v)
  
  return tree

def __sample_node(tree : Node, unif_depth : bool=True) -> Node:
  """
  Helper method that samples a random node from a tree

  Parameters
  ----------
  tree : Node
    the tree from which a random node should be sampled
  unif_depth : bool, optional
    whether the depth of the random node should be sampled uniformly at random first (default is True)

  Returns
  -------
  Node
    the randomly sampled node
  """
  nodes = tree.get_subtree()
  if unif_depth:
    nodes = __sample_uniform_depth_nodes(nodes)
  return randc(nodes)
  
def __sample_uniform_depth_nodes(nodes : list) -> list:
  """
  Helper method for `__sample_node` that returns candidate nodes that all have a depth which was sampled uniformly at random

  Parameters
  ----------
  nodes : list
    list of nodes from which to sample candidates that share a random depth (typically the result of `get_subtree()`)

  Returns
  -------
  list:
    list of nodes that share a depth that was sampled uniformly at random
  """
  depths = [n.get_depth() for n in nodes]
  possible_depths = np.unique(depths)
  d = randc(possible_depths)
  candidates = [n for i, n in enumerate(nodes) if depths[i] == d]
  return candidates
  



def generate_offspring(parent : Node, 
  crossovers : list, mutations : list, coeff_opts : list,
  donors : list, internal_nodes : list, leaf_nodes : list,
  constraints : dict={"max_tree_size": 100}) -> Node:
  """
  Generates an offspring from a given parent (possibly using a donor from the population for crossover).
  Variation operators are applied in a random order.
  The application of the variation operator is handled by `__undergo_variation_operator`

  Parameters
  ----------
  parent : Node
    the parent tree from which the offspring is generated by applying the variation operators
  crossovers : list
    list of dictionaries each specifying a type of crossover and respective hyper-parameters
  mutations : list
    list of dictionaries each specifying a type of mutation and respective hyper-parameters
  coeff_opts : list
    list of dictionaries each specifying a type of coefficient optimization and respective hyper-parameters
  donors : list
    list of Node, each representing a donor tree that can be used by crossover
  internal_nodes : list
    list of internal nodes to be used by mutation
  leaf_nodes : list
    list of internal nodes to be used by mutation
  constraints : dict, optional
    constraints the generated offspring must meet (default is {"max_size": 100})

  Returns
  -------
  Node
    the offspring after having applied the variation operators
  """
  # set the offspring to a copy (to be modified) of the parent
  offspring = deepcopy(parent)
  # create a backup for constraint violation
  backup = deepcopy(offspring)

  # apply variation operators in a random order
  all_var_ops = crossovers + mutations + coeff_opts
  random_order = np.arange(len(all_var_ops))
  shuffle(random_order)
  for i in random_order:
    var_op = all_var_ops[i]
    offspring = __undergo_variation_operator(var_op, offspring, 
      crossovers, mutations, coeff_opts,
      randc(donors), internal_nodes, leaf_nodes)
    # check offspring meets constraints, else revert to backup
    if not __check_tree_meets_all_constraints(offspring, constraints):
      # revert to backup
      offspring = deepcopy(backup)
    else:
      # update backup
      backup = deepcopy(offspring)

  return offspring


def __undergo_variation_operator(var_op : dict, offspring : Node,
  crossovers, mutations, coeff_opts,
  donor, internal_nodes, leaf_nodes) -> Node:
  # decide whether to actually do something
  if var_op["rate"] < randu():
    # nope
    return offspring

  # prepare the function to call
  var_op_fun = var_op["fun"]
  # next, we need to provide the right arguments based on the type of ops
  if var_op in crossovers:
    # we need a donor
    offspring = var_op_fun(offspring, donor, **var_op["kwargs"])
  elif var_op in mutations:
    # we need to provide node types 
    offspring = var_op_fun(offspring, internal_nodes, leaf_nodes, **var_op["kwargs"])
  elif var_op in coeff_opts:
    offspring = var_op_fun(offspring, **var_op["kwargs"])

  return offspring


def __check_tree_meets_all_constraints(tree : Node, constraints : dict=dict()) -> bool:
  """
  """
  meets = True
  for constraint_name in constraints.keys():
    if constraint_name == "max_tree_size":
      if len(tree.get_subtree()) > constraints["max_tree_size"]:
        meets = False
        break
    else:
      raise ValueError("Unrecognized constraint name: {}".format(constraint_name))
  return meets
