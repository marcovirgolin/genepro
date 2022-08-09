import inspect
from copy import deepcopy
import numpy as np

from genepro.node import Node
from genepro import node_impl
from genepro.node_impl import Feature, Constant


def compute_linear_scaling(y,p):
  """
  Computes the optimal slope and intercept that realize the affine transformation that minimizes the mean-squared-error between the label and the prediction.
  See the paper: https://doi:10.1023/B:GENP.0000030195.77571.f9

  Parameters
  ----------
  y : np.array
    the label values
  p : np.array
    the respective predictions

  Returns
  -------
  float, float
    slope and intercept that represent 
  """
  slope = np.cov(y, p)[0,1] / (np.var(p) + 1e-12)
  intercept = np.mean(y) - slope*np.mean(p)
  return slope, intercept


def tree_from_prefix_repr(prefix_repr : str) -> Node:
  """
  Creates a tree from a string representation in prefix format (that is, pre-order tree traversal);
  the symbol in the string representation need to match those in the Node implementations (in `genepro.node_impl.py`)
  
  Parameters
  ----------
  prefix_repr : str
    the string representation of the tree as a list of nodes parsed with pre-order traversal (obtainable with `str(tree.get_subtree())`)

  Returns
  -------
  Node
    the tree that corresponds to the string representation
  """
  symb_list = prefix_repr.replace("[","").replace("]","").replace(", ",",").split(",")
  # generate the tree
  node_classes = [c[1] for c in inspect.getmembers(node_impl, inspect.isclass)]
  possible_nodes = list()
  for node_cls in node_classes:
    # handle Features and Constants separetely (also, avoid base class Node)
    if node_cls == Node or node_cls == Feature or node_cls == Constant:
      continue
    node_obj = node_cls()
    possible_nodes.append(node_obj)
  tree, _ = __tree_from_symb_list_recursive(symb_list, possible_nodes)
  return tree

def __tree_from_symb_list_recursive(symb_list : list, possible_nodes : list):
  """
  Helper recursive function for `tree_from_prefix_repr`

  Parameters
  ----------
  symb_list : list
    list of str that are symbols (as per the attribute `symb` of Node)

  possible_nodes : list
    list of all possible Node objects from `genepro.node_impl`

  Returns
  -------
    Node, list
      the last-generated node and the updated symb_list, required for the recursive construction of the tree
  """
  symb = symb_list[0]
  symb_list = symb_list[1:]
  # check if it is a feature
  if symb.startswith("x_"):
    id = int(symb[2:])
    n = Feature(id)
    return n, symb_list

  # check if it is a function
  for pn in possible_nodes:
    if symb == str(pn):
      n = deepcopy(pn)
      for _ in range(n.arity):
        c, symb_list = __tree_from_symb_list_recursive(symb_list, possible_nodes)
        n.insert_child(c)
      return n, symb_list

  # if reached this line, it must be a constant
  n = Constant(float(symb))
  return n, symb_list