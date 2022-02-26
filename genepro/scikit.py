import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels

from genepro.evo import Evolution
from genepro.node_impl import Feature, Constant

class GeneProEstimator(BaseEstimator):
  def __init__(self,
    score,
    internal_nodes,
    leaf_nodes=None,
    evo_kwargs=dict(),
    ):

    self.score = score

    self.evo = Evolution(
      None,
      internal_nodes, leaf_nodes, **evo_kwargs
    )
  
  def fit(self, X, y):
    # check that X and y have correct shape
    X, y = check_X_y(X, y)
    
    self.X_ = X
    self.y_ = y

    # default generation of leaf nodes
    if not self.evo.leaf_nodes:
      self.evo.leaf_nodes = [Feature(i) for i in range(X.shape[1])] + [Constant()]

    pass

  def predict(self, X):
    # check is fit had been called
    check_is_fitted(self)

    # input validation
    X = check_array(X)

    pass


class GeneProRegressor(GeneProEstimator):
  def __init__(self,
    score,
    internal_nodes,
    leaf_nodes=None,
    use_linear_scaling=True,
    evo_kwargs=dict(),
    ):
    super(GeneProRegressor,self).__init__(score, internal_nodes, leaf_nodes, evo_kwargs)

    self.use_linear_scaling = use_linear_scaling

    # create a fitness function
    def fitness_function(tree):
      pred = tree(self.X_)
      if self.use_linear_scaling:
        slope = np.cov(self.y_, pred)[0,1] / (np.var(pred) + 1e-12)
        intercept = np.mean(self.y_) - slope*np.mean(pred)
        pred = intercept + slope*pred
      return self.score(self.y_, pred)

    self.evo.fitness_function = fitness_function
  
  def fit(self, X, y):
    super(GeneProRegressor,self).fit(X,y)
    self.evo.evolve()


  def predict(self, X, best_ever=False):
    super(GeneProRegressor,self).predict(X)
    if best_ever:
      best = self.evo.best_of_gens[np.argmax([t.fitness for t in self.evo.best_of_gens])]
    else:
      best = self.evo.best_of_gens[-1]
    pred = best(X)
    if self.use_linear_scaling:
      # compute linear scaling coefficients w.r.t. training set
      pred_ = best(self.X_)
      slope = np.cov(self.y_, pred_)[0,1] / (np.var(pred_) + 1e-12)
      intercept = np.mean(self.y_) - slope*np.mean(pred_)
      # update prediction with linear scaling coefficients
      pred = intercept + slope*pred
    return pred


class GeneProClassifier(GeneProEstimator):
  def __init__(self,
    score,
    internal_nodes,
    leaf_nodes=None,
    evo_kwargs=dict(),
    ):
    super(GeneProClassifier,self).__init__(score, internal_nodes, leaf_nodes, evo_kwargs)

    # create a fitness function
    def fitness_function(tree):
      out = tree(self.X_)
      pred = np.where(out < 0, -1, 1)
      return self.score(self.y_, pred)

    self.evo.fitness_function = fitness_function
  
  def fit(self, X, y):
    super(GeneProClassifier,self).fit(X,y)

    # store classes
    self.classes_ = unique_labels(self.y_)
    if len(self.classes_) > 2:
      raise ValueError("Only binary classification is supported (repeat a one vs. all approach for multi-class)")

    # convert y_ into -1 and +1
    self.y_ = np.where(self.y_ == self.classes_[0], -1, +1)

    self.evo.evolve()

  def predict(self, X, best_ever=False):
    super(GeneProClassifier,self).predict(X)
    if best_ever:
      best = self.evo.best_of_gens[np.argmax([t.fitness for t in self.evo.best_of_gens])]
    else:
      best = self.evo.best_of_gens[-1]
    out = best(X)
    pred = np.where(out < 0, -1, 1)
    pred = np.where(pred == -1, self.classes_[0], self.classes_[1])

    return pred