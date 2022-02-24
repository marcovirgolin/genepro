import numpy as np
from genepro.node import Node


class Plus(Node):
  def __init__(self):
    super(Plus,self).__init__()
    self.arity = 2
    self.symb = '+'

  def _get_args_repr(self, args):
    return self._get_typical_repr(args,'between')

  def get_output(self, X):
    c_outs = self._get_child_outputs(X)
    return c_outs[0] + c_outs[1]


class Minus(Node):
  def __init__(self):
    super(Minus,self).__init__()
    self.arity = 2
    self.symb = '-'

  def _get_args_repr(self, args):
    return self._get_typical_repr(args,'between')

  def get_output(self, X):
    c_outs = self._get_child_outputs(X)
    return c_outs[0] - c_outs[1]


class Times(Node):
  def __init__(self):
    super(Times,self).__init__()
    self.arity = 2
    self.symb = '*'

  def _get_args_repr(self, args):
    return self._get_typical_repr(args,'between')

  def get_output(self, X):
    c_outs = self._get_child_outputs(X)
    return np.multiply(c_outs[0], c_outs[1])

class Div(Node):
  def __init__(self):
    super(Div,self).__init__()
    self.arity = 2
    self.symb = '/'

  def _get_args_repr(self, args):
    return self._get_typical_repr(args,'between')

  def get_output(self, X):
    c_outs = self._get_child_outputs(X)
    # implements a protection to avoid dividing by 0
    sign_b = np.sign(c_outs[1])
    sign_b = np.where(sign_b == 0, 1, sign_b) 
    protected_div = sign_b * c_outs[0] / (1e-9 + np.abs(c_outs[1]))
    return protected_div

class Log(Node):
  def __init__(self):
    super(Log,self).__init__()
    self.arity = 1
    self.symb = 'log'

  def _get_args_repr(self, args):
    # let's report also protection (to some level of detail)
    return "log(abs("+args[0]+"))"

  def get_output(self, X):
    c_outs = self._get_child_outputs(X)
    # implements a protection to avoid arg <= 0
    protected_log = np.log(np.abs(c_outs[0]) + 1e-9)
    return protected_log


class Feature(Node):
  def __init__(self,id):
    super(Feature,self).__init__()
    self.arity = 0
    self.id = id
    self.symb = 'x'+str(id)

  def _get_args_repr(self, args):
    return self.symb

  def get_output(self, X):
    return X[:,self.id]

class Constant(Node):
  def __init__(self, value : float=None):
    super(Constant,self).__init__()
    self.arity = 0
    self.__value = value
    self.symb = str(value)

  def get_value(self):
    if not self.__value:
      # sample uniformly between -5 and +5
      self.__value = np.random.uniform()*10 - 5 
      self.symb = str(self.__value)
    return self.__value 

  def set_value(self, value : float):
    self.__value = value   
    self.symb = str(value)

  def _get_args_repr(self, args):
    # make sure it is initialized
    self.get_value()
    return self.symb

  def get_output(self, X : np.ndarray) -> np.ndarray:
    # make sure it is initialized
    v = self.get_value()
    return np.repeat(v, len(X))