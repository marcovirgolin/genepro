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


class Square(Node):
  def __init__(self):
    super(Square,self).__init__()
    self.arity = 1
    self.symb = '**2'

  def _get_args_repr(self, args):
    return self._get_typical_repr(args,'after')

  def get_output(self, X):
    c_outs = self._get_child_outputs(X)
    return np.square(c_outs[0])


class Cube(Node):
  def __init__(self):
    super(Cube,self).__init__()
    self.arity = 1
    self.symb = '**3'

  def _get_args_repr(self, args):
    return self._get_typical_repr(args,'after')

  def get_output(self, X):
    c_outs = self._get_child_outputs(X)
    return np.multiply(np.square(c_outs[0]), c_outs[0])


class Sqrt(Node):
  def __init__(self):
    super(Sqrt,self).__init__()
    self.arity = 1
    self.symb = 'sqrt'

  def _get_args_repr(self, args):
    # let's report also protection
    return "sqrt(abs("+args[0]+"))"

  def get_output(self, X):
    c_outs = self._get_child_outputs(X)
    # implements a protection to avoid arg <= 0
    return np.sqrt(np.abs(c_outs[0]))


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


class Exp(Node):
  def __init__(self):
    super(Exp,self).__init__()
    self.arity = 1
    self.symb = "exp"

  def _get_args_repr(self, args):
    return self._get_typical_repr(args,'before')

  def get_output(self, X):
    c_outs = self._get_child_outputs(X)
    return np.exp(c_outs[0])


class Sin(Node):
  def __init__(self):
    super(Sin,self).__init__()
    self.arity = 1
    self.symb = "sin"

  def _get_args_repr(self, args):
    return self._get_typical_repr(args,'before')

  def get_output(self, X):
    c_outs = self._get_child_outputs(X)
    return np.sin(c_outs[0])


class Cos(Node):
  def __init__(self):
    super(Cos,self).__init__()
    self.arity = 1
    self.symb = "cos"

  def _get_args_repr(self, args):
    return self._get_typical_repr(args,'before')

  def get_output(self, X):
    c_outs = self._get_child_outputs(X)
    return np.cos(c_outs[0])


class Max(Node):
  def __init__(self):
    super(Max,self).__init__()
    self.arity = 2
    self.symb = "max"

  def _get_args_repr(self, args):
    return self._get_typical_repr(args,"before")

  def get_output(self, X):
    c_outs = self._get_child_outputs(X)
    return np.where(c_outs[0]>c_outs[1], c_outs[0], c_outs[1])


class Min(Node):
  def __init__(self):
    super(Min,self).__init__()
    self.arity = 2
    self.symb = "min"

  def _get_args_repr(self, args):
    return self._get_typical_repr(args,"before")

  def get_output(self, X):
    c_outs = self._get_child_outputs(X)
    return np.where(c_outs[0]<c_outs[1], c_outs[0], c_outs[1])


class IfThenElse(Node):
  def __init__(self):
    super(IfThenElse,self).__init__()
    self.arity = 3
    self.symb = "if-then-else"

  def _get_args_repr(self, args):
    return "if("+args[0]+" >= 0)then("+args[1]+")else("+args[2]+")"

  def get_output(self, X):
    c_outs = self._get_child_outputs(X)
    return np.where(c_outs[0]>=0, c_outs[1], c_outs[2])


class Feature(Node):
  def __init__(self,id):
    super(Feature,self).__init__()
    self.arity = 0
    self.id = id
    self.symb = 'x_'+str(id)

  def _get_args_repr(self, args):
    return self.symb

  def get_output(self, X):
    return X[:,self.id]

class Constant(Node):
  def __init__(self, value : float=None):
    super(Constant,self).__init__()
    self.arity = 0
    self.__value = value
    self.symb = str(value) if value is not None else "const?"

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