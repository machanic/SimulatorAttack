from abc import ABCMeta
from abc import abstractmethod
import collections
import contextlib
import torch

def unchain_backward(state):
    """Call Variable.unchain_backward recursively."""
    if isinstance(state, collections.Iterable):
        for s in state:
            unchain_backward(s)
    elif torch.is_tensor(state):
        state.detach_()

class Recurrent(object, metaclass=ABCMeta):
    """Interface of recurrent and stateful models.

    This is an interface of recurrent and stateful models. ChainerRL supports
    recurrent neural network models as stateful models that implement this
    interface.

    To implement this interface, you need to implement three abstract methods
    of it: get_state, set_state and reset_state.
    """

    __state_stack = []

    @abstractmethod
    def get_state(self):
        """Get the current state of this model.

        Returns:
            Any object that represents a state of this model.
        """
        raise NotImplementedError()

    @abstractmethod
    def set_state(self, state):
        """Overwrite the state of this model with a given state.

        Args:
            state (object): Any object that represents a state of this model.
        """
        raise NotImplementedError()

    @abstractmethod
    def reset_state(self):
        """Reset the state of this model to the initial state.

        For typical RL models, this method is expected to be called before
        every episode.
        """
        raise NotImplementedError()

    def unchain_backward(self):
        unchain_backward(self.get_state())

    def push_state(self):
        self.__state_stack.append(self.get_state())
        self.reset_state()

    def pop_state(self):
        self.set_state(self.__state_stack.pop())

    def push_and_keep_state(self):
        self.__state_stack.append(self.get_state())

    def update_state(self, *args, **kwargs):
        """Update this model's state as if self.__call__ is called.

        Unlike __call__, stateless objects may do nothing.
        """
        self(*args, **kwargs)

    @contextlib.contextmanager
    def state_reset(self):
        self.push_state()
        yield
        self.pop_state()

    @contextlib.contextmanager
    def state_kept(self):
        self.push_and_keep_state()
        yield
        self.pop_state()



@contextlib.contextmanager
def state_kept(model):
    """Keeps the previous state of a given link.

    This is a context manager that saves saves the current state of the link
    before entering the context, and then restores the saved state after
    escaping the context.

    This will just ignore non-Recurrent links.

       .. code-block:: python

          # Suppose the model is in a state A
          assert model.get_state() is A

          with state_kept(link):
              # The model is still in a state A
              assert model.get_state() is A

              # After evaluating the link, it may be in a different state
              y1 = model(x1)
              assert model.get_state() is not A

          # After escaping from the context, the link is in a state A again
          # because of the context manager
          assert model.get_state() is A
    """
    if isinstance(model, Recurrent):
        model.push_and_keep_state()
        yield
        model.pop_state()
    else:
        yield