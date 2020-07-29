import torch
import six
import numpy

def _concat_arrays(arrays):
    # Convert `arrays` to numpy.ndarray if `arrays` consists of the built-in
    # types such as int or float.
    if not isinstance(arrays[0], numpy.ndarray):
        arrays = numpy.asarray(arrays)
    return torch.cat([array[None] for array in arrays])


def concat_examples(batch):
    """Concatenates a list of examples into array(s).
    This function converts an "array of tuples" into a "tuple of arrays".
    Specifically, given a list of examples each of which consists of
    a list of elements, this function first makes an array
    by taking the element in the same position from each example
    and concatenates them along the newly-inserted first axis
    (called `batch dimension`) into one array.
    It repeats this for all positions and returns the resulting arrays.
    The output type depends on the type of examples in ``batch``.
    For instance, consider each example consists of two arrays ``(x, y)``.
    Then, this function concatenates ``x`` 's into one array, and ``y`` 's
    into another array, and returns a tuple of these two arrays. Another
    example: consider each example is a dictionary of two entries whose keys
    are ``'x'`` and ``'y'``, respectively, and values are arrays. Then, this
    function concatenates ``x`` 's into one array, and ``y`` 's into another
    array, and returns a dictionary with two entries ``x`` and ``y`` whose
    values are the concatenated arrays.
    When the arrays to concatenate have different shapes, the behavior depends
    on the ``padding`` value. If ``padding`` is ``None`` (default), it raises
    an error. Otherwise, it builds an array of the minimum shape that the
    contents of all arrays can be substituted to. The padding value is then
    used to the extra elements of the resulting arrays.
    .. admonition:: Example
       >>> import numpy as np
       >>> from chainer import dataset
       >>> x = [([1, 2], 1),
       ...      ([3, 4], 2),
       ...      ([5, 6], 3)]
       >>> dataset.concat_examples(x)
       (array([[1, 2],
              [3, 4],
              [5, 6]]), array([1, 2, 3]))
       >>>
       >>> y = [(np.array([1, 2]), 0),
       ...      (np.array([3]), 1),
       ...      (np.array([]), 2)]
       >>> dataset.concat_examples(y, padding=100)
       (array([[  1,   2],
              [  3, 100],
              [100, 100]]), array([0, 1, 2]))
       >>>
       >>> z = [(np.array([1, 2]), np.array([0])),
       ...      (np.array([3]), np.array([])),
       ...      (np.array([]), np.array([2]))]
       >>> dataset.concat_examples(z, padding=(100, 200))
       (array([[  1,   2],
              [  3, 100],
              [100, 100]]), array([[  0],
              [200],
              [  2]]))
       >>> w = [{'feature': np.array([1, 2]), 'label': 0},
       ...      {'feature': np.array([3, 4]), 'label': 1},
       ...      {'feature': np.array([5, 6]), 'label': 2}]
       >>> dataset.concat_examples(w)  # doctest: +SKIP
       {'feature': array([[1, 2],
              [3, 4],
              [5, 6]]), 'label': array([0, 1, 2])}
    Args:
        batch (list): A list of examples. This is typically given by a dataset
            iterator.
        device (int): Device ID to which each array is sent. Negative value
            indicates the host memory (CPU). If it is omitted, all arrays are
            left in the original device.
        padding: Scalar value for extra elements. If this is None (default),
            an error is raised on shape mismatch. Otherwise, an array of
            minimum dimensionalities that can accommodate all arrays is
            created, and elements outside of the examples are padded by this
            value.
    Returns:
        Array, a tuple of arrays, or a dictionary of arrays. The type depends
        on the type of each example in the batch.
    """
    if len(batch) == 0:
        raise ValueError('batch is empty')

    first_elem = batch[0]

    if isinstance(first_elem, tuple):
        result = []


        for i in six.moves.range(len(first_elem)):
            result.append(_concat_arrays(
                [example[i] for example in batch]))

        return tuple(result)

    elif isinstance(first_elem, dict):
        result = {}
        for key in first_elem:
            result[key] =  _concat_arrays(
                [example[key] for example in batch]).cuda()

        return result


def batch_states(states,  phi):
    """The default method for making batch of observations.

    Args:
        states (list): list of observations from an environment.
        phi (callable): Feature extractor applied to observations

    Return:
        the object which will be given as input to the model.
    """
    features = [phi(s) for s in states]
    return concat_examples(features)

