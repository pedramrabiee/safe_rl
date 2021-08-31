from gym.spaces.box import Box
from gym.spaces.space import Space
import numpy as np


# gym.spaces.tuple modification: the .contains method is modified to check membership of list of vector in any of the Tuple underlying spaces
class Tuple(Space):
    """
    A tuple (i.e., product) of simpler spaces

    Example usage:
    self.observation_space = spaces.Tuple((spaces.Discrete(2), spaces.Discrete(3)))
    """

    def __init__(self, spaces):
        self.spaces = spaces
        for space in spaces:
            assert isinstance(space, Space), "Elements of the tuple must be instances of gym.Space"
        super(Tuple, self).__init__(None, None)

    def seed(self, seed=None):
        [space.seed(seed) for space in self.spaces]

    def sample(self):
        return tuple([space.sample() for space in self.spaces])

    def contains(self, x):
        single_obs = False
        if isinstance(x, np.ndarray) and np.isscalar(x[0]):
            single_obs = True
        if isinstance(x, list) or not single_obs:
            return [any(space.contains(part) for space in self.spaces) for part in x]
        return any(space.contains(x) for space in self.spaces)

    def __repr__(self):
        return "Tuple(" + ", ".join([str(s) for s in self.spaces]) + ")"

    def to_jsonable(self, sample_n):
        # serialize as list-repr of tuple of vectors
        return [space.to_jsonable([sample[i] for sample in sample_n]) \
                for i, space in enumerate(self.spaces)]

    def from_jsonable(self, sample_n):
        return [sample for sample in zip(*[space.from_jsonable(sample_n[i]) for i, space in enumerate(self.spaces)])]

    def __getitem__(self, index):
        return self.spaces[index]

    def __len__(self):
        return len(self.spaces)

    def __eq__(self, other):
        return isinstance(other, Tuple) and self.spaces == other.spaces

# def shrink_tuple_space(tuple_space, eps):
#     new_space = []
#     for box in tuple_space:
#         low = box.low + eps
#         high = box.high - eps
#         dtype = box.dtype
#         new_space.append(Box(low=low, high=high, dtype=dtype))
#     return Tuple(new_space)
