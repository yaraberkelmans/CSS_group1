import numpy as np

# Here goes the agent class and its subroutines


# Agent class definition
class Agent:
    """
    Agent contains:
    - Unique identifier id
    - Social position x in R^d
    - Opinion vector theta in R^m
    """

    def __init__(self, id, d, m):
        self.id = id
        self.x = np.zeros(d)
        self.theta = np.zeros(m)

    @property
    def d(self) -> int:
        """Dimension of social position space."""
        return self.x.shape[0]

    @property
    def m(self) -> int:
        """Dimension of opinion space."""
        return self.theta.shape[0]

    def social_distance(self, other: "Agent") -> float:
        """Compute the social distance between self and another agent."""
        return np.linalg.norm(self.x - other.x)

    def within_social_radius(self, other: "Agent", radius: float) -> bool:
        """Check if another agent is within a given social radius."""
        dx = self.x - other.x
        return bool(np.dot(dx, dx) <= radius**2)

    def reflect(self, low: float, high: float) -> None:
        """Reflect social position to stay within [low, high]."""
        if self.x[0] < low:
            self.x[0] = low + (low - self.x[0])
        if self.x[0] > high:
            self.x[0] = high - (self.x[0] - high)
        if self.x[1] < low:
            self.x[1] = low + (low - self.x[1])
        if self.x[1] > high:
            self.x[1] = high - (self.x[1] - high)
