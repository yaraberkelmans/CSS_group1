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
        return np.linalg.norm(other.x - self.x)

    def within_social_radius(self, other: "Agent", radius: float) -> bool:
        """Check if another agent is within a given social radius."""
        dx = self.x - other.x
        return np.dot(dx, dx) <= radius**2

    def social_drift(self, other: "Agent", beta: float, R_sp: float) -> np.ndarray:
        """Update social position U based on another agent's position."""

        if not self.within_social_radius(other, R_sp):
            return np.zeros_like(self.x)

        sign = float(self.theta @ other.theta)
        if abs(sign) < 1e-10:
            sign = 0.0
        else:
            sign = np.sign(sign)

        return beta * sign * (other.x - self.x)

    def opinion_drift(self, other: "Agent", alpha: float, R_op: float) -> np.ndarray:
        """Update opinion vector based V on another agent's opinion."""

        if not self.within_social_radius(other, R_op):
            return np.zeros_like(self.theta)

        return alpha * (other.theta - self.theta)
