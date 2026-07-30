class Trool:
    """
    Trinary logic representation (-1: False, 0: Uncertain, 1: True).
    """
    VFALSE = -1
    ISH = 0
    VTRUE = 1
    
    def __init__(self, value=0):
        if value not in {self.VFALSE, self.ISH, self.VTRUE}:
            raise ValueError("Trool must be -1, 0, or 1.")
        self.value = value
    
    def __neg__(self):
        """Negation operation."""
        return Trool(-self.value)
    
    def __sub__(self, other):
        """Return the sign of the difference, preserving the trinary domain."""
        if isinstance(other, Trool):
            difference = self.value - other.value
            return Trool((difference > 0) - (difference < 0))
        raise TypeError("Subtraction must be between two Trool objects.")
    
    def __repr__(self):
        return f"Trool({self.value})"


class Tri:
    """
    Represents a 3-element structure, useful for trinary-based geometric representations.
    """
    def __init__(self, a, b, c):
        self.tri = [a, b, c]
    
    def __getitem__(self, index):
        return self.tri[index]
    
    def __setitem__(self, index, value):
        self.tri[index] = value
    
    def __sub__(self, other):
        """Component-wise subtraction."""
        if not isinstance(other, Tri):
            raise TypeError("Subtraction must be between two Tri objects.")
        return Tri(
            self.tri[0] - other.tri[0],
            self.tri[1] - other.tri[1],
            self.tri[2] - other.tri[2]
        )
    
    def norm(self):
        """Computes the Euclidean norm of the Tri structure."""
        return sum(x.value**2 for x in self.tri) ** 0.5
    
    def __repr__(self):
        return f"Tri({self.tri})"


# Example usage
if __name__ == "__main__":
    t1 = Tri(Trool(1), Trool(0), Trool(-1))
    t2 = Tri(Trool(-1), Trool(1), Trool(0))
    
    diff = t1 - t2
    print("Tri difference:", diff)
    print("Norm of difference:", diff.norm())
