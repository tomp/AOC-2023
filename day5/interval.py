from dataclasses import dataclass


@dataclass(frozen=True, order=True)
class Interval:
    start: int
    end: int

    def __str__(self) -> str:
        return f"Interval[{self.start}, {self.end}]"

    def intersection(self, other: "Interval") -> "Interval":
        if self.end < other.start or other.end < self.start:
            return None

        if self.start < other.start:
            start = other.start
        else:
            start = self.start

        if self.end < other.end:
            return Interval(start, self.end)

        return Interval(start, other.end)

    def union(self, other: "Interval") -> "Interval":
        if not self.intersection(other):
            raise ValueError("Intervals need to overlap to be joined")
        return Interval(min(self.start, other.start), max(self.end, other.end))

    def size(self) -> int:
        return self.end - self.start
