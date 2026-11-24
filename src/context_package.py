"""A context package: the five roles, held as data instead of written as prose.

Semester 1 taught the five roles. This week they become fields on an object,
so that a package can be built, compared, tested and changed one role at a
time. A prompt pasted into a string cannot do any of that.

This file is copied forward into Weeks 3, 7, 9 and 13.
"""

from dataclasses import dataclass, field

ROLES = ["authority", "exemplar", "constraints", "rubric", "metadata"]


@dataclass
class ContextPackage:
    """One engineered context.

    authority   - who the model is
    exemplar    - what a good answer looks like, shown not described
    constraints - what it must not do
    rubric      - how its answer will be judged
    metadata    - what it needs to know (format, audience, length)
    """

    name: str
    authority: str = ""
    exemplar: str = ""
    constraints: list = field(default_factory=list)
    rubric: list = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def missing_roles(self):
        """Return the names of any roles left empty.

        A package with an empty role is a prompt with extra ceremony. The
        Assessment 1 grader runs this.
        """
        values = {
            "authority": self.authority.strip(),
            "exemplar": self.exemplar.strip(),
            "constraints": self.constraints,
            "rubric": self.rubric,
            "metadata": self.metadata,
        }
        return [role for role in ROLES if not values[role]]

    def build_prompt(self):
        """Assemble the five roles into one system instruction.

        The order is always the same: AUTHORITY, METADATA, EXEMPLAR,
        CONSTRAINTS, RUBRIC. Keeping it fixed is what makes two packages
        comparable - otherwise you are measuring layout, not content.
        """
        parts = [f"# AUTHORITY\n{self.authority.strip()}"]

        if self.metadata:
            facts = "\n".join(f"- {k}: {v}" for k, v in self.metadata.items())
            parts.append(f"# METADATA\n{facts}")

        parts.append(f"# EXEMPLAR\nA good answer looks like this:\n\n{self.exemplar.strip()}")

        if self.constraints:
            rules = "\n".join(f"- {c}" for c in self.constraints)
            parts.append(f"# CONSTRAINTS\nYou must not:\n{rules}")

        if self.rubric:
            criteria = "\n".join(f"{i}. {r}" for i, r in enumerate(self.rubric, 1))
            parts.append(
                "# RUBRIC\nCheck your draft against each of these before you reply.\n"
                + criteria
            )

        return "\n\n".join(parts)
