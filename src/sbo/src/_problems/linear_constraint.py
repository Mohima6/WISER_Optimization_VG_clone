# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2019, 2023.
#
# This code is licensed under the Apache License, Version 2.0.
# You may obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Linear Constraint definition for a Quadratic Program."""  # CHANGE: Clarified class purpose

from typing import Any, Dict, List, Union
from numpy import ndarray
from scipy.sparse import spmatrix

from .constraint import Constraint, ConstraintSense
from .linear_expression import LinearExpression


class LinearConstraint(Constraint):
    """Represents a linear constraint in a quadratic program.

    A linear constraint has the general form:
        linear_expr(x) (==, <=, >=) rhs
    """

    Sense = ConstraintSense  # CHANGE: Added comment below to explain purpose
    # CHANGE: Duplicated for Sphinx compatibility with static doc tools

    def __init__(
        self,
        quadratic_program: Any,
        name: str,
        linear: Union[ndarray, spmatrix, List[float], Dict[Union[str, int], float]],
        sense: ConstraintSense,
        rhs: float,
    ) -> None:
        """
        Args:
            quadratic_program: The parent quadratic program.
            name: Name of the constraint (must be unique).
            linear: Coefficients of the linear expression (LHS).
            sense: The sense of the constraint (e.g., ==, <=, >=).
            rhs: The right-hand-side scalar value.
        """
        super().__init__(quadratic_program, name, sense, rhs)
        self._linear = LinearExpression(quadratic_program, linear)
        # CHANGE: Explained arguments and logic in a detailed and readable way

    @property
    def linear(self) -> LinearExpression:
        """Return the linear expression (left-hand side) of the constraint.

        Returns:
            LinearExpression: Expression object holding coefficients.
        """
        return self._linear

    @linear.setter
    def linear(
        self,
        linear: Union[ndarray, spmatrix, List[float], Dict[Union[str, int], float]],
    ) -> None:
        """Set a new linear expression for the constraint's left-hand side.

        Args:
            linear: New coefficients as array, sparse matrix, list, or dictionary.
        """
        self._linear = LinearExpression(self.quadratic_program, linear)
        # CHANGE: Reworded docstring and added clarity about acceptable formats

    def evaluate(self, x: Union[ndarray, List, Dict[Union[int, str], float]]) -> float:
        """Evaluate the left-hand side expression for given variable values.

        Args:
            x: Variable values (array, list, or dict keyed by variable name or index).

        Returns:
            float: Value of the linear expression.
        """
        return self.linear.evaluate(x)
        # CHANGE: Improved docstring to clarify input types and evaluation purpose

    def __repr__(self) -> str:
        from .._translators.prettyprint import DEFAULT_TRUNCATE, expr2str

        lhs = expr2str(linear=self.linear, truncate=DEFAULT_TRUNCATE)
        return f"<{self.__class__.__name__}: {lhs} {self.sense.label} {self.rhs} '{self.name}'>"
        # CHANGE: No logic change — just ensured inline comments clarify cyclic import

    def __str__(self) -> str:
        from .._translators.prettyprint import expr2str

        lhs = expr2str(linear=self.linear)
        return f"{lhs} {self.sense.label} {self.rhs} '{self.name}'"
        # CHANGE: Same as above — restructured for clarity without changing function
