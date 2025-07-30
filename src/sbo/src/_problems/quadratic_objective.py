"""Quadratic Objective."""

from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

from numpy import ndarray
from scipy.sparse import spmatrix

from .exceptions import QiskitOptimizationError
from .linear_expression import LinearExpression
from .quadratic_expression import QuadraticExpression
from .quadratic_program_element import QuadraticProgramElement


class ObjSense(Enum):
    """Objective Sense Type: defines whether to minimize or maximize."""
    MINIMIZE = 1
    MAXIMIZE = -1


class QuadraticObjective(QuadraticProgramElement):
    """
    Represents a quadratic objective of the form:
        constant + linear * x + x^T * Q * x
    """

    Sense = ObjSense  # 🟢 Added alias for clarity

    def __init__(
        self,
        quadratic_program: Any,
        constant: float = 0.0,
        linear: Optional[Union[ndarray, spmatrix, List[float], Dict[Union[str, int], float]]] = None,
        quadratic: Optional[Union[
            ndarray,
            spmatrix,
            List[List[float]],
            Dict[Tuple[Union[int, str], Union[int, str]], float]
        ]] = None,
        sense: ObjSense = ObjSense.MINIMIZE,
    ) -> None:
        """
        Initializes the quadratic objective.

        Args:
            quadratic_program: The parent program this objective belongs to.
            constant: The constant offset term.
            linear: Coefficients for the linear part.
            quadratic: Coefficients for the quadratic part.
            sense: Whether to minimize or maximize.
        """
        super().__init__(quadratic_program)

        self._constant = constant

        # 🟢 Set defaults to empty dictionaries to avoid None-type issues
        self._linear = LinearExpression(quadratic_program, linear if linear is not None else {})
        self._quadratic = QuadraticExpression(quadratic_program, quadratic if quadratic is not None else {})
        self._sense = sense

    @property
    def constant(self) -> float:
        """Returns constant term."""
        return self._constant

    @constant.setter
    def constant(self, constant: float) -> None:
        """Sets constant term."""
        self._constant = constant

    @property
    def linear(self) -> LinearExpression:
        """Returns linear expression."""
        return self._linear

    @linear.setter
    def linear(self, linear: Union[ndarray, spmatrix, List[float], Dict[Union[str, int], float]]) -> None:
        """Sets linear expression."""
        self._linear = LinearExpression(self.quadratic_program, linear)

    @property
    def quadratic(self) -> QuadraticExpression:
        """Returns quadratic expression."""
        return self._quadratic

    @quadratic.setter
    def quadratic(self, quadratic: Union[
        ndarray,
        spmatrix,
        List[List[float]],
        Dict[Tuple[Union[int, str], Union[int, str]], float]
    ]) -> None:
        """Sets quadratic expression."""
        self._quadratic = QuadraticExpression(self.quadratic_program, quadratic)

    @property
    def sense(self) -> ObjSense:
        """Returns optimization direction (minimize or maximize)."""
        return self._sense

    @sense.setter
    def sense(self, sense: ObjSense) -> None:
        """Sets optimization direction."""
        self._sense = sense

    def evaluate(self, x: Union[ndarray, List, Dict[Union[int, str], float]]) -> float:
        """
        Evaluates the objective value for a given assignment.

        Args:
            x: Variable values.

        Returns:
            Objective value.

        Raises:
            QiskitOptimizationError: if shape mismatch occurs.
        """
        n = self.quadratic_program.get_num_vars()

        # 🔴 Check consistency of shapes before evaluation to avoid silent bugs
        if self.linear.coefficients.shape != (1, n) or self.quadratic.coefficients.shape != (n, n):
            raise QiskitOptimizationError(
                "The shape of the objective does not match the number of variables. "
                "Define objective after all variables are declared."
            )

        # 🟢 Split expression for clarity
        const = self.constant
        linear_val = self.linear.evaluate(x)
        quadratic_val = self.quadratic.evaluate(x)
        return const + linear_val + quadratic_val

    def evaluate_gradient(self, x: Union[ndarray, List, Dict[Union[int, str], float]]) -> ndarray:
        """
        Evaluates the gradient of the objective.

        Args:
            x: Variable values.

        Returns:
            Gradient vector.

        Raises:
            QiskitOptimizationError: if shape mismatch occurs.
        """
        n = self.quadratic_program.get_num_vars()

        if self.linear.coefficients.shape != (1, n) or self.quadratic.coefficients.shape != (n, n):
            raise QiskitOptimizationError(
                "The shape of the objective does not match the number of variables. "
                "Define objective after all variables are declared."
            )

        return self.linear.evaluate_gradient(x) + self.quadratic.evaluate_gradient(x)

    def __repr__(self):
        # 🟢 Use truncated pretty expression in developer mode
        from .._translators.prettyprint import DEFAULT_TRUNCATE, expr2str
        expr_str = expr2str(self.constant, self.linear, self.quadratic, DEFAULT_TRUNCATE)
        return f"<{self.__class__.__name__}: {self._sense.name.lower()} {expr_str}>"

    def __str__(self):
        # 🟢 Cleaner user-facing string
        from .._translators.prettyprint import expr2str
        expr_str = expr2str(self.constant, self.linear, self.quadratic)
        return f"{self._sense.name.lower()} {expr_str}"
