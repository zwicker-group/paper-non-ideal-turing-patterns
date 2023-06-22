"""
Module defining free energies

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

import itertools
import logging
import math
from abc import ABCMeta, abstractmethod, abstractproperty
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from numba.extending import register_jitable

from pde.fields.base import FieldBase
from pde.tools.numba import jit, nb
from pde.tools.typing import NumberOrArray


class SolventFractionError(RuntimeError):
    """error indicating that the solvent fraction was not in [0, 1]"""

    pass


def xlogx_scalar(x: float) -> float:
    r"""calculates :math:`x \log(x)`, including the corner case x == 0

    Args:
        x (float): The argument

    Returns:
        float: The result
    """
    if x < 0:
        return math.nan
    elif x == 0:
        return 0
    else:
        return x * np.log(x)  # type: ignore


# vectorize the function above
xlogx: Callable[[NumberOrArray], NumberOrArray] = np.vectorize(xlogx_scalar, otypes="d")


class FreeEnergyBase(metaclass=ABCMeta):
    """abstract base class for free energies"""

    dim: int
    """int: The number of independent components. For an incompressible system,
    this is typically one less than the number of components."""

    variables: List[str]
    """list: the names of the variables defining this free energy. The order in this
    list defines the order in which values are supplied to methods of this class"""

    variable_bounds: Dict[str, Tuple[float, float]]
    """dict: the bounds imposed on each variable"""

    def __init__(self) -> None:
        self._logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def chemical_potential(self, phi: NumberOrArray, t: float = 0) -> np.ndarray:
        pass

    @abstractproperty
    def expression(self) -> str:
        pass

    @abstractmethod
    def __call__(self, *args, t: float = 0):
        pass

    @abstractmethod
    def free_energy(self, phi: NumberOrArray, t: float = 0) -> NumberOrArray:
        pass

    @abstractmethod
    def _repr_data(self) -> Tuple[str, Dict[str, Any]]:
        pass

    def pressure(self, phi: NumberOrArray, t: float = 0) -> np.ndarray:
        r"""evaluate the pressure :math:`P = c\mu - f`

        Args:
            phi: volume fraction at which the pressure is evaluated
            t: simulation time at which the pressure is evaluated

        Returns:
            :class:`numpy.ndarray`: The pressure associated with volume fractions `phi`
        """
        mus = self.chemical_potential(phi, t)
        f = self.free_energy(phi, t)
        if self.dim == 1:
            return phi * mus - f
        else:
            return sum(phi[i] * mus[i] for i in range(self.dim)) - f  # type: ignore

    @abstractmethod
    def make_chemical_potential(
        self, backend: str = "numba"
    ) -> Callable[[np.ndarray, float, np.ndarray], np.ndarray]:
        pass

    def make_chemical_potential_split(
        self, backend: str = "numba"
    ) -> Tuple[
        Callable[..., np.ndarray], Callable[..., np.ndarray], Callable[..., np.ndarray]
    ]:
        raise NotImplementedError

    def __repr__(self):
        template, data = self._repr_data(formatter=repr)
        data["class"] = self.__class__.__name__
        return template.format(**data)

    def __str__(self):
        template, data = self._repr_data(formatter=str)
        data["class"] = self.__class__.__name__
        return template.format(**data)

    def _concentration_samples(self, num: int) -> np.ndarray:
        """return an array of (uniform) samples of valid concentrations

        Args:
            num (int): The number of samples per variable

        Returns:
            :class:`numpy.ndarray`: An array of concentrations. Returns num**dim (or
                less) items of length `dim`, where `dim` is the number of independent
                components.
        """
        # determine test concentration for each variable of the free energy
        c_vars = []
        for v in self.variables:
            c_min, c_max = self.variable_bounds[v]
            if np.isinf(c_min):
                c_min = -0.1
            if np.isinf(c_max):
                c_max = 1.1
            c_vars.append(np.linspace(c_min, c_max, num))

        # build all combinations of these concentrations
        cs = np.array(list(itertools.product(*c_vars)))
        self.regularize_state(cs)
        return np.squeeze(np.unique(cs, axis=0))  # type: ignore

    def regularize_state(self, phi: np.ndarray) -> float:
        """regularize a state ensuring that variables stay within bounds

        Args:
            state (:class:`~numpy.ndarray`):
                The state given as an array of local concentrations

        Returns:
            float: a measure for the corrections applied to the state
        """
        # determine the bounds for the variable
        bounds = self.variable_bounds[self.variables[0]]

        if np.all(np.isinf(bounds)):
            # there are no bounds to enforce
            return 0

        # check whether the state is finite everywhere
        if not np.all(np.isfinite(phi)):
            raise RuntimeError("State is not finite")

        # ensure all variables are positive are in (0, 1)
        np.clip(phi, *bounds, out=phi)

        # TODO: Return the correct amount of regularization applied
        return math.nan

    def make_state_regularizer(
        self, state: FieldBase, global_adjust: bool = False
    ) -> Callable[[np.ndarray], float]:
        """returns a function that can be called to regularize a state

        Args:
            state (:class:`~pde.fields.FieldBase`):
                An example for the state from which the grid and other information can
                be extracted
            global_adjust (bool):
                Flag indicating whether we attempt to preserve the total amount of
                material by adjusting the fields globally.

        Returns:
            Function that can be applied to a state to regularize it and which
            returns a measure for the corrections applied to the state
        """
        if all(np.isinf(a) and np.isinf(b) for a, b in self.variable_bounds.values()):
            self._logger.info("Skip regularizer since no bounds are present")

            # no bounds need to be enforced
            def regularizer_noop(phi: np.ndarray) -> float:
                """no-op regularizer"""
                return 0

            return regularizer_noop

        # we need to enforce bounds for multiple variables
        dim = self.dim
        bounds = np.array([self.variable_bounds[v] for v in self.variables])
        assert bounds.shape == (dim, 2)
        self._logger.info("Use regularizer enforcing bounds %s", self.variable_bounds)

        vol_sys = state.grid.volume
        vol = state.grid.make_cell_volume_compiled(flat_index=True)

        @register_jitable
        def regularizer_inner(phi: np.ndarray, phi_min: float, phi_max: float) -> float:
            """helper function ensuring a single species stays in a given bound"""
            # accumulate lower and upper bound separately; both values are positive
            amount_low = 0.0
            vol_low = 0.0
            amount_high = 0.0
            vol_high = 0.0

            # determine the deviation amount
            for i in range(phi.size):
                if phi.flat[i] < phi_min:
                    # concentration is below lower bound
                    v = vol(i)
                    amount_low += v * (phi_min - phi.flat[i])
                    vol_low += v
                    phi.flat[i] = phi_min

                elif phi.flat[i] > phi_max:
                    # concentration is above upper bound
                    v = vol(i)
                    amount_high += v * (phi.flat[i] - phi_max)
                    vol_high += v
                    phi.flat[i] = phi_max

            # correct the data if requested
            if global_adjust:
                if amount_high > amount_low:
                    # we had more points that were too high => add material elsewhere
                    conc_corr = (amount_high - amount_low) / (vol_sys - vol_high)
                    assert conc_corr > 0
                    for i in range(phi.size):
                        phi.flat[i] = min(phi_max, phi.flat[i] + conc_corr)

                elif amount_high < amount_low:
                    # we had more points that were too low => remove material elsewhere
                    conc_corr = (amount_low - amount_high) / (vol_sys - vol_low)
                    assert conc_corr > 0
                    for i in range(phi.size):
                        phi.flat[i] = max(phi_min, phi.flat[i] - conc_corr)

                # else:
                #     both amounts are equal and cancel each other

            # return the total amount that was corrected anywhere
            return amount_high + amount_low

        if self.dim == 1:
            # a single species => array is not nested
            phi_min, phi_max = bounds[0]

            def regularizer(phi: np.ndarray) -> float:
                """ensure all variables are positive are in (0, 1)"""
                return regularizer_inner(phi, phi_min, phi_max)  # type: ignore

        else:
            # multiple species => correct each individual species
            def regularizer(phi: np.ndarray) -> float:
                """ensure all variables are positive are in (0, 1)"""
                # iterate over all species
                correction = 0
                for j in nb.prange(dim):
                    phi_min, phi_max = bounds[j]
                    correction += regularizer_inner(phi[j], phi_min, phi_max)
                return correction

        return regularizer


class FreeEnergyNComponentsBase(FreeEnergyBase, metaclass=ABCMeta):
    """abstract base class for free energies of multiple components"""

    def __init__(
        self,
        variables: List[str],
        *,
        variable_bounds: Dict[str, Tuple[float, float]] = {"*": (-np.inf, np.inf)},
    ):
        """
        Args:
            variables (list of str):
                The names of the fields appearing in the free energy. The length of this
                list determines the dimension of the composition space that this free
                energy describes.
            variable_bounds (dict):
                Gives the bounds for each variable. Each entry specifies the bounds for
                one variable using a tuple of lower and upper bound. The special entry
                '*' specifies bounds for variables that are not otherwise specified. The
                default value does not impose any bounds on the variables. A typical
                bound for volume fractions can be imposed using `{"*": (0, 1))}`.
        """
        super().__init__()

        # set name of the variables
        self.variables: List[str] = variables
        self.dim = len(variables)

        # set the bounds of all variables
        self.variable_bounds = {}
        default_bound = variable_bounds.get("*", (-np.inf, np.inf))
        for variable in self.variables:
            bound = variable_bounds.get(variable, default_bound)
            if bound is None:
                raise RuntimeError(f"Could not determine bound for `{variable}`")
            self.variable_bounds[variable] = bound

    def regularize_state(self, phi: np.ndarray, sum_max: float = 1 - 1e-8) -> float:
        """regularize a state ensuring that variables stay within bounds

        The bounds for all variables are defined in the class attribute
        :attr:`variable_bounds`.

        Args:
            phi (:class:`~numpy.ndarray`):
                The state given as an array of local concentrations
            sum_max (float):
                The maximal value the sum of all concentrations may have. This can be
                used to limit the concentration of a variable that has been removed due
                to incompressibility. If this value is set to `np.inf`, the constraint
                is not applied

        Returns:
            float: a measure for the corrections applied to the state
        """
        if not np.all(np.isfinite(phi)):
            raise RuntimeError("State is not finite")

        if self.dim == 1:
            # deal with a single variable
            return super().regularize_state(phi)

        else:
            # deal with multiple variables

            # adjust each variable individually
            for i, variable in enumerate(self.variables):
                bounds = self.variable_bounds[variable]
                phi[i] = np.clip(phi[i], *bounds)
                # Note that we did not use the `out` argument, since this would not work
                # if `phi[i]` was a scalar

            # limit the sum of all variables
            if np.isfinite(sum_max):
                phis = phi.sum(axis=0)
                loc = phis > sum_max
                if np.any(loc):
                    phi[:, loc] *= sum_max / phis[loc]

        # TODO: Return the correct amount of regularization applied
        return math.nan

    def make_state_regularizer(
        self, state: FieldBase, sum_max: float = 1 - 1e-8
    ) -> Callable[[np.ndarray], float]:
        """returns a function that can be called to regularize a state

        Args:
            state (:class:`~pde.fields.FieldBase`):
                An example for the state from which the grid and other
                information can be extracted
            sum_max (float):
                The maximal value the sum of all concentrations may have. This can be
                used to limit the concentration of a variable that has been removed due
                to incompressibility. If this value is set to `np.inf`, the constraint
                is not applied

        Returns:
            Function that can be applied to a state to regularize it and which
            returns a measure for the corrections applied to the state
        """
        if self.dim == 1:
            # deal with a single variable
            return super().make_state_regularizer(state)

        else:
            # deal with multiple variables
            dim = self.dim
            bounds = np.array(
                [self.variable_bounds[variable] for variable in self.variables]
            )

            def regularizer(phi: np.ndarray) -> Callable[[np.ndarray], float]:
                """regularize a state ensuring variables stay within bounds"""
                if not isinstance(phi, (np.ndarray, nb.types.Array)):
                    raise TypeError

                if phi.ndim == 1:
                    # a single set of concentrations is given

                    def regularizer_impl(phi: np.ndarray) -> float:
                        """regularize a state ensuring variables stay within bounds"""
                        correction = 0.0

                        # adjust each variable individually
                        for i in range(dim):
                            if phi[i] < bounds[i, 0]:
                                correction += bounds[i, 0] - phi[i]
                                phi[i] = bounds[i, 0]
                            elif phi[i] > bounds[i, 1]:
                                correction += phi[i] - bounds[i, 1]
                                phi[i] = bounds[i, 1]

                        # limit the sum of all variables
                        if np.isfinite(sum_max):
                            phis = 0.0
                            for i in range(dim):
                                phis += phi[i]
                            if phis > sum_max:
                                for i in range(dim):
                                    phi[i] *= sum_max / phis

                        return correction

                else:
                    # an array of concentrations is given

                    def regularizer_impl(phi: np.ndarray) -> float:
                        """regularize a state ensuring variables stay within bounds"""
                        correction = 0.0

                        # adjust each variable individually
                        for i in range(dim):
                            for j in range(phi[0].size):
                                if phi[i].flat[j] < bounds[i, 0]:
                                    correction += bounds[i, 0] - phi[i].flat[j]
                                    phi[i, ...].flat[j] = bounds[i, 0]
                                elif phi[i].flat[j] > bounds[i, 1]:
                                    correction += phi[i].flat[j] - bounds[i, 1]
                                    phi[i, ...].flat[j] = bounds[i, 1]

                        # limit the sum of all variables
                        if np.isfinite(sum_max):
                            for j in range(phi[0].size):
                                phis = 0.0
                                for i in range(dim):
                                    phis += phi[i].flat[j]
                                if phis > sum_max:
                                    for i in range(dim):
                                        phi[i, ...].flat[j] *= sum_max / phis

                        # Note that we needed to use phi[i, ...] to write to the array
                        # also when it is 1d to circumvent a known bug:
                        # https://github.com/numpy/numpy/issues/16881

                        return correction

                return regularizer_impl

            if nb.config.DISABLE_JIT:
                # jitting is disabled => return generic python function

                # we here simply supply a 2d array so the more generic implementation
                # is chosen, which works for all cases in the case of numpy
                return regularizer(np.empty((self.dim, 2)))

            else:
                # jitting is enabled => return specialized, compiled function
                return nb.generated_jit(nopython=True)(regularizer)  # type: ignore


class FloryHugginsNComponents(FreeEnergyNComponentsBase):
    r"""Flory-Huggins free energy for arbitrary number of components

    The :math:`N` components are described by their volume fractions :math:`\phi_i`,
    where we here use the Python convention of enumeration: :math:`i=0,\ldots,N-1`. The
    local free energy density for has the form

    .. math ::
        f(\phi) = \sum_{i=0}^{N-1} \frac{\phi_i}{\nu_i} \ln(\phi_i) +
            \sum_{i,j=0}^{N-1} \frac{\chi_{ij}}{2} \phi_i \phi_j +
            \sum_{i=0}^{N-1} w_i \phi_i

    where :math:`\nu_i` are the relative molecular volumes, :math:`\chi_{ij}` is the
    Flory interaction parameter matrix, and :math:`\alpha_i` determine the internal
    energies, which can affect chemical reactions. Note that the Flory matrix must be
    symmetric, :math:`\chi_{ij} = \chi_{ji}`, with vanishing elements on the diagonal,
    :math:`\chi_{ii} = 0`.

    Since we assume an incompressible system, only :math:`N - 1` components are
    independent. We thus eliminate the last component using
    :math:`\phi_{N-1} = 1 - \sum_{i=0}^{N-2} \phi_i`. Consequently, the chemical
    potentials used in the description are exchange chemical potentials
    :math:`\bar\mu_i = \mu_i - \mu_{N-1}` describing the difference to the chemical
    potential of the removed component.
    """

    variable_bounds_default = (1e-8, 1 - 1e-8)
    explicit_time_dependence = False
    raise_solvent_fraction_error: bool = True
    """bool: determines how to deal with non-positive solvent fractions

    If the flag is `True` a SolventFractionError is raised when calculating chemical
    potentials for fractions whose sum is >= 1 (so the solvent fraction would be
    non-negative). If the flag is `False`, NaN's are returned instead of raising an
    exception.
    """

    sizes: np.ndarray
    _chis: np.ndarray
    _internal_energies: np.ndarray

    def __init__(
        self,
        num_comp: int,
        chis: NumberOrArray = 0,
        sizes: NumberOrArray = 1,
        internal_energies: NumberOrArray = 0,
        *,
        variables: Optional[Sequence[str]] = None,
        **kwargs,
    ):
        r"""
        Args:
            num_comp (int):
                Number of components described by this free energy. The number of
                independent components will be one less.
            chi (`numpy.ndarray` or float):
                Flory-interaction parameters :math:`\chi_{ij}` with shape
                `(num_comp, num_comp)`. Alternatively, a float value may be given, which
                then determines all off-diagonal entries.
            size (`numpy.ndarray` or float):
                Array of shape `(num_comp,)` determining the relative molecular volumes
                :math:`\nu_i`. Float values are broadcasted to full arrays.
            internal_energies (`numpy.ndarray` or float):
                Array with shape `(num_comp,)` setting the internal energies
                :math:`w_i`. Float values are broadcasted to full arrays.
            variables (list):
                The name of the variables in the free energy. If omitted, they will be
                named `phi#`, where # is a integer between 1 and num_comp - 1.
        """
        if num_comp < 2:
            raise ValueError("Need at least two components")
        # determine how all the components are called
        if variables is None:
            if num_comp == 2:
                variables = ["phi"]
            else:
                variables = [f"phi{i}" for i in range(1, num_comp)]
        elif len(variables) == num_comp:
            variables = list(variables[:-1])
        elif len(variables) == num_comp - 1:
            variables = list(variables)
        else:
            raise ValueError(f"`variables` must be a list of {num_comp - 1} strings")

        super().__init__(
            variables=variables, variable_bounds={"*": self.variable_bounds_default}
        )

        # set all internal variables
        self.sizes = np.broadcast_to(sizes, (num_comp,))
        self._internal_energies = np.broadcast_to(internal_energies, (self.num_comp,))
        self.chis = chis  # type: ignore
        # the last assignment also calculate the reduced chis and internal_energies

        if kwargs:
            raise ValueError(f"Did not use arguments {kwargs}")

    @property
    def num_comp(self) -> int:
        """int: the number of components in the mixture"""
        return self.dim + 1

    @classmethod
    def random_normal(
        cls,
        num_comp: int,
        chi_mean: float = 0,
        chi_std: float = 1,
        *,
        last_component_inert: bool = False,
        rng: Optional[np.random.Generator] = None,
    ) -> FloryHugginsNComponents:
        """create a Flory-Huggins free energy density with random interactions

        In essence, this function creates a random, symmetric chi-matrix with vanishing
        diagonal terms. The remaining terms are drawn independently from a normal
        distribution.

        Args:
            num_comp (int):
                Number of components described by this free energy. The number of
                independent components will be one less.
            chi_mean (float):
                The mean value of the random entries of the chi matrix
            chi_std (float):
                The standard deviation of the random entries of the chi matrix
            last_component_inert (bool):
                Flag determining whether the last component, which is removed due to
                incompressibility is inert. If this is the case, the associated entries
                in the chi matrix will be zero
            rng (:class:`~numpy.random.Generator`):
                Random number generator (default: :func:`~numpy.random.default_rng()`)

        Returns:
            :class:`~phasesep.free_energies.flory_huggins.FloryHugginsNComponents`:
            The actual object representing the free energy density
        """
        if rng is None:
            rng = np.random.default_rng()

        chis = np.zeros((num_comp, num_comp))
        num_entries = num_comp * (num_comp - 1) // 2
        chi_vals = rng.normal(chi_mean, chi_std, num_entries)
        chis[np.triu_indices(num_comp, 1)] = chi_vals
        if last_component_inert:
            chis[:, -1] = 0
        chis += chis.T  # symmetrize the chi matrix
        return cls(num_comp, chis)

    @property
    def chis(self) -> np.ndarray:
        r"""Flory interaction parameters :math:`\chi_{ij}`"""
        return self._chis

    @chis.setter
    def chis(self, value: NumberOrArray):
        """set the interaction parameters"""
        shape = (self.num_comp, self.num_comp)
        if np.isscalar(value):
            # a scalar value sets all off-diagonal entries
            chis = np.full(shape, value)

        else:
            chis = np.array(np.broadcast_to(value, shape))
            if not np.allclose(np.diag(chis), 0):
                self._logger.warning("Diagonal part of the chi matrix is not used")

        # ensure that the diagonal entries vanish
        np.fill_diagonal(chis, 0)

        # ensure that the chi matrix is symmetric
        if not np.allclose(chis, chis.T):
            self._logger.warning("Using symmetrized χ interaction-matrix")
        self._chis = 0.5 * (chis + chis.T)

        self._calculate_reduced_values()

    @property
    def internal_energies(self) -> np.ndarray:
        r"""Internal energies :math:`w_i`"""
        return self._internal_energies

    @internal_energies.setter
    def internal_energies(self, values: NumberOrArray):
        """sets the internal energies of the free energy"""
        self._internal_energies = np.broadcast_to(values, (self.num_comp,))
        self._calculate_reduced_values()

    def _calculate_reduced_values(self) -> None:
        """calculate the reduced Flory parameters and internal_energies"""
        chis = self.chis
        w = self.internal_energies
        self._chis_reduced = np.empty((self.dim, self.dim))
        self._internal_energies_reduced = np.empty(self.dim)

        n = self.num_comp - 1  # index of the component to be removed
        for i in range(self.dim):
            for j in range(self.dim):
                self._chis_reduced[i, j] = chis[i, j] - chis[i, n] - chis[n, j]
            self._internal_energies_reduced[i] = w[i] - w[n] + chis[i, n]

    def _repr_data(self, formatter=str) -> Tuple[str, Dict[str, Any]]:
        """return data useful for representing this class"""
        data = {
            "num_comp": self.num_comp,
            "chis": formatter(self.chis),
            "sizes": formatter(self.sizes),
            "internal_energies": formatter(self.internal_energies),
        }
        template = (
            "{class}(num_comp={num_comp}, chis={chis}, sizes={sizes} "
            "internal_energies={internal_energies})"
        )
        return template, data

    @property
    def expression(self) -> str:
        """str: the mathematical expression describing the free energy"""
        # gather all the variables
        var_last = f"(1 - {' - '.join(self.variables)})"
        variables = self.variables + [var_last]

        result = []
        # entropic terms
        for i, (var, size) in enumerate(zip(variables, self.sizes)):
            log_var = f"log({var[1:-1]})" if i == self.dim else f"log({var})"
            if size == 1:
                result.append(f"{var} * {log_var}")
            else:
                result.append(f"{var}/{size:g} * {log_var}")

        # quadratic enthalpic terms
        for i, vi in enumerate(variables):
            for j, vj in enumerate(variables[i:], i):
                if self.chis[i, j] != 0:
                    term = f"{self.chis[i, j]:g} * {vi} * {vj}"
                    result.append(term)

        # linear enthalpic terms
        for i, vi in enumerate(variables):
            if self.internal_energies[i] != 0:
                result.append(f"{self.internal_energies[i]:g} * {vi}")

        return " + ".join(result)

    def free_energy(self, phi: NumberOrArray, t: float = 0) -> NumberOrArray:
        """evaluate the local free energy density

        Args:
            phi: volume fraction at which the free energy is evaluated
            t: simulation time at which the free energy is evaluated

        Returns:
            the free energy associated with `phi`
        """
        phi = np.asanyarray(phi)
        assert len(phi) == self.dim, f"Require {self.dim} fields"
        phi_last: np.ndarray = 1 - phi.sum(axis=0)

        return (  # type: ignore
            np.einsum("i,i...->...", 1 / self.sizes[:-1], xlogx(phi))
            + xlogx(phi_last) / self.sizes[self.dim]
            + np.einsum("i...,ij,j...->...", phi, 0.5 * self._chis_reduced, phi)
            + np.einsum("i,i...->...", self._internal_energies_reduced, phi)
            + self.internal_energies[self.dim]
        )

    def __call__(self, *phis, t: float = 0):
        return self.free_energy(np.array(phis), t)

    def chemical_potential(
        self,
        phi: NumberOrArray,
        t: float = 0,
        *,
        out: Optional[np.ndarray] = None,
        regularize: bool = True,
    ) -> np.ndarray:
        """evaluate the local part of the chemical potential

        Args:
            phi: volume fraction at which the chemical potential is evaluated
            t: time at which the chemical potential is evaluated
            out: Array to which results are written
            regularize (bool): Whether to regularize the input

        Returns:
            the chemical potential associated with `phi`
        """
        phi = np.atleast_1d(phi)
        assert len(phi) == self.dim, f"Require {self.dim} fields"
        if regularize:
            self.regularize_state(phi)
        phi_last = 1 - phi.sum(axis=0)

        if self.raise_solvent_fraction_error and np.any(phi_last <= 0):
            raise SolventFractionError

        if out is None:
            out = np.empty_like(phi)

        for i in range(self.dim):
            out[i] = (1 + np.log(phi[i])) / self.sizes[i]
            for j in range(self.dim):
                out[i] += self._chis_reduced[i, j] * phi[j]
            out[i] += self._internal_energies_reduced[i]
        out -= (1 + np.log(phi_last)) / self.sizes[self.dim]

        return out  # type: ignore

    def make_chemical_potential(
        self, backend: str = "numba"
    ) -> Callable[[np.ndarray, float, np.ndarray], np.ndarray]:
        """return function evaluating the chemical potential

        Args:
            backend (str):
                Specifies how the functions are created. Accepted values are 'numpy'
                and 'numba'.

        Returns:
            A function that evaluates the chemical potential
        """
        if backend == "numpy":
            # use straight-forward numpy version
            mu_local = self.chemical_potential

        elif backend == "numba":
            # numba optimized version
            dim = self.dim
            sizes_inv = 1 / self.sizes
            chi_reduced = self._chis_reduced
            internal_energies_reduced = self._internal_energies_reduced
            raise_solvent_fraction_error = self.raise_solvent_fraction_error

            @jit
            def mu_local(arr: np.ndarray, t: float, out: np.ndarray) -> np.ndarray:
                for index in nb.prange(arr[0].size):
                    # determine solvent component
                    phi_last = 1
                    for i in range(dim):
                        phi_last -= arr[i].flat[index]

                    if phi_last <= 0:
                        # solvent component is negative => invalid state
                        if raise_solvent_fraction_error:
                            raise SolventFractionError  # raise error
                        else:
                            # signal error by returning not-a-number
                            for i in range(dim):
                                out[i].flat[index] = np.nan

                    else:
                        # solvent fraction is fine, so we can calculate everything
                        entropy_last = (1 + np.log(phi_last)) * sizes_inv[dim]

                        for i in range(dim):
                            # calculate chemical potential for species i
                            mu = (1 + np.log(arr[i].flat[index])) * sizes_inv[i]
                            for j in range(dim):
                                mu += chi_reduced[i, j] * arr[j].flat[index]

                            mu += internal_energies_reduced[i] - entropy_last
                            out[i].flat[index] = mu
                return out

        else:
            raise ValueError(f"Backend `{backend}` is not supported")

        return mu_local  # type: ignore
