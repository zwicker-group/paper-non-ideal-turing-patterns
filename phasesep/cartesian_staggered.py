r"""
Implements operators for staggered Cartesian grids 

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from typing import Callable, List, Optional, Sequence

import numba as nb
import numpy as np
from numba.extending import register_jitable

from pde import config
from pde.grids.boundaries import Boundaries
from pde.grids.boundaries.local import DirichletBC, NormalDirichletBC
from pde.grids.cartesian import CartesianGrid
from pde.tools.expressions import ScalarExpression
from pde.tools.numba import jit
from pde.tools.typing import OperatorType


def scalar_field_to_staggered(arr: np.ndarray, axis: int) -> np.ndarray:
    """interpolate a scalar field to a staggered grid

    Args:
        arr (:class:`numpy.ndarray`):
            The values at the support points of the original grid
        axis (int):
            Along which axis the grid should be staggered

    Returns:
        :class:`numpy.ndarray`: The values at the support points of the staggered grid
    """
    # determine valid part of the array
    window_left = [slice(1, -1)] * arr.ndim
    window_left[axis] = slice(0, -1)
    window_right = [slice(1, -1)] * arr.ndim
    window_right[axis] = slice(1, None)

    return (arr[tuple(window_left)] + arr[tuple(window_right)]) / 2  # type: ignore


def make_sc_flux_to_staggered_scipy(
    grid: CartesianGrid, mobility: Optional[ScalarExpression] = None
) -> Callable[[Optional[np.ndarray], np.ndarray], List[np.ndarray]]:
    """make function calculating the flux for a single component on a staggered grid

    Args:
        grid (:class:`pde.grids.cartesian.CartesianGrid`):
            The original Cartesian grid on which the fields are defined
        mobility (:class:`pde.tools.expressions.ScalarExpression`, optional):
            The expression that is used to evaluate the mobility. If omitted, the
            mobility is assumed to be 1.

    Returns:
        A function that can be applied to an array of values
    """
    assert isinstance(grid, CartesianGrid)
    scale = -1 / grid.discretization
    dim = grid.dim

    def calc_flux(
        phi_full: Optional[np.ndarray], mu_full: np.ndarray
    ) -> List[np.ndarray]:
        """calculate flux driven by gradients in `mu`"""
        # check input
        if phi_full is None:
            assert mobility is None
        else:
            assert phi_full.shape == grid._shape_full
        assert mu_full.shape == grid._shape_full

        fluxes = []
        for axis in range(dim):
            # calculate the gradient along the axis for the valid data
            valid = [slice(1, -1)] * dim
            valid[axis] = slice(None)
            flux = np.diff(mu_full[tuple(valid)], axis=axis) * scale[axis]

            # multiply by the mobility if requested
            if mobility is None:
                pass  # nothing to do here
            elif mobility.constant:
                flux *= mobility.value
            else:
                # space dependent mobility
                phi_staggered = scalar_field_to_staggered(phi_full, axis=axis)  # type: ignore
                flux *= mobility(phi_staggered)

            fluxes.append(flux)

        return fluxes

    return calc_flux


def make_mc_flux_to_staggered_scipy(
    grid: CartesianGrid,
    num_comp: int,
    diffusivity: Optional[np.ndarray] = None,
    mobility_model: str = "const",
) -> Callable[[np.ndarray, np.ndarray], List[List[np.ndarray]]]:
    """make function calculating the flux for multiple components on a staggered grid

    Args:
        grid (:class:`pde.grids.cartesian.CartesianGrid`):
            The Cartesian grid on which the fields are defined
        num_comp (int):
            The number of components that are described in the state. Because of
            incompressibility, this is usually one component less than in the actual
            mixtures.
        diffusivity (:class:`~numpy.ndarray`, optional):
            The matrix that determines the (cross)-diffusivities for all component. If
            omitted it is replaced by a full matrix of ones.
        mobility_model (str):
            The mobility model to use

    Returns:
        A function that can be applied to an array of values
    """
    assert isinstance(grid, CartesianGrid)
    dim = grid.dim

    # pick correct mobilities
    if mobility_model == "scaled_correct":
        if diffusivity is None:
            diff_arr = np.ones(num_comp + 1)
        else:
            diff_arr = np.asarray(diffusivity)
        assert diff_arr.shape == (num_comp + 1,)
    elif mobility_model == "scaled_diagonal":
        if diffusivity is None:
            diff_arr = np.ones(num_comp)
        else:
            diff_arr = np.asarray(diffusivity)
        if diff_arr.ndim == 2:
            diff_arr = np.diag(diff_arr)
        if len(diff_arr) == num_comp + 1:
            diff_arr = diff_arr[:-1]
        assert diff_arr.shape == (num_comp,)
    elif mobility_model == "const":
        if diffusivity is None:
            diff_arr = np.ones((num_comp, num_comp))
        else:
            diff_arr = np.asarray(diffusivity)
        assert diff_arr.shape == (num_comp, num_comp)
    else:
        raise RuntimeError(f"Unsupported mobility model `{mobility_model}`")

    calc_sc_flux = make_sc_flux_to_staggered_scipy(grid, mobility=None)

    def calc_flux(phi: np.ndarray, mu: np.ndarray) -> List[List[np.ndarray]]:
        """calculate fluxes driven by gradients in `mu`"""
        assert phi.shape == (num_comp,) + grid._shape_full
        assert mu.shape == (num_comp,) + grid._shape_full

        # calculate the negative gradients of mu on the staggered grid
        mu_grads = [calc_sc_flux(None, mu[n]) for n in range(num_comp)]

        # determine the fluxes
        fluxes = [
            [np.zeros_like(mu_grads[n][i]) for i in range(dim)] for n in range(num_comp)
        ]
        for i in range(dim):  # iterate over all dimension
            if mobility_model == "scaled_correct":
                # determine the staggered phi field for this dimension
                phis_staggered = [
                    scalar_field_to_staggered(phi[n], axis=i) for n in range(num_comp)
                ]
                # get denominator for mobility model
                denominator = diff_arr[num_comp] * (1 - sum(phis_staggered))
                for n in range(num_comp):
                    denominator += diff_arr[n] * phis_staggered[n]

                # determine the effect of the thermodynamic force on the flux
                with np.errstate(all="ignore"):
                    # errors can appear for ghost cells
                    for n in range(num_comp):  # iterate over all components
                        for m in range(num_comp):  # iterate over all components
                            mob = diff_arr[n] * phis_staggered[n]
                            mob *= diff_arr[m] * phis_staggered[m]
                            mob /= -denominator
                            if n == m:
                                mob += diff_arr[n] * phis_staggered[n]
                            fluxes[n][i] += mob * mu_grads[m][i]

            elif mobility_model == "scaled_diagonal":
                # determine the staggered phi field for this dimension
                # determine the effect of the thermodynamic force on the flux
                with np.errstate(all="ignore"):
                    # errors can appear for ghost cells
                    for n in range(num_comp):  # iterate over all components
                        phis_staggered_n = scalar_field_to_staggered(phi[n], axis=i)
                        fluxes[n][i] += diff_arr[n] * phis_staggered_n * mu_grads[n][i]

            elif mobility_model == "const":
                # determine the effect of the thermodynamic force on the flux
                with np.errstate(all="ignore"):
                    # errors can appear for ghost cells
                    for n in range(num_comp):  # iterate over all components
                        for m in range(num_comp):  # iterate over all components
                            if diff_arr[n, m] != 0:
                                fluxes[n][i] += diff_arr[n, m] * mu_grads[m][i]

        return fluxes

    return calc_flux


def _make_sc_flux_to_staggered_numba_1d(
    grid: CartesianGrid, mobility: Optional[Callable[[float], float]] = None
) -> Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]:
    """make function calculating the flux for a single component on a 1d staggered grid

    Args:
        grid (:class:`pde.grids.cartesian.CartesianGrid`):
            The Cartesian grid on which the fields are defined
        mobility (callable, optional):
            The function that is used to evaluate the mobility. If omitted, the
            mobility is assumed to be 1.

    Returns:
        A function that can be applied to an array of values. This function takes three
        arguments: the chemical potential `mu`, the phase field `phi`, and an array to
        which the result is written.
    """
    assert isinstance(grid, CartesianGrid) and grid.dim == 1
    (dim_x,) = grid.shape
    (dx,) = grid.discretization

    # use parallel processing for large enough arrays
    parallel = dim_x >= config["numba.multithreading_threshold"]

    if mobility is None:

        @register_jitable
        def get_mobility(phi: float) -> float:
            return 1.0

    else:
        get_mobility = mobility

    @jit(parallel=parallel)
    def calc_flux(phi: np.ndarray, mu: np.ndarray, out: np.ndarray) -> np.ndarray:
        """calculate flux driven by gradients in `mu`"""
        # calculate inner points
        for i in nb.prange(dim_x + 1):
            factor = get_mobility(0.5 * (phi[i + 1] + phi[i]))
            out[0, i] = -factor * (mu[i + 1] - mu[i]) / dx

        return out

    return calc_flux  # type: ignore


def _make_sc_flux_to_staggered_numba_2d(
    grid: CartesianGrid, mobility: Optional[Callable[[float], float]] = None
) -> Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]:
    """make function calculating the flux for a single component on a 2d staggered grid

    Args:
        grid (:class:`pde.grids.cartesian.CartesianGrid`):
            The Cartesian grid on which the fields are defined
        mobility (callable, optional):
            The function that is used to evaluate the mobility. If omitted, the
            mobility is assumed to be 1.

    Returns:
        A function that can be applied to an array of values. This function takes three
        arguments: the chemical potential `mu`, the phase field `phi`, and an array to
        which the result is written.
    """
    assert isinstance(grid, CartesianGrid) and grid.dim == 2
    dim_x, dim_y = grid.shape
    dx, dy = grid.discretization

    # use parallel processing for large enough arrays
    parallel = dim_x * dim_y >= config["numba.multithreading_threshold"]

    if mobility is None:

        @register_jitable
        def get_mobility(phi: float) -> float:
            return 1.0

    else:
        get_mobility = mobility

    @jit(parallel=parallel)
    def calc_flux(phi: np.ndarray, mu: np.ndarray, out: np.ndarray) -> np.ndarray:
        """calculate flux driven by gradients in `mu`"""
        for i in nb.prange(dim_x + 1):
            for j in range(1, dim_y + 1):
                factor = -get_mobility(0.5 * (phi[i + 1, j] + phi[i, j]))
                out[0, i, j] = factor * (mu[i + 1, j] - mu[i, j]) / dx

        for i in nb.prange(1, dim_x + 1):
            for j in range(dim_y + 1):
                factor = -get_mobility(0.5 * (phi[i, j + 1] + phi[i, j]))
                out[1, i, j] = factor * (mu[i, j + 1] - mu[i, j]) / dy

        return out

    return calc_flux  # type: ignore


def make_sc_flux_to_staggered_numba(
    grid: CartesianGrid, mobility: Optional[Callable[[float], float]] = None
) -> Callable[..., np.ndarray]:
    """make function calculating the flux for a single component on a staggered grid

    Args:
        grid (:class:`pde.grids.cartesian.CartesianGrid`):
            The Cartesian grid on which the fields are defined
        mobility (callable, optional):
            The function that is used to evaluate the mobility. If omitted, the
            mobility is assumed to be 1.

    Returns:
        A function that can be applied to an array of values
    """
    assert isinstance(grid, CartesianGrid)
    dim = grid.dim

    if dim == 1:
        calc_flux = _make_sc_flux_to_staggered_numba_1d(grid, mobility)
    elif dim == 2:
        calc_flux = _make_sc_flux_to_staggered_numba_2d(grid, mobility)
    else:
        raise NotImplementedError(
            f"Numba flux operator not implemented for dimension {dim}"
        )

    return calc_flux


def _make_mc_flux_to_staggered_const_numba_1d(
    grid: CartesianGrid,
    num_comp: int,
    diffusivity: Optional[np.ndarray] = None,
) -> Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]:
    """make function for multicomponent flux on a 1d staggered grid with const. mobili.

    Args:
        grid (:class:`pde.grids.cartesian.CartesianGrid`):
            The Cartesian grid on which the fields are defined
        num_comp (int):
            The number of components
        diffusivity (:class:`~numpy.ndarray`, optional):
            The matrix that determines the (cross)-diffusivities for all component. If
            omitted it is replaced by a full matrix of ones.

    Returns:
        A function that can be applied to an array of values. This function takes three
        arguments: the chemical potentials `mu`, the phase fields `phi`, and an array to
        which the result is written.
    """
    assert isinstance(grid, CartesianGrid) and grid.dim == 1
    (dim_x,) = grid.shape
    (dx,) = grid.discretization
    vector_fields_shape = (num_comp, 2) + grid._shape_full

    if diffusivity is None:
        diff_arr = np.ones((num_comp, num_comp))
    else:
        diff_arr = diffusivity
    assert diff_arr.shape == (num_comp, num_comp)

    # use parallel processing for large enough arrays
    parallel = dim_x >= config["numba.multithreading_threshold"]

    @jit(parallel=parallel)
    def calc_mu_grad(mu: np.ndarray, mu_grad: np.ndarray):
        """get phi and negative gradient of mu for a single field on staggered grid"""
        for i in nb.prange(dim_x + 1):
            mu_grad[0, i] = -(mu[i + 1] - mu[i]) / dx

    @jit(parallel=parallel)
    def calc_flux(phi: np.ndarray, mu: np.ndarray, out: np.ndarray) -> np.ndarray:
        """calculate fluxes driven by gradients in `mu`"""
        # calculate the phis and the gradients of mu on the staggered grid
        mu_grads = np.zeros(vector_fields_shape)
        for n in range(num_comp):  # iterate over all components
            calc_mu_grad(mu[n], mu_grads[n])

        # determine the fluxes from the thermodynamic force on the flux
        for n in range(num_comp):  # iterate over all components
            out[n, ...] = 0  # make sure the initial values are zero
            for m in range(num_comp):  # iterate over all components
                if diff_arr[n, m] != 0:
                    # use diffusivities directly
                    out[n, 0, ...] += diff_arr[n, m] * mu_grads[m, 0, ...]

        return out

    return calc_flux  # type: ignore


def _make_mc_flux_to_staggered_scaled_diagonal_numba_1d(
    grid: CartesianGrid,
    num_comp: int,
    diffusivity: Optional[np.ndarray] = None,
) -> Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]:
    """make function for multicomponent flux on a 1d staggered grid with const. mobili.

    Args:
        grid (:class:`pde.grids.cartesian.CartesianGrid`):
            The Cartesian grid on which the fields are defined
        num_comp (int):
            The number of components
        diffusivity (:class:`~numpy.ndarray`, optional):
            The vector that determines the diffusivities for all component. If omitted
            it is replaced by a full vector of ones.

    Returns:
        A function that can be applied to an array of values. This function takes three
        arguments: the chemical potentials `mu`, the phase fields `phi`, and an array to
        which the result is written.
    """
    assert isinstance(grid, CartesianGrid) and grid.dim == 1
    (dim_x,) = grid.shape
    (dx,) = grid.discretization

    # create the diffusivity vector
    if diffusivity is None:
        diff_arr = np.ones(num_comp)
    else:
        diff_arr = np.asanyarray(diffusivity)
    if diff_arr.ndim == 2:
        diff_arr = np.diag(diff_arr)
    if diff_arr.ndim == num_comp + 1:
        diff_arr = diff_arr[:-1]
    assert diff_arr.shape == (num_comp,)

    # use parallel processing for large enough arrays
    parallel = dim_x >= config["numba.multithreading_threshold"]

    @jit(parallel=parallel)
    def calc_flux(phi: np.ndarray, mu: np.ndarray, out: np.ndarray) -> np.ndarray:
        """calculate fluxes driven by gradients in `mu`"""
        # determine the fluxes from the thermodynamic force on the flux
        for n in range(num_comp):  # iterate over all components
            for i in nb.prange(dim_x + 1):
                phi_staggered = (phi[n, i + 1] + phi[n, i]) / 2
                mu_grad = -(mu[n, i + 1] - mu[n, i]) / dx
                out[n, 0, i] = diff_arr[n] * phi_staggered * mu_grad

        return out

    return calc_flux  # type: ignore


def _make_mc_flux_to_staggered_scaled_correct_numba_1d(
    grid: CartesianGrid,
    num_comp: int,
    diffusivity: Optional[np.ndarray] = None,
) -> Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]:
    """make function for multicomponent flux on a 1d staggered grid with scaled mob.

    Args:
        grid (:class:`pde.grids.cartesian.CartesianGrid`):
            The Cartesian grid on which the fields are defined
        num_comp (int):
            The number of components
        diffusivity (:class:`~numpy.ndarray`, optional):
            The matrix that determines the (cross)-diffusivities for all component. If
            omitted it is replaced by a full matrix of ones.

    Returns:
        A function that can be applied to an array of values. This function takes three
        arguments: the chemical potentials `mu`, the phase fields `phi`, and an array to
        which the result is written.
    """
    assert isinstance(grid, CartesianGrid) and grid.dim == 1
    (dim_x,) = grid.shape
    (dx,) = grid.discretization
    vector_fields_shape = (num_comp, 1) + grid._shape_full
    denominator_shape = grid._shape_full

    if diffusivity is None:
        diff_arr = np.ones(num_comp + 1)
    else:
        diff_arr = diffusivity
    assert diff_arr.shape == (num_comp + 1,)

    # use parallel processing for large enough arrays
    parallel = dim_x >= config["numba.multithreading_threshold"]

    @jit(parallel=parallel)
    def calc_phi_mu_grad_phi(
        phi: np.ndarray, mu: np.ndarray, phi_staggered: np.ndarray, mu_grad: np.ndarray
    ):
        """get phi and negative gradient of mu for a single field on staggered grid"""
        for i in nb.prange(dim_x + 1):
            phi_staggered[0, i] = (phi[i + 1] + phi[i]) / 2
            mu_grad[0, i] = -(mu[i + 1] - mu[i]) / dx

    @jit(parallel=parallel)
    def calc_flux(phi: np.ndarray, mu: np.ndarray, out: np.ndarray) -> np.ndarray:
        """calculate fluxes driven by gradients in `mu`"""
        # calculate the phis and the gradients of mu on the staggered grid
        phis_staggered = np.zeros(vector_fields_shape)
        mu_grads = np.zeros(vector_fields_shape)
        denominator = np.zeros(denominator_shape)
        for n in range(num_comp):  # iterate over all components
            calc_phi_mu_grad_phi(phi[n], mu[n], phis_staggered[n], mu_grads[n])
            denominator += diff_arr[n] * phis_staggered[n, 0]
        denominator += diff_arr[num_comp] * (1 - phis_staggered[:, 0].sum(axis=0))

        # determine the fluxes from the thermodynamic force on the flux
        for n in range(num_comp):  # iterate over all components
            out[n, ...] = 0  # make sure the initial values are zero
            for m in range(num_comp):  # iterate over all components
                # scale diffusivities with concentrations
                mob = diff_arr[n] * phis_staggered[n, 0, ...]
                mob *= diff_arr[m] * phis_staggered[m, 0, ...]
                mob /= -denominator
                if n == m:
                    mob += diff_arr[n] * phis_staggered[n, 0, ...]
                out[n, 0, ...] += mob * mu_grads[m][0]

        return out

    return calc_flux  # type: ignore


def _make_mc_flux_to_staggered_const_numba_2d(
    grid: CartesianGrid,
    num_comp: int,
    diffusivity: Optional[np.ndarray] = None,
) -> Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]:
    """make function for multicomponent flux on a 1d staggered grid with const. mobili.

    Args:
        grid (:class:`pde.grids.cartesian.CartesianGrid`):
            The Cartesian grid on which the fields are defined
        num_comp (int):
            The number of components
        diffusivity (:class:`~numpy.ndarray`, optional):
            The matrix that determines the (cross)-diffusivities for all component. If
            omitted it is replaced by a full matrix of ones.

    Returns:
        A function that can be applied to an array of values. This function takes three
        arguments: the chemical potentials `mu`, the phase fields `phi`, and an array to
        which the result is written.
    """
    assert isinstance(grid, CartesianGrid) and grid.dim == 2
    dim_x, dim_y = grid.shape
    dx, dy = grid.discretization
    vector_fields_shape = (num_comp, 2) + grid._shape_full

    if diffusivity is None:
        diff_arr = np.ones((num_comp, num_comp))
    else:
        diff_arr = diffusivity
    assert diff_arr.shape == (num_comp, num_comp)

    # use parallel processing for large enough arrays
    parallel = dim_x * dim_y >= config["numba.multithreading_threshold"]

    @jit(parallel=parallel)
    def calc_mu_grad(mu: np.ndarray, mu_grad: np.ndarray):
        """phi and the negative gradient of mu for a single field on staggered grid"""
        for i in nb.prange(dim_x + 1):
            for j in range(1, dim_y + 1):
                mu_grad[0, i, j] = -(mu[i + 1, j] - mu[i, j]) / dx

        for i in nb.prange(1, dim_x + 1):
            for j in range(dim_y + 1):
                mu_grad[1, i, j] = -(mu[i, j + 1] - mu[i, j]) / dy

    @jit(parallel=parallel)
    def calc_flux(phi: np.ndarray, mu: np.ndarray, out: np.ndarray) -> np.ndarray:
        """calculate fluxes driven by gradients in `mu`"""
        # calculate the phis and the gradients of mu on the staggered grid
        mu_grads = np.zeros(vector_fields_shape)
        for n in range(num_comp):  # iterate over all components
            calc_mu_grad(mu[n], mu_grads[n])

        # determine the fluxes from the thermodynamic force on the flux
        for n in range(num_comp):  # iterate over all components
            out[n, ...] = 0  # make sure the initial values are zero
            for m in range(num_comp):  # iterate over all components
                if diff_arr[n, m] != 0:
                    for i in range(2):  # iterate over all dimension
                        # use diffusivities directly
                        out[n, i, ...] += diff_arr[n, m] * mu_grads[m, i, ...]

        return out

    return calc_flux  # type: ignore


def _make_mc_flux_to_staggered_scaled_diagonal_numba_2d(
    grid: CartesianGrid,
    num_comp: int,
    diffusivity: Optional[np.ndarray] = None,
) -> Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]:
    """make function for multicomponent flux on a 1d staggered grid with const. mobili.

    Args:
        grid (:class:`pde.grids.cartesian.CartesianGrid`):
            The Cartesian grid on which the fields are defined
        num_comp (int):
            The number of components
        diffusivity (:class:`~numpy.ndarray`, optional):
            The vector that determines the diffusivities for all component. If omitted
            it is replaced by a full vector of ones.

    Returns:
        A function that can be applied to an array of values. This function takes three
        arguments: the chemical potentials `mu`, the phase fields `phi`, and an array to
        which the result is written.
    """
    assert isinstance(grid, CartesianGrid) and grid.dim == 2
    dim_x, dim_y = grid.shape
    dx, dy = grid.discretization

    # create the diffusivity vector
    if diffusivity is None:
        diff_arr = np.ones(num_comp)
    else:
        diff_arr = np.asanyarray(diffusivity)
    if diff_arr.ndim == 2:
        diff_arr = np.diag(diff_arr)
    if len(diff_arr) == num_comp + 1:
        diff_arr = diff_arr[:-1]
    assert diff_arr.shape == (num_comp,)

    # use parallel processing for large enough arrays
    parallel = dim_x * dim_y >= config["numba.multithreading_threshold"]

    @jit(parallel=parallel)
    def calc_flux(phi: np.ndarray, mu: np.ndarray, out: np.ndarray) -> np.ndarray:
        """calculate fluxes driven by gradients in `mu`"""
        # determine the fluxes from the thermodynamic force on the flux
        for n in range(num_comp):  # iterate over all components
            # first space direction
            for i in nb.prange(dim_x + 1):
                for j in range(1, dim_y + 1):
                    phi_staggered = (phi[n, i + 1, j] + phi[n, i, j]) / 2
                    mu_grad_comp = -(mu[n, i + 1, j] - mu[n, i, j]) / dx
                    out[n, 0, i, j] = diff_arr[n] * phi_staggered * mu_grad_comp

            # second space direction
            for i in nb.prange(1, dim_x + 1):
                for j in range(dim_y + 1):
                    phi_staggered = (phi[n, i, j + 1] + phi[n, i, j]) / 2
                    mu_grad_comp = -(mu[n, i, j + 1] - mu[n, i, j]) / dy
                    out[n, 1, i, j] = diff_arr[n] * phi_staggered * mu_grad_comp

        return out

    return calc_flux  # type: ignore


def _make_mc_flux_to_staggered_scaled_correct_numba_2d(
    grid: CartesianGrid,
    num_comp: int,
    diffusivity: Optional[np.ndarray] = None,
) -> Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]:
    """make function for multicomponent flux on a 1d staggered grid with scaled mob.

    Args:
        grid (:class:`pde.grids.cartesian.CartesianGrid`):
            The Cartesian grid on which the fields are defined
        num_comp (int):
            The number of components
        diffusivity (:class:`~numpy.ndarray`, optional):
            The matrix that determines the (cross)-diffusivities for all component. If
            omitted it is replaced by a full matrix of ones.

    Returns:
        A function that can be applied to an array of values. This function takes three
        arguments: the chemical potentials `mu`, the phase fields `phi`, and an array to
        which the result is written.
    """
    assert isinstance(grid, CartesianGrid) and grid.dim == 2
    dim_x, dim_y = grid.shape
    dx, dy = grid.discretization
    vector_fields_shape = (num_comp, 2) + grid._shape_full
    denominator_shape = (2,) + grid._shape_full
    phi_shape = grid._shape_full

    if diffusivity is None:
        diff_arr = np.ones(num_comp + 1)
    else:
        diff_arr = diffusivity
    assert diff_arr.shape == (num_comp + 1,)

    # use parallel processing for large enough arrays
    parallel = dim_x * dim_y >= config["numba.multithreading_threshold"]

    @jit(parallel=parallel)
    def calc_phi_mu_grad_phi(
        phi: np.ndarray, mu: np.ndarray, phi_staggered: np.ndarray, mu_grad: np.ndarray
    ):
        """phi and the negative gradient of mu for a single field on staggered grid"""
        for i in nb.prange(dim_x + 1):
            for j in range(1, dim_y + 1):
                phi_staggered[0, i, j] = (phi[i + 1, j] + phi[i, j]) / 2
                mu_grad[0, i, j] = -(mu[i + 1, j] - mu[i, j]) / dx

        for i in nb.prange(1, dim_x + 1):
            for j in range(dim_y + 1):
                phi_staggered[1, i, j] = (phi[i, j + 1] + phi[i, j]) / 2
                mu_grad[1, i, j] = -(mu[i, j + 1] - mu[i, j]) / dy

    @jit(parallel=parallel)
    def calc_flux(phi: np.ndarray, mu: np.ndarray, out: np.ndarray) -> np.ndarray:
        """calculate fluxes driven by gradients in `mu`"""
        # calculate the phis and the gradients of mu on the staggered grid
        phis_staggered = np.ones(vector_fields_shape)
        mu_grads = np.zeros(vector_fields_shape)
        for n in range(num_comp):  # iterate over all components
            calc_phi_mu_grad_phi(phi[n], mu[n], phis_staggered[n], mu_grads[n])

        # calculate denominator in mobility
        denominator = np.zeros(denominator_shape)
        phi_last = np.ones(phi_shape)
        for i in range(2):
            phi_last[:] = 1
            for n in range(num_comp):
                denominator[i] += diff_arr[n] * phis_staggered[n, i, ...]
                phi_last -= phis_staggered[n, i, ...]
            denominator[i] += diff_arr[num_comp] * phi_last

        # determine the fluxes from the thermodynamic force on the flux
        for n in range(num_comp):  # iterate over all components
            out[n, ...] = 0  # make sure the initial values are zero
            for m in range(num_comp):  # iterate over all components
                for i in range(2):  # iterate over all dimension
                    # scale diffusivities with concentrations
                    mob = diff_arr[n] * phis_staggered[n, i, ...]
                    mob *= diff_arr[m] * phis_staggered[m, i, ...]
                    mob /= -denominator[i]
                    if n == m:
                        mob += diff_arr[n] * phis_staggered[n, i, ...]
                    out[n, i, ...] += mob * mu_grads[m, i, ...]

        return out

    return calc_flux  # type: ignore


def make_mc_flux_to_staggered_numba(
    grid: CartesianGrid,
    num_comp: int,
    diffusivity: Optional[np.ndarray] = None,
    mobility_model: str = "const",
) -> Callable[..., np.ndarray]:
    """make function calculating the flux for multiple components on a staggered grid

    Args:
        grid (:class:`pde.grids.cartesian.CartesianGrid`):
            The Cartesian grid on which the fields are defined
        num_comp (int):
            The number of components
        diffusivity (:class:`~numpy.ndarray`, optional):
            The matrix that determines the (cross)-diffusivities for all component. If
            omitted it is replaced by a full matrix of ones.
        mobility_model (str):
            The mobility model to use

    Returns:
        A function that can be applied to an array of values
    """
    assert isinstance(grid, CartesianGrid)
    dim = grid.dim

    if dim == 1:
        if mobility_model == "scaled_correct":
            calc_flux = _make_mc_flux_to_staggered_scaled_correct_numba_1d(
                grid, num_comp=num_comp, diffusivity=diffusivity
            )
        elif mobility_model == "scaled_diagonal":
            calc_flux = _make_mc_flux_to_staggered_scaled_diagonal_numba_1d(
                grid, num_comp=num_comp, diffusivity=diffusivity
            )
        elif mobility_model == "const":
            calc_flux = _make_mc_flux_to_staggered_const_numba_1d(
                grid, num_comp=num_comp, diffusivity=diffusivity
            )
        else:
            raise RuntimeError(f"Unsupported mobility model `{mobility_model}`")

    elif dim == 2:
        if mobility_model == "scaled_correct":
            calc_flux = _make_mc_flux_to_staggered_scaled_correct_numba_2d(
                grid, num_comp=num_comp, diffusivity=diffusivity
            )
        elif mobility_model == "scaled_diagonal":
            calc_flux = _make_mc_flux_to_staggered_scaled_diagonal_numba_2d(
                grid, num_comp=num_comp, diffusivity=diffusivity
            )
        elif mobility_model == "const":
            calc_flux = _make_mc_flux_to_staggered_const_numba_2d(
                grid, num_comp=num_comp, diffusivity=diffusivity
            )
        else:
            raise RuntimeError(f"Unsupported mobility model `{mobility_model}`")

    else:
        raise NotImplementedError(
            f"Numba flux operator not implemented for dimension {dim}"
        )

    return calc_flux


def set_flux_bcs(flux_data: List[np.ndarray], bcs: Boundaries) -> None:
    """set the appropriate boundary values on the staggered grid

    Args:
        flux_data (list):
            The discretized fluxes on the staggered grid
        bcs (:class:`~pde.grids.boundaries.axes.Boundaries`):
            The boundary conditions on the original grid
    """
    # TODO: instead extend boundary base such that it can also return the value on the
    # boundary!

    if bcs.grid.num_axes == 1:
        # apply boundary conditions on all sides
        bc = bcs[0]  # only axis
        if bc.periodic:
            return  # nothing to do, since the phi field was periodic already

        for bc_side in bc:
            if isinstance(bc_side, NormalDirichletBC):
                # value of the flux is imposed
                if bc_side.upper:
                    flux_data[0][-1] = bc_side.value
                else:
                    # need to invert because the normal vector points left
                    flux_data[0][0] = -bc_side.value

            elif isinstance(bc_side, DirichletBC):
                # value of the flux is imposed
                if bc_side.upper:
                    flux_data[0][-1] = bc_side.value[..., 0]
                else:
                    # need to invert because the normal vector points left
                    flux_data[0][0] = -bc_side.value[..., 0]

            else:
                raise NotImplementedError("Require Dirichlet boundary conditions")

    else:
        # apply boundary conditions on all sides
        for axis, bc in enumerate(bcs):
            if bc.periodic:
                continue  # nothing to do, since the phi field was periodic already

            for bc_side in bc:
                # prepare the array of slices to index bcs
                idx_write = [slice(None)] * bcs.grid.dim
                idx_write[axis] = -1 if bc_side.upper else 0  # type: ignore

                if isinstance(bc_side, NormalDirichletBC):
                    # value of the flux is imposed
                    if bc_side.upper:
                        flux_data[axis][tuple(idx_write)] = bc_side.value
                    else:
                        # need to invert because the normal vector points left
                        flux_data[axis][tuple(idx_write)] = -bc_side.value

                elif isinstance(bc_side, DirichletBC):
                    # value of the flux is imposed
                    if bc_side.upper:
                        flux_data[axis][tuple(idx_write)] = bc_side.value[..., axis]
                    else:
                        # need to invert because the normal vector points left
                        flux_data[axis][tuple(idx_write)] = -bc_side.value[..., axis]

                else:
                    raise NotImplementedError("Require Dirichlet boundary conditions")


def make_flux_bcs_setter(bcs: Boundaries) -> Callable:
    """set the appropriate boundary values on the staggered grid

    Args:
        bcs (:class:`~pde.grids.boundaries.axes.Boundaries`):
            The boundary conditions on the original grid
    """
    grid = bcs.grid
    assert isinstance(grid, CartesianGrid)
    periodic = tuple(grid.periodic)
    size = tuple(grid.shape)

    def get_bc_value(bc_side):
        """helper function extracting the value of the flux"""
        if bc_side.periodic:
            # return arbitrary number for periodic conditions. These are not used, but
            # it is important to set the right values, so numba can compile the function
            if grid.num_axes == 1:
                return np.nan
            else:
                return np.full(grid.num_axes, np.nan)

        elif isinstance(bc_side, NormalDirichletBC):
            # value of the flux is imposed
            value = bc_side.value if bc_side.upper else -bc_side.value
            if np.array(value).size == 1:
                value = float(value)  # work-around for issue #4458 in numba
            return value

        elif isinstance(bc_side, DirichletBC):
            # value of the flux is imposed
            if bc_side.upper:
                value = bc_side.value[..., bc_side.axis]
            else:
                value = -bc_side.value[..., bc_side.axis]
            if np.array(value).size == 1:
                value = float(value)  # work-around for issue #4458 in numba
            return value
        else:
            raise NotImplementedError("Require Dirichlet boundary conditions")

    bc_values = tuple(tuple(get_bc_value(bc_side) for bc_side in bc) for bc in bcs)

    if grid.num_axes == 1:
        # single axes -> bc_values are float values

        @jit
        def set_flux_bcs(flux_data: np.ndarray) -> None:
            # apply boundary conditions along x-axis
            if not periodic[0]:
                flux_data[0, 0] = bc_values[0][0]
                flux_data[0, size[0]] = bc_values[0][1]

    elif grid.num_axes == 2:
        # two axes -> bc_values are arrays

        @jit
        def set_flux_bcs(flux_data: np.ndarray) -> None:
            # apply boundary conditions along x-axis
            if not periodic[0]:
                flux_data[0, 0, 1:-1] = bc_values[0][0]
                flux_data[0, size[0], 1:-1] = bc_values[0][1]

            # apply boundary conditions along y-axis
            if not periodic[1]:
                flux_data[1, 1:-1, 0] = bc_values[1][0]
                flux_data[1, 1:-1, size[1]] = bc_values[1][1]

    else:
        raise NotImplementedError

    return set_flux_bcs  # type: ignore


def make_divergence_from_staggered_scipy(
    grid: CartesianGrid,
) -> Callable[..., np.ndarray]:
    """make a negative divergence operator from a staggered to a Cartesian grid

    Args:
        grid (:class:`pde.grids.cartesian.CartesianGrid`):
            The Cartesian grid on which the fields are defined

    Returns:
        A function that can be applied to an array of values
    """
    # check whether the supplied boundaries are supported
    dx = grid.discretization
    assert isinstance(grid, CartesianGrid)

    def divergence(
        arr: Sequence[np.ndarray], out: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """apply divergence operator to array `arr`"""
        # need to initialize with zeros since data is added later
        if out is None:
            out = np.zeros(grid.shape)
        else:
            assert out.shape == grid.shape
            out[:] = 0

        for axis in range(grid.dim):  # iterate over all dimensions
            out -= np.diff(arr[axis], axis=axis) / dx[axis]
        return out  # type: ignore

    return divergence


def _make_divergence_from_staggered_numba_1d(grid: CartesianGrid) -> OperatorType:
    """make a negative 1d divergence operator from a staggered to a Cartesian grid

    Args:
        grid (:class:`pde.grids.cartesian.CartesianGrid`):
            The Cartesian grid on which the fields are defined

    Returns:
        A function that can be applied to an array of values
    """
    # check whether the supplied boundaries are supported
    assert isinstance(grid, CartesianGrid) and grid.dim == 1
    (dim_x,) = grid.shape
    (scale_x,) = -1 / grid.discretization

    # use parallel processing for large enough arrays
    parallel = dim_x >= config["numba.multithreading_threshold"]

    @jit(parallel=parallel)
    def divergence(
        arr: np.ndarray, out: Optional[np.ndarray] = None, args=None
    ) -> np.ndarray:
        """apply divergence operator to array `arr`"""
        if out is None:
            out = np.empty((dim_x,), dtype=arr.dtype)
        for i in nb.prange(dim_x):  # iterate over x
            out[i] = (arr[0, i + 1] - arr[0, i]) * scale_x

        return out

    return divergence  # type: ignore


def _make_divergence_from_staggered_numba_2d(grid: CartesianGrid) -> OperatorType:
    """make a negative 2d divergence operator from a staggered to a Cartesian grid

    Args:
        grid (:class:`pde.grids.cartesian.CartesianGrid`):
            The Cartesian grid on which the fields are defined

    Returns:
        A function that can be applied to an array of values
    """
    # check whether the supplied boundaries are supported
    assert isinstance(grid, CartesianGrid) and grid.dim == 2
    dim_x, dim_y = grid.shape
    scale_x, scale_y = -1 / grid.discretization

    # use parallel processing for large enough arrays
    parallel = dim_x * dim_y >= config["numba.multithreading_threshold"]

    @jit(parallel=parallel)
    def divergence(
        arr: np.ndarray, out: Optional[np.ndarray] = None, args=None
    ) -> np.ndarray:
        """apply divergence operator to array `arr`"""
        if out is None:
            out = np.empty((dim_x, dim_y), dtype=arr.dtype)
        for i in nb.prange(dim_x):  # iterate over x
            for j in range(dim_y):  # iterate over y
                d_x = (arr[0, i + 1, j + 1] - arr[0, i, j + 1]) * scale_x
                d_y = (arr[1, i + 1, j + 1] - arr[1, i + 1, j]) * scale_y
                out[i, j] = d_x + d_y

        return out

    return divergence  # type: ignore


def make_divergence_from_staggered_numba(grid: CartesianGrid) -> OperatorType:
    """make a negative divergence operator from a staggered to a Cartesian grid

    Args:
        grid (:class:`pde.grids.cartesian.CartesianGrid`):
            The Cartesian grid on which the fields are defined

    Returns:
        A function that can be applied to an array of values
    """
    if grid.dim == 1:
        divergence = _make_divergence_from_staggered_numba_1d(grid)
    elif grid.dim == 2:
        divergence = _make_divergence_from_staggered_numba_2d(grid)
    else:
        raise NotImplementedError(
            f"Numba gradient operator not implemented for dimension {grid.dim}"
        )

    return divergence
