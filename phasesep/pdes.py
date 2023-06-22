r"""
Defines a class representing the Cahn-Hilliard equation for multiple species.

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de> 
"""

from typing import Any, Callable, Dict, Optional, Union

import numpy as np

from pde.fields import FieldCollection
from pde.grids.cartesian import CartesianGrid
from pde.pdes import PDEBase
from pde.tools.numba import jit
from pde.tools.parameters import Parameter, Parameterized, ParameterListType

from .reactions import Reaction, Reactions, ReactionsBase


def is_diagonal_matrix(arr: np.ndarray) -> bool:
    """check whether the given matrix is a diagonal matrix

    Scalars are interpreted as diagonal matrices

    Args:
        arr (:class:`~numpy.ndarray`): The matrix to check

    Returns:
        bool: Whether the non-diagonal terms vanish
    """
    if np.isscalar(arr) or arr.ndim == 0:
        return True
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        return False  # not a square matrix
    m = arr.shape[0]
    p, q = arr.strides
    # extract non-diagonal elements; see https://stackoverflow.com/a/64127402/932593
    non_diag = np.lib.stride_tricks.as_strided(arr[:, 1:], (m - 1, m), (p + q, q))
    return np.allclose(non_diag, 0)


class CahnHilliardMultiplePDE(PDEBase, Parameterized):
    r"""(extended) incompressible Cahn-Hilliard equation for many components"""

    parameters_default: ParameterListType = [
        Parameter(
            "free_energy",
            None,
            object,
            "Defines the expression for the local part of the free energy density. "
            "Currently, this needs to be an instance of "
            ":class:`~phasesep.free_energies.flory_huggins.FloryHugginsNComponents`.",
        ),
        Parameter(
            "mobility",
            np.array(1.0),
            np.array,
            "The mobility matrix define how fast the fields relax by diffusive fluxes. ",
        ),
        Parameter(
            "kappa",
            np.array(0.0),
            np.array,
            "Pre-factor :math:`kappa_{ij}` for the gradient term in the free energy, "
            "which gives rise to surface tension effects. Generally, `kappa` can be a "
            "single number (all interactions have the same prefactor) or a (symmetric) "
            "matrix of dimensions `num_comp` (specifying different interactions for "
            "each gradient combination).",
        ),
        Parameter(
            "reactions",
            None,
            object,
            r"Defines the reaction rates :math:`s_i(\phi, \mu, t)` describing how the "
            "concentration of each component :math:`i` changes locally. This parameter "
            "can either be a callable function that directly calculates the rates. "
            "Alternatively, multiple reactions can be specified by supplying an "
            "instance of :class:`phasesep.reactions.Reactions`. If an instance of "
            ":class:`phasesep.reactions.Reaction` is supplied it is used as the only "
            "reaction in the system. The special value `None` corresponds to no "
            "reactions.",
        ),
    ]

    _kappa: np.ndarray
    _mobilities: np.ndarray

    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        r"""
        Args:
            parameters (dict):
                A dictionary of parameters to change the defaults. The allowed
                parameters can be obtained from
                :meth:`~Parameterized.get_parameters` or displayed by calling
                :meth:`~Parameterized.show_parameters`.
        """

        Parameterized.__init__(self, parameters)
        PDEBase.__init__(self, noise=0)

        # define the functions for evaluating the right hand side
        if self.parameters["free_energy"] is None:
            raise ValueError(
                "No free energy specified. Define one using the classes `FreeEnergy` "
                "or `FloryHugginsNComponents` and supply it using the `free_energy` "
                "parameter."
            )
        else:
            self.f_local = self.parameters["free_energy"]
        self.dim = self.f_local.dim
        self.num_comp = self.dim + 1

        # get gradient prefactor matrix
        self.kappa = self.parameters["kappa"]

        # set the mobilities
        self.mobilities = self.parameters["mobility"]

        # check the data for the chemical reactions
        param_reactions = self.parameters["reactions"]
        if isinstance(param_reactions, ReactionsBase):
            # reactions are given in the right format
            self.reactions: ReactionsBase = param_reactions

        elif isinstance(param_reactions, Reaction):
            # a single reaction needs to be wrapped in `Reactions`
            self.reactions = Reactions(self.num_comp, [param_reactions])

        else:
            # assume that a list of individual reactions is given
            self.reactions = Reactions(self.num_comp, param_reactions)
        self.explicit_time_dependence = self.reactions.explicit_time_dependence

    @property
    def kappa(self) -> np.ndarray:
        """numpy.ndarray: Pre-factors for the gradient terms"""
        return self._kappa

    @kappa.setter
    def kappa(self, value: Union[float, np.ndarray]):
        """set the kappa matrix ensuring the correct symmetries"""
        value_arr = np.broadcast_to(value, (self.num_comp, self.num_comp))

        kappa = 0.5 * (value_arr + value_arr.T)  # symmetrize array
        self._kappa = kappa

        # calculate reduced kappa matrix to be used in chemical potential
        self._kappa_reduced = np.empty((self.dim, self.dim))
        n = self.num_comp - 1  # index of the component to be removed
        for i in range(self.dim):
            for j in range(self.dim):
                # Add surface tension of the solvent
                self._kappa_reduced[i, j] = (
                    kappa[i, j] + kappa[n, n] - kappa[i, n] - kappa[n, j]
                )

    @property
    def mobilities(self) -> np.ndarray:
        """numpy.ndarray: The mobilities of the diffusive fluxes"""
        return self._mobilities

    @mobilities.setter
    def mobilities(self, mobilities: Union[float, np.ndarray]):
        """sets the mobility matrix

        Args:
            mobilities (float or :class:`~numpy.ndarray`):
                The mobilities are generally given by a matrix, where the off-diagonal
                elements correspond to cross-diffusion. When only a 1d-array is
                specified it is assumed that this array defines the diagonal elements
                while the off-diagonal elements are zero.
        """
        # the interpretation of the mobilities depends on the mobility model
        mobilities_arr = np.asarray(mobilities)

        # convert mobilities to correct rank
        if mobilities_arr.ndim == 0:
            # interpret scalar values as a uniform diagonal matrix
            mobilities_arr = mobilities_arr * np.eye(self.num_comp)
        elif mobilities_arr.ndim == 1:
            # interpret 1d data as a diagonal matrix
            mobilities_arr = np.diag(mobilities_arr)
        elif mobilities_arr.ndim == 2:
            # check that mobility matrix is diagonal
            if not is_diagonal_matrix(mobilities_arr):
                mobilities_arr = np.diag(np.diagonal(mobilities_arr))
        elif mobilities_arr.ndim > 2:
            raise ValueError("Mobility matrix must have at most 2 dimensions")

        # symmetrize the matrix
        mobilities_unsym = mobilities_arr
        mobilities_arr = 0.5 * (mobilities_arr + mobilities_arr.T)
        if not np.allclose(mobilities_arr, mobilities_unsym):
            self._logger.warning("Mobility matrix was symmetrized")

        # store the original mobilities
        self._mobilities = mobilities_arr

        # store the diagonal entries only of the retained components
        self._mobilities_reduced = np.diagonal(mobilities_arr)[: self.dim]
        assert self._mobilities_reduced.shape == (self.dim,)

    @property
    def info(self) -> Dict[str, Any]:
        """dict: information about the PDE"""
        return self.parameters.copy()

    def _check_field_type(self, state: FieldCollection):
        """checks whether the supplied field is consistent with this class

        Args:
            state (:class:`~pde.fields.FieldCollection`):
                The fields describing the concentration distributions
        """
        if not isinstance(state, FieldCollection):
            raise ValueError("Supplied state must be a FieldCollection")
        if len(state) != self.dim:
            raise ValueError(f"Expected state with {self.dim} fields")

    def _impose_mu_bcs(
        self, mu: FieldCollection, phi: FieldCollection, t: float = 0
    ) -> None:
        """impose boundary conditions on the chemical potential

        This function simply imposes boundary conditions on the field `mu`, which may
        generally depend on the concentration variable `phi` and time `t`. Subclasses
        can overwrite this method to impose more complex boundary conditions, e.g.,
        reaction fluxes that depend on the multiple fields.

        Args:
            mu (:class:`~pde.fields.FieldCollection`):
                The fields describing the chemical potentials
            phi (:class:`~pde.fields.FieldCollection`):
                The fields describing the concentration distributions
            t (float):
                The current time point
        """
        for i in range(self.dim):
            mu[i].set_ghost_cells("auto_periodic_neumann", args={"phi": phi[i], "t": t})

    def chemical_potential(
        self, state: FieldCollection, t: float = 0, *, kappa_term: bool = True
    ) -> FieldCollection:
        """return the (exchange) chemical potentials for a given state

        This method also imposes boundary conditions on the state and the returned
        chemical potential, i.e., the ghost cells of the respective fields will be set
        using the appropriate boundary conditions.

        Args:
            state (:class:`~pde.fields.FieldCollection`):
                The fields describing the concentration distributions
            t (float):
                The current time point

        Returns:
            :class:`~pde.fields.FieldCollection`:
            The (exchange) chemical potentials associated with the fields
        """
        self._check_field_type(state)

        # evaluate chemical potential
        mu = state.copy(label="chemical potential")
        mu.data[:] = self.f_local.chemical_potential(state.data, t=t)
        for i in range(self.dim):
            state[i].set_ghost_cells(
                "auto_periodic_neumann", args={"t": t}
            )  # impose BCs on phi
            if kappa_term:
                phi_i_lap = state[i].laplace(bc=None).data  # type: ignore
                for j in range(self.dim):
                    mu.data[j] -= self._kappa_reduced[j, i] * phi_i_lap

        self._impose_mu_bcs(mu, state, t)
        return mu

    def reaction_rates(
        self, state: FieldCollection, mu: Optional[FieldCollection] = None, t: float = 0
    ) -> FieldCollection:
        """return the rate of change for each field due to chemical reactions

        Args:
            state (:class:`~pde.fields.FieldCollection`):
                The fields describing the concentration distributions
            mu (:class:`~pde.fields.FieldCollection`, optional):
                The chemical potentials corresponding to the concentration fields . If
                omitted, the chemical potentials are calculated based on `state`.
            t (float):
                The current point in time

        Returns:
            :class:`~pde.fields.FieldCollection`:
            The reaction rates for all components.
        """
        self._check_field_type(state)
        result = state.copy(label="reaction fluxes")

        if self.reactions.present:
            # calculate reaction rate
            if mu is None:
                mu = self.chemical_potential(state, t=t)

            result.data[:] = self.reactions.reaction_rates(state.data, mu.data, t)

        else:
            # no reaction is given
            result.data[:] = 0

        return result

    def evolution_rate(  # type: ignore
        self, state: FieldCollection, t: float = 0, *, kappa_term: bool = True
    ) -> FieldCollection:
        """evaluate the right hand side of the PDE

        Args:
            state (:class:`~pde.fields.FieldCollection`):
                The fields describing the concentration distributions
            t (float):
                The current time point

        Returns:
            :class:`~pde.fields.FieldCollection`:
            The evolution rate of each component
        """
        # evaluate chemical potential (and impose boundary condition on state
        mu = self.chemical_potential(state, t=t, kappa_term=kappa_term)

        # pick the correct implementation of the right hand side
        rhs = state.copy(label="evolution rate")

        # calculate the diffusive flux assuming constant mobilities
        assert isinstance(state.grid, CartesianGrid), "Need Cartesian grid"

        from .cartesian_staggered import (
            make_divergence_from_staggered_scipy,
            make_mc_flux_to_staggered_scipy,
            set_flux_bcs,
        )

        get_flux_staggered = make_mc_flux_to_staggered_scipy(
            state.grid,
            num_comp=len(state),
            diffusivity=self._mobilities_reduced,
            mobility_model="scaled_diagonal",
        )
        bcs = state.grid.get_boundary_conditions("auto_periodic_dirichlet", rank=1)
        div_staggered = make_divergence_from_staggered_scipy(state.grid)

        fluxes = get_flux_staggered(state._data_full, mu._data_full)
        for n, flux in enumerate(fluxes):
            set_flux_bcs(flux, bcs)
            div_staggered(flux, rhs[n].data)

        # add the chemical reaction
        if self.reactions.present:
            rhs += self.reactions.reaction_rates(state.data, mu=mu.data, t=t)

        return rhs

    def _make_impose_mu_bcs(
        self, state: FieldCollection
    ) -> Callable[[np.ndarray, np.ndarray, float], None]:
        """impose boundary conditions on the chemical potential

        The returned function simply imposes boundary conditions on the field `mu`,
        which may generally depend on the concentration variable `phi` and time `t`.
        Subclasses can overwrite this method to impose more complex boundary conditions,
        e.g., reaction fluxes that depend on the multiple fields.

        Args:
            state (:class:`~pde.fields.FieldCollection`):
                The fields describing the concentration distributions
        """
        num_comp = self.dim
        bc = state.grid.get_boundary_conditions("auto_periodic_neumann")
        set_ghost_cells_mu = bc.make_ghost_cell_setter()

        @jit
        def impose_mu_bcs(
            mu_full: np.ndarray, phi_full: np.ndarray, t: float = 0
        ) -> None:
            for n in range(num_comp):
                set_ghost_cells_mu(mu_full[n], args={"phi": phi_full[n], "t": t})

        return impose_mu_bcs  # type: ignore

    def _make_pde_rhs_numba_staggered(
        self, state: FieldCollection
    ) -> Callable[[np.ndarray, np.ndarray, float, np.ndarray], None]:
        """handle conservative part of the rhs of the PDE using a staggered grid

        Args:
            state (:class:`~pde.fields.FieldCollection`):
                An example for the state defining the grid and data types
        """
        assert isinstance(state.grid, CartesianGrid), "Need Cartesian grid"
        self._check_field_type(state)

        num_comp = self.dim
        vector_coll_full_shape = (num_comp, state.grid.dim) + state.grid._shape_full

        from .cartesian_staggered import (
            make_divergence_from_staggered_numba,
            make_flux_bcs_setter,
            make_mc_flux_to_staggered_numba,
        )

        get_flux_staggered = make_mc_flux_to_staggered_numba(
            state.grid,
            num_comp=num_comp,
            diffusivity=self._mobilities_reduced,
            mobility_model="scaled_diagonal",
        )
        bcs = state.grid.get_boundary_conditions("auto_periodic_dirichlet", rank=1)
        set_bcs = make_flux_bcs_setter(bcs)
        get_div_staggered = make_divergence_from_staggered_numba(state.grid)

        @jit
        def pde_rhs_conserved(
            phi_full: np.ndarray, mu_full: np.ndarray, t: float, rhs: np.ndarray
        ) -> None:
            """calculate the conservative part of the right hand side of PDE"""
            fluxes = np.empty(vector_coll_full_shape)  # temporary array
            get_flux_staggered(phi_full, mu_full, out=fluxes)
            for n in range(num_comp):
                set_bcs(fluxes[n])
                get_div_staggered(fluxes[n, ...], rhs[n, ...])

        return pde_rhs_conserved  # type: ignore

    def _make_pde_rhs_numba(  # type: ignore
        self, state: FieldCollection, *, kappa_term: bool = True
    ) -> Callable[[np.ndarray, float], np.ndarray]:
        """create a compiled function evaluating the right hand side of the PDE

        Args:
            state (:class:`~pde.fields.FieldCollection`):
                An example for the state defining the grid and data types

        Returns:
            A function with signature `(state_data, t)`, which can be called with an
            instance of :class:`~numpy.ndarray` of the state data and the time to
            obtained an instance of :class:`~numpy.ndarray` giving the evolution rate.
        """
        # check whether the state is reasonable
        self._check_field_type(state)

        # check whether the state is consistent with the free energy
        state_test = state.copy()
        if not np.allclose(state.data, state_test.data):
            self._logger.warning(
                "The initial state might violate constraints imposed by the free "
                "energy density. The resulting simulation might be invalid."
            )

        grid = state.grid
        num_comp = self.dim
        scalar_shape = grid.shape
        scalar_collection_shape = (self.dim,) + grid.shape
        scalar_collection_full_shape = (self.dim,) + grid._shape_full

        get_valid = grid._make_get_valid()
        set_valid = grid._make_set_valid()

        # prepare the functions to calculate the chemical potential
        kappa_reduced = self._kappa_reduced
        mu_local = self.f_local.make_chemical_potential(backend="numba")
        laplace = grid.make_operator_no_bc("laplace")

        # prepare functions imposing boundary conditions
        bc = grid.get_boundary_conditions("auto_periodic_neumann")
        set_ghost_cells_phi = bc.make_ghost_cell_setter()
        impose_mu_bcs = self._make_impose_mu_bcs(state)

        # determine whether the laplace operator needs to be calculated
        calc_laplace_phi = np.any(kappa_reduced != 0, axis=1) & kappa_term

        # prepare the function calculating the diffusive fluxes
        pde_rhs_conserved = self._make_pde_rhs_numba_staggered(state)

        # prepare the reaction term
        reactions_present = self.reactions.present
        apply_reaction = self.reactions.make_apply_reaction_rates_compiled()

        @jit
        def pde_rhs(phi: np.ndarray, t: float) -> np.ndarray:
            """calculate the right hand side of the PDE and return it"""
            # create temporary arrays
            phi_full = np.empty(scalar_collection_full_shape)
            mu_full = np.empty(scalar_collection_full_shape)
            phi_lap_n = np.empty(scalar_shape)
            rhs = np.zeros(scalar_collection_shape)  # initialize with zeros

            # determine local part of the chemical potential
            mu_local(phi, t, out=get_valid(mu_full))

            # add the non-local part of the chemical potential
            for n in range(num_comp):
                set_valid(phi_full[n], phi[n])  # copy state
                # set bcs on phi
                set_ghost_cells_phi(phi_full[n], args={"t": t})
                if calc_laplace_phi[n]:
                    laplace(phi_full[n], out=phi_lap_n)
                    for m in range(num_comp):
                        mu_valid = get_valid(mu_full[m])
                        mu_valid -= kappa_reduced[m, n] * phi_lap_n

            # impose boundary conditions on mu
            impose_mu_bcs(mu_full, phi_full, t)

            # apply the (conservative) diffusive fluxes
            pde_rhs_conserved(phi_full, mu_full, t, rhs)

            # add the chemical reaction
            if reactions_present:
                apply_reaction(get_valid(phi_full), get_valid(mu_full), t, rhs)  # type: ignore
            return rhs

        return pde_rhs  # type: ignore
