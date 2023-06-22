r"""
Defines classes handling chemical reactions.

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de> 
"""

from __future__ import annotations

import copy
import logging
from abc import ABCMeta, abstractmethod, abstractproperty
from typing import Any, Callable, Dict, Optional, Sequence

import numpy as np

from pde.tools.cache import cached_method
from pde.tools.expressions import ExpressionType, ScalarExpression
from pde.tools.numba import jit

# dictionary with common function arguments and their synonyms
ARGUMENT_SYNONYMS = {
    "state": ["state", "phi", "c", "φ"],
    "phi": ["phi", "state", "c", "φ"],
    "mu": ["mu", "µ"],
    "c": ["c", "phi", "state", "φ"],
    "t": ["t", "time"],
    "time": ["time", "t"],
    "rhs": ["rhs", "out"],
}


class ReactionFluxExpression(ScalarExpression):
    """class representing the expression of a reaction flux"""

    def __init__(
        self,
        expression: Optional[ExpressionType] = None,
        args: Sequence[str] = ("t",),
        allow_indexed: bool = False,
        consts: Optional[Dict[str, Any]] = None,
        user_funcs: Optional[Dict[str, Any]] = None,
    ):
        r"""
        Args:
            expression (str or float):
                The expression is None or a string that sympy can interpret.
            args (sequence of str):
                Additional dependencies of the reaction flux. Typical choices are the
                chemical potential :math:`\mu` and time :math:`t`.
            allow_indexed (bool):
                Whether to allow indexing of variables. If enabled, array
                variables are allowed to be indexed using square bracket
                notation.
            consts (dict, optional):
                A dictionary with user defined constants that can be used in the
                expression.
            user_funcs (dict, optional):
                A dictionary with user defined functions that can be used in the
                expression
        """
        signature = [ARGUMENT_SYNONYMS["c"]] + list(args)

        if expression is None:
            expression = 0.0

        super().__init__(
            expression,
            signature=signature,
            allow_indexed=allow_indexed,
            consts=consts,
            user_funcs=user_funcs,
        )

    @property
    def present(self) -> bool:
        """bool: whether reaction flux is present or not"""
        return not (self.constant and self.value == 0)


class Reaction:
    """class representing a single reaction between multiple components"""

    def __init__(
        self,
        stoichiometry: np.ndarray,
        reaction_flux: Optional[ExpressionType] = None,
        *,
        conservative: bool = True,
        weights: Optional[np.ndarray] = None,
        consts: Optional[Dict[str, Any]] = None,
    ):
        r"""
        Args:
            stoichiometry (:class:`~numpy.ndarray`):
                Determines how strongly each component is affected by the reaction flux
            reaction_flux (str or :class:`ReactionFluxExpression`):
                The expression defining the reaction flux
            conservative (bool):
                Whether the total mass needs to be conserved by this reaction. In this
                case, the stoichiometry is checked for consistency.
            weights (:class:`~numpy.ndarray`):
                Weights used for checking whether the stoichiometry is conservative.
                This can be interpreted as molecular masses to check mass conservation.
            consts (dict, optional):
                A dictionary with user defined constants that can be used in the
                expression `reaction_flux`.
        """
        self._logger = logging.getLogger(self.__class__.__name__)

        self.stoichiometry = np.array(stoichiometry, np.double)
        self.num_comp = len(self.stoichiometry)

        if weights is None:
            self.weights = np.ones(self.stoichiometry.shape)
        else:
            self.weights = np.broadcast_to(weights, self.stoichiometry.shape)

        if conservative:
            if not np.isclose(self.stoichiometry @ self.weights, 0):
                self._logger.warning("Reaction stoichiometries do not add to zero")

        # prepare reaction_flux
        self.reaction_flux = ReactionFluxExpression(
            reaction_flux, args=["mu", "t"], allow_indexed=True, consts=consts
        )

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.stoichiometry}, "
            f'"{self.reaction_flux.expression}")'
        )

    def to_string(self, components: Optional[Sequence[str]] = None) -> str:
        """represent the chemical reaction as a string

        Args:
            components (list, optional):
                Gives the name of the components. If `None`, the components are simply
                enumerated

        Returns:
            str: single line representing the chemical reaction
        """
        if components is None:
            components = [f"C_{i}" for i in range(self.num_comp)]

        reactants, products = [], []
        for comp, stoch in zip(components, self.stoichiometry):
            if stoch > 0:
                products.append(comp if stoch == 1 else f"{stoch} {comp}")
            elif stoch < 0:
                reactants.append(comp if stoch == -1 else f"{-stoch} {comp}")

        if not reactants:
            reactants = ["∅"]
        if not products:
            products = ["∅"]

        return f"{' + '.join(reactants)} <-> {' + '.join(products)}"

    @property
    def present(self) -> bool:
        """bool: whether the reaction is present or not"""
        return any(self.stoichiometry) and self.reaction_flux.present

    @property
    def explicit_time_dependence(self) -> bool:
        """bool: whether the reaction depends on time explicitly"""
        return self.reaction_flux.depends_on("t")

    @property
    def conservative(self) -> bool:
        """bool: whether the reaction conserves particle numbers"""
        return abs(self.stoichiometry @ self.weights) < 1e-10  # type: ignore

    def reaction_rate(
        self, state: np.ndarray, mu: np.ndarray, t: float = 0
    ) -> np.ndarray:
        """calculate the reaction rates for all components

        Args:
            state (:class:`~numpy.ndarray`):
                The concentrations of all components
            mu (:class:`~numpy.ndarray`):
                The chemical potentials of all components
            time (float):
                The current time

        Returns:
            :class:`~numpy.ndarray`: The reaction rates for all components
        """
        reaction_flux = self.reaction_flux(state, mu, t)
        return np.multiply.outer(self.stoichiometry, reaction_flux)

    @cached_method()
    def make_reaction_rates_compiled(
        self, skip_last_component: bool = False, with_out: bool = False
    ) -> Callable[..., np.ndarray]:
        """return compiled function evaluating reaction rates

        Args:
            skip_last_component (bool):
                Flag determining whether calculations are done for all components or
                whether the last component is skipped. This affects both the expected
                input shape and the returned output.
            with_out (bool):
                Whether the returned function takes an `out` argument to which results
                are then written.

        Returns:
            function: a compiled function returning the reaction rates
        """
        if skip_last_component:
            size = self.num_comp - 1
            stoichiometry = self.stoichiometry[:-1]
        else:
            size = self.num_comp
            stoichiometry = self.stoichiometry
        reaction_flux = self.reaction_flux.get_compiled()

        @jit
        def reaction_rates_out(
            state: np.ndarray, mu: np.ndarray, t: float, out: np.ndarray
        ) -> np.ndarray:
            """inner function calculating the reaction rate"""
            flux = reaction_flux(state, mu, t)
            for i in range(size):
                out[i] = stoichiometry[i] * flux
            return out

        if with_out:
            return reaction_rates_out  # type: ignore

        @jit
        def reaction_rates(state: np.ndarray, mu: np.ndarray, t: float) -> np.ndarray:
            """inner function calculating the reaction rate"""
            out = np.empty((size,) + state.shape[1:])
            reaction_rates_out(state, mu, t, out)
            return out

        return reaction_rates  # type: ignore


class ReactionsBase(metaclass=ABCMeta):
    """base class representing many reactions"""

    present: bool
    """indicates whether the reactions are present or not"""

    @abstractproperty
    def explicit_time_dependence(self) -> bool:
        """bool: indicates whether the reactions depend on time explicitly"""

    @abstractmethod
    def reaction_rates(
        self, state: np.ndarray, mu: np.ndarray, t: float = 0
    ) -> np.ndarray:
        """calculate the reaction rates for all components"""
        raise NotImplementedError

    @abstractmethod
    def make_apply_reaction_rates_compiled(
        self,
    ) -> Callable[[np.ndarray, np.ndarray, float, np.ndarray, Optional[float]], None]:
        """make compiled function evaluating reaction rates"""
        raise NotImplementedError

    @cached_method()
    def make_reaction_rates_compiled(
        self,
    ) -> Callable[[np.ndarray, np.ndarray, float], np.ndarray]:
        """return compiled function evaluating reaction rates

        Returns:
            function: a compiled function returning the reaction rates
        """
        apply_rates = self.make_apply_reaction_rates_compiled()

        @jit
        def reaction_rates(phi: np.ndarray, mu: np.ndarray, t: float) -> np.ndarray:
            """inner function calculating the reaction rates"""
            out = np.zeros_like(phi)
            apply_rates(phi, mu, t, out)  # type: ignore
            return out

        return reaction_rates  # type: ignore


class Reactions(ReactionsBase):
    """class representing multiple reactions"""

    def __init__(self, num_comp: int, reactions: Optional[Sequence[Reaction]] = None):
        """
        Args:
            num_comp (int): The number of components in the system
            reactions (list of :class:`Reaction`): All reactions
        """
        self.num_comp = int(num_comp)

        if reactions is None:
            self.reactions: Sequence[Reaction] = []
        elif isinstance(reactions, Reactions):
            assert self.num_comp == reactions.num_comp, "Mismatch in component count"
            self.reactions = copy.deepcopy(reactions.reactions)
        else:
            self.reactions = list(reactions)

        for r in self.reactions:
            if not isinstance(r, Reaction):
                raise TypeError(f"`{r}` is not of type `Reaction`")
            if r.num_comp != self.num_comp:
                raise ValueError(f"`{r}` does work for {self.num_comp} components")

    def __len__(self) -> int:
        """int: Number of reactions"""
        return len(self.reactions)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.reactions})"

    def __getitem__(self, index: int) -> Reaction:
        return self.reactions[index]

    def to_string(self, components: Optional[Sequence[str]] = None) -> str:
        """represent the chemical reactions as a string

        Args:
            components (list, optional):
                Gives the name of the components. If `None`, the components are simply
                enumerated.

        Returns:
            str: a line per chemical reaction
        """
        return "\n".join(r.to_string(components) for r in self.reactions)

    @property
    def present(self) -> bool:  # type: ignore
        """bool: whether any reactions are present"""
        return any(r.present for r in self.reactions)

    @property
    def explicit_time_dependence(self) -> bool:
        """bool: whether the reaction depends on time explicitly"""
        return any(r.explicit_time_dependence for r in self.reactions)

    @property
    def conservative(self) -> bool:
        """bool: whether all reactions are conservative"""
        return all(r.conservative for r in self.reactions)

    @property
    def stoichiometry(self) -> np.ndarray:
        """numpy.ndarray: matrix of stoichiometric factors"""
        if len(self) == 0:
            return np.zeros((0, self.num_comp))
        else:
            return np.vstack([r.stoichiometry for r in self.reactions])

    def reaction_fluxes(
        self, state: np.ndarray, mu: np.ndarray, t: float = 0
    ) -> np.ndarray:
        """calculate the reaction fluxes for all reactions

        Args:
            state (:class:`~numpy.ndarray`):
                The concentrations of all components
            mu (:class:`~numpy.ndarray`):
                The chemical potentials of all components
            time (float):
                The current time

        Returns:
            :class:`~numpy.ndarray`: The reaction flux for all reactions
        """
        state = np.asarray(state)
        mu = np.asarray(mu)
        result = np.empty((len(self),) + state.shape[1:])

        for i, reaction in enumerate(self.reactions):
            result[i, ...] = reaction.reaction_flux(state, mu, t)

        return result

    def reaction_rates(
        self, state: np.ndarray, mu: np.ndarray, t: float = 0
    ) -> np.ndarray:
        """calculate the reaction rates for all components

        Args:
            state (:class:`~numpy.ndarray`):
                The concentrations of all components
            mu (:class:`~numpy.ndarray`):
                The chemical potentials of all components
            time (float):
                The current time

        Returns:
            :class:`~numpy.ndarray`: The reaction rates for all components
        """
        state = np.asarray(state)
        mu = np.asarray(mu)
        result = np.zeros_like(state)

        if len(result) == self.num_comp:
            stoichiometry = self.stoichiometry
        elif len(result) == self.num_comp - 1:
            stoichiometry = self.stoichiometry[:, :-1]
        else:
            raise ValueError(
                f"State must have {self.num_comp - 1} or {self.num_comp} components"
            )

        for i, reaction in enumerate(self.reactions):
            reaction_flux = reaction.reaction_flux(state, mu, t)
            reaction_flux = np.broadcast_to(reaction_flux, state.shape[1:])
            result += np.multiply.outer(stoichiometry[i], reaction_flux)

        return result

    @cached_method()
    def make_apply_reaction_rates_compiled(
        self,
    ) -> Callable[[np.ndarray, np.ndarray, float, np.ndarray, Optional[float]], None]:
        """make compiled function evaluating reaction rates

        Note that only the reaction rates of the first `num_comp - 1` components are
        returned since the function is typically used in incompressible simulations.
        """
        # determine the number of components that are described
        num_comp = self.num_comp - 1
        stoichiometry = self.stoichiometry[:, :-1]

        # check whether reactions are actually present
        if not self.present:

            def noop(
                state: np.ndarray,
                mu: np.ndarray,
                t: float,
                res: np.ndarray,
                prefactor: float = 1.0,
            ) -> None:
                pass

            return jit(noop)  # type: ignore

        # else: apply the reaction fluxes
        reaction_fluxes = tuple(r.reaction_flux.get_compiled() for r in self.reactions)

        def chain(stoichiometries, reaction_fluxes, inner=None):
            """recursive helper function for applying a unknown number of
            reactions.

            This code is inspired by the example given here:
            https://github.com/numba/numba/issues/2542#issuecomment-329209271
            """
            # run through all reactions in reverse
            s_head, s_tail = stoichiometries[-1], stoichiometries[:-1]
            r_head, r_tail = reaction_fluxes[-1], reaction_fluxes[:-1]

            if inner is None:
                # the innermost function does not need to call a child
                @jit
                def wrap(
                    state: np.ndarray,
                    mu: np.ndarray,
                    t: float,
                    res: np.ndarray,
                    prefactor: float = 1.0,
                ) -> None:
                    flux = r_head(state, mu, t)
                    for j in range(num_comp):
                        res[j] += prefactor * s_head[j] * flux

            else:
                # all other functions need to call one deeper in the chain
                @jit
                def wrap(
                    state: np.ndarray,
                    mu: np.ndarray,
                    t: float,
                    res: np.ndarray,
                    prefactor: float = 1.0,
                ) -> None:
                    inner(state, mu, t, res, prefactor)

                    flux = r_head(state, mu, t)
                    for j in range(num_comp):
                        res[j] += prefactor * s_head[j] * flux

            if r_tail:
                return chain(s_tail, r_tail, inner=wrap)
            else:
                return wrap

        # compile the recursive chain
        apply_reaction_rates = chain(stoichiometry, reaction_fluxes)
        return jit(apply_reaction_rates)  # type: ignore


__all__ = ["Reaction", "ReactionFluxExpression", "Reactions"]
