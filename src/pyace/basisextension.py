import logging
from typing import Any

from pyace.basis import BBasisConfiguration, BBasisFunctionSpecification
from pyace.const import (
    ORDERS_LMAX_KW,
    ORDERS_NRADMAX_KW,
    POTENTIAL_LMAX_KW,
    POTENTIAL_NRADMAX_KW,
)

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


def get_actual_ladder_step(
    ladder_step_param: int | float | list,
    current_number_of_funcs: int,
    final_number_of_funcs: int,
) -> int:
    ladder_discrete_step: int = 0
    ladder_frac: float = 0.0

    if isinstance(ladder_step_param, int) and ladder_step_param >= 1:
        ladder_discrete_step = int(ladder_step_param)
    elif isinstance(ladder_step_param, float) and 1.0 > ladder_step_param > 0:
        ladder_frac = float(ladder_step_param)
    elif isinstance(ladder_step_param, (list, tuple)):
        if len(ladder_step_param) > 2:
            raise ValueError(f"Invalid number of ladder step parameters: {ladder_step_param}.")
        for p in ladder_step_param:
            if p >= 1:
                ladder_discrete_step = int(p)
            elif 0 < p < 1:
                ladder_frac = float(p)
            else:
                raise ValueError(
                    f"Invalid ladder step parameter: {ladder_step_param}. Should be integer >= 1 or  0<float<1 or list of both [int, float]"
                )
    elif ladder_step_param is None:
        ladder_discrete_step = final_number_of_funcs - current_number_of_funcs
        log.info("Ladder step parameter is None - all functions will be added")
    else:
        raise ValueError(
            f"Invalid ladder step parameter: {ladder_step_param}. Should be integer >= 1 or  0<float<1 or list of both [int, float]"
        )

    ladder_frac_step = int(round(ladder_frac * current_number_of_funcs))
    ladder_step = max(ladder_discrete_step, ladder_frac_step, 1)
    log.info(
        f"Possible ladder steps: discrete - {ladder_discrete_step}, fraction - {ladder_frac_step}. Selected maximum - {ladder_step}"
    )

    if current_number_of_funcs + ladder_step > final_number_of_funcs:
        ladder_step = final_number_of_funcs - current_number_of_funcs
        log.info(f"Ladder step is too large and adjusted to {ladder_step}")

    return ladder_step


def construct_bbasisconfiguration(
    potential_config: dict,
    initial_basisconfig: BBasisConfiguration = None,
    overwrite_blocks_from_initial_bbasis=False,
) -> BBasisConfiguration:
    from pyace.multispecies_basisextension import (
        create_multispecies_basis_config,
        single_to_multispecies_converter,
    )

    # for backward compatibility with pacemaker 1.0 potential_dict format
    check_backward_compatible_parameters(potential_config)

    # if old-single specie format
    if not (
        "elements" in potential_config
        and "bonds" in potential_config  # "embeddings" in potential_config and
        and "functions" in potential_config
    ):
        # convert to new general multispecies format
        potential_config = single_to_multispecies_converter(potential_config)

    return create_multispecies_basis_config(
        potential_config,
        initial_basisconfig=initial_basisconfig,
        overwrite_blocks_from_initial_bbasis=overwrite_blocks_from_initial_bbasis,
    )


def sort_funcspecs_list(
    lst: list[BBasisFunctionSpecification],
    ladder_type: str,
) -> list[BBasisFunctionSpecification]:
    if ladder_type == "power_order":
        return list(sorted(lst, key=lambda func: len(func.ns) + sum(func.ns) + sum(func.ls)))
    elif ladder_type == "body_order":
        return list(
            sorted(
                lst,
                key=lambda func: (
                    len(tuple(func.ns)),
                    tuple(func.ns),
                    tuple(func.ls),
                    tuple(func.LS),
                    tuple(func.elements),
                ),
            )
        )
    else:
        raise ValueError(f"Specified Ladder type ({ladder_type}) is not implemented")


def extend_basis(
    initial_basis: BBasisConfiguration,
    final_basis: BBasisConfiguration,
    ladder_type: str,
    func_step: int | None = None,
    return_is_extended: bool = False,
) -> BBasisConfiguration:
    # TODO: move from here, optimize import
    from pyace.multispecies_basisextension import extend_multispecies_basis

    return extend_multispecies_basis(
        initial_basis,
        final_basis,
        ladder_type,
        func_step,
        return_is_extended=return_is_extended,
    )


def check_backward_compatible_parameters(potential_config: dict[str, Any]) -> None:
    # "nradmax" -> "nradmax_by_orders"
    # "lmax" -> "lmax_by_orders"

    if POTENTIAL_NRADMAX_KW in potential_config:
        log.warn(
            f"potential_config:'{POTENTIAL_NRADMAX_KW}' is deprecated parameter, please use '{ORDERS_NRADMAX_KW}'"
        )
        potential_config[ORDERS_NRADMAX_KW] = potential_config[POTENTIAL_NRADMAX_KW]

    if POTENTIAL_LMAX_KW in potential_config:
        log.warn(
            f"potential_config:'{POTENTIAL_LMAX_KW}' is deprecated parameter, please use '{ORDERS_LMAX_KW}'"
        )
        potential_config[ORDERS_LMAX_KW] = potential_config[POTENTIAL_LMAX_KW]
