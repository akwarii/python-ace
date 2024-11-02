import functools
import inspect
import logging
import warnings

from ase.data import atomic_numbers, covalent_radii

from pyace.atomicenvironment import aseatoms_to_atomicenvironment


def compute_nn_dist_per_bond(atoms, cutoff=3, elements_mapper_dict=None):
    ae = aseatoms_to_atomicenvironment(atoms, cutoff, elements_mapper_dict=elements_mapper_dict)
    return ae.get_minimal_nn_distance_per_bond()


def get_vpa(atoms):
    try:
        return atoms.get_volume() / len(atoms)
    except ValueError as e:
        return 0


def complement_min_dist_dict(min_dist_per_bond_dict, bond_quantile_dict, elements, verbose):
    res = min_dist_per_bond_dict.copy()
    for mu_i in range(len(elements)):
        for mu_j in range(mu_i, len(elements)):
            k = tuple(sorted([mu_i, mu_j]))
            if k not in res:
                if k in bond_quantile_dict:
                    res[k] = bond_quantile_dict[k]
                else:
                    # get some default values  as covalent radii
                    # covalent_radii[1] for H, ...
                    z_i = atomic_numbers[elements[mu_i]]
                    z_j = atomic_numbers[elements[mu_j]]
                    r_in = covalent_radii[z_i] + covalent_radii[z_j]
                    if verbose:
                        logging.warning(
                            f"No minimal distance for bond {k}, using sum of covalent  radii: {r_in:.3f}"
                        )
                    res[k] = r_in
    return res


# Source: Laurent LAPORTE's answer
# https://stackoverflow.com/questions/2536307/decorators-in-the-python-standard-lib-deprecated-specifically
def deprecated(reason):
    """
    This is a decorator which can be used to mark functions
    as deprecated. It will result in a warning being emitted
    when the function is used.
    """
    string_types = (bytes, str)

    if isinstance(reason, string_types):

        # The @deprecated is used with a 'reason'.
        #
        # .. code-block:: python
        #
        #    @deprecated("please, use another function")
        #    def old_function(x, y):
        #      pass

        def decorator(func1):

            if inspect.isclass(func1):
                fmt1 = "Call to deprecated class {name} ({reason})."
            else:
                fmt1 = "Call to deprecated function {name} ({reason})."

            @functools.wraps(func1)
            def new_func1(*args, **kwargs):
                warnings.simplefilter("always", DeprecationWarning)
                warnings.warn(
                    fmt1.format(name=func1.__name__, reason=reason),
                    category=DeprecationWarning,
                    stacklevel=2,
                )
                warnings.simplefilter("default", DeprecationWarning)
                return func1(*args, **kwargs)

            return new_func1

        return decorator

    elif inspect.isclass(reason) or inspect.isfunction(reason):

        # The @deprecated is used without any 'reason'.
        #
        # .. code-block:: python
        #
        #    @deprecated
        #    def old_function(x, y):
        #      pass

        func2 = reason

        if inspect.isclass(func2):
            fmt2 = "Call to deprecated class {name}."
        else:
            fmt2 = "Call to deprecated function {name}."

        @functools.wraps(func2)
        def new_func2(*args, **kwargs):
            warnings.simplefilter("always", DeprecationWarning)
            warnings.warn(
                fmt2.format(name=func2.__name__), category=DeprecationWarning, stacklevel=2
            )
            warnings.simplefilter("default", DeprecationWarning)
            return func2(*args, **kwargs)

        return new_func2

    else:
        raise TypeError(repr(type(reason)))
