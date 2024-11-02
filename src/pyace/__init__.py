import logging

from pyace.asecalc import PyACECalculator, PyACEEnsembleCalculator
from pyace.atomicenvironment import (
    ACEAtomicEnvironment,
    aseatoms_to_atomicenvironment,
    create_cube,
    create_linear_chain,
)
from pyace.basis import (
    ACEBBasisFunction,
    ACEBBasisSet,
    ACECTildeBasisFunction,
    ACECTildeBasisSet,
    ACERadialFunctions,
    BBasisConfiguration,
    BBasisFunctionSpecification,
    BBasisFunctionsSpecificationBlock,
    Fexp,
)
from pyace.calculator import ACECalculator
from pyace.coupling import (
    ACECouplingTree,
    expand_ls_LS,
    generate_ms_cg_list,
    is_valid_ls_LS,
    validate_ls_LS,
)
from pyace.evaluator import ACEBEvaluator, ACECTildeEvaluator, get_ace_evaluator_version
from pyace.multispecies_basisextension import create_multispecies_basis_config
from pyace.preparedata import (
    ACEDataset,
    EnergyBasedWeightingPolicy,
    UniformWeightingPolicy,
)
from pyace.pyacefit import PyACEFit
from pyace.radial import (
    RadialFunctionSmoothness,
    RadialFunctionsValues,
    RadialFunctionsVisualization,
)

from . import _version

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logging.getLogger().setLevel(logging.INFO)

__version__ = _version.get_versions()["version"]

__all__ = [
    "ACEAtomicEnvironment",
    "create_cube",
    "create_linear_chain",
    "aseatoms_to_atomicenvironment",
    "BBasisFunctionSpecification",
    "BBasisConfiguration",
    "BBasisFunctionsSpecificationBlock",
    "ACEBBasisFunction",
    "ACECTildeBasisFunction",
    "ACERadialFunctions",
    "ACECTildeBasisSet",
    "ACEBBasisSet",
    "ACECalculator",
    "ACECouplingTree",
    "generate_ms_cg_list",
    "validate_ls_LS",
    "is_valid_ls_LS",
    "expand_ls_LS",
    "ACECTildeEvaluator",
    "ACEBEvaluator",
    "PyACEFit",
    "PyACECalculator",
    "PyACEEnsembleCalculator",
    "ACEDataset",
    "Fexp",
    "get_ace_evaluator_version",
    "EnergyBasedWeightingPolicy",
    "UniformWeightingPolicy",
    "RadialFunctionsValues",
    "RadialFunctionsVisualization",
    "RadialFunctionSmoothness",
    "create_multispecies_basis_config",
]
