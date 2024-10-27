import numpy as np
import pandas as pd
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.neighborlist import neighbor_list
from ase.units import _e, _eps0
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

from pyace import PyACECalculator

# string consts
ZBL = "zbl"
EOS = "eos"
KINK = "kink"
EXTRAPOLATION = "extrapolation"

# column names
GAMMA_C = "gamma"
GAMMA_PER_ATOM_C = "gamma_per_atom"
VPA_C = "vpa"
EPA_C = "epa"
Z_C = "z"
ASE_ATOMS_C = "ase_atoms"
EPA_ZBL_C = "epa_zbl"
FORCES_ZBL_C = "forces_zbl"
EPA_EOS_C = "epa_eos"
FORCES_EOS_C = "forces_eos"
EPA_CORRECTED_C = "energy_corrected_per_atom"
FORCES_C = "forces"
E_CORRECTED_C = "energy_corrected"
NAME_C = "name"

# transformation coefficients to eV
K = _e**2 / (4 * np.pi * _eps0) / 1e-10 / _e

# coefficients of ZBL potential
phi_coefs = np.array([0.18175, 0.50986, 0.28022, 0.02817])
phi_exps = np.array([-3.19980, -0.94229, -0.40290, -0.20162])


def phi(x):
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    return np.sum(phi_coefs * np.exp(x.reshape(-1, 1) * phi_exps), axis=1)


def dphi(x):
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    return np.sum(phi_coefs * phi_exps * np.exp(x.reshape(-1, 1) * phi_exps), axis=1)


def d2phi(x):
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    return np.sum(phi_coefs * phi_exps * phi_exps * np.exp(x.reshape(-1, 1) * phi_exps), axis=1)


# common factor: K*Zi*Zj*
def fun_E_ij(nl_dist, a):
    return 1 / nl_dist * phi(nl_dist / a)


# common factor: K*Zi*Zj*
def fun_dE_ij(nl_dist, a):
    return (-1 / nl_dist**2) * phi(nl_dist / a) + 1 / nl_dist * dphi(nl_dist / a) / a


# common factor: K*Zi*Zj*
def fun_d2E_ij(nl_dist, a):
    return (
        (+2 / nl_dist**3) * phi(nl_dist / a)
        + 2 * (-1 / nl_dist**2) * dphi(nl_dist / a) / a
        + (1 / nl_dist) * d2phi(nl_dist / a) / (a**2)
    )


class ZBLCalculator(Calculator):
    """Python implementation of Ziegler-Biersack-Littmark (ZBL) potential as ASE calculator
    References
        https://docs.lammps.org/pair_zbl.html
        https://docs.lammps.org/pair_gromacs.html
        https://github.com/lammps/lammps/blob/develop/src/pair_zbl.cpp

    """

    implemented_properties = ["energy", "free_energy", "forces"]

    def __init__(
        self,
        cut_in: float = 0.0,
        cutoff: float = 3.0,
        **kwargs,
    ) -> None:
        Calculator.__init__(self, **kwargs)
        self.inner_cutoff = cut_in
        self.outer_cutoff = cutoff

        self.energy = None
        self.forces = None

    def calculate(
        self,
        atoms: Atoms | None = None,
        properties: list[str] = ["energy", "forces", "free_energy"],
        system_changes: list[str] = all_changes,
    ) -> None:
        Calculator.calculate(self, atoms, properties, system_changes)

        nl_i, nl_j, d, D = neighbor_list("ijdD", atoms, cutoff=self.outer_cutoff)

        atomic_numbers = atoms.get_atomic_numbers()  # type: ignore
        Zi = atomic_numbers[nl_i]
        Zj = atomic_numbers[nl_j]

        # ZBL potential
        a = 0.46850 / (Zi**0.23 + Zj**0.23)

        E_ij = fun_E_ij(d, a)

        Ec = fun_E_ij(self.outer_cutoff, a)
        dEc = fun_dE_ij(self.outer_cutoff, a)
        d2Ec = fun_d2E_ij(self.outer_cutoff, a)

        drcut = self.outer_cutoff - self.inner_cutoff

        A = (-3 * dEc + drcut * d2Ec) / drcut**2
        B = (2 * dEc - drcut * d2Ec) / drcut**3
        C = -Ec + 1 / 2 * drcut * dEc - 1 / 12 * drcut**2 * d2Ec

        S = A / 3 * (d - self.inner_cutoff) ** 3 + B / 4 * (d - self.inner_cutoff) ** 4 + C  # S(r)

        S[d < self.inner_cutoff] = C[d < self.inner_cutoff]
        S[d > self.outer_cutoff] = 0
        self.energy = K / 2 * np.sum(Zi * Zj * (E_ij + S))

        # forces
        dEdr = fun_dE_ij(d, a)

        dS_dr = A * (d - self.inner_cutoff) ** 2 + B * (d - self.inner_cutoff) ** 3
        dS_dr[(d < self.inner_cutoff) | (d > self.outer_cutoff)] = 0.0

        pair_forces = -(dEdr + dS_dr).reshape(-1, 1) * (D / d.reshape(-1, 1))
        pair_forces *= (Zi * Zj).reshape(-1, 1) * K / 2

        nat = len(atoms)  # type: ignore
        forces = [
            np.bincount(nl_j, weights=pair_forces[:, i], minlength=nat)
            - np.bincount(nl_i, weights=pair_forces[:, i], minlength=nat)
            for i in range(3)
        ]
        self.forces = np.vstack(forces).T

        # results
        self.results["energy"] = self.energy
        self.results["free_energy"] = self.energy
        self.results["forces"] = self.forces


def E_ER_pars(V, pars):
    E0, V0, c3, lr = pars
    xrs = (V ** (1 / 3) - V0 ** (1 / 3)) / lr
    return E0 * (1 + xrs + c3 * xrs**3) * np.exp(-xrs)


def E_ER(V, *pars):
    return E_ER_pars(V, pars)


def get_min_nn_dist(atoms: Atoms, cutoff: float = 7.0) -> float:
    """Compute minimal nearest-neighbours (NN) distance in `atoms` (maximum up to `cutoff`)"""
    min_nn_dist = np.min(neighbor_list("d", atoms, cutoff=cutoff))
    return min_nn_dist


def make_cell_for_non_periodic_structure(
    atoms: Atoms,
    wrapped: bool = True,
    scale: float = 3.0,
    alat: float | None = None,
    min_cell_len=10.0,
) -> list[list[float]]:
    """
    Function to generate a cell with sides s_i; s_i = diameter_i + (alat*scale) adapted
    from Minaam Quamar

    atoms: ase.Atoms object
        must be an atoms object with pbc=False and cell=[0,0,0]
        with both positive and negative cartesian positions.

        In case `structure` is provided as a periodic structure,
        set argument `wrapped=False` to wrap the positions around the
        origin to resemble a non-periodic struct

    alat: float
        ground state equilibrium lattice constant for the element

    scale: float
        arbit but preferably > 5 to ensure minimal interaction
        through pbc boundary

    wrapped: bool
        if input structure is periodic then set `wrapped=False`

        This will manually wrap the atoms about (0,0,0). But may shift the
        relative atomic positions slightly. So avoid unless necessary

    """
    if alat is None:
        raise ValueError("alat must be provided")

    if not wrapped:
        atoms.wrap(center=[0, 0, 0])

    # get all the x,y,z positions
    pos = atoms.positions
    x, y, z = pos.T

    # get the diameter of the cluster in x,y,z
    X = max(x) - min(x)
    Y = max(y) - min(y)
    Z = max(z) - min(z)

    # to ensure minimum diameter=1 for very small clusters
    if X < 1:
        X = 1
    if Y < 1:
        Y = 1
    if Z < 1:
        Z = 1

    # generate orthogonal cell
    cell = [
        [max(X + (scale * alat), min_cell_len), 0.0, 0.0],
        [0.0, max(Y + (scale * alat), min_cell_len), 0.0],
        [0.0, 0.0, max(Z + (scale * alat), min_cell_len)],
    ]

    return cell


def make_periodic_structure(
    atoms: Atoms,
    wrapped: bool = True,
    scale: float = 5.0,
    alat: float = 2.0,
) -> Atoms:
    atoms = atoms.copy()
    if not all(atoms.get_pbc()):
        orthogonal_cell = make_cell_for_non_periodic_structure(
            atoms, wrapped=wrapped, scale=scale, alat=alat
        )
        atoms.set_cell(orthogonal_cell)
        atoms.set_pbc(True)
    return atoms


def enforce_pbc_structure(basis_ref: Atoms) -> tuple[Atoms, bool]:
    if not all(basis_ref.get_pbc()):
        basis_ref = make_periodic_structure(basis_ref)
        enforced_pbc = True
    else:
        enforced_pbc = False
    return basis_ref, enforced_pbc


def remove_pbc(atoms: Atoms) -> None:
    atoms.set_cell(None, scale_atoms=False)
    atoms.set_pbc(False)


def generate_nndist_atoms(
    original_atoms: Atoms,
    nn_distances: list[np.ndarray],
    cutoff: float = 7.0,
) -> list[Atoms]:
    original_atoms, enforced_pbc = enforce_pbc_structure(original_atoms)
    min_nn_dist = get_min_nn_dist(original_atoms, cutoff=cutoff)

    atoms_list = []
    for z in nn_distances:
        at = original_atoms.copy()
        cell = at.get_cell() * z / min_nn_dist
        at.set_cell(cell, scale_atoms=True)

        if enforced_pbc:
            remove_pbc(at)

        atoms_list.append(at)

    return atoms_list


def fit_eos(
    vpas: pd.Series,
    epas: pd.Series,
    fit_iterations: int,
    best_rmse_threshold: float,
    seed: int,
) -> tuple[np.ndarray, float]:
    if len(vpas) < 4:
        raise RuntimeError(
            f"Number of reliable data-points ({len(vpas)}) is less than minimal required for EOS (4)"
        )

    if seed is not None:
        np.random.seed(seed)

    p0 = np.array((-5, 20, 0.5, 1))
    # random shuffle for best params optimization
    e_best_rmse = np.inf
    best_parsER = None
    for it in range(fit_iterations):
        dp0 = 0 if it == 0 else np.random.randn(4) * np.array([10, 20, 5, 5]) * 2

        try:
            parsER, _ = curve_fit(
                E_ER,
                vpas,
                epas,
                p0=p0 + dp0,
                maxfev=1000,
            )
        except Exception as e:
            print("Exception:", e)
            continue

        e_rmse = np.sqrt(np.mean((E_ER_pars(vpas, parsER) - epas) ** 2))
        if e_rmse < e_best_rmse:
            e_best_rmse = e_rmse
            best_parsER = parsER
            print("E_rmse:", e_rmse)

        if e_best_rmse < best_rmse_threshold:
            break

    return best_parsER, e_best_rmse  # type: ignore


def plot_all(
    df: pd.DataFrame,
    df_reliable: pd.DataFrame | None = None,
    df_selected: pd.DataFrame | None = None,
    plot_eos: bool = True,
    plot_zbl: bool = False,
) -> None:
    """
    Helper function to plot various energy per atom (EPA) data against the distance (z) for a given DataFrame.

    Parameters:
    -----------
    df : pd.DataFrame
        The main DataFrame containing the data to be plotted. Must include columns 'z' and 'epa'.
    df_reliable : pd.DataFrame, optional
        An optional DataFrame containing reliable EPA data to be plotted. Must include columns 'z' and 'epa'.
    df_selected : pd.DataFrame, optional
        An optional DataFrame containing selected EPA data to be plotted. Must include columns 'z' and 'energy_corrected_per_atom'.
    plot_eos : bool, default=True
        If True, plots the EOS (Equation of State) data.
    plot_zbl : bool, default=False
        If True, plots the ZBL (Ziegler-Biersack-Littmark) data.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    title = df.iloc[0]["ase_atoms"].get_chemical_formula()

    if plot_zbl:
        plt.plot(df["z"], df["epa_zbl"], "--", label="ZBL")

    if plot_eos:
        plt.plot(df["z"], df["epa_eos"], "-", label="EOS", color="gray")

    plt.plot(df["z"], df["epa"], "o-", color="red", label="ACE")

    if df_reliable is not None:
        plt.plot(
            df_reliable["z"],
            df_reliable["epa"],
            "o-",
            color="green",
            ls="--",
            label="ACE-reliable",
        )

    if df_selected is not None:
        plt.plot(
            df_selected["z"],
            df_selected["energy_corrected_per_atom"],
            "d-",
            color="blue",
            label="AUG data",
        )

    plt.title(title)
    plt.legend()

    plt.xlabel("z, A")
    plt.ylabel("E, eV/at")

    plt.yscale("symlog")
    plt.ylim(-10, None)

    plt.show()


def augment_structure_eos(
    atoms: Atoms,
    calc: PyACECalculator,
    nn_distance_range: tuple[float, float] = (1.0, 5.0),
    nn_distance_step: float = 0.1,
    reliability_criteria: str = KINK,
    augmentation_type: str = EOS,
    epa_reliable_max: float | None = None,
    epa_aug_max: float | None = None,
    epa_aug_min: float | None = None,
    gamma_max: float = 10.0,
    eos_fit_n_iter: int = 20,
    eos_fit_rmse_threshold: float = 0.5,
    eos_seed: int | None = None,
    plot_verbose: bool = False,
    plot_eos: bool = False,
    plot_zbl: bool = False,
    zbl_r_in: float = 0.0,
    zbl_r_out: float = 4.0,
) -> pd.DataFrame:
    """
    Augments the structure of atoms using Equation of State (EOS) or Ziegler-Biersack-Littmark (ZBL) methods.

    Parameters:
    -----------
    atoms : Atoms
        The atomic structure to be augmented.
    calc : PyACECalculator
        The calculator used for energy and force calculations.
    nn_distance_range : tuple[float, float], optional
        The range of nearest neighbor distances to consider, by default (1.0, 5.0).
    nn_distance_step : float, optional
        The step size for nearest neighbor distances, by default 0.1.
    reliability_criteria : str, optional
        The criteria for reliability, by default KINK.
    augmentation_type : str, optional
        The type of augmentation to perform, either EOS or ZBL, by default EOS.
    epa_reliable_max : float | None, optional
        The maximum reliable energy per atom, by default None.
    epa_aug_max : float | None, optional
        The maximum augmented energy per atom, by default None.
    epa_aug_min : float | None, optional
        The minimum augmented energy per atom, by default None.
    gamma_max : float, optional
        The maximum gamma value for reliability, by default 10.0.
    eos_fit_n_iter : int, optional
        The number of iterations for EOS fitting, by default 20.
    eos_fit_rmse_threshold : float, optional
        The RMSE threshold for EOS fitting, by default 0.5.
    eos_seed : int | None, optional
        The seed for EOS fitting, by default None.
    plot_verbose : bool, optional
        Whether to plot verbose information, by default False.
    plot_eos : bool, optional
        Whether to plot EOS information, by default False.
    plot_zbl : bool, optional
        Whether to plot ZBL information, by default False.
    zbl_r_in : float, optional
        The inner radius for ZBL, by default 0.0.
    zbl_r_out : float, optional
        The outer radius for ZBL, by default 4.0.

    Returns:
    --------
    pd.DataFrame
        A DataFrame containing the augmented atomic structures and related data.
    """
    plot_verbose = plot_verbose or (plot_eos or plot_zbl)
    compute_zbl = augmentation_type == ZBL or plot_zbl
    compute_eos = augmentation_type == EOS or plot_eos

    natoms = len(atoms)
    compute_gamma = reliability_criteria == EXTRAPOLATION
    df = compute_enn_df(
        atoms,
        calc,
        compute_zbl,
        nn_distance_range,
        nn_distance_step,
        compute_gamma,
        zbl_r_in,
        zbl_r_out,
    )

    df_reliable = select_reliable_enn_part(df, reliability_criteria, epa_reliable_max, gamma_max)

    epa_reliable_max = df_reliable[EPA_C].max()
    vpa_reliable_min = df_reliable[VPA_C].min()

    if compute_eos:
        best_parsER, e_best_rmse = fit_eos(
            df_reliable[VPA_C],
            df_reliable[EPA_C],
            fit_iterations=eos_fit_n_iter,
            best_rmse_threshold=eos_fit_rmse_threshold,
            seed=eos_seed,
        )
        print("BEST E_RMSE:", e_best_rmse)

        df[EPA_EOS_C] = E_ER_pars(df[VPA_C], best_parsER)
        df[FORCES_EOS_C] = df[ASE_ATOMS_C].map(lambda a: np.zeros((len(a), 3)))  # type: ignore

        if e_best_rmse > eos_fit_rmse_threshold and not augmentation_type == ZBL:
            if plot_verbose:
                plot_all(df, df_reliable, df_selected=None, plot_eos=True, plot_zbl=plot_zbl)
            raise RuntimeError(f"Cannot reliabley fit EOS-ER to ACE data, E-RMSE={e_best_rmse}")

    # augment data
    df_selected = df.copy()
    if augmentation_type == ZBL:
        df_selected[EPA_CORRECTED_C] = df_selected[EPA_ZBL_C]
        df_selected[FORCES_C] = df_selected[FORCES_ZBL_C]
        df_selected = df_selected[df_selected[EPA_ZBL_C] >= epa_reliable_max].copy()
    elif augmentation_type == EOS:
        df_selected[EPA_CORRECTED_C] = df_selected[EPA_EOS_C]
        df_selected[FORCES_C] = df_selected[FORCES_EOS_C]
    else:
        raise NotImplementedError(f"Unknown augmentation_type=`{augmentation_type}`")

    # 1. Volume less than min reliable volume
    df_selected = df_selected[df_selected[VPA_C] <= vpa_reliable_min]

    # 2. Epa_aug max criteria
    if epa_aug_max is not None:
        df_selected = df_selected[df_selected[EPA_CORRECTED_C] <= epa_aug_max]

    # 3. Epa_aug min criteria
    if epa_aug_min is not None:
        df_selected = df_selected[df_selected[EPA_CORRECTED_C] >= epa_aug_min]

    df_selected[E_CORRECTED_C] = df_selected[EPA_CORRECTED_C] * natoms

    df_selected[NAME_C] = "augmented/" + augmentation_type + "/" + atoms.get_chemical_formula()

    if plot_verbose:
        plot_all(
            df,
            df_reliable,
            df_selected,
            plot_zbl=plot_zbl,
            plot_eos=plot_eos,
        )

    # remove calc
    df_selected[ASE_ATOMS_C].map(lambda a: a.set_calculator(None))
    return df_selected[
        [
            NAME_C,
            ASE_ATOMS_C,
            E_CORRECTED_C,
            EPA_CORRECTED_C,
            Z_C,
            FORCES_C,
        ]
    ]


def compute_enn_df(
    atoms: Atoms,
    calc: PyACECalculator,
    compute_zbl: bool = False,
    nn_distance_range: tuple[float, float] = (1.0, 5.0),
    nn_distance_step: float = 0.1,
    compute_gamma: bool = False,
    zbl_r_in: float = 0.0,
    zbl_r_out: float = 4.0,
) -> pd.DataFrame:
    if compute_zbl:
        zblcalc = ZBLCalculator(cut_in=zbl_r_in, cutoff=zbl_r_out)

    natoms = len(atoms)
    structs = []
    epas, vpas, epa_zbls = [], [], []
    gammas, gamma_per_atom = [], []
    zs, fzbls = [], []

    nn_distances = list(np.arange(*nn_distance_range, nn_distance_step))
    for z, curr_atoms in zip(nn_distances, generate_nndist_atoms(atoms, nn_distances)):
        # compute ACE energy
        curr_atoms.set_calculator(calc)
        epas.append(curr_atoms.get_potential_energy() / natoms)

        try:
            vpas.append(curr_atoms.get_volume() / natoms)
        except Exception as e:
            vpas.append(0)
        zs.append(z)

        if compute_gamma:
            gammas.append(calc.results[GAMMA_C].max())
            gamma_per_atom.append(calc.results[GAMMA_C])

        structs.append(curr_atoms)

        if compute_zbl:
            curr_atoms.set_calculator(zblcalc)
            epa_zbls.append(curr_atoms.get_potential_energy() / natoms)
            fzbls.append(curr_atoms.get_forces())

    df = (
        pd.DataFrame({VPA_C: vpas, EPA_C: epas, Z_C: zs, ASE_ATOMS_C: structs})
        .sort_values(Z_C)
        .reset_index(drop=True)
    )

    if compute_gamma:
        df[GAMMA_C] = gammas
        df[GAMMA_PER_ATOM_C] = gamma_per_atom

    if compute_zbl:
        df[EPA_ZBL_C] = epa_zbls
        df[FORCES_ZBL_C] = fzbls

    return df


def select_reliable_enn_part(
    df: pd.DataFrame,
    reliability_criteria: str = KINK,
    epa_reliable_max: float | None = None,
    gamma_max: float = 1.5,
) -> pd.DataFrame:
    # Selection of reliable part (required for all augmentation types)
    if reliability_criteria == KINK:
        peaks, _ = find_peaks(df[EPA_C])

        if len(peaks) == 0:
            df_reliable = df.copy()
        else:
            # get up to last peak
            p = peaks[-1]
            df_reliable = df.iloc[p:].copy()

    elif reliability_criteria == EXTRAPOLATION:
        df_reliable = df[df[GAMMA_C] < gamma_max].copy()
    else:
        raise NotImplementedError(
            f"Reliability_criteria '{reliability_criteria}' is not implemented"
        )

    if epa_reliable_max is not None:
        df_reliable = df_reliable[df_reliable[EPA_C] <= epa_reliable_max].copy()

    return df_reliable
