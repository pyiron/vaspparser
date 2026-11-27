from __future__ import annotations

import warnings

import numpy as np
from ase.atoms import Atoms

from vaspparser.compatibility.input import Incar
from vaspparser.vasp.structure import vasp_sorter

VALID_LORBIT = [0, 1, 2, 5, 10, 11, 12, 13, 14]


def set_algorithm(incar: Incar, algorithm: str = "Fast", ialgo: int | None = None):
    """Electronic minimisation algorithm - equivalent to ``Vasp.set_algorithm()``."""
    if ialgo is not None:
        incar["IALGO"] = int(ialgo)
    else:
        incar["ALGO"] = str(algorithm)
        if algorithm not in ["Fast", "Accurate", "Normal", "Very Fast"]:
            warnings.warn(
                f"Algorithm {algorithm} is unusual for VASP. I hope you know what you are up to",
                stacklevel=2,
            )


def set_write_charge_density(incar: Incar, value: bool):
    incar["LCHARG"] = bool(value)


def set_write_electrostatic_potential(incar: Incar, value: bool):
    incar["LVTOT"] = bool(value)
    if bool(value):
        incar["LVHAR"] = True


def set_write_wave_funct(incar: Incar, value: bool):
    if not isinstance(value, bool):
        raise ValueError("write_wave_funct, can either be True or False.")
    incar["LWAVE"] = value


def set_write_resolved_dos(incar: Incar, value: bool | int):
    if not isinstance(value, (bool, int)):
        raise ValueError(
            "write_resolved_dos, can either be True, False or 0, 1, 2, 5, 10, 11, 12."
        )
    incar["LORBIT"] = value


def set_convergence_precision(
    incar: Incar,
    ionic_energy_tolerance: float | None = 1.0e-3,
    electronic_energy: float | None = 1.0e-7,
    ionic_force_tolerance: float | None = 1.0e-2,
):
    """Electronic and ionic convergence criteria - equivalent to ``Vasp.set_convergence_precision()``."""
    if ionic_force_tolerance is not None:
        incar["EDIFFG"] = -1.0 * abs(ionic_force_tolerance)
    elif ionic_energy_tolerance is not None:
        incar["EDIFFG"] = abs(ionic_energy_tolerance)
    else:
        incar["EDIFFG"] = -0.01 if incar["ISIF"] not in (5, 6, 7) else 0.01
    if electronic_energy is not None:
        incar["EDIFF"] = electronic_energy


def calc_static(
    incar: Incar,
    electronic_steps: int = 100,
    algorithm: str | None = None,
    retain_charge_density: bool = False,
    retain_electrostatic_potential: bool = False,
):
    """Static calculation - equivalent to ``Vasp.calc_static()``."""
    incar["IBRION"] = -1
    incar["NELM"] = electronic_steps
    incar["NSW"] = 0
    if algorithm is not None:
        set_algorithm(incar=incar, algorithm=algorithm)
    set_write_charge_density(incar=incar, value=retain_charge_density)
    set_write_electrostatic_potential(incar=incar, value=retain_electrostatic_potential)


def calc_minimize(
    incar: Incar,
    electronic_steps: int = 400,
    ionic_steps: int = 100,
    max_iter: int | None = None,
    pressure: float | None = None,
    algorithm: str | None = None,
    retain_charge_density: bool = False,
    retain_electrostatic_potential: bool = False,
    ionic_energy_tolerance: float | None = None,
    ionic_force_tolerance: float | None = None,
    volume_only: bool = False,
    cell_only: bool = False,
):
    """Structure optimisation - equivalent to ``Vasp.calc_minimize()``."""
    if volume_only:
        incar["ISIF"] = 7
    elif cell_only:
        incar["ISIF"] = 6
    elif pressure == 0.0:
        incar["ISIF"] = 3
    elif pressure is None:
        incar["ISIF"] = 2
    else:
        raise ValueError("Non-zero pressure not supported!")
    if max_iter:
        electronic_steps = max_iter
        ionic_steps = max_iter
    incar["IBRION"] = 2
    incar["NELM"] = electronic_steps
    incar["NSW"] = ionic_steps
    if algorithm is not None:
        set_algorithm(incar=incar, algorithm=algorithm)
    set_write_charge_density(incar=incar, value=retain_charge_density)
    set_write_electrostatic_potential(incar=incar, value=retain_electrostatic_potential)
    set_convergence_precision(
        incar=incar,
        ionic_force_tolerance=ionic_force_tolerance,
        ionic_energy_tolerance=ionic_energy_tolerance,
        electronic_energy=None,
    )


def calc_md(
    incar: Incar,
    temperature: float | list | None = None,
    n_ionic_steps: int = 1000,
    n_print: int = 1,
    time_step: float = 1.0,
    retain_charge_density: bool = False,
    retain_electrostatic_potential: bool = False,
    **kwargs,
):
    """Molecular dynamics - equivalent to ``Vasp.calc_md()``. Additional keyword arguments are ignored."""
    if temperature is not None:
        incar["SMASS"] = 3
        if isinstance(temperature, (int, float)):
            incar["TEBEG"] = temperature
        else:
            incar["TEBEG"] = temperature[0]
            incar["TEEND"] = temperature[-1]
    else:
        incar["SMASS"] = -3
    incar["NSW"] = n_ionic_steps
    incar["NBLOCK"] = int(n_print)
    incar["POTIM"] = time_step
    if "ISYM" not in incar:
        incar["ISYM"] = 0
    set_write_charge_density(incar=incar, value=retain_charge_density)
    set_write_electrostatic_potential(incar=incar, value=retain_electrostatic_potential)
    for key, value in kwargs.items():
        if value is not None:
            warnings.warn(f"Tag {key} not relevant for vasp", stacklevel=2)


def set_occupancy_smearing(
    incar: Incar,
    smearing: str | None = None,
    width: float | None = None,
    order: int = 1,
    ismear: int | None = None,
):
    """Smearing of the occupancies - equivalent to ``Vasp.set_occupancy_smearing()``."""
    if ismear is not None:
        incar["ISMEAR"] = int(ismear)
    elif smearing is None:
        raise ValueError("Either smearing or ismear has to be specified.")
    elif smearing.lower().startswith("meth") or smearing.lower().startswith("mp"):
        incar["ISMEAR"] = int(order)
    elif smearing.lower().startswith("fermi"):
        incar["ISMEAR"] = -1
    elif smearing.lower().startswith("gauss"):
        incar["ISMEAR"] = 0
    else:
        raise ValueError(
            f"Smearing scheme {smearing} is not available. Only types 'Fermi', 'Gaussian', and 'Methfessel-Paxton'"
        )
    if width is not None:
        incar["SIGMA"] = width


def set_electric_field(
    incar: Incar, e_field: float = 0.1, direction: int = 2, dipole_center=None
):
    """External electric field - equivalent to ``Vasp.set_electric_field()``."""
    if direction not in range(3):
        raise ValueError("direction has to be 0, 1 or 2")
    incar["ISYM"] = 0
    incar["LORBIT"] = 11
    incar["IDIPOL"] = direction + 1
    incar["LDIPOL"] = True
    incar["EFIELD"] = e_field
    if dipole_center is not None:
        incar["DIPOL"] = " ".join(str(val) for val in dipole_center)


def set_dipole_correction(incar: Incar, direction: int = 2, dipole_center=None):
    """Dipole correction - equivalent to ``Vasp.set_dipole_correction()``."""
    set_electric_field(
        incar=incar, e_field=0, direction=direction, dipole_center=dipole_center
    )


def set_fft_mesh(
    incar: Incar, nx: int | None = None, ny: int | None = None, nz: int | None = None
):
    """Fine FFT mesh - equivalent to ``Vasp.set_fft_mesh()``."""
    if nx is not None:
        incar["NGXF"] = int(nx)
    if ny is not None:
        incar["NGYF"] = int(ny)
    if nz is not None:
        incar["NGZF"] = int(nz)


def set_mixing_parameters(
    incar: Incar,
    method: str | None = None,
    n_pulay_steps: int | None = None,
    density_mixing_parameter: float | None = None,
    spin_mixing_parameter: float | None = None,
    density_residual_scaling: float | None = None,
    spin_residual_scaling: float | None = None,
):
    """Density mixing - equivalent to ``Vasp.set_mixing_parameters()``."""
    if density_residual_scaling is not None or spin_residual_scaling is not None:
        raise NotImplementedError("Residual scaling is not implemented in VASP")
    if method is None:
        method = "PULAY"
    if method.upper() == "PULAY":
        incar["IMIX"] = 4
    if method.upper() == "KERKER":
        incar["IMIX"] = 1
    if n_pulay_steps is not None:
        incar["MAXMIX"] = n_pulay_steps
    if density_mixing_parameter is not None:
        incar["AMIX"] = density_mixing_parameter


def set_empty_states(incar: Incar, n_elect: float, n_empty_states: int | None = None):
    """Number of bands - equivalent to ``Vasp.set_empty_states()``."""
    if n_empty_states is not None:
        if n_empty_states < 0:
            raise ValueError(
                f"Number of empty states must be a positive integer or zero, not {n_empty_states}!"
            )
        incar["NBANDS"] = int(round(n_elect / 2)) + int(n_empty_states)


def set_rwigs(incar: Incar, structure: Atoms, rwigs_dict: dict):
    """Wigner-Seitz radii - equivalent to ``Vasp.set_rwigs()``."""
    if not isinstance(rwigs_dict, dict):
        raise ValueError("'rwigs_dict' has to be a dict!")
    if not all(isinstance(val, (int, float)) for val in rwigs_dict.values()):
        raise ValueError("The values of 'rwigs_dict' has to be floats!")
    species = sorted(set(structure.get_chemical_symbols()))
    for k in species:
        if k not in rwigs_dict:
            raise ValueError(f"'{k}' is not in rwigs_dict!")
    incar["RWIGS"] = " ".join(map(str, [rwigs_dict[k] for k in species]))


def set_spin_constraint(
    incar: Incar,
    structure: Atoms,
    lamb: float,
    rwigs_dict: dict,
    direction: bool = False,
    norm: bool = False,
):
    """Constrained magnetic moments - equivalent to ``Vasp.set_spin_constraint()``."""
    if not isinstance(direction, bool):
        raise ValueError("'direction' has to be a bool!")
    if not isinstance(norm, bool):
        raise ValueError("'norm' has to be a bool!")
    if not isinstance(lamb, float):
        raise ValueError("'lamb' has to be a float!")
    if direction and norm:
        incar["I_CONSTRAINED_M"] = 2
    elif direction:
        incar["I_CONSTRAINED_M"] = 1
    elif norm:
        raise ValueError("Constraining norm only is not possible.")
    else:
        raise ValueError(
            "You have to constrain either direction or norm and direction."
        )
    incar["LAMBDA"] = lamb
    set_rwigs(incar=incar, structure=structure, rwigs_dict=rwigs_dict)


def set_coulomb_interactions(
    incar: Incar,
    structure: Atoms,
    ldauu: dict | None = None,
    ldaul: dict | None = None,
    ldauj: dict | None = None,
    interaction_type: int = 2,
    ldau_print: bool = True,
):
    """
    On-site Coulomb interaction (DFT+U) - equivalent to ``Vasp.set_coulomb_interactions()``, with the species tags
    of pyiron replaced by dictionaries indexed by the chemical symbol.
    """
    ldauu = {} if ldauu is None else ldauu
    ldaul = {} if ldaul is None else ldaul
    ldauj = {} if ldauj is None else ldauj
    lst_l, lst_u, lst_j, needed = [], [], [], False
    for el in sorted(set(structure.get_chemical_symbols())):
        conditions = [el in ldauu, el in ldaul, el in ldauj]
        if not any(conditions):
            lst_l.append("-1")
            lst_u.append("0")
            lst_j.append("0")
        elif not all(conditions):
            raise ValueError(
                "All three tags ldauu, ldauj and ldaul have to be specified"
            )
        else:
            needed = True
            lst_l.append(str(ldaul[el]))
            lst_u.append(str(ldauu[el]))
            lst_j.append(str(ldauj[el]))
    if needed:
        incar["LDAU"] = True
        incar["LDAUTYPE"] = interaction_type
        incar["LDAUL"] = " ".join(lst_l)
        incar["LDAUU"] = " ".join(lst_u)
        incar["LDAUJ"] = " ".join(lst_j)
        if ldau_print:
            incar["LDAUPRINT"] = 2


def write_magmoms(incar: Incar, structure: Atoms, sorted_indices=None):
    """
    Write the initial magnetic moments of the structure to the INCAR - equivalent to ``Vasp.write_magmoms()``:
    ``ISPIN = 2``, ``MAGMOM``, ``LORBIT = 10`` (if not set), ``LNONCOLLINEAR`` for vector moments and the checks for
    constrained moments (``M_CONSTR``, ``ISYM = 0``).
    """
    if "initial_magmoms" not in structure.arrays:
        return
    if sorted_indices is None:
        sorted_indices = vasp_sorter(structure)
    spin_constraints = (
        incar["I_CONSTRAINED_M"] is not None and incar["I_CONSTRAINED_M"] >= 1
    )
    if "ISPIN" not in incar:
        incar["ISPIN"] = 2
        if "LORBIT" not in incar or incar["LORBIT"] not in VALID_LORBIT:
            incar["LORBIT"] = 10
    if incar["ISPIN"] == 1:
        return
    magmoms = structure.get_initial_magnetic_moments()
    final_cmd = "   ".join(
        [
            (
                " ".join([str(spinmom) for spinmom in spin])
                if isinstance(spin, (list, np.ndarray))
                else str(spin)
            )
            for spin in magmoms[sorted_indices]
        ]
    )
    if "MAGMOM" not in incar:
        incar["MAGMOM"] = final_cmd
    if any(isinstance(spin, (list, np.ndarray)) for spin in magmoms):
        incar["LNONCOLLINEAR"] = True
        if spin_constraints and "M_CONSTR" not in incar:
            incar["M_CONSTR"] = final_cmd
        if (spin_constraints or "M_CONSTR" in incar) and "ISYM" not in incar:
            incar["ISYM"] = 0
        if spin_constraints and "LAMBDA" not in incar:
            raise ValueError(
                "LAMBDA is not specified but it is necessary for non collinear calculations."
            )
        if spin_constraints and "RWIGS" not in incar:
            raise ValueError(
                "Parameter RWIGS has to be set for spin constraint calculations"
            )
    if spin_constraints and not incar["LNONCOLLINEAR"]:
        raise ValueError(
            "Spin constraints are only avilable for non collinear calculations."
        )
    if "LORBIT" not in incar or incar["LORBIT"] not in VALID_LORBIT:
        incar["LORBIT"] = 10


def get_icharg_value(
    icharg: int | None = None,
    self_consistent_calc: bool | None = None,
    previous_incar=None,
):
    """
    ICHARG for a restart - equivalent to ``Vasp.get_icharg_value()``. ``previous_incar`` is the INCAR of the
    calculation the charge density comes from: if it was non-self-consistent (``ICHARG > 9``), 11 is returned.
    """
    if icharg is None:
        if self_consistent_calc is True:
            return 1
        if self_consistent_calc is False:
            return 11
        if (
            previous_incar is not None
            and previous_incar["ICHARG"] is not None
            and int(previous_incar["ICHARG"]) > 9
        ):
            return 11
        return 1
    if icharg not in [0, 1, 2, 4, 10, 11, 12]:
        raise ValueError(
            f"The value '{icharg}' is not a proper input for 'icharg'. Look at VASP manual."
        )
    return icharg


# Band structure paths


def get_band_structure_path(
    structure: Atoms, num_points: int = 10, path_type: str = "full", prec: float = 1e-5
):
    """
    k-points along a band structure path in Cartesian coordinates (units of 2π/Å) - equivalent to
    ``pyiron_atomistics.dft.waves.bandstructure.Bandstructure(structure).get_path()``, which is used by
    ``Vasp.set_for_band_structure_calc()``. The path depends on the space group (fcc, bcc, sc, ...); like in pyiron,
    unknown space groups (e.g. diamond, 227) fall back to a simplified fcc path.

    Returns:
        list: q-points along the path
    """
    import spglib

    bmat = np.linalg.inv(np.array(structure.cell)).T
    dataset = spglib.get_symmetry_dataset(
        (
            np.array(structure.cell).tolist(),
            structure.get_scaled_positions().tolist(),
            structure.get_atomic_numbers().tolist(),
        ),
        symprec=prec,
    )
    if dataset is None:
        raise ValueError("The space group of the structure could not be determined.")
    space_group = dataset.number
    b1, b2, b3 = bmat
    special_points = {
        "Gamma": [0, 0, 0],
        "G'": b2,
        "L": 0.5 * (b1 + b2 + b3),
        "K": 1.0 / 8.0 * (3 * b1 + 6 * b2 + 3 * b3),
        "U": 1.0 / 8.0 * (2 * b1 + 5 * b2 + 5 * b3),
        "X": 0.5 * (b2 + b3),
        "W": 0.25 * b1 + 0.75 * b2 + 0.5 * b3,
        "X'": 0.5 * (b1 + 2 * b2 + b3),
        "M": 0.5 * (b1 + b2),
        "X1": 0.5 * b1,
        "X2": 0.5 * b2,
    }
    fallback = {
        "very_short": ["L", "Gamma", "X"],
        "full": ["Gamma", "X", "Gamma", "L"],
        "full_20": ["Gamma", "X", "Gamma", "L"],
    }
    path_dict = {
        225: {
            "very_short": ["L", "Gamma", "X"],
            "full": ["Gamma", "X", "U", "L", "Gamma", "K"],
            "full_20": ["G", "X", "U", "Gamma", "L"],
            "full_CM": ["G'", "X'", "K", "Gamma", "L"],
        },
        229: {"very_short": ["Gamma", "X"], "full": ["Gamma", "X", "L", "W", "Gamma"]},
        221: {
            "very_short": ["L", "Gamma", "X1"],
            "full": ["Gamma", "X1", "X", "L", "Gamma"],
        },
        129: {
            "very_short": ["Gamma", "X1"],
            "full": ["Gamma", "X1", "M", "X2", "Gamma"],
        },
        123: {
            "very_short": ["X2", "Gamma", "X1"],
            "full": ["Gamma", "X1", "M", "X2", "Gamma"],
        },
        166: fallback,
        167: fallback,
    }.get(space_group, fallback)
    if space_group == 186:
        special_points = {
            "Gamma": [0, 0, 0],
            "A": 0.5 * b3,
            "L": 0.5 * (b1 + b3),
            "M": 0.5 * b1,
            "K": (1.0 / 0.3) * (b1 + b2),
            "H": (1.0 / 0.3) * (b1 + b2) + 0.5 * b3,
        }
        path_dict = {"full": ["A", "L", "M", "Gamma", "A"]}
    if path_type not in path_dict:
        raise ValueError(
            f"path type {path_type} does not exist for space group {space_group}"
        )
    q_path = np.array([special_points[q] for q in path_dict[path_type]], dtype=float)
    q_vec_list = [q - q_path[i] for i, q in enumerate(q_path[1:])]
    delta_q = sum(np.linalg.norm(d_q) for d_q in q_vec_list) / num_points
    q_point_list = []
    for i, q_vec in enumerate(q_vec_list):
        n_points = int(np.linalg.norm(q_vec) / delta_q + 0.5) + 1
        delta = 2 if i == len(q_vec_list) - 1 else 1
        for j in range(n_points + delta):
            q_point_list.append(q_path[i] + j * q_vec / (n_points + 1))
    return q_point_list


def create_line_mode_structure(
    structure: Atoms,
    with_time_reversal: bool = True,
    recipe: str = "hpkot",
    threshold: float = 1e-07,
    symprec: float = 1e-05,
    angle_tolerance: float = -1.0,
):
    """
    Standardised primitive cell with high symmetry points and path from seekpath - equivalent to
    ``pyiron_atomistics.Atoms.create_line_mode_structure()``. Requires the optional ``seekpath`` package.

    Returns:
        Atoms, dict, dict: structure, high symmetry points, high symmetry path ``{"full": [...]}``
    """
    import seekpath

    numbers = structure.get_atomic_numbers()
    species = list(dict.fromkeys(structure.get_chemical_symbols()))
    indices = [species.index(s) for s in structure.get_chemical_symbols()]
    sp_dict = seekpath.get_path(
        structure=(np.array(structure.cell), structure.get_scaled_positions(), indices),
        with_time_reversal=with_time_reversal,
        recipe=recipe,
        threshold=threshold,
        symprec=symprec,
        angle_tolerance=angle_tolerance,
    )
    _ = numbers
    struc_new = Atoms(
        symbols=[species[i] for i in sp_dict["primitive_types"]],
        scaled_positions=sp_dict["primitive_positions"],
        cell=sp_dict["primitive_lattice"],
        pbc=structure.pbc,
    )
    return struc_new, sp_dict["point_coords"], {"full": sp_dict["path"]}
