from __future__ import annotations

import os

import numpy as np
from ase.atoms import Atoms


def structure_from_parsed_output(
    initial_structure: Atoms,
    parsed_output: dict,
    *,
    wrap: bool = True,
    magnetic_moments: bool = True,
    index: int = -1,
) -> Atoms:
    """
    Final (or any) structure of a VASP calculation - the equivalent of ``job.get_structure()``, which
    ``job.restart()`` uses as the structure of the new calculation.

    Args:
        initial_structure: The input structure of the calculation, used as template.
        parsed_output: The parsed output of :func:`vasp_file_interface_function`.
        wrap: Wrap the atoms into the unit cell like pyiron (``center_coordinates_in_unit_cell()``: relative
            coordinates in [-1e-4, 1 - 1e-4), so atoms at the cell boundary are mapped to 0).
        magnetic_moments: Use the final magnetic moments (if available) as initial magnetic moments, like pyiron.
        index: Ionic step (default: the last one).

    Returns:
        Atoms: the structure
    """
    atoms = initial_structure.copy()
    atoms.set_cell(parsed_output["generic"]["cells"][index], scale_atoms=False)
    atoms.set_positions(parsed_output["generic"]["positions"][index])
    if wrap and any(atoms.pbc):
        eps = 1e-4
        atoms.set_scaled_positions(
            np.mod(atoms.get_scaled_positions(wrap=False) + eps, 1) - eps
        )
    if magnetic_moments:
        magmoms = parsed_output["generic"].get("dft", {}).get("final_magmoms")
        if magmoms is not None and len(magmoms) > 0:
            atoms.set_initial_magnetic_moments(np.array(magmoms[index]))
    return atoms


def read_incar(working_directory: str) -> dict:
    """Read the INCAR of a calculation as dictionary (values as strings)."""
    incar = {}
    with open(os.path.join(working_directory, "INCAR")) as f:
        for raw_line in f:
            line = raw_line.split("#")[0].split("!")[0]
            if "=" in line:
                key, value = line.split("=", 1)
                incar[key.strip()] = value.strip()
    return incar


def check_convergence(
    working_directory: str, parsed_output: dict, incar: dict | None = None
) -> bool:
    """
    Electronic and ionic convergence check - equivalent to ``Vasp.convergence_check()``: the highest band has to be
    empty, static calculations have to be converged electronically, structure optimisations must finish before NSW
    with a converged last ionic step. MD calculations are always considered converged.
    """
    if incar is None:
        incar = read_incar(working_directory)
    occ_matrix = parsed_output.get("electronic_structure", {}).get("occ_matrix")
    if occ_matrix is not None and not np.all(
        np.isclose(np.asarray(occ_matrix)[:, :, -1], 0)
    ):
        return False
    ibrion = int(incar.get("IBRION", 0))
    max_e_steps = int(incar.get("NELM", 60))
    max_i_steps = int(incar.get("NSW", 0))
    if str(incar.get("ALGO", "")).upper() in ["EIGENVAL", "EXACT"] and max_e_steps == 1:
        return True
    scf_energies = parsed_output["generic"]["dft"]["scf_energy_free"]
    e_steps_converged = [len(step) < max_e_steps for step in scf_energies]
    is_md = ibrion == 0 and max_i_steps != 0  # MD: convergence is not checked
    static_converged = max_i_steps == 0 and bool(np.all(e_steps_converged))
    minimize_converged = (
        max_i_steps > 0
        and len(scf_energies) < max_i_steps
        and e_steps_converged[-1]  # only the last ionic step has to be converged
    )
    return is_md or static_converged or minimize_converged
