from __future__ import annotations

import os
import shutil
import subprocess
import warnings
from dataclasses import asdict, is_dataclass
from typing import Any

import numpy as np
from ase.atoms import Atoms
from ase.constraints import FixAtoms, FixCartesian

from vaspparser.compatibility.calculate import (
    calc_md,
    calc_minimize,
    calc_static,
    get_band_structure_path,
    get_icharg_value,
    set_algorithm,
    set_convergence_precision,
    set_coulomb_interactions,
    set_dipole_correction,
    set_electric_field,
    set_empty_states,
    set_fft_mesh,
    set_mixing_parameters,
    set_occupancy_smearing,
    set_rwigs,
    set_spin_constraint,
    set_write_charge_density,
    set_write_electrostatic_potential,
    set_write_resolved_dos,
    set_write_wave_funct,
    write_magmoms,
)
from vaspparser.compatibility.data import (
    BandStructureInput,
    CalcMDInput,
    CalcMinimizeInput,
    CalcStaticInput,
    RestartInput,
)
from vaspparser.compatibility.input import (
    Incar,
    Kpoints,
    get_nelect,
    get_potcar_content,
    get_potcar_files,
    get_resource_path,
    set_kpoints,
)
from vaspparser.vasp.output import parse_vasp_output
from vaspparser.vasp.structure import get_poscar_content, vasp_sorter


def _to_dict(value) -> dict:
    """Accept ``None``, a dictionary or a dataclass instance and return a (copied) dictionary."""
    if value is None:
        return {}
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if isinstance(value, dict):
        return dict(value)
    raise TypeError(f"Expected a dict or a dataclass instance, not {type(value)}.")


def _get_calc_mode_and_kwargs(
    calc_mode: str, calc_kwargs: Any, calc_dataclass: Any
) -> tuple[str, dict]:
    if calc_dataclass is not None:
        if isinstance(calc_dataclass, CalcStaticInput):
            return "static", asdict(calc_dataclass)
        if isinstance(calc_dataclass, CalcMinimizeInput):
            return "minimize", asdict(calc_dataclass)
        if isinstance(calc_dataclass, CalcMDInput):
            return "md", asdict(calc_dataclass)
        # duck typing for the dataclasses of lammpsparser.compatibility.data
        name = type(calc_dataclass).__name__
        if name in ["CalcMDInput", "CalcMinimizeInput"] and is_dataclass(
            calc_dataclass
        ):
            mode = "md" if name == "CalcMDInput" else "minimize"
            allowed = (
                CalcMDInput.__dataclass_fields__
                if mode == "md"
                else CalcMinimizeInput.__dataclass_fields__
            )
            instance: Any = calc_dataclass
            defaults = asdict(type(instance)())
            kwargs, ignored = {}, []
            for key, value in asdict(instance).items():
                if mode == "minimize" and key == "max_iter":
                    # LAMMPS iterations are ionic steps, NELM stays unchanged
                    kwargs["ionic_steps"] = value
                elif mode == "minimize" and key == "ionic_energy_tolerance":
                    # 0.0 disables the energy criterion in LAMMPS
                    kwargs[key] = None if value == 0.0 else value
                elif key in allowed:
                    kwargs[key] = value
                elif value != defaults[key]:
                    ignored.append(key)
            if len(ignored) > 0:
                warnings.warn(
                    f"The parameters {ignored} of {name} are not used for VASP.",
                    stacklevel=3,
                )
            return mode, kwargs
        raise TypeError(
            "calc_dataclass must be an instance of CalcStaticInput, CalcMinimizeInput or CalcMDInput"
        )
    return calc_mode, _to_dict(calc_kwargs)


def _prepare_structure(structure: Atoms) -> Atoms:
    """Convert FixAtoms constraints to FixCartesian, which vaspparser writes as selective dynamics."""
    if len(structure.constraints) == 0:
        return structure
    structure = structure.copy()
    constraints = []
    for constraint in structure.constraints:
        if isinstance(constraint, FixAtoms):
            constraints.append(
                FixCartesian(a=constraint.get_indices(), mask=(True, True, True))
            )
        else:
            constraints.append(constraint)
    structure.set_constraint(constraints)
    return structure


def _resolve_restart_file(path: str | None, file_name: str) -> str | None:
    if path is None:
        return None
    path = os.path.abspath(os.path.expanduser(path))
    if os.path.isdir(path):
        path = os.path.join(path, file_name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"The restart file {path} does not exist.")
    return path


def get_vasp_input(
    structure: Atoms,
    *,
    job_name: str = "vasp_job",
    plane_wave_cutoff: float | None = None,
    exchange_correlation_functional: str | None = None,
    potentials: dict | None = None,
    spin_constraints: int | None = None,
    write_electrostatic_potential: bool | None = None,
    write_charge_density: bool | None = None,
    write_wave_funct: bool | None = None,
    write_resolved_dos: bool | int | None = None,
    sorted_indices: list | np.ndarray | None = None,
    coulomb_interactions_kwargs=None,
    algorithm_kwargs: dict | None = None,
    calc_mode: str = "static",
    calc_kwargs=None,
    calc_dataclass=None,
    band_structure_calc_kwargs=None,
    convergence_precision_kwargs=None,
    dipole_correction_kwargs: dict | None = None,
    electric_field_kwargs=None,
    occupancy_smearing_kwargs=None,
    fft_mesh_kwargs: dict | None = None,
    mixing_parameters_kwargs=None,
    n_empty_states: int | None = None,
    rwigs_kwargs: dict | None = None,
    spin_constraint_kwargs=None,
    kpoints_kwargs=None,
    restart_kwargs=None,
    incar_file: str | list | dict | None = None,
    kpoints_file: str | list | dict | None = None,
    resource_path: str | None = None,
) -> dict:
    """
    Generate the VASP input files without executing VASP. The arguments are the same as for
    :func:`vasp_file_interface_function`.

    Returns:
        dict: ``{"files_to_create": {file_name: content}, "files_to_copy": {file_name: source_path}}``
    """
    structure = _prepare_structure(structure)
    resource_path = get_resource_path(resource_path=resource_path)
    calc_mode, calc_kwargs = _get_calc_mode_and_kwargs(
        calc_mode, calc_kwargs, calc_dataclass
    )
    xc = "PBE"
    if exchange_correlation_functional is not None:
        if exchange_correlation_functional in ["PBE", "pbe", "GGA", "gga"]:
            xc = "PBE"
        elif exchange_correlation_functional in ["LDA", "lda"]:
            xc = "LDA"
        else:
            raise ValueError("exchange_correlation_functional has to be PBE or LDA")
    if sorted_indices is None:
        sorted_indices = vasp_sorter(structure)

    incar, kpoints = Incar(job_name=job_name), Kpoints()

    # The order follows pyiron_atomistics.vasp.vasp.vasp_function()
    if plane_wave_cutoff is not None:
        incar["ENCUT"] = plane_wave_cutoff
    if spin_constraints is not None:
        incar["I_CONSTRAINED_M"] = spin_constraints
    if write_electrostatic_potential is not None:
        set_write_electrostatic_potential(
            incar=incar, value=write_electrostatic_potential
        )
    if write_charge_density is not None:
        set_write_charge_density(incar=incar, value=write_charge_density)
    if write_wave_funct is not None:
        set_write_wave_funct(incar=incar, value=write_wave_funct)
    if write_resolved_dos is not None:
        set_write_resolved_dos(incar=incar, value=write_resolved_dos)
    coulomb = _to_dict(coulomb_interactions_kwargs)
    if len(coulomb) > 0:
        set_coulomb_interactions(incar=incar, structure=structure, **coulomb)
    if algorithm_kwargs:
        set_algorithm(incar=incar, **algorithm_kwargs)
    band_structure = _to_dict(band_structure_calc_kwargs)
    if len(band_structure) > 0:
        band_structure = {**asdict(BandStructureInput()), **band_structure}
        if band_structure["read_charge_density"]:
            incar["ICHARG"] = 11
        bs_structure = (
            band_structure["structure"]
            if band_structure["structure"] is not None
            else structure
        )
        q_point_list = get_band_structure_path(
            structure=bs_structure, num_points=band_structure["num_points"]
        )
        set_kpoints(
            incar=incar,
            kpoints=kpoints,
            structure=structure,
            scheme="Manual",
            symmetry_reduction=False,
            manual_kpoints=np.array(q_point_list),
            weights=None,
            reciprocal=False,
        )
    convergence = _to_dict(convergence_precision_kwargs)
    if len(convergence) > 0:
        set_convergence_precision(incar=incar, **convergence)
    if dipole_correction_kwargs:
        set_dipole_correction(incar=incar, **dipole_correction_kwargs)
    efield = _to_dict(electric_field_kwargs)
    if len(efield) > 0:
        set_electric_field(incar=incar, **efield)
    smearing = _to_dict(occupancy_smearing_kwargs)
    if len(smearing) > 0:
        set_occupancy_smearing(incar=incar, **smearing)
    if fft_mesh_kwargs:
        set_fft_mesh(incar=incar, **fft_mesh_kwargs)
    mixing = _to_dict(mixing_parameters_kwargs)
    if len(mixing) > 0:
        set_mixing_parameters(incar=incar, **mixing)
    if n_empty_states is not None:
        set_empty_states(
            incar=incar,
            n_elect=get_nelect(structure=structure, xc=xc, resource_path=resource_path),
            n_empty_states=n_empty_states,
        )
    if rwigs_kwargs:
        set_rwigs(incar=incar, structure=structure, rwigs_dict=rwigs_kwargs)
    spin_constraint = _to_dict(spin_constraint_kwargs)
    if len(spin_constraint) > 0:
        set_spin_constraint(incar=incar, structure=structure, **spin_constraint)
    kpoints_dict = _to_dict(kpoints_kwargs)
    if len(kpoints_dict) > 0:
        set_kpoints(incar=incar, kpoints=kpoints, structure=structure, **kpoints_dict)
    if calc_mode == "static":
        calc_static(incar=incar, **calc_kwargs)
    elif calc_mode == "md":
        calc_md(incar=incar, **calc_kwargs)
    elif calc_mode == "minimize":
        calc_minimize(incar=incar, **calc_kwargs)
    else:
        raise ValueError(
            f"calc_mode must be one of: static, md or minimize, not {calc_mode}"
        )

    # Restart files - equivalent to Vasp.restart_from_charge_density() / restart_from_wave_functions() / ...
    restart = _to_dict(restart_kwargs)
    files_to_copy = {}
    for file_name in restart.get("restart_file_list", []):
        files_to_copy[os.path.basename(file_name)] = os.path.abspath(file_name)
    wave_function_file = _resolve_restart_file(
        restart.get("wave_function_file"), "WAVECAR"
    )
    charge_density_file = _resolve_restart_file(
        restart.get("charge_density_file"), "CHGCAR"
    )
    if wave_function_file is not None:
        files_to_copy["WAVECAR"] = wave_function_file
    if restart.get("istart") is not None:
        incar["ISTART"] = restart["istart"]
    if charge_density_file is not None:
        files_to_copy["CHGCAR"] = charge_density_file
    if (
        restart.get("icharg") is not None
        or restart.get("self_consistent_calc") is not None
    ):
        incar["ICHARG"] = get_icharg_value(
            icharg=restart.get("icharg"),
            self_consistent_calc=restart.get("self_consistent_calc"),
        )

    # Direct modifications of the input files
    if isinstance(incar_file, dict):
        for key, value in incar_file.items():
            incar[key] = value
    elif incar_file is not None:
        incar.load_string(incar_file)
    if isinstance(kpoints_file, dict):
        for line, value in kpoints_file.items():
            kpoints[line] = value
    elif kpoints_file is not None:
        kpoints.load_string(kpoints_file)

    # Done by pyiron when the input is written: magnetic moments and coulomb interactions
    write_magmoms(incar=incar, structure=structure, sorted_indices=sorted_indices)

    files_to_create = {
        "INCAR": "".join(incar.get_string_lst()),
        "POTCAR": get_potcar_content(
            get_potcar_files(
                structure=structure,
                xc=xc,
                potentials=potentials,
                resource_path=resource_path,
            )
        ),
        "POSCAR": "".join(
            get_poscar_content(structure=structure, write_species=True, cartesian=True)
        ),
    }
    if "KSPACING" in incar:
        warnings.warn(
            "'KSPACING' found in INCAR, no KPOINTS file written", stacklevel=2
        )
    else:
        files_to_create["KPOINTS"] = "".join(kpoints.get_string_lst())
    return {"files_to_create": files_to_create, "files_to_copy": files_to_copy}


def write_vasp_input(working_directory: str, input_dict: dict):
    """Write the files of :func:`get_vasp_input` to the working directory and copy the restart files."""
    os.makedirs(working_directory, exist_ok=True)
    for file_name, content in input_dict["files_to_create"].items():
        with open(os.path.join(working_directory, file_name), "w") as f:
            f.writelines(content)
    for file_name, source in input_dict["files_to_copy"].items():
        shutil.copy(source, os.path.join(working_directory, file_name))


def collect_errors(working_directory: str, file_name: str = "error.out") -> list[str]:
    """
    Check the VASP output for the EDDRMM and ZBRENT errors, like ``Vasp.collect_errors()``.

    Returns:
        list: warning messages
    """
    messages: list[str] = []
    path = os.path.join(working_directory, file_name)
    if not os.path.exists(path):
        return messages
    with open(path) as f:
        lines = f.readlines()
    eddrmm = [
        i
        for i, line in enumerate(lines)
        if "WARNING in EDDRMM: call to ZHEGV failed, returncode =" in line
    ]
    if len(eddrmm) > 0:
        step = len([line for line in lines[: eddrmm[0]] if "E0=" in line])
        messages.append(
            f"EDDRMM warnings occured {len(eddrmm)} times, first in ionic step {step}."
        )
    if any("ZBRENT: fatal error in bracketing" in line for line in lines):
        messages.append(
            "'ZBRENT: fatal error in bracketing' occured. Please check VASP manual for details."
        )
    return messages


def vasp_file_interface_function(
    working_directory: str,
    structure: Atoms,
    *,
    job_name: str = "vasp_job",
    plane_wave_cutoff: float | None = None,
    exchange_correlation_functional: str | None = None,
    potentials: dict | None = None,
    spin_constraints: int | None = None,
    write_electrostatic_potential: bool | None = None,
    write_charge_density: bool | None = None,
    write_wave_funct: bool | None = None,
    write_resolved_dos: bool | int | None = None,
    sorted_indices: list | np.ndarray | None = None,
    eddrmm_handling: str = "ignore",
    coulomb_interactions_kwargs=None,
    algorithm_kwargs: dict | None = None,
    calc_mode: str = "static",
    calc_kwargs=None,
    calc_dataclass=None,
    band_structure_calc_kwargs=None,
    convergence_precision_kwargs=None,
    dipole_correction_kwargs: dict | None = None,
    electric_field_kwargs=None,
    occupancy_smearing_kwargs=None,
    fft_mesh_kwargs: dict | None = None,
    mixing_parameters_kwargs=None,
    n_empty_states: int | None = None,
    rwigs_kwargs: dict | None = None,
    spin_constraint_kwargs=None,
    kpoints_kwargs=None,
    restart_kwargs=None,
    server_kwargs: dict | None = None,
    vasp_command: str | None = None,
    executable_path: str | None = None,
    incar_file: str | list | dict | None = None,
    kpoints_file: str | list | dict | None = None,
    resource_path: str | None = None,
    accept_crash: bool = False,
):
    """
    Run a VASP calculation with the same input options as the VASP job object of ``pyiron_atomistics``, without
    depending on ``pyiron_base`` or ``pyiron_atomistics``. The arguments follow
    ``pyiron_atomistics.vasp.vasp.vasp_function()``; every ``*_kwargs`` argument accepts a dictionary with the
    arguments of the corresponding method of the job object or one of the dataclasses from
    :mod:`vaspparser.compatibility.data`.

    Examples:

    >>> import os
    >>> from ase.build import bulk
    >>> from vaspparser.compatibility.file import vasp_file_interface_function
    >>>
    >>> shell_output, parsed_output, job_crashed = vasp_file_interface_function(
    ...     working_directory=os.path.abspath("vasp_working_directory"),
    ...     structure=bulk("Al", a=4.04),
    ...     plane_wave_cutoff=300,
    ...     kpoints_kwargs={"mesh": [8, 8, 8]},
    ...     occupancy_smearing_kwargs={"smearing": "methfessel_paxton", "width": 0.1},
    ...     calc_mode="static",
    ...     vasp_command="mpirun -np 2 vasp_std",
    ... )
    >>> parsed_output["generic"]["energy_tot"]

    Args:
        working_directory (str): directory in which VASP is executed
        structure (Atoms): ase.atoms.Atoms - atomistic structure, initial magnetic moments (scalar or vector) are
            written to MAGMOM, FixCartesian/FixAtoms constraints as selective dynamics
        job_name (str): written to the SYSTEM tag (pyiron uses the job name)
        plane_wave_cutoff (float): ENCUT in eV - ``job.set_encut()``
        exchange_correlation_functional (str): "PBE" (default) or "LDA" - selects the POTCAR set
        potentials (dict): non-default POTCARs ``{element: name or absolute path}`` - ``job.potential.Fe = "Fe_pv"``
        spin_constraints (int): I_CONSTRAINED_M - ``job.spin_constraints``
        write_electrostatic_potential (bool): LVTOT/LVHAR - ``job.write_electrostatic_potential``
        write_charge_density (bool): LCHARG - ``job.write_charge_density`` (note that like in pyiron the calc_*
            functions reset LCHARG, use ``calc_kwargs={"retain_charge_density": True}``)
        write_wave_funct (bool): LWAVE - ``job.write_wave_funct``
        write_resolved_dos (bool/int): LORBIT - ``job.write_resolved_dos``
        sorted_indices (list): order of the atoms in the POSCAR (default: species by species)
        eddrmm_handling (str): "ignore" or "warn" - ``job.set_eddrmm_handling()``
        coulomb_interactions_kwargs (dict/CoulombInteractionsInput): DFT+U - ``job.set_coulomb_interactions()``
        algorithm_kwargs (dict): "algorithm", "ialgo" - ``job.set_algorithm()``
        calc_mode (str): "static", "minimize" or "md"
        calc_kwargs (dict): arguments of ``job.calc_static()``, ``job.calc_minimize()`` or ``job.calc_md()``
        calc_dataclass (CalcStaticInput/CalcMinimizeInput/CalcMDInput): alternative to calc_mode and calc_kwargs,
            the CalcMDInput and CalcMinimizeInput dataclasses of lammpsparser are accepted as well
        band_structure_calc_kwargs (dict/BandStructureInput): ``job.set_for_band_structure_calc()``
        convergence_precision_kwargs (dict/ConvergencePrecisionInput): ``job.set_convergence_precision()``
        dipole_correction_kwargs (dict): "direction", "dipole_center" - ``job.set_dipole_correction()``
        electric_field_kwargs (dict/ElectricFieldInput): ``job.set_electric_field()``
        occupancy_smearing_kwargs (dict/OccupancySmearingInput): ``job.set_occupancy_smearing()``
        fft_mesh_kwargs (dict): "nx", "ny", "nz" - ``job.set_fft_mesh()``
        mixing_parameters_kwargs (dict/MixingParametersInput): ``job.set_mixing_parameters()``
        n_empty_states (int): ``job.set_empty_states()``
        rwigs_kwargs (dict): ``{element: radius}`` - ``job.set_rwigs()``
        spin_constraint_kwargs (dict/SpinConstraintInput): ``job.set_spin_constraint()``
        kpoints_kwargs (dict/KpointsInput): ``job.set_kpoints()``
        restart_kwargs (dict/RestartInput): restart files and ICHARG/ISTART - ``job.restart_from_charge_density()``,
            ``job.restart_from_wave_functions()``, ``job.append_charge_density()``, ... see :func:`restart_from_charge_density`
        server_kwargs (dict): "cores", "threads" - only used together with executable_path
        vasp_command (str): shell command to execute VASP, e.g. "mpirun -np 2 vasp_std". Defaults to the
            ASE_VASP_COMMAND or VASP_COMMAND environment variables, or "vasp_std"
        executable_path (str): pyiron-style run script, called as ``<executable_path> <cores> <threads>``
        incar_file (str/list/dict): direct modifications of the INCAR - ``job.input.incar[key] = value``
        kpoints_file (str/list/dict): direct modifications of the KPOINTS file - ``job.input.kpoints``
        resource_path (str): directory with the potential tables and POTCAR directories
        accept_crash (bool): return ``job_crashed=True`` instead of raising an error if VASP fails

    Returns:
        str, dict, bool: Tuple consisting of the shell output (str), the parsed output (dict) and a boolean flag if
                         the execution raised an accepted error.
    """
    if eddrmm_handling not in ["ignore", "warn", "restart"]:
        raise ValueError("eddrmm_handling has to be 'ignore', 'warn' or 'restart'")
    structure = _prepare_structure(structure)
    if sorted_indices is None:
        sorted_indices = vasp_sorter(structure)
    input_dict = get_vasp_input(
        structure=structure,
        job_name=job_name,
        plane_wave_cutoff=plane_wave_cutoff,
        exchange_correlation_functional=exchange_correlation_functional,
        potentials=potentials,
        spin_constraints=spin_constraints,
        write_electrostatic_potential=write_electrostatic_potential,
        write_charge_density=write_charge_density,
        write_wave_funct=write_wave_funct,
        write_resolved_dos=write_resolved_dos,
        sorted_indices=sorted_indices,
        coulomb_interactions_kwargs=coulomb_interactions_kwargs,
        algorithm_kwargs=algorithm_kwargs,
        calc_mode=calc_mode,
        calc_kwargs=calc_kwargs,
        calc_dataclass=calc_dataclass,
        band_structure_calc_kwargs=band_structure_calc_kwargs,
        convergence_precision_kwargs=convergence_precision_kwargs,
        dipole_correction_kwargs=dipole_correction_kwargs,
        electric_field_kwargs=electric_field_kwargs,
        occupancy_smearing_kwargs=occupancy_smearing_kwargs,
        fft_mesh_kwargs=fft_mesh_kwargs,
        mixing_parameters_kwargs=mixing_parameters_kwargs,
        n_empty_states=n_empty_states,
        rwigs_kwargs=rwigs_kwargs,
        spin_constraint_kwargs=spin_constraint_kwargs,
        kpoints_kwargs=kpoints_kwargs,
        restart_kwargs=restart_kwargs,
        incar_file=incar_file,
        kpoints_file=kpoints_file,
        resource_path=resource_path,
    )
    write_vasp_input(working_directory=working_directory, input_dict=input_dict)

    server_kwargs = {} if server_kwargs is None else server_kwargs
    if executable_path is not None:
        command = f"{executable_path} {server_kwargs.get('cores', 1)} {server_kwargs.get('threads', 1)}"
    elif vasp_command is not None:
        command = vasp_command
    else:
        command = os.getenv("ASE_VASP_COMMAND", os.getenv("VASP_COMMAND", "vasp_std"))
    environment = os.environ.copy()
    environment.update(
        {
            "PYIRON_CORES": str(server_kwargs.get("cores", 1)),
            "PYIRON_THREADS": str(server_kwargs.get("threads", 1)),
        }
    )
    job_crashed = False
    try:
        shell_output = subprocess.run(
            command,
            cwd=working_directory,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=True,
            env=environment,
        ).stdout
    except subprocess.CalledProcessError as error:
        shell_output = error.output
        if not accept_crash:
            with open(os.path.join(working_directory, "error.out"), "w") as f:
                f.write(shell_output)
            raise
        job_crashed = True
    with open(os.path.join(working_directory, "error.out"), "w") as f:
        f.write(shell_output)
    if eddrmm_handling != "ignore":
        for message in collect_errors(working_directory=working_directory):
            warnings.warn(message, stacklevel=2)
    parsed_output = None
    if not job_crashed:
        parsed_output = parse_vasp_output(
            working_directory=working_directory,
            structure=structure,
            sorted_indices=sorted_indices,
        )
    return shell_output, parsed_output, job_crashed


# Restarts - functional equivalents of the restart methods of the pyiron job object


def _read_incar(working_directory: str) -> Incar | None:
    path = os.path.join(working_directory, "INCAR")
    if not os.path.exists(path):
        return None
    incar = Incar()
    with open(path) as f:
        incar.load_string(f.read())
    return incar


def restart_from_charge_density(
    working_directory: str,
    icharg: int | None = None,
    self_consistent_calc: bool | None = None,
) -> RestartInput:
    """Copy the CHGCAR of a previous calculation and set ICHARG - ``job.restart_from_charge_density()``."""
    return RestartInput(
        charge_density_file=os.path.join(working_directory, "CHGCAR"),
        icharg=get_icharg_value(
            icharg=icharg,
            self_consistent_calc=self_consistent_calc,
            previous_incar=_read_incar(working_directory),
        ),
    )


def restart_from_wave_functions(
    working_directory: str, istart: int = 1
) -> RestartInput:
    """Copy the WAVECAR of a previous calculation and set ISTART - ``job.restart_from_wave_functions()``."""
    return RestartInput(
        wave_function_file=os.path.join(working_directory, "WAVECAR"), istart=istart
    )


def restart_from_wave_and_charge(
    working_directory: str,
    icharg: int | None = None,
    self_consistent_calc: bool | None = None,
    istart: int = 1,
) -> RestartInput:
    """Copy CHGCAR and WAVECAR and set ISTART and ICHARG - ``job.restart_from_wave_and_charge()``."""
    return RestartInput(
        charge_density_file=os.path.join(working_directory, "CHGCAR"),
        wave_function_file=os.path.join(working_directory, "WAVECAR"),
        istart=istart,
        icharg=get_icharg_value(
            icharg=icharg,
            self_consistent_calc=self_consistent_calc,
            previous_incar=_read_incar(working_directory),
        ),
    )


def restart_for_band_structure_calculations(working_directory: str) -> RestartInput:
    """Copy the CHGCAR and set ICHARG = 11 - ``job.restart_for_band_structure_calculations()``."""
    return restart_from_charge_density(working_directory=working_directory, icharg=11)
