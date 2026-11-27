from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CalcStaticInput:
    """
    Input parameters for a static (single point) VASP calculation, equivalent to ``Vasp.calc_static()``.

    Attributes:
        electronic_steps: Maximum number of electronic steps (``NELM``).
        algorithm: Electronic minimisation algorithm (``ALGO``), e.g. ``"Fast"`` or ``"Normal"``.
        retain_charge_density: Write the charge density ``CHGCAR`` (``LCHARG``).
        retain_electrostatic_potential: Write the electrostatic potential ``LOCPOT`` (``LVTOT`` and ``LVHAR``).
    """

    electronic_steps: int = 100
    algorithm: str | None = None
    retain_charge_density: bool = False
    retain_electrostatic_potential: bool = False


@dataclass
class CalcMinimizeInput:
    """
    Input parameters for a VASP structure optimisation, equivalent to ``Vasp.calc_minimize()``.

    The field names follow ``lammpsparser.compatibility.data.CalcMinimizeInput`` where the meaning is the same
    (``ionic_energy_tolerance``, ``ionic_force_tolerance``, ``max_iter``, ``pressure``).

    Attributes:
        electronic_steps: Maximum number of electronic steps per ionic step (``NELM``).
        ionic_steps: Maximum number of ionic steps (``NSW``).
        max_iter: If set, overwrites both ``electronic_steps`` and ``ionic_steps``.
        pressure: ``None`` relaxes the positions only (``ISIF = 2``), ``0.0`` relaxes positions and cell
            (``ISIF = 3``). Non-zero pressures are not supported.
        algorithm: Electronic minimisation algorithm (``ALGO``).
        retain_charge_density: Write the charge density ``CHGCAR`` (``LCHARG``).
        retain_electrostatic_potential: Write the electrostatic potential ``LOCPOT`` (``LVTOT`` and ``LVHAR``).
        ionic_energy_tolerance: Energy convergence criterion in eV (positive ``EDIFFG``).
        ionic_force_tolerance: Force convergence criterion in eV/Å (negative ``EDIFFG``), has priority over the
            energy criterion.
        volume_only: Relax only the volume (``ISIF = 7``).
        cell_only: Relax only the cell shape and volume (``ISIF = 6``).
    """

    electronic_steps: int = 400
    ionic_steps: int = 100
    max_iter: int | None = None
    pressure: float | None = None
    algorithm: str | None = None
    retain_charge_density: bool = False
    retain_electrostatic_potential: bool = False
    ionic_energy_tolerance: float | None = None
    ionic_force_tolerance: float | None = None
    volume_only: bool = False
    cell_only: bool = False


@dataclass
class CalcMDInput:
    """
    Input parameters for a VASP molecular dynamics calculation, equivalent to ``Vasp.calc_md()``.

    The field names follow ``lammpsparser.compatibility.data.CalcMDInput`` where the meaning is the same
    (``temperature``, ``n_ionic_steps``, ``time_step``, ``n_print``).

    Attributes:
        temperature: Target temperature in K for a Nosé thermostat (``SMASS = 3``, ``TEBEG``) or a two element
            list ``[T_start, T_end]`` for a linear ramp (``TEBEG``/``TEEND``). ``None`` runs NVE (``SMASS = -3``).
        n_ionic_steps: Number of MD steps (``NSW``).
        n_print: Output interval (``NBLOCK``).
        time_step: Time step in fs (``POTIM``).
        retain_charge_density: Write the charge density ``CHGCAR`` (``LCHARG``).
        retain_electrostatic_potential: Write the electrostatic potential ``LOCPOT`` (``LVTOT`` and ``LVHAR``).
    """

    temperature: float | list | None = None
    n_ionic_steps: int = 1000
    n_print: int = 1
    time_step: float = 1.0
    retain_charge_density: bool = False
    retain_electrostatic_potential: bool = False


@dataclass
class KpointsInput:
    """
    k-point settings, equivalent to ``Vasp.set_kpoints()``.

    Attributes:
        mesh: Size of the mesh for the ``"MP"`` and ``"GC"`` schemes.
        scheme: ``"MP"`` (Monkhorst-Pack), ``"GC"`` (Gamma centred), ``"GP"`` (Gamma point only), ``"Manual"`` or
            ``"Line"``.
        center_shift: Shift of the mesh in relative coordinates.
        symmetry_reduction: ``False`` switches off the symmetry reduction (``ISYM = -1``).
        manual_kpoints: k-points for the ``"Manual"`` scheme.
        weights: Weights of the manual k-points (default 1.0).
        reciprocal: Manual k-points in reciprocal (direct) instead of Cartesian coordinates.
        k_mesh_spacing: k-point spacing in 2π/Å, overwrites ``mesh``.
        n_path: Number of points per segment for the ``"Line"`` scheme.
        path_name: Name of the high symmetry path for the ``"Line"`` scheme.
        high_symmetry_points: Dictionary of high symmetry points (relative coordinates) for the ``"Line"`` scheme.
        high_symmetry_path: Dictionary of paths, each a list of (start, end) labels, for the ``"Line"`` scheme.
    """

    mesh: list | None = None
    scheme: str = "MP"
    center_shift: list | None = None
    symmetry_reduction: bool = True
    manual_kpoints: list | None = None
    weights: list | None = None
    reciprocal: bool = True
    k_mesh_spacing: float | None = None
    n_path: int | None = None
    path_name: str | None = None
    high_symmetry_points: dict | None = None
    high_symmetry_path: dict | None = None


@dataclass
class OccupancySmearingInput:
    """
    Smearing of the occupancies, equivalent to ``Vasp.set_occupancy_smearing()``.

    Attributes:
        smearing: ``"Fermi"``, ``"Gaussian"`` or ``"Methfessel-Paxton"``.
        width: Smearing width in eV (``SIGMA``).
        order: Order of the Methfessel-Paxton smearing.
        ismear: Set ``ISMEAR`` directly (deprecated in pyiron), overwrites ``smearing``.
    """

    smearing: str | None = None
    width: float | None = None
    order: int = 1
    ismear: int | None = None


@dataclass
class ConvergencePrecisionInput:
    """
    Convergence criteria, equivalent to ``Vasp.set_convergence_precision()``.

    Attributes:
        ionic_energy_tolerance: Ionic energy criterion in eV (positive ``EDIFFG``).
        electronic_energy: Electronic energy criterion in eV (``EDIFF``).
        ionic_force_tolerance: Ionic force criterion in eV/Å (negative ``EDIFFG``), has priority.
    """

    ionic_energy_tolerance: float | None = 1.0e-3
    electronic_energy: float | None = 1.0e-7
    ionic_force_tolerance: float | None = 1.0e-2


@dataclass
class MixingParametersInput:
    """
    Density mixing, equivalent to ``Vasp.set_mixing_parameters()``.

    Attributes:
        method: ``"PULAY"`` (``IMIX = 4``) or ``"KERKER"`` (``IMIX = 1``).
        n_pulay_steps: ``MAXMIX``.
        density_mixing_parameter: ``AMIX``.
    """

    method: str | None = None
    n_pulay_steps: int | None = None
    density_mixing_parameter: float | None = None


@dataclass
class ElectricFieldInput:
    """
    Dipole correction / external electric field, equivalent to ``Vasp.set_electric_field()`` and
    ``Vasp.set_dipole_correction()`` (``e_field = 0``).

    Attributes:
        e_field: Electric field in eV/Å (``EFIELD``).
        direction: Direction of the field 0, 1 or 2 (``IDIPOL = direction + 1``).
        dipole_center: Centre of the dipole in relative coordinates (``DIPOL``).
    """

    e_field: float = 0.1
    direction: int = 2
    dipole_center: list | None = None


@dataclass
class SpinConstraintInput:
    """
    Constrained magnetic moments, equivalent to ``Vasp.set_spin_constraint()``.

    Attributes:
        lamb: Penalty factor (``LAMBDA``).
        rwigs_dict: Wigner-Seitz radius per species (``RWIGS``).
        direction: Constrain the direction of the moments.
        norm: Constrain the norm of the moments (only together with ``direction``).
    """

    lamb: float = 10.0
    rwigs_dict: dict = field(default_factory=dict)
    direction: bool = False
    norm: bool = False


@dataclass
class CoulombInteractionsInput:
    """
    On-site Coulomb interaction (DFT+U), equivalent to ``Vasp.set_coulomb_interactions()``. In pyiron the values are
    species tags, here they are dictionaries indexed by the chemical symbol.

    Attributes:
        ldauu: Effective U per species (``LDAUU``).
        ldaul: Angular momentum per species (``LDAUL``), species without entry get ``-1``.
        ldauj: J per species (``LDAUJ``).
        interaction_type: 1 Liechtenstein, 2 Dudarev (``LDAUTYPE``).
        ldau_print: Write the occupancy matrices (``LDAUPRINT = 2``).
    """

    ldauu: dict = field(default_factory=dict)
    ldaul: dict = field(default_factory=dict)
    ldauj: dict = field(default_factory=dict)
    interaction_type: int = 2
    ldau_print: bool = True


@dataclass
class BandStructureInput:
    """
    Non-self-consistent band structure along a path, equivalent to ``Vasp.set_for_band_structure_calc()``.

    Attributes:
        num_points: Number of k-points along the full path.
        structure: Structure used to construct the path (default: the input structure).
        read_charge_density: Read the charge density of a previous calculation (``ICHARG = 11``).
    """

    num_points: int = 20
    structure: object | None = None
    read_charge_density: bool = True


@dataclass
class RestartInput:
    """
    Restart files, equivalent to ``Vasp.restart_from_charge_density()``, ``Vasp.restart_from_wave_functions()``,
    ``Vasp.restart_from_wave_and_charge()``, ``Vasp.append_charge_density()`` and ``Vasp.append_wave_function()``.

    Attributes:
        charge_density_file: Path to a ``CHGCAR`` file or a directory containing one.
        wave_function_file: Path to a ``WAVECAR`` file or a directory containing one.
        icharg: ``ICHARG`` value. Only set if given, or derived from ``self_consistent_calc`` when a charge density
            file is given (see :func:`get_icharg_value`).
        self_consistent_calc: ``True`` gives ``ICHARG = 1``, ``False`` gives ``ICHARG = 11``.
        istart: ``ISTART`` value, only set if given (pyiron's restart functions use ``ISTART = 1``).
        restart_file_list: Additional files to copy into the working directory.
    """

    charge_density_file: str | None = None
    wave_function_file: str | None = None
    icharg: int | None = None
    self_consistent_calc: bool | None = None
    istart: int | None = None
    restart_file_list: list = field(default_factory=list)
