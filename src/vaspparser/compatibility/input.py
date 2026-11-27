from __future__ import annotations

import ast
import csv
import os
import warnings

import numpy as np
from ase.atoms import Atoms

DEFAULT_INCAR = {
    "SYSTEM": "vasp_job",
    "PREC": "Accurate",
    "ALGO": "Fast",
    "LREAL": False,
    "LWAVE": False,
    "LORBIT": 0,
}

DEFAULT_KPOINTS = [
    "Kpoints file generated with pyiron_atomistics",
    "0",
    "Monkhorst_Pack",
    "4 4 4",
    "0 0 0",
]


class Incar:
    """
    Ordered collection of INCAR tags. New tags are appended, existing tags keep their position - the same behaviour
    and the same file format (``KEY=VALUE``, Fortran booleans) as the ``Incar`` class of ``pyiron_atomistics``.
    Missing tags return ``None``.
    """

    _bool_dict = {True: ".TRUE.", False: ".FALSE."}

    def __init__(self, job_name: str = "vasp_job", parameters: dict | None = None):
        self._data = dict(DEFAULT_INCAR)
        self._data["SYSTEM"] = job_name
        self._comments = {"SYSTEM": "jobname"}
        if parameters is not None:
            self.update(parameters)

    def __getitem__(self, key):
        return self._data.get(key)

    def __setitem__(self, key, value):
        self._data[key] = _str_to_bool(value)

    def __delitem__(self, key):
        del self._data[key]
        self._comments.pop(key, None)

    def __contains__(self, key):
        return key in self._data

    def keys(self):
        return self._data.keys()

    def items(self):
        return self._data.items()

    def update(self, parameters: dict):
        for key, value in parameters.items():
            self[key] = value

    def load_string(self, input_str: str | list):
        """Read INCAR content (string or list of lines) and add/overwrite the tags."""
        if isinstance(input_str, str):
            input_str = input_str.splitlines()
        for raw_line in input_str:
            line = raw_line.split("#")[0].split("!")[0]
            if "=" in line:
                key, value = line.split("=", 1)
                self[key.strip()] = value.strip()

    def get_string_lst(self) -> list[str]:
        lines = []
        for key, value in self._data.items():
            if isinstance(value, (bool, np.bool_)):
                value_str = self._bool_dict[bool(value)]
            else:
                value_str = str(value)
            if key in self._comments:
                lines.append(f"{key}={value_str} #{self._comments[key]}\n")
            else:
                lines.append(f"{key}={value_str}\n")
        return lines

    def to_dict(self) -> dict:
        return dict(self._data)


def _str_to_bool(value):
    if isinstance(value, str):
        for true_false, options in {
            True: [".TRUE.", ".True.", ".true.", "True", "T"],
            False: [".FALSE.", ".False.", ".false.", "False", "F"],
        }.items():
            if value.strip() in options:
                return true_false
    return value


class Kpoints:
    """
    Line based KPOINTS file, using the same defaults and the same line manipulations as the ``Kpoints`` class of
    ``pyiron_atomistics``.
    """

    def __init__(self, lines: list | None = None):
        self._lines = (
            list(DEFAULT_KPOINTS) if lines is None else [str(line) for line in lines]
        )

    def set_value(self, line: int, val):
        if line < len(self._lines):
            self._lines[line] = str(val)
        elif line == len(self._lines):
            self._lines.append(str(val))
        else:
            raise IndexError(
                f"Line {line} cannot be set in a KPOINTS file with {len(self._lines)} lines."
            )

    def __getitem__(self, line: int):
        return self._lines[line]

    def __setitem__(self, line: int, value):
        self.set_value(line=line, val=value)

    def load_string(self, input_str: str | list):
        if isinstance(input_str, str):
            input_str = input_str.splitlines()
        self._lines = [line.rstrip("\n") for line in input_str]

    def set_kpoints_file(
        self, method=None, size_of_mesh=None, shift=None, n_path=None, path=None
    ):
        if n_path is not None:
            if path is None:
                raise ValueError("trace have to be defined")
            self.set_value(line=1, val=n_path)
            self.set_value(line=3, val="rec")
            for i, t in enumerate(path):
                val = " ".join([str(ii) for ii in t[0]])
                val = val + " !" + t[1]
                self.set_value(line=i + 4, val=val)
        if method is not None:
            self.set_value(line=2, val=method)
        if size_of_mesh is not None:
            self.set_value(line=3, val=" ".join([str(i) for i in size_of_mesh]))
        if shift is not None:
            self.set_value(line=4, val=" ".join([str(i) for i in shift]))

    def get_string_lst(self) -> list[str]:
        return [line + "\n" for line in self._lines]


def get_k_mesh_by_density(cell, k_mesh_spacing: float = 0.5) -> list[int]:
    """
    k-point mesh for a given k-point spacing in units of 2π/Å (same as ``pyiron_atomistics``).
    """
    cell = np.asarray(cell)
    omega = np.linalg.det(cell)
    l1, l2, l3 = cell
    g1 = 2 * np.pi / omega * np.cross(l2, l3)
    g2 = 2 * np.pi / omega * np.cross(l3, l1)
    g3 = 2 * np.pi / omega * np.cross(l1, l2)
    kmesh = np.rint(
        np.array([np.linalg.norm(g) for g in [g1, g2, g3]]) / k_mesh_spacing
    )
    kmesh[kmesh < 1] = 1
    return [int(k) for k in kmesh]


def set_kpoints(
    incar: Incar,
    kpoints: Kpoints,
    structure: Atoms,
    mesh=None,
    scheme: str = "MP",
    center_shift=None,
    symmetry_reduction: bool = True,
    manual_kpoints=None,
    weights=None,
    reciprocal: bool = True,
    k_mesh_spacing: float | None = None,
    n_path: int | None = None,
    path_name: str | None = None,
    high_symmetry_points: dict | None = None,
    high_symmetry_path: dict | None = None,
):
    """
    Set up the KPOINTS file - equivalent to ``Vasp.set_kpoints()``. For the ``"Line"`` scheme the high symmetry
    points and paths, which pyiron stores on the structure object, are given explicitly (see
    :func:`create_line_mode_structure`).
    """
    if k_mesh_spacing is not None:
        if mesh is not None:
            warnings.warn("mesh value is overwritten by k_mesh_spacing", stacklevel=2)
        mesh = get_k_mesh_by_density(cell=structure.cell, k_mesh_spacing=k_mesh_spacing)
    if mesh is not None and np.min(mesh) <= 0:
        raise ValueError("mesh values must be larger than 0")
    if center_shift is not None and (
        np.min(center_shift) < 0 or np.max(center_shift) > 1
    ):
        warnings.warn("center_shift is given in relative coordinates", stacklevel=2)
    if not symmetry_reduction:
        incar["ISYM"] = -1
    if scheme not in ["MP", "GC", "GP", "Line", "Manual"]:
        raise ValueError(
            f"scheme has to be one of MP, GC, GP, Line or Manual, not {scheme}."
        )
    if scheme == "MP":
        if mesh is None:
            mesh = [int(val) for val in kpoints[3].split()]
        kpoints.set_kpoints_file(size_of_mesh=mesh, shift=center_shift)
    elif scheme == "GC":
        if mesh is None:
            mesh = [int(val) for val in kpoints[3].split()]
        kpoints.set_kpoints_file(
            size_of_mesh=mesh, shift=center_shift, method="Gamma centered"
        )
    elif scheme == "GP":
        kpoints.set_kpoints_file(size_of_mesh=[1, 1, 1], method="Gamma Point")
    elif scheme == "Line":
        if n_path is None:
            raise ValueError("n_path has to be defined")
        if high_symmetry_points is None or high_symmetry_path is None:
            raise ValueError(
                "high_symmetry_points and high_symmetry_path have to be defined"
            )
        if path_name not in high_symmetry_path:
            raise ValueError("path_name is not a valid key of high_symmetry_path")
        k_trace = []
        for start, end in high_symmetry_path[path_name]:
            k_trace.append((high_symmetry_points[start], start))
            k_trace.append((high_symmetry_points[end], end))
        kpoints.set_kpoints_file(method="Line", n_path=n_path, path=k_trace)
    elif scheme == "Manual":
        if manual_kpoints is None:
            raise ValueError(
                "For the manual mode, the kpoints list should be specified"
            )
        if weights is not None and len(manual_kpoints) != len(weights):
            raise ValueError("manual_kpoints and weights have to have the same length")
        kpoints.set_value(line=1, val=str(len(manual_kpoints)))
        kpoints.set_value(line=2, val="Reciprocal" if reciprocal else "Cartesian")
        for i, kpt in enumerate(manual_kpoints):
            wt = weights[i] if weights is not None else 1.0
            kpoints.set_value(
                line=3 + i,
                val=" ".join([str(kpt[0]), str(kpt[1]), str(kpt[2]), str(wt)]),
            )


# Pseudopotentials


def get_resource_path(resource_path: str | None = None) -> str:
    """
    Locate the directory with the pyiron-style potential tables ``potentials_vasp.csv``,
    ``potentials_vasp_pbe_default.csv`` and ``potentials_vasp_lda_default.csv`` next to the ``potpaw_PBE`` and
    ``potpaw`` directories.

    The directory can be given directly or as a pyiron resource directory (containing ``vasp/potentials``).
    Without an argument, the ``VASPPARSER_RESOURCE_PATH`` environment variable and the ``share/pyiron`` directory
    of the active conda environment are checked.
    """
    candidates = []
    if resource_path is not None:
        candidates.append(os.path.abspath(os.path.expanduser(resource_path)))
    else:
        if "VASPPARSER_RESOURCE_PATH" in os.environ:
            candidates.append(
                os.path.abspath(
                    os.path.expanduser(os.environ["VASPPARSER_RESOURCE_PATH"])
                )
            )
        for key in ["CONDA_PREFIX", "CONDA_DIR"]:
            if key in os.environ:
                candidates.append(os.path.join(os.environ[key], "share", "pyiron"))
    for path in candidates:
        for directory in [
            path,
            os.path.join(path, "vasp", "potentials"),
            os.path.join(path, "potentials"),
        ]:
            if os.path.exists(os.path.join(directory, "potentials_vasp.csv")):
                return directory
    raise ValueError(
        f"No VASP potential tables (potentials_vasp.csv) found in {candidates}. Set resource_path or the "
        "VASPPARSER_RESOURCE_PATH environment variable."
    )


def _read_csv(file_name: str) -> list[dict]:
    with open(file_name) as f:
        return list(csv.DictReader(f))


def _parse_list(value: str) -> list:
    value = value.strip()
    if value.startswith("["):
        return ast.literal_eval(value)
    return [value]


def get_potential_table(xc: str = "PBE", resource_path: str | None = None) -> dict:
    """
    Default potential for every element for the given exchange correlation functional.

    Returns:
        dict: ``{element: {"Name": ..., "Filename": ..., "n_elect": ..., "ENMAX": ...}}``
    """
    directory = get_resource_path(resource_path=resource_path)
    if xc in ["PBE", "GGA"]:
        default_file, model = "potentials_vasp_pbe_default.csv", "gga-pbe"
    elif xc == "LDA":
        default_file, model = "potentials_vasp_lda_default.csv", "lda"
    else:
        raise ValueError(
            'The exchange correlation functional has to be set and it can either be "LDA" or "PBE"'
        )
    potentials = {
        row["Name"]: row
        for row in _read_csv(os.path.join(directory, "potentials_vasp.csv"))
        if row["Model"] == model
    }
    table = {}
    for row in _read_csv(os.path.join(directory, default_file)):
        element, name = row[list(row.keys())[0]], row["Name"]
        if name in potentials:
            pot = potentials[name]
            table[element] = {
                "Name": name,
                "Filename": _parse_list(pot["Filename"])[0],
                "n_elect": float(pot["n_elect"]),
                "ENMAX": (
                    float(pot["ENMAX"]) if pot.get("ENMAX") not in (None, "") else None
                ),
            }
    return table


def get_potcar_files(
    structure: Atoms,
    xc: str = "PBE",
    potentials: dict | None = None,
    resource_path: str | None = None,
) -> list[str]:
    """
    Paths of the POTCAR files for the species of the structure in VASP (alphabetical) order - equivalent to the
    ``Potcar`` class of ``pyiron_atomistics``.

    Args:
        structure: Atomistic structure
        xc: ``"PBE"`` (or ``"GGA"``) or ``"LDA"``
        potentials: Non-default potentials, ``{element: name}`` (e.g. ``{"Fe": "Fe_pv"}``) or
            ``{element: absolute_path_to_potcar}`` - equivalent to ``job.potential.Fe = "Fe_pv"``.
        resource_path: Directory with the potential tables, see :func:`get_resource_path`.
    """
    directory = get_resource_path(resource_path=resource_path)
    table = get_potential_table(xc=xc, resource_path=directory)
    potentials = {} if potentials is None else potentials
    path_lst = []
    for element in sorted(set(structure.get_chemical_symbols())):
        if element in potentials and potentials[element] is not None:
            new_element = potentials[element]
            if os.path.isabs(new_element):
                path_lst.append(new_element)
                continue
            path_list = table[element]["Filename"].split("/")
            path_list[-2] = new_element
            file_name = "/".join(path_list)
        else:
            file_name = table[element]["Filename"]
        path = os.path.join(directory, file_name)
        if not os.path.isfile(path):
            raise ValueError(f"The POTCAR file {path} for {element} does not exist.")
        path_lst.append(path)
    return path_lst


def get_nelect(
    structure: Atoms, xc: str = "PBE", resource_path: str | None = None
) -> float:
    """Number of valence electrons based on the default potentials (as ``Vasp.get_nelect()``)."""
    table = get_potential_table(xc=xc, resource_path=resource_path)
    symbols = structure.get_chemical_symbols()
    return sum(table[el]["n_elect"] * symbols.count(el) for el in sorted(set(symbols)))


def get_potcar_content(potcar_files: list[str]) -> str:
    content = []
    for file_name in potcar_files:
        with open(file_name) as f:
            content.append(f.read())
    return "".join(content)
