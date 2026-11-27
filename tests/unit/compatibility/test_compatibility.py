import importlib.util
import os
import subprocess
import sys
import tempfile
import unittest
import warnings
from dataclasses import dataclass

import numpy as np
from ase.build import bulk
from ase.constraints import FixAtoms

from vaspparser.compatibility.calculate import (
    calc_md,
    calc_minimize,
    calc_static,
    get_icharg_value,
    set_convergence_precision,
    set_coulomb_interactions,
    set_electric_field,
    set_empty_states,
    set_fft_mesh,
    set_mixing_parameters,
    set_occupancy_smearing,
    set_spin_constraint,
    write_magmoms,
)
from vaspparser.compatibility.data import (
    CalcMDInput,
    CalcMinimizeInput,
    CalcStaticInput,
    CoulombInteractionsInput,
    KpointsInput,
    OccupancySmearingInput,
    RestartInput,
)
from vaspparser.compatibility.file import (
    collect_errors,
    get_vasp_input,
    restart_for_band_structure_calculations,
    restart_from_charge_density,
    restart_from_wave_and_charge,
    restart_from_wave_functions,
    vasp_file_interface_function,
)
from vaspparser.compatibility.input import (
    Incar,
    Kpoints,
    get_k_mesh_by_density,
    get_nelect,
    get_potcar_files,
    get_resource_path,
    set_kpoints,
)
from vaspparser.compatibility.output import (
    check_convergence,
    read_incar,
    structure_from_parsed_output,
)

STATIC = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../static"))
RESOURCE_PATH = os.path.join(STATIC, "vasp")
FULL_JOB = os.path.join(STATIC, "vasp_test_files", "full_job_sample")


def fe_bcc():
    structure = bulk("Fe", "bcc", a=2.8, cubic=True)
    structure.set_initial_magnetic_moments(None)
    return structure


def incar_dict(text):
    incar = Incar()
    incar.load_string(text)
    return incar.to_dict()


class TestIncarKpoints(unittest.TestCase):
    def test_incar_format(self):
        incar = Incar(job_name="test")
        incar["ENCUT"] = 400
        incar["LCHARG"] = True
        incar["EDIFF"] = 1e-6
        lines = incar.get_string_lst()
        self.assertEqual(lines[0], "SYSTEM=test #jobname\n")
        self.assertIn("LREAL=.FALSE.\n", lines)
        self.assertIn("LCHARG=.TRUE.\n", lines)
        self.assertIn("EDIFF=1e-06\n", lines)
        self.assertIsNone(incar["ISPIN"])
        incar["PREC"] = "Normal"
        self.assertEqual(incar.get_string_lst()[1], "PREC=Normal\n")
        del incar["ENCUT"]
        self.assertNotIn("ENCUT", incar)

    def test_incar_load_string(self):
        incar = Incar()
        incar.load_string("ISPIN = 2 # spin\nLWAVE = .TRUE.\nKGAMMA=T\n")
        self.assertEqual(incar["ISPIN"], "2")
        self.assertTrue(incar["LWAVE"])
        self.assertTrue(incar["KGAMMA"])

    def test_kpoints_schemes(self):
        structure = bulk("Al", a=4.0)
        incar, kpoints = Incar(), Kpoints()
        self.assertEqual(kpoints.get_string_lst()[2], "Monkhorst_Pack\n")
        set_kpoints(
            incar,
            kpoints,
            structure,
            mesh=[3, 4, 5],
            scheme="GC",
            center_shift=[0.5, 0, 0],
        )
        self.assertEqual(
            kpoints.get_string_lst()[2:], ["Gamma centered\n", "3 4 5\n", "0.5 0 0\n"]
        )
        set_kpoints(incar, kpoints, structure, scheme="GP")
        self.assertEqual(kpoints.get_string_lst()[2:4], ["Gamma Point\n", "1 1 1\n"])
        self.assertIsNone(incar["ISYM"])
        set_kpoints(
            incar, Kpoints(), structure, mesh=[2, 2, 2], symmetry_reduction=False
        )
        self.assertEqual(incar["ISYM"], -1)
        kpoints = Kpoints()
        set_kpoints(
            incar,
            kpoints,
            structure,
            scheme="Manual",
            manual_kpoints=[[0, 0, 0], [0.5, 0, 0]],
            weights=[1, 3],
        )
        self.assertEqual(
            "".join(kpoints.get_string_lst()),
            "Kpoints file generated with pyiron_atomistics\n2\nReciprocal\n0 0 0 1\n0.5 0 0 3\n",
        )
        kpoints = Kpoints()
        set_kpoints(
            incar,
            kpoints,
            structure,
            scheme="Line",
            n_path=10,
            path_name="p",
            high_symmetry_points={"G": [0, 0, 0], "X": [0.5, 0, 0.5]},
            high_symmetry_path={"p": [("G", "X")]},
        )
        self.assertEqual(
            kpoints.get_string_lst()[1:],
            ["10\n", "Line\n", "rec\n", "0 0 0 !G\n", "0.5 0 0.5 !X\n"],
        )
        with self.assertRaises(ValueError):
            set_kpoints(
                incar, Kpoints(), structure, scheme="Line", n_path=10, path_name="p"
            )
        with self.assertRaises(ValueError):
            set_kpoints(incar, Kpoints(), structure, scheme="unknown")
        with self.assertRaises(ValueError):
            set_kpoints(incar, Kpoints(), structure, mesh=[0, 1, 1])

    def test_k_mesh_by_density(self):
        self.assertEqual(
            get_k_mesh_by_density(np.eye(3) * 4.0, k_mesh_spacing=0.5), [3, 3, 3]
        )
        self.assertEqual(
            get_k_mesh_by_density(np.eye(3) * 100.0, k_mesh_spacing=0.5), [1, 1, 1]
        )


class TestPotentials(unittest.TestCase):
    def test_resource_path(self):
        expected = os.path.join(RESOURCE_PATH, "potentials")
        self.assertEqual(get_resource_path(RESOURCE_PATH), expected)
        self.assertEqual(get_resource_path(expected), expected)
        with self.assertRaises(ValueError):
            get_resource_path(STATIC + "/does_not_exist")

    def test_potcar_files(self):
        files = get_potcar_files(fe_bcc(), resource_path=RESOURCE_PATH)
        self.assertTrue(files[0].endswith(os.path.join("potpaw_PBE", "Fe", "POTCAR")))
        files = get_potcar_files(
            bulk("Al", a=4.0), potentials={"Al": "Al_GW"}, resource_path=RESOURCE_PATH
        )
        self.assertTrue(files[0].endswith(os.path.join("Al_GW", "POTCAR")))
        files = get_potcar_files(
            fe_bcc(), potentials={"Fe": files[0]}, resource_path=RESOURCE_PATH
        )
        self.assertTrue(files[0].endswith(os.path.join("Al_GW", "POTCAR")))
        with self.assertRaises(
            ValueError
        ):  # the LDA potentials are not part of the test resources
            get_potcar_files(fe_bcc(), xc="LDA", resource_path=RESOURCE_PATH)
        self.assertEqual(get_nelect(fe_bcc(), resource_path=RESOURCE_PATH), 16.0)


class TestCalculate(unittest.TestCase):
    def test_calc_static(self):
        incar = Incar()
        calc_static(
            incar,
            retain_charge_density=True,
            retain_electrostatic_potential=True,
            algorithm="Normal",
        )
        self.assertEqual((incar["IBRION"], incar["NELM"], incar["NSW"]), (-1, 100, 0))
        self.assertTrue(incar["LCHARG"] and incar["LVTOT"] and incar["LVHAR"])
        self.assertEqual(incar["ALGO"], "Normal")

    def test_calc_minimize(self):
        for kwargs, isif in [
            ({}, 2),
            ({"pressure": 0.0}, 3),
            ({"volume_only": True}, 7),
            ({"cell_only": True}, 6),
        ]:
            incar = Incar()
            calc_minimize(incar, **kwargs)
            self.assertEqual(incar["ISIF"], isif)
            self.assertEqual(incar["NELM"], 400)
            self.assertEqual(incar["EDIFFG"], 0.01 if isif in (6, 7) else -0.01)
        incar = Incar()
        calc_minimize(incar, max_iter=5, ionic_energy_tolerance=1e-4)
        self.assertEqual((incar["NELM"], incar["NSW"], incar["EDIFFG"]), (5, 5, 1e-4))
        with self.assertRaises(ValueError):
            calc_minimize(Incar(), pressure=1.0)

    def test_calc_md(self):
        incar = Incar()
        calc_md(
            incar, temperature=[300, 600], n_ionic_steps=10, n_print=2, time_step=2.0
        )
        self.assertEqual(
            (incar["SMASS"], incar["TEBEG"], incar["TEEND"]), (3, 300, 600)
        )
        self.assertEqual(
            (incar["NSW"], incar["NBLOCK"], incar["POTIM"], incar["ISYM"]),
            (10, 2, 2.0, 0),
        )
        incar = Incar()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            calc_md(incar, pressure=0.0)
        self.assertEqual(incar["SMASS"], -3)
        self.assertEqual(len(w), 1)

    def test_setters(self):
        incar = Incar()
        set_convergence_precision(incar, electronic_energy=1e-5)
        self.assertEqual((incar["EDIFF"], incar["EDIFFG"]), (1e-5, -0.01))
        for smearing, ismear in [
            ("fermi", -1),
            ("gaussian", 0),
            ("methfessel_paxton", 1),
        ]:
            set_occupancy_smearing(incar, smearing=smearing, width=0.1)
            self.assertEqual(incar["ISMEAR"], ismear)
        with self.assertRaises(ValueError):
            set_occupancy_smearing(incar, smearing="unknown")
        set_electric_field(incar, e_field=0.2, direction=1, dipole_center=[0, 0.5, 0])
        self.assertEqual(
            (incar["IDIPOL"], incar["EFIELD"], incar["DIPOL"], incar["ISYM"]),
            (2, 0.2, "0 0.5 0", 0),
        )
        set_fft_mesh(incar, nx=10, nz=12)
        self.assertEqual((incar["NGXF"], incar["NGYF"], incar["NGZF"]), (10, None, 12))
        set_mixing_parameters(
            incar, method="kerker", n_pulay_steps=10, density_mixing_parameter=0.2
        )
        self.assertEqual((incar["IMIX"], incar["MAXMIX"], incar["AMIX"]), (1, 10, 0.2))
        set_empty_states(incar, n_elect=16, n_empty_states=4)
        self.assertEqual(incar["NBANDS"], 12)
        with self.assertRaises(ValueError):
            set_empty_states(incar, n_elect=16, n_empty_states=-1)

    def test_magnetic_moments(self):
        structure = fe_bcc()
        structure.set_initial_magnetic_moments([2.0, -2.0])
        incar = Incar()
        write_magmoms(incar, structure)
        self.assertEqual(
            (incar["ISPIN"], incar["MAGMOM"], incar["LORBIT"]), (2, "2.0   -2.0", 0)
        )
        structure.set_initial_magnetic_moments(None)
        structure.set_initial_magnetic_moments([[0, 0, 2], [0, 0, -2]])
        incar = Incar()
        set_spin_constraint(
            incar, structure, lamb=10.0, rwigs_dict={"Fe": 1.2}, direction=True
        )
        write_magmoms(incar, structure)
        self.assertTrue(incar["LNONCOLLINEAR"])
        self.assertEqual(incar["M_CONSTR"], "0.0 0.0 2.0   0.0 0.0 -2.0")
        self.assertEqual(
            (incar["I_CONSTRAINED_M"], incar["RWIGS"], incar["ISYM"]), (1, "1.2", 0)
        )
        with self.assertRaises(ValueError):
            set_spin_constraint(
                Incar(), structure, lamb=10.0, rwigs_dict={"Fe": 1.2}, norm=True
            )
        incar = Incar(parameters={"I_CONSTRAINED_M": 1})
        with self.assertRaises(ValueError):  # LAMBDA missing
            write_magmoms(incar, structure)
        incar = Incar()
        write_magmoms(incar, fe_bcc())
        self.assertIsNone(incar["ISPIN"])

    def test_coulomb_interactions(self):
        incar = Incar()
        set_coulomb_interactions(
            incar, fe_bcc(), ldauu={"Fe": 2.0}, ldaul={"Fe": 2}, ldauj={"Fe": 0.0}
        )
        self.assertEqual(
            (
                incar["LDAU"],
                incar["LDAUTYPE"],
                incar["LDAUL"],
                incar["LDAUU"],
                incar["LDAUJ"],
                incar["LDAUPRINT"],
            ),
            (True, 2, "2", "2.0", "0.0", 2),
        )
        with self.assertRaises(ValueError):
            set_coulomb_interactions(Incar(), fe_bcc(), ldauu={"Fe": 2.0})
        incar = Incar()
        set_coulomb_interactions(incar, fe_bcc())
        self.assertIsNone(incar["LDAU"])

    def test_icharg(self):
        self.assertEqual(get_icharg_value(), 1)
        self.assertEqual(get_icharg_value(self_consistent_calc=False), 11)
        self.assertEqual(
            get_icharg_value(previous_incar=Incar(parameters={"ICHARG": 11})), 11
        )
        self.assertEqual(get_icharg_value(icharg=2), 2)
        with self.assertRaises(ValueError):
            get_icharg_value(icharg=3)

    @unittest.skipIf(
        importlib.util.find_spec("spglib") is None, "spglib is not installed"
    )
    def test_band_structure_path(self):
        from vaspparser.compatibility.calculate import get_band_structure_path

        q_points = get_band_structure_path(bulk("Al", a=4.0), num_points=20)
        self.assertTrue(np.allclose(q_points[0], 0))
        self.assertGreater(len(q_points), 20)

    @unittest.skipIf(
        importlib.util.find_spec("seekpath") is None, "seekpath is not installed"
    )
    def test_line_mode_structure(self):
        from vaspparser.compatibility.calculate import create_line_mode_structure

        structure, points, path = create_line_mode_structure(bulk("Si", a=5.43))
        self.assertEqual(len(structure), 2)
        self.assertIn("GAMMA", points)
        self.assertIn("full", path)


class TestVaspInput(unittest.TestCase):
    def test_get_vasp_input(self):
        structure = fe_bcc()
        structure.set_initial_magnetic_moments([2.0, 2.0])
        structure.set_constraint(FixAtoms(indices=[0]))
        input_dict = get_vasp_input(
            structure=structure,
            job_name="fe",
            plane_wave_cutoff=300,
            kpoints_kwargs=KpointsInput(mesh=[4, 4, 4], scheme="GC"),
            occupancy_smearing_kwargs=OccupancySmearingInput(smearing="mp", width=0.1),
            convergence_precision_kwargs={"electronic_energy": 1e-6},
            coulomb_interactions_kwargs=CoulombInteractionsInput(
                ldauu={"Fe": 2.0}, ldaul={"Fe": 2}, ldauj={"Fe": 0.0}
            ),
            calc_dataclass=CalcMinimizeInput(ionic_force_tolerance=0.02),
            n_empty_states=2,
            write_wave_funct=True,
            incar_file={"NCORE": 2},
            resource_path=RESOURCE_PATH,
        )
        files = input_dict["files_to_create"]
        incar = incar_dict(files["INCAR"])
        self.assertEqual(incar["SYSTEM"], "fe")
        self.assertEqual(incar["ISIF"], "2")
        self.assertEqual(incar["EDIFFG"], "-0.02")
        self.assertEqual(incar["NBANDS"], "10")
        self.assertEqual(incar["MAGMOM"], "2.0   2.0")
        self.assertEqual(incar["NCORE"], "2")
        self.assertTrue(incar["LWAVE"] and incar["LDAU"])
        self.assertIn("Gamma centered", files["KPOINTS"])
        self.assertIn("Selective dynamics", files["POSCAR"])
        self.assertIn("F F F", files["POSCAR"])
        self.assertTrue(files["POTCAR"].startswith("# This is not a real POTCAR"))
        self.assertEqual(input_dict["files_to_copy"], {})

    def test_kspacing_and_overwrites(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            files = get_vasp_input(
                structure=bulk("Al", a=4.0),
                potentials={"Al": "Al_GW"},
                incar_file="KSPACING = 0.3\n",
                resource_path=RESOURCE_PATH,
            )["files_to_create"]
        self.assertNotIn("KPOINTS", files)
        self.assertTrue(any("KSPACING" in str(m.message) for m in w))
        files = get_vasp_input(
            structure=fe_bcc(),
            calc_mode="md",
            calc_kwargs={"temperature": 300, "n_ionic_steps": 5},
            kpoints_file={3: "2 2 2"},
            resource_path=RESOURCE_PATH,
        )["files_to_create"]
        self.assertEqual(files["KPOINTS"].splitlines()[3], "2 2 2")
        self.assertEqual(incar_dict(files["INCAR"])["SMASS"], "3")
        with self.assertRaises(ValueError):
            get_vasp_input(
                structure=fe_bcc(), calc_mode="unknown", resource_path=RESOURCE_PATH
            )
        with self.assertRaises(ValueError):
            get_vasp_input(
                structure=fe_bcc(),
                exchange_correlation_functional="HSE",
                resource_path=RESOURCE_PATH,
            )

    def test_lammps_dataclasses(self):
        @dataclass
        class CalcMDInput:  # same name and fields as lammpsparser.compatibility.data.CalcMDInput
            temperature: float | None = None
            n_ionic_steps: int = 1
            time_step: float = 1.0
            n_print: int = 100
            seed: int = 80996

        @dataclass
        class CalcMinimizeInput:  # same name and fields as lammpsparser.compatibility.data.CalcMinimizeInput
            ionic_energy_tolerance: float = 0.0
            ionic_force_tolerance: float = 1e-4
            max_iter: int = 100000
            style: str = "cg"

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            files = get_vasp_input(
                structure=fe_bcc(),
                calc_dataclass=CalcMDInput(temperature=500, n_ionic_steps=3, seed=1),
                resource_path=RESOURCE_PATH,
            )["files_to_create"]
        incar = incar_dict(files["INCAR"])
        self.assertEqual(
            (incar["TEBEG"], incar["NSW"], incar["NBLOCK"]), ("500", "3", "100")
        )
        self.assertTrue(any("seed" in str(m.message) for m in w))
        files = get_vasp_input(
            structure=fe_bcc(),
            calc_dataclass=CalcMinimizeInput(),
            resource_path=RESOURCE_PATH,
        )["files_to_create"]
        incar = incar_dict(files["INCAR"])
        self.assertEqual(
            (incar["NSW"], incar["NELM"], incar["EDIFFG"]), ("100000", "400", "-0.0001")
        )
        with self.assertRaises(TypeError):
            get_vasp_input(
                structure=fe_bcc(),
                calc_dataclass=KpointsInput(),
                resource_path=RESOURCE_PATH,
            )
        with self.assertRaises(TypeError):
            get_vasp_input(
                structure=fe_bcc(),
                kpoints_kwargs=[1, 2, 3],
                resource_path=RESOURCE_PATH,
            )

    def test_restart(self):
        with tempfile.TemporaryDirectory() as directory:
            for file_name in ["CHGCAR", "WAVECAR"]:
                with open(os.path.join(directory, file_name), "w") as f:
                    f.write("data")
            with open(os.path.join(directory, "INCAR"), "w") as f:
                f.write("ICHARG = 11\n")
            self.assertEqual(restart_from_charge_density(directory).icharg, 11)
            self.assertEqual(
                restart_from_charge_density(
                    directory, self_consistent_calc=True
                ).icharg,
                1,
            )
            self.assertEqual(
                restart_for_band_structure_calculations(directory).icharg, 11
            )
            self.assertEqual(restart_from_wave_functions(directory).istart, 1)
            restart = restart_from_wave_and_charge(directory, icharg=1)
            input_dict = get_vasp_input(
                structure=fe_bcc(), restart_kwargs=restart, resource_path=RESOURCE_PATH
            )
            incar = incar_dict(input_dict["files_to_create"]["INCAR"])
            self.assertEqual((incar["ISTART"], incar["ICHARG"]), ("1", "1"))
            self.assertEqual(sorted(input_dict["files_to_copy"]), ["CHGCAR", "WAVECAR"])
            input_dict = get_vasp_input(
                structure=fe_bcc(),
                restart_kwargs=RestartInput(
                    charge_density_file=directory,
                    restart_file_list=[directory + "/INCAR"],
                ),
                resource_path=RESOURCE_PATH,
            )
            self.assertNotIn(
                "ICHARG", incar_dict(input_dict["files_to_create"]["INCAR"])
            )
            self.assertEqual(sorted(input_dict["files_to_copy"]), ["CHGCAR", "INCAR"])
            with self.assertRaises(FileNotFoundError):
                get_vasp_input(
                    structure=fe_bcc(),
                    restart_kwargs={"wave_function_file": directory + "/missing"},
                    resource_path=RESOURCE_PATH,
                )


class TestVaspFileInterfaceFunction(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.script = os.path.join(self.tmp.name, "fake_vasp.py")
        with open(self.script, "w") as f:
            f.write(
                "import shutil\n"
                f"for name in ['vasprun.xml', 'OUTCAR', 'OSZICAR', 'CONTCAR']:\n"
                f"    shutil.copy(r'{os.path.join(FULL_JOB, '')}' + name, name)\n"
                "print('fake vasp finished')\n"
            )
        self.command = f'"{sys.executable}" "{self.script}"'

    def tearDown(self):
        self.tmp.cleanup()

    def test_run(self):
        working_directory = os.path.join(self.tmp.name, "calc")
        structure = fe_bcc()
        shell_output, parsed_output, job_crashed = vasp_file_interface_function(
            working_directory=working_directory,
            structure=structure,
            calc_dataclass=CalcStaticInput(),
            vasp_command=self.command,
            resource_path=RESOURCE_PATH,
            eddrmm_handling="warn",
        )
        self.assertIn("fake vasp finished", shell_output)
        self.assertFalse(job_crashed)
        self.assertEqual(len(parsed_output["generic"]["energy_tot"]), 1)
        self.assertTrue(os.path.exists(os.path.join(working_directory, "error.out")))
        self.assertEqual(read_incar(working_directory)["IBRION"], "-1")
        final = structure_from_parsed_output(structure, parsed_output)
        self.assertEqual(len(final), 2)
        self.assertTrue(check_convergence(working_directory, parsed_output))

    def test_executable_path_and_crash(self):
        working_directory = os.path.join(self.tmp.name, "crash")
        failing = f'"{sys.executable}" -c "import sys; sys.exit(1)"'
        shell_output, parsed_output, job_crashed = vasp_file_interface_function(
            working_directory=working_directory,
            structure=fe_bcc(),
            vasp_command=failing,
            resource_path=RESOURCE_PATH,
            accept_crash=True,
        )
        self.assertTrue(job_crashed)
        self.assertIsNone(parsed_output)
        with self.assertRaises(subprocess.CalledProcessError):
            vasp_file_interface_function(
                working_directory=working_directory,
                structure=fe_bcc(),
                vasp_command=failing,
                resource_path=RESOURCE_PATH,
            )
        with self.assertRaises(ValueError):
            vasp_file_interface_function(
                working_directory=working_directory,
                structure=fe_bcc(),
                eddrmm_handling="unknown",
                resource_path=RESOURCE_PATH,
            )

    def test_collect_errors(self):
        with open(os.path.join(self.tmp.name, "error.out"), "w") as f:
            f.write(
                " E0= 1\n WARNING in EDDRMM: call to ZHEGV failed, returncode = 6\n"
                " ZBRENT: fatal error in bracketing\n"
            )
        messages = collect_errors(self.tmp.name)
        self.assertEqual(len(messages), 2)
        self.assertIn("first in ionic step 1", messages[0])
        self.assertEqual(collect_errors(os.path.join(self.tmp.name, "missing")), [])


if __name__ == "__main__":
    unittest.main()
