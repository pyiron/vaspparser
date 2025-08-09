# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under theterms of "New BSD License", see the LICENSE file.

import os
import unittest
import numpy as np
from pyiron_vasp.vasp.procar import Procar


class TestProcarParser(unittest.TestCase):
    def setUp(self):
        self.parser = Procar()
        self.file_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "../static/vasp_test_files/PROCAR_for_test",
        )

    def test_from_file(self):
        self.assertIsNone(self.parser._check_if_spin_polarized("dummy"))
        es_obj = self.parser.from_file(self.file_path)
        self.assertEqual(len(es_obj.kpoints), 1)
        self.assertEqual(len(es_obj.kpoints[0].bands[0]), 2)
        self.assertEqual(es_obj.kpoints[0].bands[0][0].eigenvalue, 1.0)
        self.assertEqual(es_obj.kpoints[0].bands[0][0].occupancy, 1.0)
        self.assertTrue(
            np.allclose(
                es_obj.kpoints[0].bands[0][0].resolved_dos_matrix,
                np.array([[0.1, 0.1, 0.1]]),
            )
        )
        self.assertTrue(
            np.allclose(
                es_obj.kpoints[0].bands[0][0].atom_resolved_dos, np.array([0.3])
            )
        )
        self.assertTrue(
            np.allclose(
                es_obj.kpoints[0].bands[0][0].orbital_resolved_dos,
                np.array([0.1, 0.1, 0.1]),
            )
        )
        self.assertEqual(es_obj.kpoints[0].bands[0][1].eigenvalue, 2.0)
        self.assertEqual(es_obj.kpoints[0].bands[0][1].occupancy, 0.0)
        self.assertTrue(
            np.allclose(
                es_obj.kpoints[0].bands[0][1].resolved_dos_matrix,
                np.array([[0.2, 0.2, 0.2]]),
            )
        )
        self.assertTrue(
            np.allclose(
                es_obj.kpoints[0].bands[0][1].atom_resolved_dos, np.array([0.6])
            )
        )
        self.assertTrue(
            np.allclose(
                es_obj.kpoints[0].bands[0][1].orbital_resolved_dos,
                np.array([0.2, 0.2, 0.2]),
            )
        )


if __name__ == "__main__":
    unittest.main()
