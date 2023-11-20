import unittest
import pyiron_vasp


class TestVersion(unittest.TestCase):
    def test_version(self):
        version = pyiron_vasp.__version__
        print(version)
        self.assertTrue(version.startswith('0'))
