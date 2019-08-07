import unittest
import RamseyDecoupling as qml
import qutip as qtp

class TestDistanceStats(unittest.TestCase):
    def test_fidelity(self):
        self.assertEqual(1.0,
            qml.DistanceStats.unitary_diff_fidelity(qtp.identity(2), qtp.identity(2))
            )

if __name__ == '__main__':
    unittest.main()
