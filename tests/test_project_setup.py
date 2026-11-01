import unittest
import researchpilot

class TestProjectSetup(unittest.TestCase):
    def test_package_import(self):
        self.assertIsNotNone(researchpilot)
        
if __name__=="__main__":
    unittest.main()