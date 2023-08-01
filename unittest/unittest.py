import unittest
import main

class TestSample(unittest.TestCase):
    
    def assertArea(self):
        pass
    
    def assertMeshSplit(self):
        pass
    
    def assertMeshValidity(self, meshes):
        for mesh in meshes:
            assert mesh.IsValid

    def assertMinimumBuildingArea(self):
        pass

if __name__ == '__main__':
    unittest.main()
    