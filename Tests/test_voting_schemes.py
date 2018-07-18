import unittest
from Ensemble.Combination.VotingSchemes import majority_voting


class MyTestCase(unittest.TestCase):
    def test_majority_voting(self):
        list1 = [0,1,1,1,0]

        self.assertEqual(majority_voting(list1),1)

        list2 = [0,1]

        self.assertTrue(majority_voting(list2) == 0 or majority_voting(list2) == 1)

        list3 = [0,1,2,3,4,4,5]

        self.assertEqual(majority_voting(list3),4)

if __name__ == '__main__':
    unittest.main()
