import unittest
from algos.multiagent.CNN_PPO import RolloutBuffer, MapsBuffer, Actor, PPO

# https://github.com/cgoldberg/python-unittest-tutorial

class UnitTestModule(unittest.TestCase):
    def test_dummy(self):
        print('\n:: test_dummy ::')
        self.assertEqual(2+2,4)


class Unit_RolloutBuffer(unittest.TestCase):
    def setUp(self):
        self.buffer = RolloutBuffer()

    def tearDown(self):
        del self.buffer

    def test_actions(self):
        self.buffer.actions.append(3)
        self.assertEqual(len(self.buffer.actions), 1)
        self.assertEqual(self.buffer.actions[0], 3)
        self.buffer.actions.append(4)
        self.buffer.clear()
        self.assertEqual(len(self.buffer.actions), 0)
        
    def test_states(self):
        self.buffer.states.append(3)
        self.assertEqual(len(self.buffer.states), 1)
        self.assertEqual(self.buffer.states[0], 3)   
        self.buffer.states.append(4)             
        self.buffer.clear()
        self.assertEqual(len(self.buffer.states), 0)
                
    def test_logprobs(self):
        self.buffer.logprobs.append(3)
        self.assertEqual(len(self.buffer.logprobs), 1)
        self.assertEqual(self.buffer.logprobs[0], 3) 
        self.buffer.logprobs.append(4)               
        self.buffer.clear()
        self.assertEqual(len(self.buffer.logprobs), 0)
        
    def test_rewards(self):
        self.buffer.rewards.append(3)
        self.assertEqual(len(self.buffer.rewards), 1)
        self.assertEqual(self.buffer.rewards[0], 3)    
        self.buffer.rewards.append(4)            
        self.buffer.clear()
        self.assertEqual(len(self.buffer.rewards), 0)
                
    def test_is_terminals(self):
        self.buffer.is_terminals.append(3)
        self.assertEqual(len(self.buffer.is_terminals), 1)
        self.assertEqual(self.buffer.is_terminals[0], 3)    
        self.buffer.is_terminals.append(4)            
        self.buffer.clear()
        self.assertEqual(len(self.buffer.is_terminals), 0)
                
    def test_mapstacks(self):
        self.buffer.mapstacks.append(3)
        self.assertEqual(len(self.buffer.mapstacks), 1)
        self.assertEqual(self.buffer.mapstacks[0], 3)     
        self.buffer.mapstacks.append(4)           
        self.buffer.clear()
        self.assertEqual(len(self.buffer.mapstacks), 0)
                
    def test_full_clear(self):
        self.buffer.actions.append(3)


if __name__ == '__main__':
    unittest.main()