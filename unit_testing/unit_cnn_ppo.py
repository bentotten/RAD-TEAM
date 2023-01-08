import pytest
import unittest
import torch
from dataclasses import field


from algos.multiagent.CNN_PPO import RolloutBuffer, MapsBuffer, Actor, PPO

# https://github.com/cgoldberg/python-unittest-tutorial

class UnitTestModule(unittest.TestCase):
    def test_always_passes(self):
        self.assertEqual(2+2, 4)

class Unit_RolloutBuffer(unittest.TestCase):
    def setUp(self):
        self.buffer = field(default_factory = lambda: RolloutBuffer())
        self.buffer2 = field(default_factory = lambda: RolloutBuffer())

    def tearDown(self):
        del self.buffer
        del self.buffer2

    def test_actions(self):
        sample = torch.tensor(2)
        self.buffer.actions.append(sample)
        self.assertEqual(len(self.buffer.actions), 1)
        self.assertEqual(self.buffer.actions[0], 2)
        self.buffer.actions.append(torch.tensor([3]))
        self.assertFalse(len(self.buffer.actions), (len(self.buffer2.actions))        
        self.buffer.clear()
        self.assertEqual(len(self.buffer.actions), 0)
        
    def test_states(self):
        sample = {
            0: {
                'id': 0, 
                'state': [
                    1.44600000e+03, 3.33333333e-01, 3.33333333e-01, 0.00000000e+00,
                    0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                    0.00000000e+00, 0.00000000e+00, 0.00000000e+00
                    ],
                'reward':-0.27, 
                'done':False, 
                'error':{'out_of_bounds': False, 'out_of_bounds_count': 0}
                }
            }
        self.buffer.states.append(sample)
        self.assertEqual(len(self.buffer.states), 1)
        self.assertEqual(len(self.buffer.states[0][0]), 5)   
        self.buffer.states.append(4)             
        self.buffer.clear()
        self.assertEqual(len(self.buffer.states), 0)
                
    def test_logprobs(self):
        sample = torch.Tensor([-1.4719])
        self.buffer.logprobs.append(sample)
        self.assertEqual(len(self.buffer.logprobs), 1)
        self.assertEqual(self.buffer.logprobs[0], -1.4719) 
        self.buffer.logprobs.append(4)               
        self.buffer.clear()
        self.assertEqual(len(self.buffer.logprobs), 0)
        
    def test_rewards(self):
        self.buffer.rewards.append(-0.3)
        self.assertEqual(len(self.buffer.rewards), 1)
        self.assertEqual(self.buffer.rewards[0], -0.3)    
        self.buffer.rewards.append(4)            
        self.buffer.clear()
        self.assertEqual(len(self.buffer.rewards), 0)
                
    def test_is_terminals(self):
        self.buffer.is_terminals.append(False)
        self.assertEqual(len(self.buffer.is_terminals), 1)
        self.assertEqual(self.buffer.is_terminals[0], False)    
        self.buffer.is_terminals.append(True)            
        self.buffer.clear()
        self.assertEqual(len(self.buffer.is_terminals), 0)
                
    def test_mapstacks(self):
        sample = torch.zeros(1, 4, 20, 20)
        self.buffer.mapstacks.append(sample)
        self.assertEqual(len(self.buffer.mapstacks), 1)
        self.assertEqual(len(self.buffer.mapstacks[0].size()), len(sample.size()))     
        self.buffer.mapstacks.append(4)           
        self.buffer.clear()
        self.assertEqual(len(self.buffer.mapstacks), 0)

    def test_readings(self):
        key = (0.3333333333333333, 0.3333333333333333)
        sample = [1446.0]
        self.buffer.readings[key] = sample
        self.assertEqual(len(self.buffer.readings), 1)
        self.assertEqual(self.buffer.readings[key][0], 1446.0)     
        self.buffer.clear()
        self.assertEqual(len(self.buffer.readings), 0)
                
    def test_full_clear(self):
        self.buffer.actions.append(3)


if __name__ == '__main__':
    unittest.main()