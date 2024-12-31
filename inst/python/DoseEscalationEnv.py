import gymnasium as gym
import numpy as np
from gymnasium.spaces import Discrete, Box, Space
from typing import List

class DoseEscalationEnv(gym.Env):
    """Custom environment for reinforcement leaning"""
    
    J: int
    N_total: int
    N_cohort: int
    current_dose: int
    prob_scenarios: List[List[float]]
    MTD_scenarios: List[List[int]]
    weight_scenarios: List[float]
    num_scenarios: int
    action_space: Space
    observation_space: Space        
    
    def __init__(self, config):
        self.J        = config["J"]
        self.N_total  = config["N_total"]
        self.N_cohort = config["N_cohort"]
        self.prob_scenarios    = config["prob_scenarios"]
        self.MTD_scenarios     = config["MTD_scenarios"]
        self.weight_scenarios  = config["weight_scenarios"]
        self.num_scenarios     = len(self.MTD_scenarios)
        self.action_space      = Discrete(3 + self.J)
        self.observation_space = self._create_observation_space()

    def _create_observation_space(self):
        """Create an observation space.
        
        The observation space is the space of possible values for state s.
        
        Returns:
            gymnasium.spaces.Space: Observation space.
        """
        current_dose_lo = np.repeat(0.0, 1)
        current_dose_hi = np.repeat(1.0, 1)
        prop_N_lo = np.repeat(0.0, self.J)
        prop_N_hi = np.repeat(1.0, self.J)
        prop_DLT_lo = np.repeat(0.0, self.J)
        prop_DLT_hi = np.repeat(1.0, self.J)
        sum_N_lo = np.repeat(0.0, 1)
        sum_N_hi = np.repeat(1.0, 1)
        sum_DLT_lo = np.repeat(0.0, 1)
        sum_DLT_hi = np.repeat(1.0, 1)
        is_final_lo = np.repeat(0.0, 1)
        is_final_hi = np.repeat(1.0, 1)

        observation_space = Box(
            low=np.concatenate((current_dose_lo, prop_N_lo, prop_DLT_lo, sum_N_lo, sum_DLT_lo, is_final_lo)),
            high=np.concatenate((current_dose_hi, prop_N_hi, prop_DLT_hi, sum_N_hi, sum_DLT_hi, is_final_hi)),
            dtype=np.float32
        )
        return observation_space

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # draw the true scenario
        scenario_ind = self.np_random.choice(list(range(self.num_scenarios)), p=self.weight_scenarios)
        self.prob_true = self.prob_scenarios[scenario_ind]
        self.MTD_true = self.MTD_scenarios[scenario_ind]
        
        # initialize simulated data
        self.Ns   = np.zeros(self.J, dtype=int)
        self.DLTs = np.zeros(self.J, dtype=int)
        draw_dose = 0
        draw_DLT = self.np_random.binomial(n=self.N_cohort, p=self.prob_true[draw_dose])
        self.Ns[draw_dose] += self.N_cohort
        self.DLTs[draw_dose] += draw_DLT
        self.current_dose = draw_dose
        return self._compute_state(), {}
        
    def _compute_state(self):
        """Compute state s from simulated data.
        
        State s is the vector described in Sect. 2.2 of the original paper, 
        plus the information whether it is the final cohort or not.
        
        Returns:
            The specific value of the state s.
        """
        is_final = 0.1 if np.sum(self.Ns) == self.N_total else 0.0
        
        return np.concatenate((
            np.array([self.current_dose / (self.J - 1)]),
            self.Ns / self.N_total,
            self.DLTs / self.N_total,
            np.sum(self.Ns, keepdims=True) / self.N_total,
            np.sum(self.DLTs, keepdims=True) / self.N_total,
            np.array([is_final])
        ))

    def step(self, action):
        if action >= 3:  # determine MTD
            terminated = True
            MTD = action - 3
            reward = 1 if MTD in self.MTD_true else 0
            return self._compute_state(), reward, terminated, False, {}
        
        if np.sum(self.Ns) < self.N_total:
            terminated = False
            reward = 0
            
            # escalation
            if action == 0:  # down
                if self.current_dose == 0:
                    terminated = True
                    MTD = -1  # this means "no MTD"
                    reward = 1 if MTD in self.MTD_true else 0
                    return self._compute_state(), reward, terminated, False, {}
                else:
                    draw_dose = self.current_dose - 1
            elif action == 1:  # stay
                draw_dose = self.current_dose
            elif action == 2:  # up
                draw_dose = self.current_dose + 1 if self.current_dose <= self.J - 2 else self.current_dose
            
            # update simulated data
            draw_DLT = self.np_random.binomial(n=self.N_cohort, p=self.prob_true[draw_dose])
            self.Ns[draw_dose] += self.N_cohort
            self.DLTs[draw_dose] += draw_DLT
            self.current_dose = draw_dose
        else:
            terminated = True
            reward = 0
            
        return self._compute_state(), reward, terminated, False, {}
