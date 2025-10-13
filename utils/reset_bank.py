import pickle
import numpy as np
from copy import deepcopy

class ResetBank:
    def __init__(self):
        """
        Bank (List of Dict): Each list contains a saved reset state
            Keys:
                data stretch joint information
                handle positions
                goal drawer (0, 1, 2)
        """
        self.bank = []

    def add_from_sim(self, qpos, ctrl, goal):
        state_dict = {}
        state_dict["qpos"] = qpos
        state_dict["ctrl"] = ctrl
        state_dict["goal"] = goal
        self.bank.append(state_dict)
        return self.bank
    
    def add(self, entry):
        self.bank.append(entry)
        return self.bank
    
    def extend(self, bank):
        self.bank.append(deepcopy(bank))
        return self.bank

    def sample(self, ind=None, pop=True):
        if len(self.bank) == 0:
            print("Reset bank empty")
            return None
        if ind is None:
            ind = np.random.randint(0, len(self.bank))
        if pop:
            reset_state = self.bank.pop(ind)
        else:
            reset_state = self.bank[ind]
        return reset_state

    def __len__(self):
        return len(self.bank)

    def save(self, file_path):
        with open(file_path, 'wb') as file:
            pickle.dump(self, file)
    
    def load(self, file_path):
        with open(file_path, 'rb') as file:
            self = pickle.load(file)
    