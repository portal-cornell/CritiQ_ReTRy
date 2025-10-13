import torch
import torch.nn as nn
from setuptools.dist import sequence
from torch.utils.data import Dataset
import numpy as np
import torch.optim as optim
import random
# from continuous_dqn import state


# Define a simple neural network for BC

def set_seed(seed):
    """
    Set the random seed for reproducibility.
    """
    torch.manual_seed(seed)  # PyTorch
    np.random.seed(seed)     # NumPy
    random.seed(seed)        # Python's random
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)  # For CUDA
        torch.cuda.manual_seed_all(seed)  # If multiple GPUs are used
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

class BCNetwork(nn.Module):
    def __init__(self, input_dim=19, output_dim=4, hidden_dim=256, seed=42, lr=1e-3, device="cpu", optim="adam"):
        super(BCNetwork, self).__init__()
        set_seed(seed)

        # self.model = nn.Sequential(
        #     nn.Linear(input_dim, hidden_dim),
        #     nn.Tanh(),
        #     nn.Linear(hidden_dim, hidden_dim),
        #     nn.Tanh(),
        #     nn.Linear(hidden_dim, hidden_dim),
        #     nn.Tanh(),
        #     nn.Linear(hidden_dim, hidden_dim),
        #     nn.Tanh(),
        #     nn.Linear(hidden_dim, output_dim),
        # )
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Tanh(),
        )
        # self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.optimizer = self.get_optimizer(optim, self.parameters(), lr)
        self.criterion = nn.MSELoss()
        self.device=device
        self.state_dim = input_dim
        self.action_dim = output_dim

    def get_optimizer(self,optimizer_name, model_parameters, lr):
        """옵티마이저를 문자열로 받아 동적으로 생성하는 함수"""
        optimizers = {
            "sgd": optim.SGD,
            "adam": optim.Adam,
            "rmsprop": optim.RMSprop,
            "adagrad": optim.Adagrad
        }

        if optimizer_name.lower() not in optimizers:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}. Choose from {list(optimizers.keys())}")

        return optimizers[optimizer_name.lower()](model_parameters, lr=lr)
    def forward(self, ob):
        ob = torch.cat((
            ob["jnt_states"], 
            ob["target_0"], 
            ob["target_1"],
            ob["target_2"],
            ob["red_box"], 
            ob["red_history"],
        ), dim=-1)

        if not isinstance(ob, torch.Tensor):
            # Convert to Tensor
            ob = torch.tensor(ob, dtype=torch.float32)
        
        # x = torch.cat((
        #     ob["jnt_states"], 
        #     ob["student_handle_pos_0"], 
        #     ob["handle_displacement_0"],
        #     ob["handle_0_status"],
        #     ob["student_handle_pos_1"], 
        #     ob["handle_displacement_1"],
        #     ob["handle_1_status"],
        #     ob["student_handle_pos_2"], 
        #     ob["handle_displacement_2"],
        #     ob["handle_2_status"],
        # ), dim=-1).to(self.device)
        return self.model(ob)
    def predict(self, obs_seq, deterministic=True):
        self.eval()
        with torch.no_grad():
            actions = self.forward(obs_seq)
            # output = actions.view(actions.shape[0], self.action_dim)
            return actions
   
    def train_model(self, dataloader, num_epochs=10):
        """
        Trains the BC model.

        Args:
        - dataloader (DataLoader): DataLoader for training data
        - criterion (loss function): Loss function to minimize
        - num_epochs (int): Number of epochs to train for
        - device (str): Device to use ('cpu' or 'cuda')

        Returns:
        - None
        """
        self.to(self.device)  # Move model to device (CPU/GPU)

        for epoch in range(num_epochs):
            self.train()  # Set model to training mode
            epoch_loss = 0

            for state_batch, action_batch in dataloader:
                # Move data to the same device as the model
                state_batch, action_batch = state_batch.to(self.device), action_batch.to(self.device)

                # Forward pass
                predicted_actions = self(state_batch)
                loss = self.criterion(predicted_actions, action_batch)

                # Backward pas
                self.optimizer.zero_grad()  # Reset gradients
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

            # Print epoch loss
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss / len(dataloader):.4f}")
    def load_model(self, path: str):
        # Load a pre-trained student policy
        
        self.load_state_dict(torch.load(path, weights_only=True))
        

class LSTMBC(nn.Module):
    def __init__(self, input_dim=19, output_dim=5, lstm_layers=2, hidden_dim = 128, sequence_length=10, seed=42, lr=1e-4, device="cpu", optim="adam"):
        super(LSTMBC, self).__init__()
        set_seed(seed)
        self.seq_len=sequence_length
        self.lstm = nn.LSTM(input_size=input_dim,
                            hidden_size=hidden_dim,
                            num_layers=lstm_layers,
                            batch_first=True)
        self.model = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim*sequence_length)
        )
        self.lstm = self.lstm.to(device)
        self.model = self.model.to(device)
        # self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.optimizer = self.get_optimizer(optim, self.parameters(), lr)
        self.criterion = nn.MSELoss()
        self.device=device
        self.state_dim = input_dim
        self.action_dim = output_dim


    def get_optimizer(self,optimizer_name, model_parameters, lr):
        """옵티마이저를 문자열로 받아 동적으로 생성하는 함수"""
        optimizers = {
            "sgd": optim.SGD,
            "adam": optim.Adam,
            "rmsprop": optim.RMSprop,
            "adagrad": optim.Adagrad
        }

        if optimizer_name.lower() not in optimizers:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}. Choose from {list(optimizers.keys())}")

        return optimizers[optimizer_name.lower()](model_parameters, lr=lr)

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            # Convert to Tensor
            x = torch.tensor(x, dtype=torch.float32)
        batch_size = x.size(0)
        x = x.to(self.device)
        x = x.to(self.device)  # Move input to device
        lstm_out, _ = self.lstm(x)  # Output: (batch, seq_len, hidden_dim)
        last_hidden = lstm_out[:, -1, :]  # Take last hidden state (batch_size, hidden_dim)
        action_output = self.model(last_hidden)
        return action_output.view(batch_size, self.seq_len, -1)

    def _predict(self, obs_seq, deterministic=False):
        self.eval()
        with torch.no_grad():
            actions = self.forward(obs_seq)
            output = actions.view(actions.shape[0], self.seq_len, self.action_dim)
            return output
            # if deterministic:
            #     with torch.no_grad():
            #         lstm_out, _ = self.lstm(obs_seq)  # Pass through LSTM
            #         output = self.model(lstm_out)  # Pass through MLP
            #         output = output.view(output.shape[0], self.seq_len, self.action_dim)  # Reshape output
            # else:
            #     return actions.sample().numpy()
    
    def predict(self, obs, deterministic=False):
        return self._predict(obs, deterministic)
    
    def train_model(self, dataloader, num_epochs=10):
        """Train BC and Discriminator for multiple epochs."""
        torch.autograd.set_detect_anomaly(True)
        for epoch in range(num_epochs):
            bc_loss_total = 0
            disc_loss_total = 0

            for state_seq_batch, action_seq_batch in dataloader:
                state_seq_batch, action_seq_batch = state_seq_batch.to(self.device), action_seq_batch.to(self.device)
                bc_action_seq = self.forward(state_seq_batch)
                mse_loss = self.criterion(bc_action_seq, action_seq_batch)
                bc_state_seq = state_seq_batch

                bc_loss = mse_loss 

                # Update BC network
                self.optimizer.zero_grad()
                bc_loss.backward()
                self.optimizer.step()

                
                bc_loss_total += bc_loss.item()
                
            print(
                f"Epoch {epoch + 1}/{num_epochs}, BC Loss: {bc_loss_total / len(dataloader):.4f}, Disc Loss: {disc_loss_total / len(dataloader):.4f}")
    def load_model(self, path: str):
        # Load a pre-trained student policy
        
        self.load_state_dict(torch.load(path, weights_only=True))
        

# class BCDiscriminator(nn.Module):
#     def __init__(self, bc_network, discriminator, env, seq_len=10, device="cpu"):
#         super(BCDiscriminator, self).__init__()
#         self.bc_network = bc_network.to(device)
#         self.discriminator = discriminator.to(device)
#         self.env = env  # Environment to roll out BC policy
#         self.seq_len = seq_len
#         self.device = device
#         self.bc_lstm = False
#         for module in bc_network.modules():
#             if isinstance(module, nn.LSTM):
#                 self.bc_lstm = True



#     def rollout_policy(self, state_batch):
#         """Rollout BC policy to generate a sequence of (s, a) pairs"""
#         batch_size = state_batch.shape[0]
#         state_seq = torch.zeros((batch_size, self.seq_len, state_batch.shape[1]), device=self.device)
#         action_seq = torch.zeros((batch_size, self.seq_len, self.bc_network.action_dim), device=self.device)

#         state = state_batch.clone()
#         for t in range(self.seq_len):
#             action = self.bc_network.model(state)  # Predict action
#             state_seq[:, t, :] = state  # Store state
#             action_seq[:, t, :] = action  # Store action

#             # Get next state (assuming deterministic environment transition)
#             # state = self.env.step(state, action)  # Custom env step function
#             state = state + action
#             if self.env.done(state.detach().cpu().numpy()):
#                 break
#         return state_seq, action_seq

#     def train_model(self, dataloader, num_epochs=10):
#         """Train BC and Discriminator for multiple epochs."""
#         torch.autograd.set_detect_anomaly(True)
#         for epoch in range(num_epochs):
#             bc_loss_total = 0
#             disc_loss_total = 0

#             for state_seq_batch, action_seq_batch in dataloader:
#                 state_seq_batch, action_seq_batch = state_seq_batch.to(self.device), action_seq_batch.to(self.device)

#                 # 🔹 Extract initial state & action for BC
#                 if self.bc_lstm:
#                     bc_action_seq = self.bc_network.forward(state_seq_batch)
#                     mse_loss = self.bc_network.criterion(bc_action_seq, action_seq_batch)
#                     bc_state_seq = state_seq_batch

#                 else:
#                     state_batch = state_seq_batch[:, 0, :].clone().detach()
#                     action_batch = action_seq_batch[:, 0, :].clone().detach()

#                     # 🔹 BC Loss: MSE Loss + Discriminator Reward
#                     pred_action = self.bc_network.model(state_batch)
#                     mse_loss = self.bc_network.criterion(pred_action, action_batch)

#                     # Generate BC rollouts
#                     bc_state_seq, bc_action_seq = self.rollout_policy(state_batch)

#                 # Discriminator Loss
#                 expert_label = torch.ones((state_seq_batch.size(0), 1), device=self.device)*0.8

#                 bc_label = torch.ones((state_seq_batch.size(0), 1), device=self.device)*0.2

#                 expert_prob = self.discriminator(torch.cat([state_seq_batch, action_seq_batch], dim=-1))
#                 bc_prob = self.discriminator(torch.cat([bc_state_seq.detach(), bc_action_seq.detach()], dim=-1))
#                 with torch.no_grad():
#                     discriminator_reward = 1 - bc_prob.mean() # Avoid backprop graph issues

#                 bc_loss = mse_loss + discriminator_reward # Detach to avoid graph reuse

#                 # Update BC network
#                 self.bc_network.optimizer.zero_grad()
#                 bc_loss.backward()
#                 self.bc_network.optimizer.step()

#                 # Compute Discriminator loss
#                 disc_loss = self.discriminator.criterion(expert_prob, expert_label) + \
#                             self.discriminator.criterion(bc_prob, bc_label)

#                 # Update Discriminator
#                 if epoch % 3 == 0:
#                     self.discriminator.optimizer.zero_grad()
#                     disc_loss.backward()
#                     self.discriminator.optimizer.step()

#                 bc_loss_total += bc_loss.item()
#                 disc_loss_total += disc_loss.item()

#             print(
#                 f"Epoch {epoch + 1}/{num_epochs}, BC Loss: {bc_loss_total / len(dataloader):.4f}, Disc Loss: {disc_loss_total / len(dataloader):.4f}")


# class TrajectoryDataset(Dataset):
#     def __init__(self, dataset):
#         states = []
#         actions = []
#         for demonstration in dataset:
#             for state, action in demonstration:
#                 states.append(state)
#                 actions.append(action)

#         # Convert to NumPy arrays for compatibility with PyTorch
#         states = np.array(states, dtype=np.float32)
#         actions = np.array(actions, dtype=np.float32)
#         self.states = torch.tensor(states, dtype=torch.float32)
#         self.actions = torch.tensor(actions, dtype=torch.float32)

#     def __len__(self):
#         return len(self.states)

#     def __getitem__(self, idx):
#         return self.states[idx], self.actions[idx]


# class SequentialTrajectoryDataset(Dataset):
#     def __init__(self, dataset, state_dim, action_dim, seq_len=10):
#         """
#         Args:
#             dataset: List of demonstrations, where each demonstration is a list of (state, action) tuples.
#             state_dim: Dimension of the state space.
#             action_dim: Dimension of the action space.
#             seq_len: Desired sequence length for LSTM input.
#         """
#         self.seq_len = seq_len
#         self.state_dim = state_dim
#         self.action_dim = action_dim
#         self.states, self.actions = self.process_data(dataset)

#     def process_data(self, dataset):
#         states, actions = [], []
#         for demonstration in dataset:
#             demo_states, demo_actions = zip(*demonstration)
#             demo_states = np.array(demo_states, dtype=np.float32)
#             demo_actions = np.array(demo_actions, dtype=np.float32)
#             states.append(demo_states)
#             actions.append(demo_actions)
#         return states, actions

#     def __len__(self):
#         """Total number of time steps available across all trajectories"""
#         return sum(len(traj) for traj in self.states)

#     def __getitem__(self, idx):
#         """Returns a sequence starting from idx, with padding if necessary."""
#         # Find which trajectory this index belongs to
#         traj_idx = 0
#         # while idx >= len(self.states[traj_idx]) - self.seq_len + 1:
#         #     idx -= len(self.states[traj_idx]) - self.seq_len + 1
#         #     traj_idx += 1

#         while idx >= len(self.states[traj_idx])  + 1:
#             idx -= len(self.states[traj_idx])+ 1
#             traj_idx += 1

#         # Extract the current trajectory
#         traj_states = self.states[traj_idx]
#         traj_actions = self.actions[traj_idx]

#         # Construct the sequence i ~ i+seq_len (with zero padding if necessary)
#         end_idx = min(idx + self.seq_len, len(traj_states))
#         pad_len = max(0, (idx + self.seq_len) - len(traj_states))

#         seq_states = np.zeros((self.seq_len, self.state_dim), dtype=np.float32)
#         seq_actions = np.zeros((self.seq_len, self.action_dim), dtype=np.float32)

        
#         seq_states[:self.seq_len - pad_len] = traj_states[idx:end_idx]
#         seq_actions[:self.seq_len - pad_len] = traj_actions[idx:end_idx]

#         return torch.tensor(seq_states, dtype=torch.float32), torch.tensor(seq_actions, dtype=torch.float32)
