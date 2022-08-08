import random
import copy
from collections import namedtuple, deque
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import model


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

STATE_SIZE = 24  #State size for each agent 
ACTION_SIZE = 2  #Acion size for each agent

BUFFER_SIZE = int(1e4)  # replay buffer size
BATCH_SIZE = 256        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-4         # learning rate of the actor 
LR_CRITIC = 1e-3        # learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay
NOISE_DECAY = 1.0      # Noise decay factor
NOISE_START = 0.5      #Noise weight factor at start
LEARN_EVERY = 2       # How often to learn from batch memory & update network
NOISE_STOP = 40000     #What timestep to stop using noise at

class MADDPG_DualAgent(): #Dual agent class that instantiates 2 agents for us

    def __init__(self, 
                 action_size = ACTION_SIZE, 
                 seed = 2022, 
                 n_agents = 2,
                 buffer_size = BUFFER_SIZE,
                 batch_size = BATCH_SIZE,
                 gamma = GAMMA,
                 learn_every = LEARN_EVERY,
                 noise_start = NOISE_START,
                 noise_decay = NOISE_DECAY,
                 t_stop_noise = NOISE_STOP):
        """
        Params
        ======
            action_size (int): dimension of each action
            seed (int): Random seed
            n_agents (int): number of distinct agents
            buffer_size (int): replay buffer size
            batch_size (int): minibatch size
            gamma (float): discount factor
            noise_start (float): initial noise weighting factor
            noise_decay (float): noise decay rate
            learn_every (int): how often to learn and update the network
            t_stop_noise (int): max number of timesteps with noise applied in training
        """

        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.learn_every = learn_every
        self.gamma = gamma
        self.n_agents = n_agents
        self.noise_weight = noise_start
        self.noise_decay = noise_decay
        self.t_step = 0
        self.noise_on = True
        self.t_stop_noise = t_stop_noise

        
        models = [model.DoubleAgent(n_agents=n_agents) for _ in range(n_agents)]
        self.agents = [Agent(k, models[k]) for k in range(n_agents)]
        # create shared replay buffer - code reused from Project 2
        self.memory = ReplayBuffer(action_size, self.buffer_size, self.batch_size, seed)

    def step(self, all_states, all_actions, all_rewards, all_next_states, all_dones):
        '''
        Reshape states into vectors, add all to memory. 
        Learn from experience buffer. 
        Step forwards!
        '''
        all_states = all_states.reshape(1, -1)
        all_next_states = all_next_states.reshape(1, -1)
        self.memory.add(all_states, all_actions, all_rewards, all_next_states, all_dones)
        
        #Turn off noise once we reach the t_stop_noise steps!!! VERY IMPORTANT!!!!
        if self.t_step > self.t_stop_noise:
            self.noise_on = False
        #Update timestep 
        self.t_step = self.t_step + 1
        
        if self.t_step % self.learn_every == 0:
            # If enough samples are available in memory, sample and learn
            if len(self.memory) > self.batch_size:
                experiences = [self.memory.sample() for _ in range(self.n_agents)]
                self.learn(experiences, self.gamma)

    def act(self, all_states, add_noise=True):
        '''
        Calculate each actor's action based on environment state
        '''
        all_actions = []
        for agent, state in zip(self.agents, all_states):
            action = agent.act(state, noise_weight=self.noise_weight, add_noise=self.noise_on)
            self.noise_weight *= self.noise_decay
            all_actions.append(action)
        return np.array(all_actions).reshape(1, -1) 

    def learn(self, experiences, gamma):
        '''
        Learn: calculate each actor's next action. 
        '''
        all_next_actions = []
        all_actions = []
        for k, agent in enumerate(self.agents):
            states, _, _, next_states, _ = experiences[k]
            agent_id = torch.tensor([k]).to(device)
            #Get action for agents 
            state = states.reshape(-1, 2, 24).index_select(1, agent_id).squeeze(1)
            action = agent.actor_local(state)
            all_actions.append(action)
            #Get next state for agents 
            next_state = next_states.reshape(-1, 2, 24).index_select(1, agent_id).squeeze(1)
            next_action = agent.actor_target(next_state)
            all_next_actions.append(next_action)
                       
        
        for k, agent in enumerate(self.agents):
            agent.learn(k, experiences[k], gamma, all_next_actions, all_actions)
            
    def save_agents(self):
        # save models for local actor and critic of each agent
        for i, agent in enumerate(self.agents):
            torch.save(agent.actor_local.state_dict(),  f"checkpoint_actor_agent_{i}.pth")
            torch.save(agent.critic_local.state_dict(), f"checkpoint_critic_agent_{i}.pth")


class Agent():
    '''
    Original single DDPG Agent
    '''

    def __init__(self, 
                 agent_id,
                 model,
                 action_size = ACTION_SIZE,
                 seed = 2022,
                 tau = TAU,
                 lr_actor = LR_ACTOR,
                 lr_critic = LR_CRITIC,
                 weight_decay = WEIGHT_DECAY,
                 noise_decay = NOISE_DECAY):
        """
        Params
        ======
            model: model object
            action_size (int): dimension of action space
            seed (int): Random seed
            tau (float): for soft update of target parameters
            lr_actor (float): learning rate for actor
            lr_critic (float): learning rate for critic
            weight_decay (float): L2 weight decay
        """
        random.seed(seed)
        self.id = agent_id
        self.action_size = action_size
        self.tau = tau
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.noise_decay = noise_decay

       
        #Actor Network (w/ target network - modified to account for DualAgent function)
        self.actor_local = model.actor_local
        self.actor_target = model.actor_target
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr = lr_actor)
        
        #Critic Network (w/ target network - modified to account for DualAgent function)
        self.critic_local = model.critic_local
        self.critic_target = model.critic_target
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr = lr_critic, weight_decay = weight_decay)
        
        self.hard_copy_weights(self.actor_target, self.actor_local)
        self.hard_copy_weights(self.critic_target, self.critic_local)
        
        #Noise process - unchanged from Udacity provided code. 
        self.noise = OUNoise(action_size, seed)

    def hard_copy_weights(self, target, source): #Unchanged from Udacity code
        """ copy weights from source to target network (part of initialization)"""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)
    
    def act(self, state, noise_weight = 0.5, add_noise=True):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            self.noise_val = self.noise.sample() * noise_weight
            action += self.noise_val
        return np.clip(action, -1, 1)

    def reset(self):
        self.noise.reset()

    def learn(self, agent_id, experiences, gamma, all_next_actions, all_actions): #mostly unchanged from Udacity code 
        """Update policy and value parameters using given batch of experience tuples.
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
            all_next_actions (list): each agent's next_action (as calculated by its actor)
            all_actions (list): each agent's action (as calculated by its actor)
        """

        states, actions, rewards, next_states, dones = experiences

        # ---------------------------- update critic ---------------------------- #
        # get predicted next-state actions and Q values from target models
        self.critic_optimizer.zero_grad()
        
        actions_next = torch.cat(all_next_actions, dim=1).to(device)
        agent_id = torch.tensor([agent_id]).to(device)
        
        with torch.no_grad():
            q_targets_next = self.critic_target(next_states, actions_next)
        #Calculate Q targets y_i
        q_exp = self.critic_local(states, actions)
        q_targets = rewards.index_select(1, agent_id) + (gamma * q_targets_next * (1 - dones.index_select(1, agent_id)))
        #Compute critic loss
        critic_loss = F.mse_loss(q_exp, q_targets.detach())
        #Minimise the loss
        critic_loss.backward()
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        #Compute actor loss
        self.actor_optimizer.zero_grad()
        
        #Build actor predictions vector
        actions_pred = [actions if k == self.id else actions.detach() for k, actions in enumerate(all_actions)]
        actions_pred = torch.cat(actions_pred, dim=1).to(device)
        
        #Compute and minimise actor loss 
        actor_loss = -self.critic_local(states, actions_pred).mean()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, self.tau)
        self.soft_update(self.actor_local, self.actor_target, self.tau)


    def soft_update(self, local_model, target_model, tau): #Unchanged from Udacity code 
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


class OUNoise: #Unchanged from Udacity code except for not using array in sample()
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        random.seed(seed)
        np.random.seed(seed)
        self.size = size
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.size)
        self.state = x + dx
        return self.state


class ReplayBuffer(): #Unchanged from Udacity code 
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): Random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)


    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)