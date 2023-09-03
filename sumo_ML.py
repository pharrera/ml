import traci
import time
import traci.constants as tc
import pytz
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import random

# Function to get current datetime
def get_datetime():
    utc_now = pytz.utc.localize(datetime.datetime.utcnow())
    currentDT = utc_now.astimezone(pytz.timezone("America/Los_Angeles"))
    DATIME = currentDT.strftime("%Y-%m-%d %H:%M:%S")
    return DATIME

# Function to flatten a 2D list
def flatten_list(_2d_list):
    flat_list = [item for sublist in _2d_list for item in (sublist if isinstance(sublist, list) else [sublist])]
    return flat_list

# PyTorch Model
class TrafficAutoEncoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(TrafficAutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, input_size),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# PyTorch Dataset
class TrafficDataset(Dataset):
    def __init__(self, data, max_sample_size):
        self.data = data
        self.max_sample_size = max_sample_size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        tensor_sample = [float(elem) if isinstance(elem, (int, float)) else 0.0 for elem in sample[1:]]
        # Pad with zeros to match the maximum size
        if len(tensor_sample) < self.max_sample_size:
            tensor_sample.extend([0.0] * (self.max_sample_size - len(tensor_sample)))
        return torch.tensor(tensor_sample, dtype=torch.float32)

# Function to control traffic signals based on fixed-time plan
def control_traffic_signals(phase_green_duration, phase_yellow_duration):
    for tlight in traci.trafficlight.getIDList():
        current_phase = traci.trafficlight.getPhase(tlight)
        current_time = traci.simulation.getTime()

        if current_phase == 0 and current_time % (phase_green_duration + phase_yellow_duration) == 0:
            traci.trafficlight.setPhase(tlight, 1)  # Switch to phase 1 (yellow)
        elif current_phase == 1 and current_time % (phase_green_duration + phase_yellow_duration) == phase_yellow_duration:
            traci.trafficlight.setPhase(tlight, 0)  # Switch to phase 0 (green)

# Class to represent the traffic state at an intersection
class TrafficState:
    def __init__(self, speed, queue_length, waiting_time):
        self.speed = speed
        self.queue_length = queue_length
        self.waiting_time = waiting_time

# Function to get traffic state at each intersection
def get_traffic_state(intersection_id):
    traffic_state = {}
    for tlight in traci.trafficlight.getIDList():
        tlight_lane_ids = traci.trafficlight.getControlledLanes(tlight)
        speed = sum(traci.lane.getLastStepMeanSpeed(lane_id) for lane_id in tlight_lane_ids) / len(tlight_lane_ids)
        queue_length = sum(traci.lane.getLastStepHaltingNumber(lane_id) for lane_id in tlight_lane_ids)
        waiting_time = sum(traci.lane.getWaitingTime(lane_id) for lane_id in tlight_lane_ids)
        traffic_state[tlight] = TrafficState(speed, queue_length, waiting_time)
    return traffic_state

# Function to update Q-values using Bellman equation
def update_q_values(reward, current_state, next_state, action, q_table, learning_rate, discount_factor):
    current_state_q_values = q_table.get(tuple(current_state), [0.0, 0.0])
    next_state_q_values = q_table.get(tuple(next_state), [0.0, 0.0])
    current_q_value = current_state_q_values[action]
    max_next_q_value = max(next_state_q_values)
    updated_q_value = current_q_value + learning_rate * (reward + discount_factor * max_next_q_value - current_q_value)
    current_state_q_values[action] = updated_q_value
    q_table[tuple(current_state)] = current_state_q_values

# Function to choose an action based on epsilon-greedy policy
def choose_action(epsilon, q_table, state):
    if random.random() < epsilon:
        return random.choice([0, 1])
    else:
        return max(enumerate(q_table.get(tuple(state), [0.0, 0.0])), key=lambda x: x[1])[0]

def main():
    # Define the fixed-time control plan for traffic lights
    phase_green_duration = 30
    phase_yellow_duration = 5

    # Define the maximum number of simulation steps
    max_steps = 10_000

    # Define the Q-learning parameters
    learning_rate = 0.1
    discount_factor = 0.9
    exploration_rate = 1.0
    min_exploration_rate = 0.01
    exploration_decay_rate = 0.001

    # Define the intersection ID you want to control the traffic signals for
    intersection_id = "cluster_123161178_3876198906_8689261198_8689261199_#2more"

    # Initialize Q-table
    q_table = {}

    # Initialize the simulation
    traci.start(["sumo", "-c", "osm.sumocfg", "--no-warnings", "--time-to-teleport", "-1"])

    # Get the initial state
    current_state = get_traffic_state(intersection_id)


    for step in range(max_steps):
        # Choose an action based on epsilon-greedy policy
        action = choose_action(exploration_rate, q_table, current_state)

        # Perform the action (control the traffic lights)
        phase_duration = phase_green_duration if action == 0 else phase_yellow_duration
        control_traffic_signals(phase_green_duration, phase_yellow_duration)
        traci.simulationStep()

        # Get the next state and the reward
        next_state = get_traffic_state(intersection_id)
        reward = -next_state[intersection_id].waiting_time

        # Update Q-values using Bellman equation
        update_q_values(reward, current_state, next_state, action, q_table, learning_rate, discount_factor)

        # Update the current state to the next state
        current_state = next_state

        # Decay the exploration rate
        exploration_rate = max(min_exploration_rate, exploration_rate - exploration_decay_rate)

    # End the simulation
    traci.close()

if __name__ == "__main__":
    main()
