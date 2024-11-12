import re
import ast
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np

log_file_path = "logs/mineflayer/20241106_173404.log"
with open(log_file_path, 'r') as file:
    log_content = file.read()

# Regex to find all 'position' entries and their values
position_pattern = re.compile(r"position:\s*(null|\{.*?\})")



# Extract positions
positions = position_pattern.findall(log_content)

#filter positions
positions = [pos for pos in positions if pos != 'null']


#parse positions into actual dicts

# Function to parse coordinate strings into dictionaries
def parse_position(position_str):
    pattern = r'(\w+):\s*(-?\d+\.?\d*)'
    matches = re.findall(pattern, position_str)
    return {key: float(value) for key, value in matches}

# Parse all positions
parsed_positions = [parse_position(pos) for pos in positions]

# Function to calculate Euclidean distance
def euclidean_distance(p1, p2):
    return math.sqrt((p2['x'] - p1['x'])**2 + (p2['y'] - p1['y'])**2 + (p2['z'] - p1['z'])**2)

# Calculate distances and cumulative distances
distances = []
cumulative_distances = [0]

for i in range(1, len(parsed_positions)):
    d = euclidean_distance(parsed_positions[i - 1], parsed_positions[i])
    distances.append(d)
    cumulative_distances.append(cumulative_distances[-1] + d)

# Plotting the cumulative distance as a time series
# plt.plot(cumulative_distances, marker='o')
# plt.title('Cumulative Distance Over Time')
# plt.xlabel('Time Step')
# plt.ylabel('Cumulative Distance')
# plt.grid(True)
# plt.show()

# for the circle plot visualization
def calculate_center_and_radius(agent_positions):
    print(agent_positions)
    x_coords = [pos['x'] for pos in agent_positions]
    z_coords = [pos['z'] for pos in agent_positions]
    
    #Calculate center (mean of x and z)
    center_x = np.mean(x_coords)
    center_z = np.mean(z_coords)
    
    #Calculate radius (max distance from center)
    distances = [np.sqrt((x - center_x)**2 + (z - center_z)**2) for x, z in zip(x_coords, z_coords)]
    radius = max(distances)
    
    return (center_x, center_z), radius

# Calculate centers and radii for each agent
agents_circles = {} #TODO multiple bots
agent = "Voyager (1)" #TODO need to look into handling multiple bots
center, radius = calculate_center_and_radius(parsed_positions)
agents_circles[agent] = {'center': center, 'radius': radius}

# Create the plot
fig, ax = plt.subplots(figsize=(10, 6))

# Define unique colors for each agent
colors = {
    'Voyager (1)': 'orange',
    #TODO add more colors for other bots
}

# Plot circles for each agent
for agent, data in agents_circles.items():
    circle = Circle(data['center'], data['radius'], color=colors[agent], alpha=0.4, label=agent)
    ax.add_patch(circle)

# Set limits based on the data
ax.set_xlim(-30, 30)
ax.set_ylim(-30, 30)

# Add legend and labels
ax.legend()
ax.set_title('Agent Movement Coverage (Based on Minecraft Coordinates)')
ax.set_xlabel('X Coordinate')
ax.set_ylabel('Z Coordinate')
ax.grid(True)

# Display the plot
plt.show()

