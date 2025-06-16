import gym
import csv
import warnings
import numpy as np

warnings.filterwarnings("ignore", category=DeprecationWarning, message="np.bool8 is a deprecated alias")

# --- Limits from CartPole environment ---
CART_POS_LIMIT = 4.8
POLE_ANGLE_LIMIT = 0.418
def voltage_for_value(value, min_val, max_val, v_min=0.1e-3, v_max=1e-3):
    norm = (value - min_val) / (max_val - min_val)
    norm = np.clip(norm, 0, 1)
    voltage = v_min + norm * (v_max - v_min)
    return voltage
# --- Simulation parameters ---
NUM_EPISODES = 1000
MAX_STEPS = 200
dt = 0.02  # Time between steps in seconds

# PWL spike parameters
duration_per_step = 5e-12  # 5 ps per logical "step"


# Poisson spike encoder function
def poisson_spike_times(value, min_val, max_val, duration, rate_max=100e9):
    norm = (value - min_val) / (max_val - min_val)
    norm = np.clip(norm, 0, 1)
    rate = norm * rate_max
    if rate <= 0:
        return []
    num_spikes = np.random.poisson(rate * duration)
    return np.sort(np.random.uniform(0, duration, num_spikes))


env = gym.make('CartPole-v1')
all_data = []
pwl_data = []

for episode in range(NUM_EPISODES):
    observation, info = env.reset()
    done = False
    step = 0

    while not done and step < MAX_STEPS:
        cart_pos, cart_vel, pole_angle, pole_vel = observation
        time_sec = step * dt

        # Record balanced step label (1)
        all_data.append([episode, step, time_sec, cart_pos, cart_vel, pole_angle, pole_vel, 1])

        # Generate time base for PWL spikes for this step
        time_base = (episode * MAX_STEPS + step) * duration_per_step

        # Encode each state variable as Poisson spike times on 4 channels
        channels = [
            poisson_spike_times(cart_pos, -4.8, 4.8, duration_per_step),
            poisson_spike_times(cart_vel, -5, 5, duration_per_step),
            poisson_spike_times(pole_angle, -0.418, 0.418, duration_per_step),
            poisson_spike_times(pole_vel, -5, 5, duration_per_step),
        ]

        channel_values = [cart_pos, cart_vel, pole_angle, pole_vel]
        channel_ranges = [(-4.8, 4.8), (-5, 5), (-0.418, 0.418), (-5, 5)]

        for ch, spikes in enumerate(channels):
            v_min, v_max = channel_ranges[ch]
            val = channel_values[ch]
            volt = voltage_for_value(val, v_min, v_max)  # scaled voltage
            for spike in spikes:
                spike_time = time_base + spike
                pwl_data.append([spike_time, ch, volt])

        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)

        done = terminated or truncated

        # If episode ended, label fallen step (0)
        if done or abs(observation[0]) > CART_POS_LIMIT or abs(observation[2]) > POLE_ANGLE_LIMIT:
            time_sec = (step + 1) * dt
            cart_pos, cart_vel, pole_angle, pole_vel = observation
            all_data.append([episode, step + 1, time_sec, cart_pos, cart_vel, pole_angle, pole_vel, 0])
            break

        step += 1

env.close()

# Save labeled CartPole data
with open('cartpole_labeled_data.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['episode', 'step', 'time_sec', 'cart_pos', 'cart_vel', 'pole_angle', 'pole_vel', 'label'])
    writer.writerows(all_data)

# Save Poisson spike PWL data
with open('cartpole_poisson_pwl.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['time_sec', 'channel', 'voltage'])
    writer.writerows(sorted(pwl_data))

# ---- PWL for JoSIM-compatible input ----
pwl_by_channel = {0: [], 1: [], 2: [], 3: []}
for t, ch, v in sorted(pwl_data):
    pwl_by_channel[ch].append((t, v))

with open('cartpole_input_pwl.txt', 'w') as f:
    for ch in range(4):
        f.write(f"* Channel {ch}\n")
        f.write(f"VCH{ch} in{ch} 0 PWL(")
        seq = []
        for t, v in pwl_by_channel[ch]:
            t_ps = round(t * 1e12, 3)
            v_mv = round(v * 1e3, 6)
            seq.append(f"{t_ps}p {v_mv}mV")
        f.write(" ".join(seq) + ")\n\n")

print(f"Data for {NUM_EPISODES} episodes saved to 'cartpole_labeled_data.csv'")
print(f"Poisson PWL spike data saved to 'cartpole_poisson_pwl.csv'")
print(f"PWL input voltage source file written to 'cartpole_input_pwl.txt'")
