import numpy as np
import math

from dataclasses import dataclass, field
from typing import List

np.random.seed(0)

N_PARTICLES = 3
MOVEMENT_PROBABILITY = 0.2 # Pārvietošanās nenoteiktība
MEASUREMENT_PROBABILITY = 0.1 # Mērījuma nenoteiktībai




# State
current_robot_position = 12
particles = []


# Particles
X_robot = np.array([])




#  ---   1) Robots atrodas pozīcijā 12

# move_robot()

# particle_locations = np.zeros((N_PARTICLES))
# particle_locations = np.random.rand(N_PARTICLES)
particle_locations = np.array([current_robot_position] * N_PARTICLES)
particle_weights = np.array([1.0] * N_PARTICLES)

# show_location_progabilities(particle_locations)


# ------------------        2) 

movement_step = 4  # Robots pārvietojas 4m pa labi
measurement_detected = -2 # uztver objektu1 2m pa kreisi

x_before_spread = particle_locations + movement_step # X (pirms izkliedes)

# Te random  -0.8 līdz 0.8
spread = np.array([-1.368, -0.118, 0.3775])

# err_interval = movement_step * MOVEMENT_PROBABILITY
# errors = np.random.uniform(-err_interval, err_interval,(N_PARTICLES, 12))
# spread = np.sum(errors, axis=1) / 2

x_after_spread = x_before_spread + spread  # X (pēc izkliedes)




O_1 = x_after_spread + measurement_detected  # O1 (objekta1 poz.)
Prob_O_1 = np.array([MEASUREMENT_PROBABILITY * abs(measurement_detected)] * N_PARTICLES)
particle_weights = particle_weights

# ---------------             3) 
movement_step = 2  #Robots pārvietojas 2m pa labi 

x_before_spread = x_after_spread + movement_step # X (pirms izkliedes)
spread = np.array([0.6035, -0.747, 0.556]) # (Izkliede)

# err_interval = movement_step * MOVEMENT_PROBABILITY
# errors = np.random.uniform(-0.8, 0.8,(N_PARTICLES, 12))
# spread = np.sum(errors, axis=1) / 2

x_after_spread = x_before_spread + spread   # (X (pēc izkliedes))
O_1 = O_1 # iekopējam iepriekšējās, jo šajā solī nav bijis mērījums
Prob_O_1 = Prob_O_1 # iekopējam iepriekšējās, jo šajā solī nav bijis mērījums
particle_weights = particle_weights


# -------------   4) TE OBJEKTS 1 Tiek ieraudzīts OTRO REIZI!
measurement_detected = -2.5 # Robots uztver objektu O1 2.5m pa kreisi


X = x_after_spread 
O1_forecast = O_1 # O1 prognoze
Z = X + measurement_detected # Kāds ir mūsu mērījums?
Y = Z - O1_forecast # Inovācija
Q = abs(measurement_detected) * MEASUREMENT_PROBABILITY # Mērījuma nenoteiktība
S = Prob_O_1 + Q # Inovācijas kovariance
K =  Prob_O_1 * (1 / S) # Kalmana ieguvums
O_1 = O1_forecast + K * Y
Prob_O_1 = (1 - K) * Prob_O_1 #Jaunās iezīmes pozīcija
particle_weights = pow(abs((2 * math.pi * S)), (-1 / 2)) * np.exp(-1 / 2 * Y * (1 / S) * Y)

intervals = np.cumsum(particle_weights) # intervāli
intervals = np.insert(intervals, 0, 0, axis=0)
step = np.sum(particle_weights) / N_PARTICLES

indicies = []
for n in range(N_PARTICLES):
    v = n * step
    for idx, (previous, current) in enumerate(zip(intervals, intervals[1:])):
        if v >= previous and v <= current:
            print(idx, previous, current, v)
            indicies.append(idx)

if len(indicies) != N_PARTICLES:
    print('Bad')
    exit(0)

# reasign values based on picked indicies
for old, new in zip(range(N_PARTICLES), indicies):
    X[old] = X[new]
    O_1[old] = O_1[new]
    Prob_O_1[old] = Prob_O_1[new]

x = 0