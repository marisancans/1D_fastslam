import numpy as np
import math
import cv2
import copy
from dataclasses import dataclass, field
from typing import List, ClassVar

np.random.seed(0)

@dataclass
class Particle:
    nr_objects: ClassVar[int] = 0 # static class variable, tracks number of total objects
    X: float
    W: float = 1.0
    object_positions: List[float] = field(default_factory = lambda: [])
    object_probabilities: List[float] = field(default_factory = lambda: [])
    Z: List[float] = field(default_factory = lambda: [])
    Y: List[float] = field(default_factory = lambda: [])
    Q: List[float] = field(default_factory = lambda: [])
    S: List[float] = field(default_factory = lambda: [])
    K: List[float] = field(default_factory = lambda: [])


N_PARTICLES = 50
MOVEMENT_PROBABILITY = 0.2 # Pārvietošanās nenoteiktība
MEASUREMENT_PROBABILITY = 0.1 # Mērījuma nenoteiktībai
ROBOT_START_POSITION = 5
DELTA = 1.0

def imshow(img, w_name='window', waitKey=0):
    cv2.namedWindow(w_name, cv2.WINDOW_NORMAL)
    cv2.imshow(w_name, img)
    cv2.waitKey(waitKey)

def show_location_progabilities(particles: List[Particle], idx):
    cell_w = 600
    cell_h = 200
    block = cell_w / 11
    img = np.zeros((cell_h, cell_w, 3))

    for loc in [2, 5 ,10]:
        X = int(loc * block)
        img = cv2.line(img, (X, 0), (X, cell_h), (0, 0, 1.0), 1)
        img = cv2.putText(img, f'{loc:.0f}', (X, cell_h//2), 0, 0.5, (0, 0, 1.0), 2, cv2.LINE_AA) 

    img_robot_x = img.copy()
    for particle in particles:
        X = int(particle.X * block)
        img_robot_x = cv2.line(img_robot_x, (X, 0), (X, cell_h), (1.0, 1.0, 0), 1)

    
    partrticles_x = img.copy()
    for particle in particles:
        for obj_x in particle.object_positions:
            obj_offset = int(obj_x * block)
            partrticles_x = cv2.line(partrticles_x, (obj_offset, 0), (obj_offset, cell_h), (0, 1.0, 0), 1)

    partrticles_x_avg = img.copy()
    if Particle.nr_objects:
        positions_np = np.array([x.object_positions for x in particles])
        avg = np.average(positions_np, axis=0)

        for a in avg:
            obj_offset = int(a * block)
            partrticles_x_avg = cv2.line(partrticles_x_avg, (obj_offset, 0), (obj_offset, cell_h), (0, 1.0, 1.00), 3)


    img_final = np.vstack([img_robot_x, partrticles_x, partrticles_x_avg])
    
    imshow(img_final)
    cv2.imwrite(f'./images/{idx}.png', img_final)

def move_robot(particles: List[Particle], movement_steps):
    err_interval = abs(movement_steps * MOVEMENT_PROBABILITY) # 0.02 ==> 0.2 * N_SOLI

    for i, _ in enumerate(particles):
        error = np.random.normal(0, err_interval, 12)
        spread = np.sum(error) / 2
        particles[i].X += (spread + movement_steps)

    return particles


def push_object_probabilities(particles: List[Particle], object_distance):
    p = abs(object_distance * MEASUREMENT_PROBABILITY)

    for i, _ in enumerate(particles):
        particles[i].object_probabilities.append(p)

        print('\----')
        for a, _ in enumerate(particles):
            print(particles[a].object_probabilities, i)
            x=0
        x=0

def push_object_positions(particles: List[Particle], positions):
    for i, _ in enumerate(particles): 
        p = positions[i]
        particles[i].object_positions.append(p)

def push_object_others(particles: List[Particle]):
    for i, _ in enumerate(particles): 
        particles[i].Z.append(0)
        particles[i].Y.append(0)
        particles[i].Q.append(0)
        particles[i].S.append(0)
        particles[i].K.append(0)


def new_object(particles: List[Particle], positions, object_distance):
    push_object_probabilities(particles, object_distance)
    push_object_positions(particles, positions)
    push_object_others(particles)
    Particle.nr_objects += 1

def update_object(particles: List[Particle], object_idx):
    for i, _ in enumerate(particles): 
        particle = particles[i]
        Z = particle.X + object_distance
        particle.Z[object_idx] = Z

        prob = particle.object_probabilities[object_idx]
        pos = particle.object_positions[object_idx]

        Y = Z - pos  # Inovācija
        particle.Y[object_idx] = Y
        
        Q = abs(object_distance) * MEASUREMENT_PROBABILITY # Mērījuma nenoteiktība
        particle.Q[object_idx] = Q

        S = prob  + Q # Inovācijas kovariance
        particle.S[object_idx] = S

        K = prob * (1 / S) # Kalmana ieguvums
        particle.K[object_idx] = K

        particle.object_positions[object_idx] = pos + K * Y
        particle.object_probabilities[object_idx] = (1 - K) * prob #Jaunās iezīmes pozīcija
        W = pow(abs((2 * math.pi * S)), (-1 / 2)) * np.exp(-1 / 2 * Y * (1 / S) * Y)
        particle.W = W



def object_detected(particles: List[Particle], object_distance):
    positions = [p.X + object_distance for p in particles]

    # No objects at all
    if not Particle.nr_objects:
        new_object(particles, positions, object_distance)
        return

    # is this new object or already seen
    pos = [p.object_positions for p in particles]
    pos_np = np.array(pos)
    pos_np = np.swapaxes(pos_np, 1, 0)
    
    pos_avg = np.average(pos_np, axis=1)
    pos_unk = np.average(np.array(positions))

    is_new = True
    object_idx = -1
    for idx, other_pos in enumerate(pos_avg):
        if abs(pos_unk - other_pos) < DELTA:
            is_new = False
            object_idx = idx
            break

    if is_new:
        new_object(particles, positions, object_distance)
    else:
        print('Not new')
        update_object(particles, object_idx)

        intervals = np.cumsum([x.W for x in particles])
        intervals = np.insert(intervals, 0, 0, axis=0)
        step = sum([x.W for x in particles]) / N_PARTICLES

        indicies = []
        for n in range(N_PARTICLES):
            v = n * step
            for idx, (previous, current) in enumerate(zip(intervals, intervals[1:])):
                if v >= previous and v <= current:
                    indicies.append(idx)

        if len(indicies) != N_PARTICLES:
            print('Bad')
            exit(0)

        # reasign values based on picked indicies
        particles_copy = copy.deepcopy(particles)
        for old, new in zip(range(N_PARTICLES), indicies):
            particles[old] = copy.deepcopy(particles_copy[new])
    


    x = 0

# State
particles = [Particle(X=ROBOT_START_POSITION) for x in range(N_PARTICLES)]

show_location_progabilities(particles, 0)

# 1.) Robots parvietojas 3m pa labi
movement_steps = 3
move_robot(particles, movement_steps)
show_location_progabilities(particles, 1)

# 2.) Robots uztver objektu 2m pa pabi
object_distance = 2
object_detected(particles, object_distance)
show_location_progabilities(particles, 2)

# 3. ) Robots pārvietojas 4m pa kreisi
movement_steps = -4
move_robot(particles, movement_steps)
show_location_progabilities(particles, 3)

# 4.) Robots uztver objektu 2m pa kreisi
object_distance = -2
object_detected(particles, object_distance)
show_location_progabilities(particles, 4)

# 5. ) Robots pārvietojas 3m pa labi
movement_steps = 3
move_robot(particles, movement_steps)
show_location_progabilities(particles, 5)

# 6.) Robots VLREIZ uztver objektu 3m pa labi
object_distance = 3
object_detected(particles, object_distance)
show_location_progabilities(particles, 6)

# 7. ) Robots pārvietojas 1m pa kreisi
movement_steps = -1
move_robot(particles, movement_steps)
show_location_progabilities(particles, 7)

# 8.) Robots uztver objektu 3m pa kreisi
object_distance = -1
object_detected(particles, object_distance)
show_location_progabilities(particles, 8)