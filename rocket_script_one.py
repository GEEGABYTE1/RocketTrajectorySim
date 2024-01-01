from poliastro.plotting.static import StaticOrbitPlotter
from poliastro.bodies import Earth
from poliastro.twobody import Orbit
from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u


# Rocket parameters
diam = 10.1  # m rocket diameter
A = np.pi / 4 * (diam) ** 2  # m^2 frontal area
CD = 0.5  # Drag Coefficient (approximation)
mprop = 131000  # kg propellant mass (first stage)
mpl = 46000  # kg payload mass
mstruc = 131000  # kg structure mass (first stage)
m0 = mprop + mstruc + mpl  # total lift-off mass (kg)
tburn = 168  # Burn time (s)
Thrust = 34000000  # N rocket thrust of Saturn V (first stage)
gravity_turn_start = 10  # s, time to start gravity turn
gravity_turn_rate = 0.1  # rad/s gravity turn rate
hturn = 1000 #pitchover height
# Additional parameters
deg = np.pi / 180  # Conversion factor from degrees to radians
t_max = 600  # s, how long the sim runs
t = np.linspace(0, t_max, 10000)  # time vector with finer intervals
v0 = 0  # m/s initial velocity
psi0 = 0.3 * deg  # rad initial flight path angle
theta0 = 0.0  # rad initial downrange angle


# Function representing the rocket dynamics with gravity turn and variable thrust
# Function representing the rocket dynamics with gravity turn and variable thrust
# ... (previous code remains unchanged)

# Simplified thrust profile
def thrust_profile(t):
    return Thrust * (1 - t / tburn) if t < tburn else 0.0

# Modified rocket dynamics function
def rocket_dynamics_variable_thrust(y, t):
    h, v, psi, theta = y
    rho = 1.225 * np.exp(-h / 8000)

    # Drag force
    drag = 0.5 * rho * v**2 * A * CD

    # Gravitational force
    gravity = 9.81 * (m0 / (m0 - mprop / tburn * np.max(t)))**2 if np.max(t) > 0 else 0

    # Variable thrust profile
    #thrust = thrust_profile(t)
    if t < tburn:
        m = m0 - ((mprop/tburn) * t)
        T = Thrust
    else:
        m = m0 - ((mprop / tburn) * tburn)
        T = 0

    if h <= hturn:
        psi_dot = 0
        v_dot = T/m - drag/m - gravity
        theta_dot = 0
        h_dot = v
    else:
        phi_dot = gravity * np.sin(psi) / v
        v_dot = T/m - drag/m - gravity * np.cos(psi)
        h_dot = v * np.cos(psi)
        theta_dot = v * np.sin(psi) / (Re + h)


# Solve the differential equations with variable thrust
y0 = [0, 0, psi0, theta0]
sol_variable_thrust = odeint(rocket_dynamics_variable_thrust, y0, t)

# Extract results for variable thrust
altitude_variable_thrust = sol_variable_thrust[:, 0]
velocity_variable_thrust = sol_variable_thrust[:, 1]
flight_path_angle_variable_thrust = sol_variable_thrust[:, 2]
downrange_angle_variable_thrust = sol_variable_thrust[:, 3]

# ... (rest of the code remains unchanged)

# Function to integrate and calculate the orbit with variable thrust
def stage(t, orbit, thrust_angle, burn_rate):
    state = np.zeros_like(orbit.rv())
    state[:3] = orbit.rv()[:3]
    state[3:] = rocket_dynamics_variable_thrust(state, t)[:-1]
    return state

# Initial conditions


# Solve the differential equations with variable thrust
sol_variable_thrust = odeint(rocket_dynamics_variable_thrust, y0, t)



# Calculate downrange distance for variable thrust
downrange_distance_variable_thrust = np.cumsum(
    velocity_variable_thrust * np.sin(flight_path_angle_variable_thrust) * (t_max / len(t))
)

# Print some diagnostic information
print("Size of t:", len(t))
print("Size of altitude_variable_thrust:", len(altitude_variable_thrust))

# Check the first few altitude values
if len(altitude_variable_thrust) > 0:
    print("First 10 altitude values:", altitude_variable_thrust[:10])

# Plot height vs downrange distance for variable thrust
plt.figure(figsize=(10, 6))
plt.plot(downrange_distance_variable_thrust, altitude_variable_thrust)
plt.title('Rocket Trajectory (Variable Thrust)')
plt.xlabel('Downrange Distance (m)')
plt.ylabel('Height (m)')
plt.grid(True)
plt.show()

# Plot height vs time for variable thrust
plt.figure(figsize=(10, 6))
plt.plot(t, altitude_variable_thrust)
plt.title('Rocket Height vs Time (Variable Thrust)')
plt.xlabel('Time (s)')
plt.ylabel('Height (m)')
plt.grid(True)
plt.show()


