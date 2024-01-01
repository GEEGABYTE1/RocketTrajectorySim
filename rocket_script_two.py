from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt



diam = 3.05 #m rocket diameter
A = np.pi/4 * (diam)**2 #m^2 frontal area
CD = 0.3 #Drag Coefficient
mprop = 111130 #kg propellant mass
mpl = 32000 #kg payload mass
mstruc = 6736 #kg structure mass
m0 = mprop + mstruc + mpl # total lift off mass (kg)
tburn = 356 #Burn time (s)
m_dot = mprop/tburn
Thrust = 1900000 # N rocket thrust of titan 2
hturn = 1000 #m pitchover height
deg = np.pi / 180
psi0 = 0.3*deg # rad initial flight path angle

#differential equation inputs
t_max = 8000 #s, how long the sim runs
t = np.linspace(0, t_max, 100000) # time_vector
v0 = 0 #m/s initial velocity

theta0 = 0 #read initial downrange angle
h0 = 0 # km initial altitude
Re = 6378100 # Radius of Earth in KM

def derivatives(t, y):
    v = y[0]
    psi = y[1]
    theta = y[2]
    h = y[3]

    #determining the gravity and drag
    g = 9.81 / (1 + h/Re) ** 2
    rho = 1.225 * np.exp(-h / 8000)
    D = 1/2 * rho * v ** 2 * A * CD
    print('h: ', h)
    print('D: ', D)
    #update thrust and mass based on if vehicle is still burning
    if t < tburn:
        m = m0 - m_dot*t
        T = Thrust
    else:
        m = m0 - m_dot * tburn
        T = 0

    #defining outputs
    if h <= hturn:
        psi_dot =0
        v_dot = T / m - (D/m) -g
        theta_dot = 0
        h_dot = v
    else:
        phi_dot = g * np.sin(psi) / v
        v_dot = T/m - D/m - g * np.cos(psi)
        h_dot = v * np.cos(psi)
        theta_dot = v * np.sin(psi) / (Re + h)
        psi_dot = phi_dot - theta_dot

    return [v_dot, psi_dot, theta_dot, h_dot]

sol = solve_ivp(derivatives, [t[0], t[-1]], [v0, psi0, theta0, h0], max_step=1)

vrel = sol.y[0]/1000 # % km/s velocity WITHOUT rotation of earth
psi = sol.y[1] # rad flight path angle
psideg = psi/deg
theta = sol.y[2] # rad downrange angle
dr = theta*Re/1000 # km downrange distance
h = sol.y[3]/1000 # km altitude
htot = h + Re/1000 #km total
t = sol.t
print(sol)


# Plot downrange distance vs time
plt.figure(figsize=(10, 6))
plt.plot(t, dr)
plt.title('Downrange Distance vs Time')
plt.xlabel('Time (s)')
plt.ylabel('Downrange Distance (km)')
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(t, vrel)
plt.title('Velocity vs Time')
plt.xlabel('Time (s)')
plt.ylabel('Velocity (km/s)')
plt.grid(True)


plt.show()


plt.figure(figsize=(10, 6))
plt.plot(t, psideg)
plt.title('flight path angle vs Time')
plt.xlabel('Time (s)')
plt.ylabel('flight path angle')
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(t, htot)
plt.title('height vs Time')
plt.xlabel('Time (s)')
plt.ylabel('downrange angle')
plt.grid(True)
plt.show()

plt.figure(figsize=(8, 8))
plt.polar(theta, htot, label='Trajectory')
plt.title('Rocket Trajectory (Polar Plot)')
plt.legend()
plt.show()