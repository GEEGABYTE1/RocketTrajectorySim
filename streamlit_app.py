import streamlit as st
from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt

# Sidebar input widgets
st.sidebar.title("Rocket Parameters")
diam = st.sidebar.number_input("Rocket Diameter (m)", value=3.05)
A = np.pi/4 * (diam)**2
CD = st.sidebar.number_input("Drag Coefficient", value=0.3)
mprop = st.sidebar.number_input("Propellant Mass (kg)", value=111130)
mpl = st.sidebar.number_input("Payload Mass (kg)", value=32000)
mstruc = st.sidebar.number_input("Structure Mass (kg)", value=6736)
m0 = mprop + mstruc + mpl
tburn = st.sidebar.number_input("Burn Time (s)", value=356)
m_dot = mprop / tburn
Thrust = st.sidebar.number_input("Rocket Thrust (N)", value=1900000)
hturn = st.sidebar.number_input("Pitchover Height (m)", value=1000)
deg = np.pi / 180
psi0 = st.sidebar.number_input("Initial Flight Path Angle (deg)", value=0.3) * deg

# Main content
st.title("Rocket Trajectory Simulation 🚀")
st.write(
    "A Rocket Trajectory Simulation inspired by my work at UTAT created from scratch. " 
    "With the given rocket parameters on the left, one can find relevant information about the behaviour of the rocket upon liftoff under the **'Results'** section of this page. "
    "Please note that the simulation only considers **single stage** rockets and does **not consider re-entry**. " 

)
st.write("However, the model was trained with the Gemini-Titan II Rocket, and was compared with the Gemini mission data in 1964 from NASA. Based on the comparisions, the simulation gives an error of **7%**.  "
    "Limitations and improvements of the model are found at the bottom of the page under the **'Discrepencies'** section.")

st.subheader("The Math/Physics Behind the Simulation ⚛️")
st.write(
    "The motion of the rocket is described by a set of ordinary differential equations (ODEs) that govern its "
    "behavior through the atmosphere. The key equations used in the simulation are found below."
)

st.write("**The Rocket Velocity ODE**")
st.latex(r"""
\begin{align*}
    \frac{dv}{dt} &= \frac{T}{m} - \frac{D}{m} - g 
\end{align*}
""")
st.write("This equation represents the rate of change of rocket velocity with respect to time. It is influenced by three main factors: thrust ($T$) divided by the instantaneous mass ($m$), drag force ($D$) divided by the instantaneous mass, and gravitational acceleration ($g$).")
st.write("- **Thrust Component**: The term $T$/$m$ represents the contribution of rocket thrust to the acceleration. As long as the rocket is burning fuel (during the burn time), thrust contributes positively to the acceleration.")
st.write("- **Drag Component**: The term $D$/$m$ represents the negative contribution of aerodynamic drag to the acceleration. Drag opposes the rocket's motion and depends on the square of velocity ($v$).")
st.write("- **Gravity Component**: The term $g$ represents the acceleration due to gravity. It decreases with altitude and is inversely proportional to the square of the distance from the Earth's center.")

st.write("**The Flight Path Angle ODE**")
st.latex(r"""
\begin{align*}
    \frac{d\psi}{dt} &= \phi' - \theta' \\
\end{align*}
""")
st.write("This equation represents the rate of change of the flight path angle ($\psi$) with respect to time. It is influenced by the pitch rate ($\phi'$) and the rate of change of the downrange angle ($\\theta'$), adjusting for the tangent of the flight path angle.")
st.write("- **Pitch Rate Component**: The term $\phi'$ represents the pitch rate, influencing the change in the flight path angle. It accounts for the rotation of the rocket around its axis.")
st.write("- **Downrange Angle Component**: The term $\\theta'$ adjusts the flight path angle change based on the rate of change of the downrange angle. It corrects for the effect of the rocket's downrange motion on the flight path angle.")


st.write("**Downrange Angle ODE**")
st.latex(r"""
\begin{align*}
    \frac{d\theta}{dt} &= \frac{v \sin(\psi)}{R_e + h} \\
\end{align*}
""")
st.write("This equation represents the rate of change of the downrange angle ($\\theta$) with respect to time. It is influenced by the rocket's velocity ($v$), the sine of the flight path angle ($\sin(\psi)$), and the inverse of the sum of the Earth's radius ($R_e$) and altitude ($h$).")
st.write("- **Velocity Component**: The term $v \sin(\psi)$ represents the component of rocket velocity perpendicular to the flight path. It contributes to the downrange motion.")
st.write("- **Altitude Component**: The denominator $(R_e + h)$ adjusts the downrange angle change based on the rocket's altitude. The higher the altitude, the smaller the impact of downrange motion.")

st.write("**Altitude ODE**")
st.latex(r"""
\begin{align*}
    \frac{dh}{dt} &= v \cos(\psi)
\end{align*}
""")
st.write("This equation represents the rate of change of altitude ($h$) with respect to time. It is influenced by the rocket's velocity ($v$) and the cosine of the flight path angle ($\cos(\psi)$).")
st.write("- **Velocity Component**: The term $v \cos(\psi)$ represents the component of rocket velocity parallel to the flight path. It contributes to changes in altitude.")
st.write("- **Flight Path Angle Component**: The cosine factor adjusts the altitude change based on the angle of the rocket's trajectory. It considers the vertical component of velocity.")

st.write("Where: ")
st.write("&emsp;" + "&emsp;" + "$v$: Rocket velocity")
st.write("&emsp;" + "&emsp;" + "&emsp;" + "&emsp;" + "- Represents how the rocket's velocity changes over time. It considers the influence of thrust, drag, and gravity on the rocket's acceleration.")
st.write("&emsp;" + "&emsp;" +  "$\\psi$: Flight path angle")
st.write("&emsp;" + "&emsp;" + "&emsp;" + "&emsp;" + "- The change in flight path angle is influenced by the difference between the rocket's pitch rate ($\\phi'$) and downrange rate ($\\theta'$).\n")
st.write("&emsp;" + "&emsp;" + "$\\theta$: Downrange angle")
st.write("&emsp;" + "&emsp;" + "&emsp;" + "&emsp;" + "- Describes how the downrange angle changes with time. It depends on the rocket's velocity and flight path angle.\n")
st.write("&emsp;" + "&emsp;" + "$h$: Altitude")
st.write("&emsp;" + "&emsp;" + "&emsp;" + "&emsp;" + "- Determines how the rocket's altitude changes over time, influenced by the rocket's vertical velocity.\n")
st.write("&emsp;" + "&emsp;" + "$T$: Rocket thrust")
st.write("&emsp;" + "&emsp;" + "$m$: Instantaneous rocket mass")
st.write("&emsp;" + "&emsp;" + "&emsp;" + "&emsp;" + "- Or the mass of the rocket overall before takeoff.")
st.write("&emsp;" + "&emsp;" + "$D$: Drag force")
st.write("&emsp;" + "&emsp;" + "$g$: Gravitational acceleration")
st.write("&emsp;" + "&emsp;" + "$R_e$: Radius of the Earth")


st.subheader("Some Physical Principles Considered (but not all)")
st.write(
    "The simulation considers several physical principles including:\n"
    "   - **Gravity and Drag:** Effects of gravity and air drag are incorporated. Gravity pulls the rocket downward, while "
    "drag opposes its motion through the atmosphere.\n"
    "   - **Thrust and Mass:** Thrust force propels the rocket upward, and the changing mass due to propellant burn is "
    "considered. These factors impact the rocket's acceleration.\n"
    "   - **Pitchover Height:** The simulation accounts for a pitchover height, beyond which the trajectory changes. This "
    "is a crucial point in the rocket's ascent.\n"
    "   - **Atmospheric Conditions:** Atmospheric density is modeled using the exponential atmosphere model. This accounts "
    "for variations in air density with altitude.\n"
)



st.title("Your Results 🤩")


# Differential equation inputs
t_max = 1400
t = np.linspace(0, t_max, 100000)
v0 = 0
theta0 = 0
h0 = 0
Re = 6378100

def derivatives(t, y):
    # (unchanged code)
    v = y[0]
    psi = y[1]
    theta = y[2]
    h = y[3]

    #determining the gravity and drag
    g = 9.81 / (1 + h/Re) ** 2
    rho = 1.225 * np.exp(-h / 8000)
    D = 1/2 * rho * v ** 2 * A * CD
    #print('h: ', h)
    #print('D: ', D)
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

# Solve the differential equation
sol = solve_ivp(derivatives, [t[0], t[-1]], [v0, psi0, theta0, h0], max_step=1)

# Extract data for plotting
vrel = sol.y[0] / 1000
psi = sol.y[1]
psideg = psi / deg
theta = sol.y[2]
dr = theta * Re / 1000
h = sol.y[3] / 1000
htot = h + Re / 1000
t = sol.t


# Plotting
st.subheader("Downrange Distance vs Time")
plt.figure(figsize=(10, 6))
plt.plot(t, dr)
plt.title('Downrange Distance vs Time')
plt.xlabel('Time (s)')
plt.ylabel('Downrange Distance (km)')
plt.grid(True)
st.pyplot(plt)

st.subheader("Velocity vs Time")
plt.figure(figsize=(10, 6))
plt.plot(t, vrel)
plt.title('Velocity vs Time')
plt.xlabel('Time (s)')
plt.ylabel('Velocity (km/s)')
plt.grid(True)
st.pyplot(plt)

st.subheader("Flight Path Angle vs Time")
plt.figure(figsize=(10, 6))
plt.plot(t, psideg)
plt.title('Flight Path Angle vs Time')
plt.xlabel('Time (s)')
plt.ylabel('Flight Path Angle (deg)')
plt.grid(True)
st.pyplot(plt)

st.subheader("Height vs Time")
plt.figure(figsize=(10, 6))
plt.plot(t, htot)
plt.title('Height vs Time')
plt.xlabel('Time (s)')
plt.ylabel('Height (km)')
plt.grid(True)
st.pyplot(plt)

st.subheader("Rocket Trajectory Polar Plot")
plt.figure(figsize=(8, 8))
plt.polar(theta, htot, label='Trajectory')
plt.title('Rocket Trajectory')
plt.legend()
st.pyplot(plt)

st.title("Discrepencies of the Model ⚙️")
st.write(
    "The simulation model has certain assumptions and limitations. It does not consider various "
    "real-world factors such as atmospheric variations, wind effects, and structural complexities. As a "
    "simplified model, discrepancies may exist between the simulation and actual rocket behavior. Improvements "
    "could involve incorporating more advanced aerodynamic models, considering varying atmospheric conditions, "
    "and accounting for external disturbances. This project serves as a starting point for exploration and learning."
)

st.write("Certain Limitations (but not all) in this simulation include, which may have overestimated or underestimated the results: ")
st.write("- The Drag Equation is simplistic and is assumed constant.")
st.write("- The simulation assumes the rocket has **only one stage**.")
st.write('- The thrust is constant. If this changes, the simulation would have to consider a trajectory optimization, which is another project on its own.')
st.write("- The gravity turn information may not be exact.")
st.write("- IVP Solution Accuracy may hinder based on inputs.")

st.write("I do aim to improve this simulation. If you have any ideas or want to work on this simulation, feel free to reach out to me! ")

footer = """
---
"""
st.markdown(footer)
st.markdown("<i><p style='text-align: center;'>Made by <a href='www.jaivalpatel.com'>Jaival Patel</a> with ❤️</p></i>", unsafe_allow_html=True)