import datetime

import numpy as np
import matplotlib.pyplot as plt
from gtsimulation.MagneticFields import magnetic_field
from gtsimulation.Algos import BunemanBorisSimulator
from gtsimulation.Global import Constants, Units, Regions, codes
from gtsimulation.MagneticFields import Uniform
from gtsimulation.Particle.Generators import Distributions, Spectrums
from numba import jit
from gtsimulation import GTSimulator
from gtsimulation.Particle import ConvertT2R, GetAntiParticle, Flux
from package import is_out, UpdatableBunemanBorisSimulator, static_dipole, rotating_dipole




time = 2e-3
dt = 1e-9
n_steps = 2e6
m0 = 0.5e30 #амплитуда момента магнитного поля
w = 300 #частота вращения [Гц]
x = 20 / 180 * np.pi #отклонение магнитного поля от оси вращения [rad]
latitude = 60
border = 1000000
precision = 0.05 * Units.TeV
R_star = 10000
step = {"UseAdaptiveStep": True, "InitialStep": 1}
T = 10 * Units.TeV
r_0 = np.array([np.cos(latitude), 0, np.sin(latitude)]) * R_star
v_0 = np.array([np.cos(latitude), 0, np.sin(latitude)])


particle = Flux(
    Spectrum=Spectrums.Monolines(energy=T),
    Distribution=Distributions.UserInput(
        R0=r_0,
        V0=v_0
    ),
    Names="e-",
    Nevents=1
)

b_field = rotating_dipole(m0, w, x)

simulator = UpdatableBunemanBorisSimulator(
    Bfield=b_field,
    Region=Regions.Undefined,
    Medium=None,
    Particles=particle,
    InteractNUC=None,
    UseDecay=False,
    Date=datetime.datetime(2000, 1, 1),
    Step=step,
    Num=n_steps,
    ForwardTrck=1,
    Save=[1, {"Coordinates": True, "Velocities": True, "Energy": True}],
    Output=None,
    Verbose=2,
    BreakCondition={"Rmax": border},
    RadLosses=True
)
track = simulator()[0][0]

r = track["Track"]["Coordinates"]
v = track["Track"]["Velocities"]
e = track["Track"]["Energy"]

fig = plt.figure(figsize=(12, 6))

ax3d = fig.add_subplot(projection='3d')

ax3d.plot(*r.T)


ax3d.scatter(*r[0], label='Electron initial position')

# radius = R_star  # Радиус 5 см (0.05 метра)
# x_center, y_center, z_center = 0.0, 0.0, 0.0  # Центр сферы
#
# u = np.linspace(0, 2 * np.pi, 30)  # Долгота
# p = np.linspace(0, np.pi, 30)      # Широта
#
# x = x_center + radius * np.outer(np.cos(u), np.sin(p))
# y = y_center + radius * np.outer(np.sin(u), np.sin(p))
# z = z_center + radius * np.outer(np.ones(np.size(u)), np.cos(p))
#
# ax3d.plot_surface(x, y, z, color='b', alpha=0.3, edgecolor='navy', linewidth=0.5)


ax3d.set_xlabel("X [m]")
ax3d.set_ylabel("Y [m]")
ax3d.set_zlabel("Z [m]")
ax3d.axis('equal')
# ax3d.set_aspect('equal')
ax3d.legend()

#fig.subplots_adjust(wspace=0.3)

# ax2d = fig.add_subplot(2, 1, 2)
# ax2d.plot(e)

plt.show()
