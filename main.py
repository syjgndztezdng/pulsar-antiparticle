from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from gtsimulation.Global import Units
from package import is_out, UpdatableBunemanBorisSimulator, static_dipole, rotating_dipole, SimulatorBuilder, energy_on_latitude



time = 1e-2
dt = 2e-8
n_steps = int(time / dt)
latitude = 45 / 180 * np.pi
m0 = 0.5e30 #амплитуда момента магнитного поля
w = 300 #частота вращения [Гц]
x = 20 / 180 * np.pi #отклонение магнитного поля от оси вращения [rad]
border = 1000000
precision = 0.05 * Units.TeV

R_star = 10000 #10km [m]

b_field = rotating_dipole(m0, w, x)

sim_b = SimulatorBuilder(UpdatableBunemanBorisSimulator)
sim_b.give_parametrs(Num=n_steps, Step=dt, Bfield=b_field, Save=[1, {"Coordinates": True, "Velocities": True, "Energy": True}], Verbose=0)

E = energy_on_latitude(sim_b, latitude, precision, R_star,  border)

print(f"На широте в {round(latitude / np.pi * 180, 1)} гр. максимальная энергия удержания составила: {round(E / Units.TeV, 2)} ТэВ")
