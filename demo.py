import numpy as np
from numba import njit

from gtsimulation.Global import Units
from gtsimulation.MagneticFields import AbsBfield
from gtsimulation.Particle.Generators import Distributions, Spectrums


class dipole_field(AbsBfield):
    def __init__(self, m0: float) -> None:
        self.m0 = m0
        self.m_vector = np.array([0, 0, 1]) * m0
        self.use_meters = True
        self.use_tesla = True


    def CalcBfield(self, x, y, z):
        return np.array(self.__calc_bfield(x, y, z, self.m_vector))
    
    @staticmethod
    @njit(fastmath=True)
    def __calc_bfield(x, y, z, m_vector):
        r_mag2 = x**2 + y**2 + z**2
        r_mag = np.sqrt(r_mag2)
    
        if r_mag == 0.0:
            return 0.0, 0.0, 0.0
        
        dot_product = m_vector[0]*x + m_vector[1]*y + m_vector[2]*z
    
        r3_inv = 1.0 / (r_mag2 * r_mag)       # 1 / r^3
        r5_inv = r3_inv / r_mag2              # 1 / r^5
    
        factor1 = 3.0 * dot_product * r5_inv  
        factor2 = r3_inv                       
    
        Bx = 1e-7 * (factor1 * x - factor2 * m_vector[0])
        By = 1e-7 * (factor1 * y - factor2 * m_vector[1])
        Bz = 1e-7 * (factor1 * z - factor2 * m_vector[2])
    
        return Bx, By, Bz

    def UpdateState(self, newDate):
        pass

    def to_string(self):
        s = f"""Type: Dipole Magnetic Field
        Version: 1.0"""

        return s
dt = 1e-9
n_steps = 1e3

m0 = 0.5e50 #амплитуда момента магнитного поля
border = 1000000
R_star = 10000
step = {"UseAdaptiveStep": True, "InitialStep": 1}
radLosses = [True, {"Photons": True, "MinE": Units.GeV, "MaxE": 1e20}]

```python
from PhotonSimulator import PhotonSimulator

T = 1000000 * Units.MeV
latitude = 0
r_0 = np.array([10000, 0, 0]) 
v_0 = np.array([np.cos(latitude), 0, np.sin(latitude)])

ph = Flux(
    Spectrum=Spectrums.Monolines(energy=T),
    Distribution=Distributions.UserInput(
        R0=r_0,
        V0=v_0
    ),
    PDGcode=22,
    Nevents=1
)

b_field = dipole_field(m0)

sim = PhotonSimulator(
    Bfield=b_field,
    Particles=ph,
    Step=dt,
    Num=n_steps,
    Save=[1, {"Coordinates": True, "Velocities": True, "Energy": True}],
    Verbose=2,
    PMConversion=True,
)
```

```python
track = sim()[0][0]
```

```python
r = track["Track"]["Coordinates"]
ch = track["Child"]
    
print(ch)
```

```python

```
