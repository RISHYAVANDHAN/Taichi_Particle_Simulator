import numpy as np
import taichi as ti
import time

from numpy.linalg import norm
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

ti.init(arch=ti.gpu, default_ip=ti.i64, default_fp=ti.f64)

@ti.data_oriented
class Particle:
    def __init__(self, mass, r0, v0=None):
        self.mass = mass
        self.r0 = np.array(r0, dtype=np.float64)
        self.v0 = np.zeros(3) if v0 is None else np.array(v0, dtype=np.float64)
        self.DOF = None

    def slice(self, r):
        return r[self.DOF]

class ForceLaw:
    def la(self, t, l, l_dot):
        pass

class Spring(ForceLaw):
    def __init__(self, stiffness, undeformed_length):
        self.c = stiffness
        self.l0 = undeformed_length

    def la(self, t, l, l_dot):
        return -self.c * (l - self.l0)

@ti.data_oriented
class TwoPointInteraction:
    def __init__(self, particle1, particle2, force_law):
        self.p1 = particle1
        self.p2 = particle2
        self.force_law = force_law
        self.DOF = np.concatenate([particle1.DOF, particle2.DOF])

    def l(self, r):
        return norm(self.p2.slice(r) - self.p1.slice(r))

    def n(self, r):
        r12 = self.p2.slice(r) - self.p1.slice(r)
        return r12 / norm(r12)

    def l_dot(self, r, v):
        return np.dot(self.n(r), self.p2.slice(v) - self.p1.slice(v))

    def F(self, t, r, v):
        force = np.zeros(6)
        la = self.force_law.la(t, self.l(r), self.l_dot(r, v))
        n = self.n(r)
        force[:3] = -la * n
        force[3:] = la * n
        return force

class System:
    def __init__(self, t0=0, gravity=np.array([0, 0, -9.81])):
        self.particles = []
        self.interactions = []
        self.t0 = t0
        self.r0, self.v0, self.m, self.F0 = [], [], [], []
        self.gravity = gravity
        self.nDOF = 0  

    def add_particles(self, particles):
        for p in particles:
            p.DOF = np.arange(3) + self.nDOF
            self.nDOF += 3
            self.particles.append(p)
            self.r0.extend(p.r0)
            self.v0.extend(p.v0)
            self.m.extend(p.mass * np.ones(3))
            self.F0.extend(p.mass * self.gravity)

    def add_interactions(self, interactions):
        self.interactions.extend(interactions)

    def assemble(self):
        self.r0 = np.array(self.r0)
        self.v0 = np.array(self.v0)
        self.m = np.array(self.m)
        self.F0 = np.array(self.F0)

    def F(self, t, r, v):
        F = self.F0.copy()
        for i in self.interactions:
            F[i.DOF] += i.F(t, r, v)
        return F

def explicit_euler(system, h, tf):
    N = int((tf - system.t0) / h)
    t, r, v = np.zeros(N+1), np.zeros((N+1, len(system.r0))), np.zeros((N+1, len(system.v0)))

    t[0], r[0], v[0] = system.t0, system.r0, system.v0
    M_inv = 1 / system.m

    for k in range(N):
        t[k + 1] = t[k] + h
        r[k + 1] = r[k] + h * v[k]
        v[k + 1] = v[k] + h * M_inv * system.F(t[k], r[k], v[k])
    
    return t, r, v

if __name__ == "__main__":
    system = System()

    m, c, L = 1, 1, 1
    system.add_particles([
        Particle(m, [0, 0, 0]),
        Particle(m, [0, 1.1 * L, 0]),
        Particle(m, [1.2 * L, 0, 0])
    ])

    system.add_interactions([
        TwoPointInteraction(system.particles[0], system.particles[1], Spring(c, L)),
        TwoPointInteraction(system.particles[1], system.particles[2], Spring(c, L)),
        TwoPointInteraction(system.particles[2], system.particles[0], Spring(c, L))
    ])

    system.assemble()

    tf, h = 3, 1e-2

    start_time = time.time()
    t, r, v = explicit_euler(system, h, tf)
    end_time = time.time()
    print(f"Explicit Euler simulation with TAICHI parallelisation completed in {end_time - start_time:.4f} seconds")

    lengths = [[i.l(ri) for ri in r] for i in system.interactions]

    plt.figure(figsize=(10, 5))
    for i, length in enumerate(lengths, start=1):
        plt.plot(t, length, label=f'$l_{i}{(i+1)%3+1}$')
    plt.xlabel('Time (s)')
    plt.ylabel('Length (m)')
    plt.title('Spring Lengths Over Time')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Animation
    fps = 20
    N_frames = int((tf - system.t0) * fps)
    frac = max(1, int(np.ceil(len(t) / N_frames)))
    r_frames = r[::frac]

    fig, ax = plt.subplots()
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    particles, = ax.plot([], [], 'bo', markersize=8)

    def init():
        particles.set_data([], [])
        return particles,

    def update(frame):
        x, y = r_frames[frame, 0::3], r_frames[frame, 1::3]
        particles.set_data(x, y)
        return particles,

    ani = FuncAnimation(fig, update, frames=len(r_frames), init_func=init, blit=True, interval=20)
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.title('Three Particle Animation')
    plt.show()
