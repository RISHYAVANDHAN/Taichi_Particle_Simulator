import numpy as np

from numpy.linalg import norm
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

class Particle:
    def __init__(self, mass, r0=np.zeros(3), v0=np.zeros(3)):
        self.mass = mass
        self.r0 = r0
        self.v0 = v0
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
        return - self.c * (l - self.l0)


class TwoPointInteraction:
    def __init__(self, particle1, particle2, force_law):
        self.particle1 = particle1
        self.particle2 = particle2
        self.force_law = force_law
        self.DOF = np.concatenate([particle1.DOF, particle2.DOF])
    
    def l(self, r):
        r12 = self.particle2.slice(r) - self.particle1.slice(r)
        return norm(r12)
    
    def n(self, r):
        r12 = self.particle2.slice(r) - self.particle1.slice(r)
        return r12 / norm(r12)

    def l_dot(self, r, v):
        v12 = self.particle2.slice(v) - self.particle1.slice(v)
        return np.dot(self.n(r), v12)

    def F(self, t, r, v):
        F = np.zeros(6)
        l = self.l(r)
        l_dot = self.l_dot(r, v)
        la = self.force_law.la(t, l, l_dot)
        n = self.n(r)
        F[:3] = - la * n
        F[3:] = la * n
        return F

class System:
    def __init__(self, t0=0, gravity=np.array([0, 0, -9.81])):
        self.particles = []
        self.interactions = []
        self.t0 = t0
        self.r0 = []
        self.v0 = []
        self.F0 = []
        self.m = []
        self.gravity = gravity
        self.last_particle_index = 0
        self.nDOF = 0 # number of degrees of freedom

    def add_particles(self, particles):
        for p in particles:
            p.DOF = np.arange(3) + self.last_particle_index
            self.last_particle_index += 3
            self.nDOF += 3
            self.particles.append(p)
            self.r0.extend(p.r0)
            self.v0.extend(p.v0)
            self.m.extend(p.mass * np.ones(3))
            self.F0.extend(p.mass * self.gravity)

    def add_interactions(self, interactions):
        # TODO: maybe just use extend here?
        for i in interactions:
            self.interactions.append(i)

    def assemble(self):
        # assemble diagonal of global mass matrix
        self.r0 = np.array(self.r0)
        self.v0 = np.array(self.v0)
        self.m = np.array(self.m)
        self.F0 = np.array(self.F0)

    def F(self, t, r, v):
        # compute global force vector for given state
        F = self.F0.copy()
        for i in self.interactions:
            F[i.DOF] += i.F(t, r, v)
        return F

def explicit_euler(system, h, tf):
    t0 = system.t0
    r0 = system.r0
    v0 = system.v0
    # Allocate memory for saving the solution
    N = int((tf - t0) / h)  # Number of time steps
    nt = N + 1              # Number of time nodes
    nr = len(r0)            # Number of position coordinates
    nv = len(v0)            # Number of velocity coordinates
    t = np.zeros(nt)        # Array of time instants
    r = np.zeros((nt, nr))  # Array of positions
    v = np.zeros((nt, nv))  # Array of velocities

    # Set initial conditions
    t[0] = t0
    r[0] = r0
    v[0] = v0

    # Inverse mass matrix
    M_inv = 1 / system.m

    # Begin the simulation for loop
    for k in range(N):
        # Update time
        t[k + 1] = t[k] + h
        # Update position using explicit Euler
        r[k + 1] = r[k] + h * v[k]
        # Update velocity using explicit Euler
        v[k + 1] = v[k] + h * M_inv * system.F(t[k], r[k], v[k])
    return t, r, v


if __name__ == "__main__":

    system = System()

    m = 1
    c = 1
    L = 1

    system.add_particles(
        [
            Particle(m, r0=np.zeros(3)), 
            Particle(m, r0=np.array([0, 1.1 * L, 0])),
            Particle(m, r0=np.array([1.2 * L, 0, 0]))
        ]
        )
    
    system.add_interactions(
        [
            TwoPointInteraction(system.particles[0], system.particles[1], Spring(c, L)),
            TwoPointInteraction(system.particles[1], system.particles[2], Spring(c, L)),
            TwoPointInteraction(system.particles[2], system.particles[0], Spring(c, L))
        ]
    )
    system.assemble()

    # Define the simulation parameters
    tf = 3               # Final time (s)
    h = 1e-2             # Time step (s)

    # Simulate using both methods
    t, r, v = explicit_euler(system, h, tf)

    # Compute lengths
    lengths_expl1 = [system.interactions[0].l(ri) for ri in r]
    lengths_expl2 = [system.interactions[1].l(ri) for ri in r]
    lengths_expl3 = [system.interactions[2].l(ri) for ri in r]

    # Plot the lengths over time on the same plot
    plt.figure(figsize=(10, 5))
    plt.plot(t, lengths_expl1, label='$l_12$', linestyle='-')
    plt.plot(t, lengths_expl2, label='$l_23$', linestyle='-')
    plt.plot(t, lengths_expl3, label='$l_31$', linestyle='-')
    plt.xlabel('Time (s)')
    plt.ylabel('Length (m)')
    plt.title('Length of the Springs Over Time')
    plt.legend()
    plt.grid(True)

    plt.show()


    ###########
    # Animation
    fps = 20
    N_frames = (tf - system.t0) * fps
    N = int((tf - system.t0) / h)
    frac = int(np.ceil(N / N_frames))
    r_frames = r[::frac]
    N_frames = r_frames.shape[0]

    fig, ax = plt.subplots()
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)

    trail_length = 2

    particles, = ax.plot([], [], 'bo', markersize=8)

    def init():
        particles.set_data([], [])
        return particles,

    def update(frame):
        x = r[frame, 0::3]
        y = r[frame, 1::3]
        particles.set_data(x, y)

        return particles,

    ani = FuncAnimation(fig, update, frames=len(t), init_func=init, blit=True, interval=20)
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.title('Three Particle Animation with Fading Trace')
    plt.show()