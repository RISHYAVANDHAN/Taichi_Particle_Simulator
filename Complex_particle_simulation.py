import numpy as np
import taichi as ti
from numpy.linalg import norm
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Initialize Taichi with GPU acceleration if available
ti.init(arch=ti.gpu, default_fp=ti.f32, debug=False)

# Define Taichi data structures
@ti.data_oriented
class TaichiSystem:
    def __init__(self, n_particles, n_interactions, n_steps, dt):
        # System parameters
        self.n_particles = n_particles
        self.n_interactions = n_interactions
        self.n_steps = n_steps
        self.dt = dt
        
        # Particle data
        self.mass = ti.field(ti.f32, shape=n_particles)
        self.pos = ti.Vector.field(3, ti.f32, shape=(n_steps, n_particles))
        self.vel = ti.Vector.field(3, ti.f32, shape=(n_steps, n_particles))
        self.force = ti.Vector.field(3, ti.f32, shape=n_particles)
        
        # Interaction data
        self.spring_connections = ti.field(ti.i32, shape=(n_interactions, 2))
        self.spring_stiffness = ti.field(ti.f32, shape=n_interactions)
        self.spring_rest_length = ti.field(ti.f32, shape=n_interactions)
        self.spring_lengths = ti.field(ti.f32, shape=(n_steps, n_interactions))
        
        # Gravity
        self.gravity = ti.Vector([0.0, 0.0, -9.81])

    @ti.kernel
    def initialize(self):
        # Reset forces
        for i in range(self.n_particles):
            self.force[i] = self.mass[i] * self.gravity

    @ti.kernel
    def compute_forces(self, step: ti.i32):
        # Reset forces to gravity
        for i in range(self.n_particles):
            self.force[i] = self.mass[i] * self.gravity
        
        # Compute spring forces
        for i in range(self.n_interactions):
            p1 = self.spring_connections[i, 0]
            p2 = self.spring_connections[i, 1]
            
            # Get positions of the two particles
            pos1 = self.pos[step, p1]
            pos2 = self.pos[step, p2]
            
            # Calculate the spring vector and its length
            r12 = pos2 - pos1
            length = r12.norm()
            
            # Store the current length
            self.spring_lengths[step, i] = length
            
            # Calculate the force direction
            direction = r12.normalized()
            
            # Hooke's law F = -k(l - l0)
            force_magnitude = self.spring_stiffness[i] * (length - self.spring_rest_length[i])
            force_vector = direction * force_magnitude
            
            # Apply forces to both particles (action-reaction)
            self.force[p1] += force_vector
            self.force[p2] -= force_vector

    @ti.kernel
    def explicit_euler_step(self, step: ti.i32):
        # Update positions and velocities using explicit Euler
        for i in range(self.n_particles):
            # Update position
            self.pos[step+1, i] = self.pos[step, i] + self.dt * self.vel[step, i]
            
            # Update velocity
            self.vel[step+1, i] = self.vel[step, i] + self.dt * (self.force[i] / self.mass[i])

    def run_simulation(self):
        self.initialize()
        
        for step in range(self.n_steps - 1):
            self.compute_forces(step)
            self.explicit_euler_step(step)

# Original classes from the provided code
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

    def to_taichi_system(self, n_steps, dt):
        # Create a TaichiSystem from this System
        ts = TaichiSystem(len(self.particles), len(self.interactions), n_steps, dt)
        
        # Copy particle data
        for i, p in enumerate(self.particles):
            ts.mass[i] = p.mass
            ts.pos[0, i] = ti.Vector(p.r0)
            ts.vel[0, i] = ti.Vector(p.v0)
        
        # Copy interaction data
        for i, interaction in enumerate(self.interactions):
            ts.spring_connections[i, 0] = self.particles.index(interaction.particle1)
            ts.spring_connections[i, 1] = self.particles.index(interaction.particle2)
            ts.spring_stiffness[i] = interaction.force_law.c
            ts.spring_rest_length[i] = interaction.force_law.l0
            
        return ts

# Taichi visualization class
@ti.data_oriented
class TaichiVisualizer:
    def __init__(self, taichi_system, window_size=512):
        self.ts = taichi_system
        self.window_size = window_size
        self.gui = ti.GUI("Particle System", res=(window_size, window_size))
        self.scale = 100  # Scaling factor for visualization
        self.center = np.array([0.5, 0.5])  # Center of the window
        
    def visualize(self, step):
        # Extract 2D positions (ignoring z-coordinate)
        positions = []
        for i in range(self.ts.n_particles):
            pos = self.ts.pos[step, i].to_numpy()
            # Scale and center the position for visualization
            pos_2d = np.array([pos[0], pos[1]]) * self.scale + self.window_size * self.center
            positions.append(pos_2d / self.window_size)  # Normalize to [0, 1]
        
        # Draw particles
        self.gui.circles(np.array(positions), radius=8, color=0x0000FF)
        
        # Draw springs
        for i in range(self.ts.n_interactions):
            p1 = self.ts.spring_connections[i, 0]
            p2 = self.ts.spring_connections[i, 1]
            
            pos1 = self.ts.pos[step, p1].to_numpy()
            pos2 = self.ts.pos[step, p2].to_numpy()
            
            # Scale and center the positions
            pos1_2d = np.array([pos1[0], pos1[1]]) * self.scale + self.window_size * self.center
            pos2_2d = np.array([pos2[0], pos2[1]]) * self.scale + self.window_size * self.center
            
            # Normalize to [0, 1]
            pos1_2d = pos1_2d / self.window_size
            pos2_2d = pos2_2d / self.window_size
            
            self.gui.line(pos1_2d, pos2_2d, radius=2, color=0xFF0000)
        
        return self.gui.show()

# Integration function using Taichi
def taichi_explicit_euler(system, h, tf):
    t0 = system.t0
    n_steps = int((tf - t0) / h) + 1
    
    # Convert to Taichi system
    ts = system.to_taichi_system(n_steps, h)
    
    # Run simulation
    ts.run_simulation()
    
    # Create time array
    t = np.linspace(t0, tf, n_steps)
    
    # Extract position and velocity data
    r = np.zeros((n_steps, system.nDOF))
    v = np.zeros((n_steps, system.nDOF))
    
    # Transfer data from Taichi fields to numpy arrays
    for step in range(n_steps):
        for i in range(len(system.particles)):
            pos = ts.pos[step, i].to_numpy()
            vel = ts.vel[step, i].to_numpy()
            
            r[step, i*3:i*3+3] = pos
            v[step, i*3:i*3+3] = vel
    
    # Extract spring lengths
    lengths = np.zeros((n_steps, len(system.interactions)))
    for step in range(n_steps):
        for i in range(len(system.interactions)):
            lengths[step, i] = ts.spring_lengths[step, i]
    
    return t, r, v, ts, lengths

# Main function
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

    # Simulate using Taichi
    t, r, v, ts, lengths = taichi_explicit_euler(system, h, tf)

    # Plot the lengths over time on the same plot
    plt.figure(figsize=(10, 5))
    plt.plot(t, lengths[:, 0], label='$l_12$', linestyle='-')
    plt.plot(t, lengths[:, 1], label='$l_23$', linestyle='-')
    plt.plot(t, lengths[:, 2], label='$l_31$', linestyle='-')
    plt.xlabel('Time (s)')
    plt.ylabel('Length (m)')
    plt.title('Length of the Springs Over Time')
    plt.legend()
    plt.grid(True)
    plt.savefig('spring_lengths.png')
    plt.show()

    # Visualize with Taichi
    visualizer = TaichiVisualizer(ts)
    
    # Run the visualization loop
    for step in range(ts.n_steps):
        if not visualizer.visualize(step):
            break

    # Also do a matplotlib animation for comparison
    fig, ax = plt.subplots()
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)

    particles, = ax.plot([], [], 'bo', markersize=8)
    springs, = ax.plot([], [], 'r-', linewidth=2)

    def init():
        particles.set_data([], [])
        springs.set_data([], [])
        return particles, springs

    def update(frame):
        x = r[frame, 0::3]
        y = r[frame, 1::3]
        particles.set_data(x, y)
        
        # Draw springs
        spring_x = []
        spring_y = []
        for i in range(len(system.interactions)):
            p1 = system.particles.index(system.interactions[i].particle1)
            p2 = system.particles.index(system.interactions[i].particle2)
            
            spring_x.extend([x[p1], x[p2], None])
            spring_y.extend([y[p1], y[p2], None])
        
        springs.set_data(spring_x, spring_y)