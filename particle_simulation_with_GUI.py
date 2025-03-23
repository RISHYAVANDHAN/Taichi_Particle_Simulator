import taichi as ti
import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Configure Taichi
ti.init(arch=ti.gpu)  # Use GPU if available, otherwise falls back to CPU

@ti.data_oriented
class ParticleSystem:
    def __init__(self, num_particles=3, t0=0, gravity=np.array([0, 0, -9.81])):
        # System parameters
        self.t0 = t0
        self.gravity = gravity
        self.num_particles = num_particles
        self.dim = 3  # 3D simulation
        
        # Taichi fields for particle properties
        self.mass = ti.field(dtype=float, shape=num_particles)
        self.position = ti.Vector.field(self.dim, dtype=float, shape=num_particles)
        self.velocity = ti.Vector.field(self.dim, dtype=float, shape=num_particles)
        self.force = ti.Vector.field(self.dim, dtype=float, shape=num_particles)
        
        # Spring connections
        self.num_springs = 0
        self.spring_a = ti.field(dtype=int, shape=10)  # Allow up to 10 springs
        self.spring_b = ti.field(dtype=int, shape=10)  # Allow up to 10 springs
        self.spring_stiffness = ti.field(dtype=float, shape=10)
        self.spring_rest_length = ti.field(dtype=float, shape=10)
        
        # Temporary arrays for simulation results
        self.positions_np = None
        self.velocities_np = None
        
    def add_particle(self, index, mass, position, velocity=None):
        if velocity is None:
            velocity = np.zeros(3)
            
        self.mass[index] = mass
        self.position[index] = position
        self.velocity[index] = velocity
        
    def add_spring(self, particle_a, particle_b, stiffness, rest_length):
        if self.num_springs < 10:  # Check if we have space
            self.spring_a[self.num_springs] = particle_a
            self.spring_b[self.num_springs] = particle_b
            self.spring_stiffness[self.num_springs] = stiffness
            self.spring_rest_length[self.num_springs] = rest_length
            self.num_springs += 1
        else:
            print("Warning: Maximum number of springs reached")
    
    @ti.kernel
    def compute_forces(self):
        # Reset forces to gravity
        for i in range(self.num_particles):
            self.force[i] = ti.Vector([self.gravity[0], self.gravity[1], self.gravity[2]]) * self.mass[i]
        
        # Add spring forces
        for s in range(self.num_springs):
            a = self.spring_a[s]
            b = self.spring_b[s]
            
            # Calculate spring vector and length
            r_ab = self.position[b] - self.position[a]
            l = r_ab.norm()
            
            # Skip if particles are at the same position
            if l < 1e-10:
                continue
            
            # Calculate normalized direction
            n = r_ab / l
            
            # Calculate spring force (Hooke's law)
            f_spring = self.spring_stiffness[s] * (l - self.spring_rest_length[s]) * n
            
            # Apply forces (action-reaction)
            self.force[a] += f_spring
            self.force[b] -= f_spring
    
    @ti.kernel
    def explicit_euler_step(self, dt: float):
        # Update velocity and position for each particle
        for i in range(self.num_particles):
            self.velocity[i] += dt * self.force[i] / self.mass[i]
            self.position[i] += dt * self.velocity[i]
    
    def simulate(self, dt, total_time):
        # Number of steps
        steps = int(total_time / dt)
        
        # Arrays to store results
        self.positions_np = np.zeros((steps + 1, self.num_particles, 3))
        self.velocities_np = np.zeros((steps + 1, self.num_particles, 3))
        times = np.zeros(steps + 1)
        
        # Copy initial state
        for i in range(self.num_particles):
            self.positions_np[0, i] = self.position[i].to_numpy()
            self.velocities_np[0, i] = self.velocity[i].to_numpy()
        
        # Simulation loop
        start_time = time.time()
        
        for step in range(steps):
            # Update time
            times[step + 1] = times[step] + dt
            
            # Compute forces
            self.compute_forces()
            
            # Update positions and velocities
            self.explicit_euler_step(dt)
            
            # Store results
            for i in range(self.num_particles):
                self.positions_np[step + 1, i] = self.position[i].to_numpy()
                self.velocities_np[step + 1, i] = self.velocity[i].to_numpy()
        
        end_time = time.time()
        print(f"Simulation completed in {end_time - start_time:.4f} seconds")
        
        return times, self.positions_np, self.velocities_np
    
    def compute_spring_lengths(self):
        lengths = np.zeros((len(self.positions_np), self.num_springs))
        
        for step in range(len(self.positions_np)):
            for s in range(self.num_springs):
                a = self.spring_a[s]
                b = self.spring_b[s]
                r_ab = self.positions_np[step, b] - self.positions_np[step, a]
                lengths[step, s] = np.linalg.norm(r_ab)
                
        return lengths
        
    def visualize(self, times, spring_lengths=None):
        # Plot spring lengths over time
        if spring_lengths is not None:
            plt.figure(figsize=(10, 5))
            for s in range(self.num_springs):
                plt.plot(times, spring_lengths[:, s], label=f'Spring {s+1}')
            plt.xlabel('Time (s)')
            plt.ylabel('Length (m)')
            plt.title('Length of the Springs Over Time')
            plt.legend()
            plt.grid(True)
            plt.show()
            
    def animate(self, times, positions):
        # Create animation
        fps = 20
        dt = times[1] - times[0]
        frac = max(1, int(1/(fps * dt)))
        
        pos_frames = positions[::frac]
        
        fig, ax = plt.subplots()
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        
        particles, = ax.plot([], [], 'bo', markersize=8)
        lines = []
        
        for s in range(self.num_springs):
            line, = ax.plot([], [], 'r-')
            lines.append(line)
        
        def init():
            particles.set_data([], [])
            for line in lines:
                line.set_data([], [])
            return [particles] + lines
        
        def update(frame):
            x = pos_frames[frame, :, 0]
            y = pos_frames[frame, :, 1]
            particles.set_data(x, y)
            
            for s in range(self.num_springs):
                a = self.spring_a[s]
                b = self.spring_b[s]
                lines[s].set_data([pos_frames[frame, a, 0], pos_frames[frame, b, 0]],
                                 [pos_frames[frame, a, 1], pos_frames[frame, b, 1]])
            
            return [particles] + lines
        
        ani = FuncAnimation(fig, update, frames=len(pos_frames), init_func=init, blit=True, interval=50)
        plt.xlabel('X (m)')
        plt.ylabel('Y (m)')
        plt.title('Particle Animation with Springs')
        plt.show()


if __name__ == "__main__":
    # Create system
    system = ParticleSystem(num_particles=3)
    
    # System parameters
    m = 1.0  # mass
    c = 10.0  # spring stiffness
    L = 1.0   # rest length
    
    # Add particles
    system.add_particle(0, m, np.array([0.0, 0.0, 0.0]))
    system.add_particle(1, m, np.array([0.0, 1.1 * L, 0.0]))
    system.add_particle(2, m, np.array([1.2 * L, 0.0, 0.0]))
    
    # Add springs
    system.add_spring(0, 1, c, L)
    system.add_spring(1, 2, c, L)
    system.add_spring(2, 0, c, L)
    
    # Simulation parameters
    dt = 1e-2    # Time step (s)
    tf = 3.0     # Final time (s)
    
    # Run simulation
    times, positions, velocities = system.simulate(dt, tf)
    
    # Compute and visualize spring lengths
    spring_lengths = system.compute_spring_lengths()
    system.visualize(times, spring_lengths)
    
    # Animate the system
    system.animate(times, positions)