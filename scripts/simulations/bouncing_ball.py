import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import random

# Simulation parameters
gravity = 0.2
bounce_strength = 1.1
dt = 0.1
trail_length = 20
rotation_speed = 0.05
ball_speed = 2

particle_lifetime_frames = 30  # 🔹 Extra frames after ball stops
frames_after_ball_stop = 0     # 🔹 Tracks frames after ball stops

def random_color():
    return (random.random(), random.random(), random.random())

def generate_polygon_vertices(sides, radius, angle_offset=0):
    angles = np.linspace(0, 2 * np.pi, sides, endpoint=False) + angle_offset
    x = radius * np.cos(angles)
    y = radius * np.sin(angles)
    x = np.append(x, x[0])
    y = np.append(y, y[0])
    return x, y

def growing_ball_simulation(container_shape="circle", container_size=10, ball_speed=2.0):
    ball_pos = np.array([0.0, 0.0])
    ball_vel = np.array([ball_speed, -ball_speed])
    ball_radius = 0.5
    ball_color = random_color()
    trail = []
    
    global rotation_speed, frames_after_ball_stop
    rotation_angle = 0
    shapes = {
        "triangle": 3, "square": 4, "pentagon": 5,
        "hexagon": 6, "heptagon": 7, "octagon": 8
    }
    
    if container_shape == "circle":
        theta = np.linspace(0, 2 * np.pi, 500)
        container_x = container_size * np.cos(theta)
        container_y = container_size * np.sin(theta)
    else:
        container_x, container_y = generate_polygon_vertices(shapes[container_shape], container_size)
    
    particles = []
    game_over = False
        
    def create_particles(position, num_particles=100):
        for _ in range(num_particles):
            angle = random.uniform(0, 2 * np.pi)
            speed = random.uniform(1, 5)
            particle_plot, = ax.plot([], [], 'o', markersize=0.1, color=random_color())
            
            particles.append({
                'position': np.array(position, dtype=np.float64),
                'velocity': np.array([speed * np.cos(angle), speed * np.sin(angle)], dtype=np.float64),
                'size': random.uniform(0.1, 0.5),
                'color': random_color(),
                'lifetime': random.uniform(0.5, 1.5),
                'plot': particle_plot
            })

    def update_particles():
        for particle in particles[:]:  
            np.add(particle['position'], particle['velocity'] * dt, out=particle['position'], casting="unsafe")   
            particle['lifetime'] -= dt  

            # 🔹 Update the plot marker position & size
            particle['plot'].set_data([particle['position'][0]], [particle['position'][1]])  
            particle['plot'].set_markersize(particle['size'] * 10)  
            particle['plot'].set_color(particle['color'])  

            # 🔹 Ensure particle['lifetime'] is a scalar before removing
            if isinstance(particle['lifetime'], (int, float)) and particle['lifetime'] <= 0:
                particle['plot'].remove()  # Remove marker when particle dies
                particles.remove(particle)
    
    def update(frame):
        global frames_after_ball_stop  # 🔹 Use global instead of nonlocal
        nonlocal ball_pos, ball_vel, ball_radius, ball_color, trail, rotation_angle, container_x, container_y, game_over

        
        if game_over:
            if frames_after_ball_stop < particle_lifetime_frames:
                update_particles()
                frames_after_ball_stop += 1
            else:
                print("Simulation ended: Particles fully animated.")
                ani.event_source.stop()  # 🔹 Stop after particles animate
            return ball, trail_line, container_plot, *[particle['plot'] for particle in particles]
        
        ball_vel[1] -= gravity * dt
        ball_pos += ball_vel * dt
        trail.append(ball_pos.copy())
        if len(trail) > trail_length:
            trail.pop(0)
        
        collision_occurred = False
        if container_shape == "circle":
            dist_from_center = np.linalg.norm(ball_pos)
            if dist_from_center + ball_radius >= container_size:
                normal = ball_pos / dist_from_center
                ball_vel = ball_vel - (1 + bounce_strength) * np.dot(ball_vel, normal) * normal
                ball_pos = ball_pos - normal * (dist_from_center + ball_radius - container_size)
                ball_radius += 0.2
                ball_color = random_color()
                collision_occurred = True
        else:
            vertices = np.column_stack((container_x, container_y))
            closest_dist = float("inf")
            collision_normal = None
            
            for i in range(len(vertices) - 1):
                v1 = vertices[i]
                v2 = vertices[i + 1]
                edge = v2 - v1
                edge_normal = np.array([-edge[1], edge[0]]) / np.linalg.norm(edge)
                relative_pos = ball_pos - v1
                dist_to_edge = np.dot(relative_pos, edge_normal)
                
                if 0 <= np.dot(relative_pos, edge) <= np.dot(edge, edge):
                    if abs(dist_to_edge) < closest_dist:
                        closest_dist = abs(dist_to_edge)
                        collision_normal = edge_normal
            
            if collision_normal is not None and closest_dist <= ball_radius:
                ball_vel = ball_vel - (1 + bounce_strength) * np.dot(ball_vel, collision_normal) * collision_normal
                ball_radius += 0.2
                ball_color = random_color()
                ball_pos -= collision_normal * (closest_dist - ball_radius)
                collision_occurred = True
        
        if ball_radius >= container_size:
            game_over = True
            create_particles(ball_pos, num_particles=200)
            create_particles([0, 0], num_particles=200)
            ax.text(0, 0, 'Game Over', fontsize=30, color='white', ha='center', va='center')
            print("Simulation ended: Ball no longer fits in the container.")
        
        ball.set_data([ball_pos[0]], [ball_pos[1]])
        ball.set_markersize(ball_radius * 20)
        ball.set_color(ball_color)
        
        trail_x, trail_y = zip(*trail)
        trail_line.set_data(trail_x, trail_y)
        
        if collision_occurred:
            global rotation_speed
            rotation_speed *= -1
        
        rotation_angle += rotation_speed
        if container_shape != "circle":
            container_x, container_y = generate_polygon_vertices(shapes[container_shape], container_size, rotation_angle)
        container_plot.set_data(container_x, container_y)
        
        if game_over:
            update_particles()
            return ball, trail_line, container_plot, *[particle['plot'] for particle in particles]

        return ball, trail_line, container_plot
    
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.set_xlim(-container_size - 1, container_size + 1)
    ax.set_ylim(-container_size - 1, container_size + 1)
    ax.set_facecolor('black')
    fig.patch.set_facecolor('black')
    
    container_plot, = ax.plot(container_x, container_y, color='white')
    ball, = ax.plot([], [], 'o', color=ball_color, markersize=ball_radius * 20)
    trail_line, = ax.plot([], [], '-', color='white', linewidth=1, alpha=0.5)
    
    ani = FuncAnimation(fig, update, frames=500, interval=20, blit=True)
    plt.show()

def main():
    shapes = ["circle", "square", "triangle", "pentagon", "hexagon", "heptagon", "octagon"]
    container_shape = random.choice(shapes)
    growing_ball_simulation(container_shape=container_shape, ball_speed=ball_speed)

if __name__ == "__main__":
    main()