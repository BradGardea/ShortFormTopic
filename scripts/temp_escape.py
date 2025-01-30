import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import random

# Simulation parameters
gravity = 0.2  # Downward force
bounce_strength = 1.1  # Strength of the bounce
dt = 0.1  # Time step
ball_speed = 2.0
ball_radius = 0.5
trail_length = 20  # Maximum length of the trail
num_layers = 1  # Only one layer for debugging
gap_size = 0.5  # Size of the gap in radians


def random_color():
    return (random.random(), random.random(), random.random())


def generate_polygon_vertices(sides, radius, angle_offset=0, gap_size=0):
    angles = np.linspace(0, 2 * np.pi, sides, endpoint=False) + angle_offset
    x = radius * np.cos(angles)
    y = radius * np.sin(angles)

    # Introduce a gap in the polygon
    if gap_size > 0:
        gap_start = np.pi  # Fixed gap position for debugging (top of the shape)
        gap_end = gap_start + gap_size
        mask = (angles < gap_start) | (angles > gap_end)
        x = x[mask]
        y = y[mask]

    x = np.append(x, x[0])  # Close the shape
    y = np.append(y, y[0])
    return x, y


def escape_ball_simulation(container_shape="circle", container_size=8, ball_speed=2.0):
    ball_pos = np.array([0.0, 0.0])
    ball_vel = np.array([ball_speed, -ball_speed])
    ball_color = random_color()
    trail = []

    rotation_angle = 0  # Initial rotation angle

    shapes = {
        "triangle": 3, "square": 4, "pentagon": 5,
        "hexagon": 6, "heptagon": 7, "octagon": 8
    }

    # Initialize containers
    containers = []
    for layer in range(num_layers):
        size = container_size + layer * ball_radius * 2  # Gradual increase of layer size
        if container_shape == "circle":
            theta = np.linspace(0, 2 * np.pi, 500)
            container_x = size * np.cos(theta)
            container_y = size * np.sin(theta)
        else:
            container_x, container_y = generate_polygon_vertices(shapes[container_shape], size, gap_size=gap_size)
        containers.append({"x": container_x, "y": container_y, "size": size, "active": True})

    # Function to handle rotation speed decay with distance from center
    def get_rotation_speed(layer):
        return max(0.005, 0.05 - 0.005 * layer)  # Slower decay for rotation speed

    def check_collision(ball_pos, container):
        vertices = np.column_stack((container["x"], container["y"]))
        closest_dist = float("inf")
        collision_normal = None

        for i in range(len(vertices) - 1):
            v1 = vertices[i]
            v2 = vertices[i + 1]
            edge = v2 - v1
            edge_normal = np.array([-edge[1], edge[0]]) / np.linalg.norm(edge)
            relative_pos = ball_pos - v1
            dist_to_edge = np.dot(relative_pos, edge_normal)

            # Check if the ball is within the edge segment
            if 0 <= np.dot(relative_pos, edge) <= np.dot(edge, edge):
                if abs(dist_to_edge) < closest_dist:
                    closest_dist = abs(dist_to_edge)
                    collision_normal = edge_normal

        # Check if the ball is in the gap
        if container_shape != "circle":
            angle = np.arctan2(ball_pos[1], ball_pos[0]) % (2 * np.pi)
            gap_start = np.pi  # Gap is fixed at the top for debugging
            gap_end = gap_start + gap_size
            if gap_start <= angle <= gap_end:
                return False  # Ball is in the gap

        # If a collision occurred, reflect the ball's velocity
        if collision_normal is not None and closest_dist <= ball_radius:
            ball_vel = ball_vel - (1 + bounce_strength) * np.dot(ball_vel, collision_normal) * collision_normal
            ball_pos -= collision_normal * (closest_dist - ball_radius)
            return True  # Collision occurred

        return False  # No collision

    def update(frame):
        nonlocal ball_pos, ball_vel, ball_color, trail, rotation_angle, containers

        ball_vel[1] -= gravity * dt
        ball_pos += ball_vel * dt
        trail.append(ball_pos.copy())
        if len(trail) > trail_length:
            trail.pop(0)

        # Check for collisions with active containers
        for layer, container in enumerate(containers):
            if not container["active"]:
                continue

            if check_collision(ball_pos, container):
                # Collision occurred, bounce the ball
                pass
            else:
                # Ball escaped through the gap, deactivate the container
                container["active"] = False
                print(f"Ball escaped layer {layer}!")

        ball.set_data([ball_pos[0]], [ball_pos[1]])
        ball.set_markersize(ball_radius * 20)
        ball.set_color(ball_color)

        trail_x, trail_y = zip(*trail)
        trail_line.set_data(trail_x, trail_y)

        # Update rotation for each active container
        for layer, container in enumerate(containers):
            if container["active"]:
                rotation_speed = get_rotation_speed(layer)
                rotation_angle += rotation_speed
                if container_shape != "circle":
                    container["x"], container["y"] = generate_polygon_vertices(
                        shapes[container_shape], container["size"], rotation_angle, gap_size
                    )
                container_plots[layer].set_data(container["x"], container["y"])
            else:
                container_plots[layer].set_data([], [])  # Hide inactive containers

        return [ball, trail_line] + container_plots  # Return all Artist objects

    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.set_xlim(-container_size - num_layers * ball_radius * 2, container_size + num_layers * ball_radius * 2)
    ax.set_ylim(-container_size - num_layers * ball_radius * 2, container_size + num_layers * ball_radius * 2)
    ax.set_facecolor('black')
    fig.patch.set_facecolor('black')

    # Plot containers
    container_plots = []
    for container in containers:
        plot, = ax.plot(container["x"], container["y"], color='white')
        container_plots.append(plot)

    ball, = ax.plot([], [], 'o', color=ball_color, markersize=ball_radius * 20)
    trail_line, = ax.plot([], [], '-', color='white', linewidth=1, alpha=0.5)

    ani = FuncAnimation(fig, update, frames=500, interval=20, blit=True)
    plt.show()


if __name__ == "__main__":
    escape_ball_simulation(container_shape="pentagon", container_size=8, ball_speed=ball_speed)