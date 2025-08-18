import numpy as np
import matplotlib.pyplot as plt
import os

"""
Generate the data for the line recognition task.
"""

def generate_straight_line(num_points = 20):
    # y = mx + b
    m = np.random.uniform(-3,3)
    b = np.random.uniform(-10,10)
    x = np.linspace(-10, 10, num_points)
    y = m*x + b

    # We can add some noise to the line
    noise = np.random.normal(0, 0.5, num_points)
    y += noise
    return x,y

def generate_parabola(num_points = 20):
    # y = ax^2 + bx + c
    a = np.random.uniform(-3,3)
    b = np.random.uniform(-5,5)
    c = np.random.uniform(-10,10)
    x = np.linspace(-10, 10, num_points)
    y = a*x**2 + b*x + c
    
    # We can add some noise to the line
    noise = np.random.normal(0, 0.5, num_points)
    y += noise
    return x,y

def save_image(x,y, filepath):
    plt.figure(figsize=(1,1))  # small figure
    plt.plot(x, y, color='black', linewidth=2)
    plt.axis('off')  # remove axes
    plt.xlim(-10, 10)
    plt.savefig(filepath, bbox_inches='tight', pad_inches=0)
    plt.close()


if __name__ == "__main__":
    num_samples = 200

    for i in range(1, num_samples+1):
        # Generate straight line
        x, y = generate_straight_line()
        save_image(x,y, f"data/training/straight/straight_{i}.png")

        # Generate parabola
        x,y = generate_parabola()
        save_image(x,y, f"data/training/parabola/parabola_{i}.png")