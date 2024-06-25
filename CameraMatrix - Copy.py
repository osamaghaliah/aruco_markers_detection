'''
This program generates the following camera matrix:

[728.49671263, 0.000000, 640.000000],
    [0.000000, 728.49671263, 360.000000],
    [0.000000, 0.000000, 1.000000]
'''

import numpy as np

# Camera parameters
resolution_width = 1280
resolution_height = 720
FoV = 82.6  # Field of View in degrees

# Calculate the focal length in pixels
f_x = f_y = resolution_width / (2 * np.tan(np.deg2rad(FoV / 2)))

# Update camera matrix
camera_matrix = np.array([
    [f_x, 0, resolution_width / 2],
    [0, f_y, resolution_height / 2],
    [0, 0, 1]
])

dist_coeffs = np.zeros(5)  # Assuming no lens distortion

print("Updated Camera Matrix:")
print(camera_matrix)