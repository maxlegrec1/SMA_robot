"""
  ____  __  __    _              _____    _        __  ____     __
 / ___||  \/  |  / \            | ____|  / \      |  \/  \ \   / /
 \___ \| |\/| | / _ \    _____  |  _|   / _ \     | |\/| |\ \ / / 
  ___) | |  | |/ ___ \  |_____| | |___ / ___ \ _  | |  | | \ V /  
 |____/|_|  |_/_/   \_\         |_____/_/   \_( ) |_|  |_|  \_/   
                                              |/                  
Authors:
   Maxime Vanderbeken
   Etienne Andrier

Date : 2025-03-19

License:
   This file is open source and may be freely used and modified,
   provided that proper credit is given to the original authors.
"""
import numpy as np
import unittest

def expand_grid(grid, direction):
    """
    Expands a 3D grid in the specified direction.
    
    Parameters:
    grid (numpy.ndarray): 3D array of shape (n, p, d)
    direction (tuple): One of (0,1), (0,-1), (1,0), (-1,0) representing 
                       south, north, east, west respectively
    
    Returns:
    numpy.ndarray: Expanded grid with the original data preserved in the proper location
    """
    # Get original dimensions
    n, p, d = grid.shape
    
    # Create the new dimensions based on direction
    if direction == (1, 0):  # East
        new_shape = (n + 1, p, d)
        # Place the original data at the top
        indices = (slice(0, n), slice(0, p), slice(0, d))
    elif direction == (-1, 0):  # West
        new_shape = (n + 1, p, d)
        # Place the original data at the bottom
        indices = (slice(1, n + 1), slice(0, p), slice(0, d))
    elif direction == (0, 1):  # North
        new_shape = (n, p + 1, d)
        # Place the original data on the left
        indices = (slice(0, n), slice(0, p), slice(0, d))
    elif direction == (0, -1):  # South
        new_shape = (n, p + 1, d)
        # Place the original data on the right
        indices = (slice(0, n), slice(1, p + 1), slice(0, d))
    else:
        raise ValueError("Direction must be one of (0,1), (0,-1), (1,0), or (-1,0)")
    
    # Create new grid filled with zeros
    new_grid = np.zeros(new_shape, dtype=grid.dtype)
    
    # Copy the original data to the correct position
    new_grid[indices] = grid
    
    return new_grid


def main():
    # Run the unit tests
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
    

if __name__ == "__main__":
    main()