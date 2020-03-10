import numpy as np

def placement_fn(grid_size, placement_grid, obj_size, num_tries=10):
    for _ in range(num_tries):
        pos = np.array([np.random.randint(1, grid_size - obj_size[0] - 1), # Random x cooridinate
                        np.random.randint(1, grid_size - obj_size[1] - 1)]) # Random y coordinate

        good = True
        for i in range(obj_size[0]):
            for j in range(obj_size[1]):
                if(placement_grid[pos[0] + i][pos[0] + j] == 1):
                    good = False

        if good:
            return pos