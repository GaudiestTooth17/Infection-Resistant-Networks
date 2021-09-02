from typing import List
from tqdm import tqdm
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from networkgen import MakeLazySpatialNetwork, make_random_spatial_configuration


def sensitivity_to_initial_configuration():
    """Test network creation speed and sensitivity of number of components to grid configuration."""
    reaches = np.linspace(0, 100, 100)
    all_num_comps: List[List[int]] = []
    for seed in tqdm(range(100)):
        configuration = make_random_spatial_configuration((500, 500), 500,
                                                          np.random.default_rng(seed))
        spatial_nets = MakeLazySpatialNetwork(configuration)
        num_comps = [nx.number_connected_components(spatial_nets(reach).G) for reach in reaches]
        all_num_comps.append(num_comps)

    num_comp_data = np.array(all_num_comps)
    quartiles = np.quantile(num_comp_data, (.25, .75), axis=0, interpolation='midpoint')
    y_coords = np.mean(num_comp_data, axis=0)
    plt.xlabel('Reach')
    plt.ylabel('Number of Components')
    plt.plot(reaches, y_coords)
    plt.fill_between(reaches, quartiles[0], quartiles[1], alpha=.4)
    plt.show()


if __name__ == '__main__':
    sensitivity_to_initial_configuration()
