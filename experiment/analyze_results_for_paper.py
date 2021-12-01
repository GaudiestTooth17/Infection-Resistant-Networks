import sys
sys.path.append('')
import fileio as fio
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np


BA_FILES = ('BarabasiAlbert Global Flicker .5 (i0=5)',
            'BarabasiAlbert Global Flicker .25 (i0=5)',
            'BarabasiAlbert Local Flicker .5 (i0=5)',
            'BarabasiAlbert Local Flicker .25 (i0=5)',
            'BarabasiAlbert Partially Isolate Infected Agents (i0=5)',
            'BarabasiAlbert Proposed Optimal Mitigation (i0=5)',
            'BarabasiAlbert Totally Isolate Infected Agents (i0=5)')


def main():
    # scalar_data_accessors = (
    #     ('Survival Rate', lambda r: r.survival_rate),
    #     ('Max Edges Rmed', lambda r: r.max_num_edges_removed),
    #     ('Temporal Avg Edges Rmed', lambda r: r.temporal_average_edges_removed),
    #     ('Avg edge rmal duration', lambda r: r.avg_edge_removal_duration),
    #     ('Max Num Infectious', lambda r: r.max_num_infectious)
    # )
    scalar_data_accessors = (
        ('Survival Rate', lambda r: r.survival_rate),
        # ('Max Edges Rmed', lambda r: np.max(r.num_edges_removed_per_step)),
        # ('Temporal Avg Edges Rmed', lambda r: np.average(r.num_edges_removed_per_step)),
        ('Avg edge rmal duration', lambda r: np.average(r.edge_removal_durations)),
        ('Max Num Infectious', lambda r: r.max_num_infectious)
    )
    vector_data_accessors = (
        ('Diameter at Step', lambda r: r.diameter_at_step),
        ('Num Comps at Step', lambda r: r.num_comps_at_step),
        ('Avg Comp Size at Step', lambda r: r.avg_comp_size_at_step),
        ('Perc Edges Node Loses at Step', lambda r: r.percent_edges_node_loses_at_step)
    )

    for name in tqdm(BA_FILES):
        results = fio.load_sim_results(name)
        for data_name, accessor in scalar_data_accessors:
            plt.title(f'{name}\n{data_name}')
            plt.boxplot([accessor(result) for result in results])
            plt.figure()
        for data_name, accessor in vector_data_accessors:
            pass

    plt.show()


if __name__ == '__main__':
    main()
