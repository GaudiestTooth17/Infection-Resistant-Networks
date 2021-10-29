import sys
sys.path.append('')
import fileio as fio
import networkx as nx
import matplotlib.pyplot as plt
from analysis import visualize_network


def investigate_sociopatterns(name):
    # 20 seconds to 15 mins
    threshold_range = list(x*20 for x in range(1, 46))
    nets = {i: fio.read_socio_patterns_network(f'networks/{name}', i)
            for i in threshold_range}
    N = nets[20].N
    print(f'N = {N}')
    title = f'{name} N={N}'
    threshold_to_visualize = 5*60  # visualize the network were it takes 5 minutes to form an edge

    def calc_net_metrics(net, step_threshold):
        comps = list(nx.connected_components(net.G))
        n_comps = len(comps)
        largest_comp: nx.Graph = nx.subgraph(net.G, max(comps, key=len))
        N_lc = len(largest_comp.nodes)
        diam_lc = nx.diameter(largest_comp)
        return n_comps, diam_lc, N_lc / net.N, nx.average_clustering(largest_comp)

    metrics = [calc_net_metrics(net, step_threshold) for step_threshold, net in nets.items()]
    n_comps, diams, perc_nodes, clusterings = zip(*metrics)
    make_plot(title, threshold_range, n_comps, 'Number of Components')
    make_plot(title, threshold_range, diams, 'Diameter of Largest Component')
    make_plot(title, threshold_range, perc_nodes, 'Percentage of Nodes in Largest Component')
    make_plot(title, threshold_range, clusterings, 'Average Clustering')
    visualize_network(nets[threshold_to_visualize].G, nets[threshold_to_visualize].layout,
                      f'{title} {threshold_to_visualize} s to form edge', save=True)


def make_plot(title, xs, ys, y_label):
    plt.figure()
    plt.title(title)
    plt.plot(xs, ys)
    plt.ylabel(y_label)
    plt.xlabel('Seconds to Form an Edge')
    plt.savefig(f'{title} {y_label}.png')


if __name__ == '__main__':
    all_net_names = ('workplace_1st_deployment.sp', 'workplace_2nd_deployment.sp',
                     'sp_data_school_day_1_g.gexf', 'sp_data_school_day_2_g.gexf')
    for name in all_net_names:
        investigate_sociopatterns(name)
