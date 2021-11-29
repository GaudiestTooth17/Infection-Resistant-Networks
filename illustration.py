import networkx as nx
import matplotlib.pyplot as plt
from networkgen import make_social_circles_network, Agent
import numpy as np


def main():
    N = 200
    N_purple = int(N * .1)
    N_blue = int(N * .2)
    N_green = N - N_purple - N_blue
    agents = {Agent('green', 30): N_green,
              Agent('blue', 40): N_blue,
              Agent('purple', 50): N_purple}
    # grid_dim = (int(N/.005), int(N/.005))  # the denominator is the desired density
    grid_dim = int(N*1.25), int(N*1.25)
    net, _ = make_social_circles_network(agents, grid_dim,
                                         rng=np.random.default_rng(0))  # type: ignore
    colored_edges = net.intercommunity_edges

    def make_cmap():
        return ['black' if edge not in colored_edges else 'red' for edge in net.edges]

    n = nx.draw_networkx_nodes(net.G, pos=net.layout, node_size=150, node_color='dimgrey')
    n.set_edgecolor('black')
    nx.draw_networkx_edges(net.G, pos=net.layout, width=2, edge_color='black')
    plt.show(block=True)
    # while True:
    #     i = input(': ')
    #     if i == 'done':
    #         break
    #     x, y = map(int, i.split())
    #     if y in G[x]:
    #         # G.remove_edge(x, y)
    #         colored_edges.add((x, y))
    #         colored_edges.add((y, x))
    #     else:
    #         G.add_edge(x, y)
    #     plt.clf()
    #     nx.draw_networkx(G, pos=layout, node_size=150, edge_color=make_cmap())
    #     plt.show(block=False)
    # nx.draw(G, pos=layout, node_size=150, edge_color=make_cmap())
    # plt.show()


if __name__ == '__main__':
    main()
