from customtypes import Number
from typing import Callable, Generic, Sequence, Tuple, TypeVar, Union
import numpy as np
import networkx as nx
import fileio as fio
import matplotlib.pyplot as plt
from tqdm import tqdm
import csv
from analyzer import visualize_network
from networkgen import _connected_community as cc
import networkx as nx
# from mpl_toolkits import mplot3d
from matplotlib import pyplot as plt
import socialgood as sg
import analyzer
from network import Network

def make_plot_from_txt_file():
    data_file = open('social-good-id0:20-od0:10_actual_degrees.txt', 'r')
    xyz = []
    actual = []
    min_ = []
    max_ = []
    while True:
        data = data_file.readline()
        if data == '':
            break
        data = data.split()
        x, y, z, d, zmin, zmax = int(data[0]), int(data[1]), float(data[2]), float(data[3]), float(data[4]), float(data[5])

        # Tried Degree
        xyz.append([x, y, z])
        # Actual Degree
        actual.append([d, y, z])
        # Min
        min_.append([d, y, zmin])
        # Max
        max_.append([d, y, zmax])

    data_file.close()
    plt.figure()
    ax = plt.axes(projection='3d')


    p_list = actual
    x_ = [x for x, y, z in p_list]
    y_ = [y for x, y, z in p_list]
    z_ = [z for x, y, z in p_list]
    ax.scatter3D(y_, x_, z_, color='black')

    p_list = min_
    x_ = [x for x, y, z in p_list]
    y_ = [y for x, y, z in p_list]
    z_ = [z for x, y, z in p_list]
    ax.scatter3D(y_, x_, z_, color='red')

    p_list = max_
    x_ = [x for x, y, z in p_list]
    y_ = [y for x, y, z in p_list]
    z_ = [z for x, y, z in p_list]
    ax.scatter3D(y_, x_, z_, color='blue')

    plt.ylabel('Average Inner Degree')
    plt.xlabel('Average Outer Degree')
    plt.show()

def validate_connected_community_network():
    d_map = {}
    d_dist = []
    for i in range(10):
        for _ in range(100):
            ideg = np.array([i for _ in range(10)])
            odeg = np.array([0 for _ in range(10)])
            g, _ = cc.make_connected_community_network(ideg, odeg)
            for _, d in g.degree():
                if d not in d_map:
                    d_map[d] = 0
                d_map[d] += 1
                d_dist.append(d)
        
        s = sum(d_map.values())
        for d, v in d_map.items():
            d_map[d] = v / s
        # print(d_map)
        plt.hist(d_dist)
        plt.figure()
        
    plt.show()

def social_good_giant_component_barabasi_albert():
    social_goods = []
    giant_comp_sizes = []
    for i in range(1, 100):
        print(i)
        g = nx.barabasi_albert_graph(100, i)
        social_good = sg.rate_social_good(g)
        social_goods.append(social_good)
        giant_comp_size = analyzer.get_giant_component_size(g, 0.9, 10)
        giant_comp_sizes.append(giant_comp_size / 100)
    plt.xlabel('Number of Edges')
    plt.ylabel('Social Good')
    plt.title('Barabasi-Albert Social Good Analysis')
    plt.plot(range(1, 100), social_goods, 'o', color='blue')
    plt.plot(range(1, 100), giant_comp_sizes, 'o', color='red')
    plt.show()

def social_good_giant_component_watts_strogatz():
    num_nodes = 100
    social_goods = []
    giant_comp_sizes = []

    x = tuple(range(2, num_nodes, 1))
    y = (i/100 for i in range(0, 100, 5))

    for p in y:
        for k in x:
            print(f'k: {k} p: {p}')
            g = nx.watts_strogatz_graph(num_nodes, k, p)
            social_good = sg.rate_social_good(g)
            social_goods.append((k, p, social_good))
            giant_comp_size = analyzer.get_giant_component_size(g, 0.95, 100)
            giant_comp_sizes.append((k, p, giant_comp_size / num_nodes))

    plt.figure()
    ax = plt.axes(projection='3d')

    p_list = social_goods
    x_ = [x for x, y, z in p_list]
    y_ = [y for x, y, z in p_list]
    z_ = [z for x, y, z in p_list]
    ax.scatter3D(x_, y_, z_, color='blue')

    p_list = giant_comp_sizes
    x_ = [x for x, y, z in p_list]
    y_ = [y for x, y, z in p_list]
    z_ = [z for x, y, z in p_list]
    ax.scatter3D(x_, y_, z_, color='red')
    
    plt.xlabel('Number of Neighbors Connected Too')
    plt.ylabel('Probability of Rewiring')
    plt.title('Watts-Strogatz Social Good Analysis')
    plt.show()


def social_good_giant_component_connected_community():
    RAND = np.random.default_rng()

    num_nodes = 200
    social_goods = []
    giant_comp_sizes = []

    x = tuple(range(10))
    y = tuple(range(20))

    for i in y:
        for j in x:
            print(f'i: {i} j: {j}')
            for _ in range(100):
                inner_degrees = np.round(RAND.poisson(i, 10))
                outer_degrees = np.round(RAND.poisson(j, 20))
                if np.sum(inner_degrees) % 2 == 1:
                    inner_degrees[np.argmin(inner_degrees)] += 1
                if np.sum(outer_degrees) % 2 == 1:
                    outer_degrees[np.argmin(outer_degrees)] += 1

                g, _ = cc.make_connected_community_network(inner_degrees, outer_degrees, RAND)
                social_good = sg.rate_social_good(g)
                social_goods.append((i, j, social_good))
                giant_comp_size = analyzer.get_giant_component_size(g, 0.9, 10)
                giant_comp_sizes.append((i, j, giant_comp_size / num_nodes))

    plt.figure()
    ax = plt.axes(projection='3d')

    p_list = social_goods
    x_ = [x for x, y, z in p_list]
    y_ = [y for x, y, z in p_list]
    z_ = [z for x, y, z in p_list]
    ax.scatter3D(x_, y_, z_, color='blue')

    p_list = giant_comp_sizes
    x_ = [x for x, y, z in p_list]
    y_ = [y for x, y, z in p_list]
    z_ = [z for x, y, z in p_list]
    ax.scatter3D(x_, y_, z_, color='red')
    
    plt.xlabel('Number of Neighbors Connected Too')
    plt.ylabel('Probability of Rewiring')
    plt.title('Connected Community Social Good Analysis')
    plt.show()

if __name__ == '__main__':
    social_good_giant_component_connected_community()

        