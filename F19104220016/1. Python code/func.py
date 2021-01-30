import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy as sp
import scipy.io as spio
import sklearn as skl
import networkx as nx
import heapq
import copy
import time
import math
from dubins import dubins_path_planning

DATA1_PATH = './data_set1.xlsx'


def cal_angle(vec1, vec2):
    return (vec2 - vec1) / np.linalg.norm(vec2 - vec1)


def cal_angle_of_two_vec(vec1, vec2):
    angle = vec1.dot(vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    return angle


def cal_dubins_dis(g: nx.DiGraph, sender, receiver, sender_angle, receiver_angle, c):
    _, clen, dis_2d = dubins_path_planning(g.nodes[sender]['pos'][0],
                                           g.nodes[sender]['pos'][1],
                                           sender_angle,
                                           g.nodes[receiver]['pos'][0],
                                           g.nodes[receiver]['pos'][1],
                                           receiver_angle,
                                           c=c)
    dubins_dis_3d = np.sqrt(np.power((clen + dis_2d), 2) +
                            np.power(g.nodes[sender]['pos'][2] - g.nodes[receiver]['pos'][2], 2))
    return dubins_dis_3d


def plt_cosmos(a_pos, b_pos, modify_horizon_pos, modify_vertical_pos):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(a_pos[0], a_pos[1], zs=a_pos[2], marker='o', s=200, c='r')
    ax.scatter(b_pos[0], b_pos[1], zs=b_pos[2], marker='o', s=200, c='b')
    modify_horizon_pos_x = modify_horizon_pos[:, 0]
    modify_horizon_pos_y = modify_horizon_pos[:, 1]
    modify_horizon_pos_z = modify_horizon_pos[:, 2]
    ax.scatter(modify_horizon_pos_x,
               modify_horizon_pos_y,
               zs=modify_horizon_pos_z,
               marker='_',
               s=20)
    modify_vertical_pos_x = modify_vertical_pos[:, 0]
    modify_vertical_pos_y = modify_vertical_pos[:, 1]
    modify_vertical_pos_z = modify_vertical_pos[:, 2]
    ax.scatter(modify_vertical_pos_x,
               modify_vertical_pos_y,
               zs=modify_vertical_pos_z,
               marker='^',
               s=20)
    plt.savefig('./1.png', dpi=1024)
    plt.show()


def output_all_path(path):
    p_list = []
    for k, v in path.items():
        p_list.append(v['path'])
        print(v['path'])
    # p_pd = pd.DataFrame(p_list, columns=None, index=None)
    # p_pd.to_excel('./pathes', header=False, index=False, )


def output_result(path, points_graph, data_name):
    node_array = np.array(path['path']).reshape((-1, 1))
    error_array = np.array(path['pre_modify_error']).reshape((-1, 2))
    type_list = []
    for n in path['path']:
        type_list.append(points_graph.nodes[n]['type'])
    type_array = np.array(type_list).reshape((-1, 1))
    result_array = np.concatenate([node_array, error_array, type_array], axis=1)
    result_df = pd.DataFrame(result_array, index=None, columns=None)
    result_df.to_excel(data_name, header=False, index=False)


def output_edges(g: nx.DiGraph):
    edges = []
    for sender, receiver in g.edges():
        dis = cal_distance(g.nodes[sender]['pos'], g.nodes[receiver]['pos'])
        edges.append([sender, receiver, dis])
    edges = np.array(edges)
    distances = edges[:, -1]
    distances_mean = np.mean(distances)
    distances_std = np.std(distances)
    distances_max = np.max(distances)
    distances_min = np.min(distances)
    plt.hist(distances, bins=100)
    plt.title('附件2数据 距离分布图')
    plt.xlabel('距离/m')
    plt.ylabel('频数')
    plt.show()
    print(distances_mean, distances_std, distances_max, distances_min)
    edges_df = pd.DataFrame(edges, columns=None, index=None)
    edges_df.to_excel('./edges.xlsx', header=False, index=False)


def output_edges_with_pos(g: nx.DiGraph):
    edges = []
    for sender, receiver in g.edges():
        edges.append([sender, g.nodes[sender]['pos'][0], g.nodes[sender]['pos'][1], g.nodes[sender]['pos'][2],
                      receiver, g.nodes[receiver]['pos'][0], g.nodes[receiver]['pos'][1], g.nodes[receiver]['pos'][2]])
    edges = np.array(edges)
    edges = {'edges': edges}
    spio.savemat('./edges_with_pos.mat', edges)
    # edges_df = pd.DataFrame(edges, columns=None, index=None)
    # edges_df.to_excel('./edges_with_pos.xlsx', header=False, index=False)


def add_node_num_weight(points_graph, num_punishment_factor):
    for sender, receiver in points_graph.edges():
        points_graph.adj[sender][receiver]['distance'] += num_punishment_factor
    return points_graph


def cal_next_sender_horizontal_direction(sender, reciever):
    direction = (reciever - sender) / np.linalg.norm(reciever - sender)
    return direction


def modify_vertical_error(error,
                          alpha1,
                          alpha2):
    if error[0] < alpha1 and error[1] < alpha2:
        error[0] = 0
    return error


def failed_modify_vertical_error(error,
                                 alpha1,
                                 alpha2):
    if error[0] < alpha1 and error[1] < alpha2:
        error[0] = min(error[0], 5)
    return error


def modify_vertical_error_possible(error,
                                   alpha1,
                                   alpha2):
    if error[0] < alpha1 and error[1] < alpha2:
        error[0] = min(error[0], 5) * 0.2
    return error


def modify_horizontal_error(error,
                            beta1,
                            beta2):
    if error[0] < beta1 and error[1] < beta2:
        error[1] = 0
    return error


def failed_modify_horizontal_error(error,
                                   beta1,
                                   beta2):
    if error[0] < beta1 and error[1] < beta2:
        error[1] = min(error[1], 5)
    return error


def modify_horizontal_error_possible(error,
                                     beta1,
                                     beta2):
    if error[0] < beta1 and error[1] < beta2:
        error[1] = min(error[1], 5) * 0.2
    return error


def examine_error(error,
                  theta=30.):
    return error[0] < theta and error[1] < theta


def get_adj_matrix(points_graph: nx.DiGraph):
    adj_matrix = np.ones((points_graph.number_of_nodes(), points_graph.number_of_nodes())) * INF
    for sender, receiver in points_graph.edges():
        adj_matrix[sender][receiver] = points_graph.adj[sender][receiver]['distance']
    # nx.draw(points_graph, with_labels=True, node_size=10)
    # plt.show()
    adj_pd = pd.DataFrame(adj_matrix, index=None, columns=None)
    adj_pd.to_excel('./adj.xlsx', header=False, index=False)
    return adj_matrix


def build_adj_matrix(graph: nx.DiGraph,
                     delta=0.001,
                     alpha1=25.,
                     alpha2=15.,
                     beta1=20.,
                     beta2=25.,
                     theta=30.):
    for sender in graph.nodes:
        for receiver in graph.nodes:
            if sender == receiver:
                continue
            distance = cal_distance(graph.nodes[sender]['pos'],
                                    graph.nodes[receiver]['pos'])
            error = delta * distance
            # if error <= theta:
            #     graph.add_edge(sender, receiver,
            #                    distance=distance,
            #                    edge_error=error)
            if graph.nodes[receiver]['type'] == 0:
                if error <= beta1 and error <= beta2:
                    graph.add_edge(sender, receiver,
                                   distance=distance,
                                   edge_error=error)
            elif graph.nodes[receiver]['type'] == 1:
                if error <= alpha1 and error <= alpha2:
                    graph.add_edge(sender, receiver,
                                   distance=distance,
                                   edge_error=error)
            elif graph.nodes[receiver]['type'] == 2:
                if error <= theta:
                    graph.add_edge(sender, receiver,
                                   distance=distance,
                                   edge_error=error)
    return graph


def cal_cumulate_distance(g: nx.DiGraph, path: list):
    dis = 0
    for i in range(len(path) - 1):
        dis += g[path[i]][path[i + 1]]['distance']
    return dis


def cal_distance(v1, v2):
    return np.linalg.norm(v1 - v2)


def load_dataset(path, k=-2):
    points = pd.read_excel(path).values
    a = points[0, :]
    a[-2] = -1
    a = a.astype(np.float32)
    b = points[-1, :]
    b[-2] = 2
    b = b.astype(np.float32)
    correction_points = points[1:k + 1, :]
    correction_points = correction_points.astype(np.float32)
    return a, b, correction_points


def plt_cosmos(a_pos, b_pos, modify_horizon_pos, modify_vertical_pos):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(a_pos[0], a_pos[1], zs=a_pos[2], marker='o', s=200, c='r')
    ax.scatter(b_pos[0], b_pos[1], zs=b_pos[2], marker='o', s=200, c='b')
    modify_horizon_pos_x = modify_horizon_pos[:, 0]
    modify_horizon_pos_y = modify_horizon_pos[:, 1]
    modify_horizon_pos_z = modify_horizon_pos[:, 2]
    ax.scatter(modify_horizon_pos_x,
               modify_horizon_pos_y,
               zs=modify_horizon_pos_z,
               marker='_',
               s=20)
    modify_vertical_pos_x = modify_vertical_pos[:, 0]
    modify_vertical_pos_y = modify_vertical_pos[:, 1]
    modify_vertical_pos_z = modify_vertical_pos[:, 2]
    ax.scatter(modify_vertical_pos_x,
               modify_vertical_pos_y,
               zs=modify_vertical_pos_z,
               marker='^',
               s=20)
    plt.savefig('./1.png', dpi=1024)
    plt.show()
