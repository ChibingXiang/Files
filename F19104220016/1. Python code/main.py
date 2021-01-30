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
import dubins
import itertools

from func import *

# dataset1 recommended punishment factor is 200
# dataset2 recommended punishment factor is 50
plt.rcParams['font.sans-serif'] = ['SimHei']
INF = np.inf
DATA1_PATH = './data_set1.xlsx'
DATA2_PATH = './data_set2.xlsx'
NODE_NUM_PUNISHMENT_CORR = 0
para1 = {'alpha1': 25,
         'alpha2': 15,
         'beta1': 20,
         'beta2': 25,
         'theta': 30,
         'delta': 0.001}
para2 = {'alpha1': 20,
         'alpha2': 10,
         'beta1': 15,
         'beta2': 20,
         'theta': 20,
         'delta': 0.001}

def main():
    a, b, correction_points = load_dataset(DATA2_PATH)
    points_graph = nx.DiGraph()
    points_graph.add_node(int(a[0]), pos=a[1:4], type=a[4], third_q_type=a[5])
    for i in range(correction_points.shape[0]):
        points_graph.add_node(int(correction_points[i, 0]),
                              pos=correction_points[i, 1:4],
                              type=correction_points[i, 4],
                              third_q_type=correction_points[i, 5])
    points_graph.add_node(int(b[0]), pos=b[1:4], type=b[4], third_q_type=b[5])
    points_graph = build_adj_matrix(points_graph, **para1)
    output_edges(points_graph)
    output_edges_with_pos(points_graph)
    points_graph = add_node_num_weight(points_graph, NODE_NUM_PUNISHMENT_CORR)
    path, cost_time = dijkstra_with_constrain8(points_graph, 0,
                                               612,
                                               **para1,
                                               punishment_factor=200)
    path_pos = get_path_pos(points_graph, path['path'])

    path = get_dubins_path(points_graph, path_pos, path['path'])
    dis = cal_cumulate_distance(points_graph, path['path'])
    output_result(path, points_graph, './q1_result1.xlsx')
    print(dis)
    print(path)
    output_all_path(path)
    print(path)
    path_iter = nx.shortest_path(points_graph, 0,
                                 # points_graph.number_of_nodes() - 1
                                 612, weight='distance')
    print(path, path_iter)
    # adj_matrix = get_adj_matrix(points_graph)
    for punishment_factor in range(0, 300, 10):
        k_weight_test(-2, punishment_factor, DATA1_PATH, para1)
    for k in list(range(50, 201, 50)) + [288, 326]:
        for punishment in range(0, 300, 10):
            k_weight_test(k, punishment, DATA2_PATH, para2)
    for i in range(200, 1200, 200):
        c_test(-2, 50, DATA2_PATH, para2, 1 / i)
    for punishment_factor in np.linspace(0, 0.1, 10):
        path, cost_time = dijkstra_with_possible_constrain(points_graph, 0,
                                                           326,
                                                           **para2, punishment_factor=punishment_factor)
        dis = cal_cumulate_distance(points_graph, path['path'])
        poss = cal_possibility(points_graph, path['path'], **para2)
        third_q_type = [points_graph.nodes[p]['third_q_type'] for p in path['path']]
        output_result(path, points_graph, 'third_q_result2/pos_{}.xlsx'.format(poss))
        print(path['path'], poss)
    return 0


def cal_possibility(g: nx.DiGraph, path,
                    delta=0.001,
                    alpha1=25.,
                    alpha2=15.,
                    beta1=20.,
                    beta2=25.,
                    theta=30.,
                    ):
    possibility = 1
    third_q_types = [g.nodes[p]['third_q_type'] for p in path]
    third_q_type_num = 0
    for third_q_type in third_q_types:
        if third_q_type == 1:
            third_q_type_num += 1
    for case in itertools.product([True, False], repeat=third_q_type_num):
        type_index = 0
        cumulate_error = np.zeros((2,))
        true_num = 0
        for c in case:
            if c:
                true_num += 1
        case_possility = np.power(0.8, true_num) * np.power(0.2, len(case) - true_num)
        for i in range(1, len(path)):
            cumulate_error += delta * cal_distance(g.nodes[path[i]]['pos'], g.nodes[path[i - 1]]['pos'])
            if g.nodes[path[i]]['third_q_type'] == 0:
                if case[type_index]:
                    if g.nodes[path[i]]['type'] == 0:
                        cumulate_error = modify_horizontal_error(cumulate_error, beta1, beta2)
                    if g.nodes[i]['type'] == 1:
                        cumulate_error = modify_vertical_error(cumulate_error, alpha1, alpha2)
                else:
                    if g.nodes[path[i]]['type'] == 0:
                        cumulate_error = failed_modify_horizontal_error(cumulate_error, beta1, beta2)
                    if g.nodes[i]['type'] == 1:
                        cumulate_error = failed_modify_horizontal_error(cumulate_error, alpha1, alpha2)
                type_index += 1
            else:
                if g.nodes[path[i]]['type'] == 0:
                    cumulate_error = modify_horizontal_error(cumulate_error, beta1, beta2)
                if g.nodes[path[i]]['type'] == 1:
                    cumulate_error = modify_vertical_error(cumulate_error, alpha1, alpha2)
            if not examine_error(cumulate_error):
                possibility -= case_possility
                break
    return possibility


def c_test(k, punishment_factor, file_path, para, c):
    a, b, correction_points = load_dataset(file_path, k)
    points_graph = nx.DiGraph()
    points_graph.add_node(int(a[0]), pos=a[1:4], type=a[4], third_q_type=a[5])
    for i in range(correction_points.shape[0]):
        points_graph.add_node(int(correction_points[i, 0]),
                              pos=correction_points[i, 1:4],
                              type=correction_points[i, 4],
                              third_q_type=correction_points[i, 5])
    points_graph.add_node(int(b[0]), pos=b[1:4], type=b[4], third_q_type=b[5])
    points_graph = build_adj_matrix(points_graph, **para)
    # points_graph = add_node_num_weight(points_graph, punishment_factor)
    path, cost_time = dijkstra_with_constrain8(points_graph, 0,
                                               326, **para,
                                               punishment_factor=punishment_factor,
                                               c=c)

    dis = cal_cumulate_distance(points_graph, path['path'])
    node_num = len(path['path']) - 2
    output_result(path, points_graph, './result_c_test/result_2/%f_%f_%dpath.xlsx' % (c,
                                                                                      dis,
                                                                                      node_num))
    print(path['path'], dis)


def get_dubins_path(g: nx.DiGraph, path_pos, path):
    path_pos_2d = path_pos[:, :2]
    sender_row = 0
    sender_angle = 0
    pos_x = []
    pos_y = []
    while sender_row < path_pos_2d.shape[0] - 1:
        if sender_row == 0:
            sender = path_pos_2d[sender_row, :]
            reciever = path_pos_2d[sender_row + 1, :]
            sender_angle = cal_next_sender_horizontal_direction(sender, reciever)

        else:
            sender = path_pos_2d[sender_row, :]
            reciever = path_pos_2d[sender_row + 1, :]
            reciever_angle = cal_next_sender_horizontal_direction(sender, reciever)
            sender_angle_x = cal_angle_of_two_vec(np.array((1, 0), dtype=float),
                                                  sender_angle)
            reciever_angle_x = cal_angle_of_two_vec(np.array((1, 0), dtype=float),
                                                    reciever_angle)
            px, py, _, _, _, _ = dubins.dubins_path_planning(sender[0],
                                                             sender[1],
                                                             sender_angle_x,
                                                             reciever[0],
                                                             reciever[1],
                                                             reciever_angle_x,
                                                             c=1 / 200)
            px = np.array(px)
            py = np.array(py)
            pos_x.append(px)
            pos_y.append(py)
            sender_angle = reciever_angle
        sender_row += 1
    all_pos_x = np.hstack(pos_x)
    all_pos_y = np.hstack(pos_y)
    # plt.figure(figsize=(60, 20))
    plt.plot(all_pos_x, all_pos_y, color='black', linewidth=1)
    plt.plot([g.nodes[path[0]]['pos'][0], g.nodes[path[1]]['pos'][0]],
             [g.nodes[path[0]]['pos'][1], g.nodes[path[1]]['pos'][1]],
             color='black', linewidth=1)
    plt.scatter(path_pos_2d[0, 0], path_pos_2d[0, 1], color='black', s=40)
    plt.scatter(path_pos_2d[-1, 0], path_pos_2d[-1, 1], color='red', s=40)
    horizontal_points = []
    vertical_points = []
    for p in path:
        if g.nodes[p]['type'] == 0:
            horizontal_points.append(p)
        elif g.nodes[p]['type'] == 1:
            vertical_points.append(p)
    horizontal_points_pos = get_path_pos(g, horizontal_points)[:, :2]
    vertical_points = get_path_pos(g, vertical_points)[:, :2]
    plt.scatter(horizontal_points_pos[:, 0], horizontal_points_pos[:, 1], color='green', s=20)
    plt.scatter(vertical_points[:, 0], vertical_points[:, 1], color='blue', s=20)
    plt.xlabel('x/m')
    plt.xlim(0, 100000)
    plt.ylabel('y/m')
    # plt.title('局部放大图')
    plt.title('全局路线总览')
    plt.ylim(0, 100000)
    plt.savefig('局部图.png', dpi=1024)
    plt.show()
    return all_pos_x, all_pos_y



def get_path_pos(g: nx.DiGraph, path_pos: list):
    p_list = []
    for p in path_pos:
        p_list.append(g.nodes[p]['pos'])
    p_arr = np.array(p_list).reshape((-1, 3))
    return p_arr


def k_weight_test(k, punishment_factor, file_path, para):
    a, b, correction_points = load_dataset(file_path, k)
    points_graph = nx.DiGraph()
    points_graph.add_node(int(a[0]), pos=a[1:4], type=a[4], third_q_type=a[5])
    for i in range(correction_points.shape[0]):
        points_graph.add_node(int(correction_points[i, 0]),
                              pos=correction_points[i, 1:4],
                              type=correction_points[i, 4],
                              third_q_type=correction_points[i, 5])
    points_graph.add_node(int(b[0]), pos=b[1:4], type=b[4], third_q_type=b[5])
    points_graph = build_adj_matrix(points_graph, **para)
    # points_graph = add_node_num_weight(points_graph, punishment_factor)
    path, cost_time = dijkstra_with_constrain8(points_graph, 0,
                                               612, **para,
                                               punishment_factor=punishment_factor)
    dis = cal_cumulate_distance(points_graph, path['path'])
    node_num = len(path['path']) - 2
    edge_num = points_graph.number_of_edges()
    output_result(path, points_graph, './result_k_test/result_1_q2/%d_%d_%f_%dpath.xlsx' % (k, punishment_factor,
                                                                                            dis,
                                                                                            node_num))
    print(path['path'], dis, cost_time, edge_num)


def time_series_statistics():
    time_series = []
    k_series = np.arange(50, 612, 50)
    for k in k_series:
        a, b, correction_points = load_dataset(DATA1_PATH, k)
        points_graph = nx.DiGraph()
        points_graph.add_node(int(a[0]), pos=a[1:4], type=a[4], third_q_type=a[5])
        for i in range(correction_points.shape[0]):
            points_graph.add_node(int(correction_points[i, 0]),
                                  pos=correction_points[i, 1:4],
                                  type=correction_points[i, 4],
                                  third_q_type=correction_points[i, 5])
        points_graph.add_node(int(b[0]), pos=b[1:4], type=b[4], third_q_type=b[5])
        points_graph = build_adj_matrix(points_graph, **para1)
        _, cost_time = dijkstra_with_constrain(points_graph, 0, 612, **para1)
        print(cost_time)
        time_series.append(cost_time)
    time_series = np.array(time_series)
    plt.plot(k_series, time_series, '-o', color='black')
    plt.xlabel('矫正点数量')
    plt.ylabel('花费时间/秒')
    plt.title('算法花费时间统计')
    plt.show()



def dijkstra_with_possible_constrain(g: nx.DiGraph, start, destination,
                                     delta=0.001,
                                     alpha1=25.,
                                     alpha2=15.,
                                     beta1=20.,
                                     beta2=25.,
                                     theta=30.,
                                     punishment_factor=0):
    dist = {}
    err_dist = {}
    path = {}
    visited = {}
    # current_error = {}
    start_time = time.time()

    for node in g.nodes:
        dist[node] = np.inf
        err_dist[node] = np.inf
        path[node] = {'path': [],
                      'cumulative_error': [],
                      'pre_modify_error': []}
        visited[node] = False
    dist[start] = 0
    err_dist[start] = 0
    path[start]['path'] = [start]
    path[start]['cumulative_error'] = [np.zeros((2,))]
    path[start]['pre_modify_error'] = [np.zeros((2,))]
    pq = []
    heapq.heappush(pq, [dist[start], start])

    while len(pq):
        error_distance, v = heapq.heappop(pq)
        if visited[v]:
            continue
        visited[v] = True
        p = copy.deepcopy(path[v])
        temp_e = p['cumulative_error'][-1].copy()
        for n in g.adj[v]:
            # for k in g.adj[v]:
            #     print(v, k, g[v][k]['distance'])
            current_dis = g[v][n]['distance']
            new_distance = dist[v] + current_dis
            temp_e += (g[v][n]['edge_error'], g[v][n]['edge_error'])
            pre_modified_e = temp_e.copy()
            p['pre_modify_error'].append(pre_modified_e)
            # print(new_distance < dist[n], not visited[n], examine_error(temp_e, theta=theta))
            # error_distance = pre_modified_e[0] * (50 + punishment_factor * 0.05) + \
            #                  pre_modified_e[1] * (50 + punishment_factor * 0.05) + \
            #                  new_distance + \
            #                  punishment_factor * len(path[n]['path'])

            error_distance = pre_modified_e[0] + pre_modified_e[1] + new_distance * punishment_factor
                             # new_distance + \
                             # punishment_factor * len(path[n]['path'])
            if error_distance < err_dist[n] and (not visited[n]) and examine_error(temp_e, theta=theta):
                # print(new_distance < dist[n], not visited[n], examine_error(temp_e, theta=theta))
                err_dist[n] = error_distance
                dist[n] = new_distance
                if g.nodes[n]['third_q_type'] == 0:
                    if g.nodes[n]['type'] == 0:
                        temp_e = modify_horizontal_error_possible(temp_e, beta1, beta2)
                    if g.nodes[n]['type'] == 1:
                        temp_e = modify_vertical_error_possible(temp_e, alpha1, alpha2)
                    heapq.heappush(pq, (err_dist[n], n))
                    temp = p.copy()
                    temp['path'].append(n)
                    temp['cumulative_error'].append(temp_e)
                    path[n] = temp
                else:
                    if g.nodes[n]['type'] == 0:
                        temp_e = modify_horizontal_error(temp_e, beta1, beta2)
                    if g.nodes[n]['type'] == 1:
                        temp_e = modify_vertical_error(temp_e, alpha1, alpha2)

                    heapq.heappush(pq, (err_dist[n], n))
                    temp = p.copy()
                    temp['path'].append(n)
                    temp['cumulative_error'].append(temp_e)
                    path[n] = temp
            p = copy.deepcopy(path[v])
            temp_e = p['cumulative_error'][-1].copy()
    end_time = time.time()
    cost_time = end_time - start_time
    # return path
    return path[destination], cost_time


def dijkstra_with_constrain8(g: nx.DiGraph, start, destination,
                             delta=0.001,
                             alpha1=25.,
                             alpha2=15.,
                             beta1=20.,
                             beta2=25.,
                             theta=30.,
                             punishment_factor=100,
                             c=1 / 200):
    dist = {}
    # dubins_dist = {}
    objective = {}
    path = {}
    visited = {}
    # current_error = {}
    start_time = time.time()

    for node in g.nodes:
        dist[node] = np.inf
        # dubins_dist[node] = np.inf
        objective[node] = np.inf
        path[node] = {
            'path': [],
            'cumulative_error': [],
            'pre_modify_error': [],
            'sender_direction': [],
        }
        visited[node] = False
    dist[start] = 0
    # dubins_dist[start] = 0
    objective[start] = 0
    path[start]['path'] = [start]
    path[start]['cumulative_error'] = [np.zeros((2,))]
    path[start]['pre_modify_error'] = [np.zeros((2,))]
    pq = []
    heapq.heappush(pq, [dist[start], start])

    loop_count = 0
    while len(pq):
        obj, v = heapq.heappop(pq)
        if visited[v]:
            continue
        loop_count += 1
        visited[v] = True
        p = copy.deepcopy(path[v])
        temp_e = p['cumulative_error'][-1].copy()
        for n in g.adj[v]:
            # for k in g.adj[v]:
            #     print(v, k, g[v][k]['distance'])
            # straight_dis = g[v][n]['distance']
            if v == 0:
                current_dis = g[v][n]['distance']
                next_sender_horizontal_direction = cal_next_sender_horizontal_direction(
                    g.nodes[v]['pos'][:2],
                    g.nodes[n]['pos'][:2]
                )
                new_distance = dist[v] + current_dis
                temp_e += (delta * current_dis, delta * current_dis)
                pre_modified_e = temp_e.copy()
                p['pre_modify_error'].append(pre_modified_e)
                # print(new_distance < dist[n], not visited[n], examine_error(temp_e, theta=theta))
                obj = pre_modified_e[0] * (50 + punishment_factor * 0.05) + \
                      pre_modified_e[1] * (50 + punishment_factor * 0.05) + \
                      new_distance + \
                      punishment_factor * len(path[n]['path'])
                if obj < objective[n] and (not visited[n]) and examine_error(temp_e, theta=theta):
                    # print(new_distance < dist[n], not visited[n], examine_error(temp_e, theta=theta))
                    dist[n] = new_distance
                    # dubins_dist[n] = current_dis.copy()
                    objective[n] = obj.copy()
                    if g.nodes[n]['type'] == 0:
                        temp_e = modify_horizontal_error(temp_e, beta1, beta2)
                    if g.nodes[n]['type'] == 1:
                        temp_e = modify_vertical_error(temp_e, alpha1, alpha2)
                    heapq.heappush(pq, (objective[n], n))
                    temp = p.copy()
                    temp['path'].append(n)
                    temp['cumulative_error'].append(temp_e)
                    temp['sender_direction'].append(next_sender_horizontal_direction)
                    path[n] = temp
                p = copy.deepcopy(path[v])
                temp_e = p['cumulative_error'][-1].copy()
            else:
                # sender_angle = cal_angle_of_two_vec(np.array(g.nodes[n]['pos'][0] - g.nodes[v]['pos'][0],
                #                                              g.nodes[n]['pos'][1] - g.nodes[v]['pos'][1]),
                #                                     p['sender_direction'][-1])
                next_sender_horizontal_direction = cal_next_sender_horizontal_direction(g.nodes[v]['pos'][:2],
                                                                                        g.nodes[n]['pos'][:2])
                sender_angle = cal_angle_of_two_vec(np.array((1, 0), dtype=float), p['sender_direction'][-1])
                receiver_angle = cal_angle_of_two_vec(np.array((1, 0), dtype=float), next_sender_horizontal_direction)
                current_dis = cal_dubins_dis(g, v, n, sender_angle, receiver_angle, c)
                new_distance = dist[v] + current_dis
                temp_e += (delta * current_dis, delta * current_dis)
                pre_modified_e = temp_e.copy()
                p['pre_modify_error'].append(pre_modified_e)
                # print(new_distance < dist[n], not visited[n], examine_error(temp_e, theta=theta))
                obj = temp_e[0] * (50 + punishment_factor * 0.05) + \
                      temp_e[1] * (50 + punishment_factor * 0.05) + \
                      new_distance + \
                      punishment_factor * len(path[n]['path'])
                if obj < objective[n] and (not visited[n]) and examine_error(temp_e, theta=theta):
                    # print(new_distance < dist[n], not visited[n], examine_error(temp_e, theta=theta))
                    # dubins_dist[n] = current_dis.copy()
                    dist[n] = new_distance
                    if g.nodes[n]['type'] == 0:
                        temp_e = modify_horizontal_error(temp_e, beta1, beta2)
                    if g.nodes[n]['type'] == 1:
                        temp_e = modify_vertical_error(temp_e, alpha1, alpha2)

                    objective[n] = obj.copy()
                    heapq.heappush(pq, (objective[n], n))
                    temp = p.copy()
                    temp['path'].append(n)
                    temp['cumulative_error'].append(temp_e)
                    temp['sender_direction'].append(next_sender_horizontal_direction)
                    path[n] = temp
                p = copy.deepcopy(path[v])
                temp_e = p['cumulative_error'][-1].copy()
    end_time = time.time()
    cost_time = end_time - start_time
    # return path
    print('dubins_distance:', dist[612])
    return path[destination], cost_time


def dijkstra_with_constrain(g: nx.DiGraph, start, destination,
                            delta=0.001,
                            alpha1=25.,
                            alpha2=15.,
                            beta1=20.,
                            beta2=25.,
                            theta=30.,
                            punishment_factor=0):
    dist = {}
    err_dist = {}
    path = {}
    visited = {}
    # current_error = {}
    start_time = time.time()

    for node in g.nodes:
        dist[node] = np.inf
        err_dist[node] = np.inf
        path[node] = {'path': [],
                      'cumulative_error': [],
                      'pre_modify_error': []}
        visited[node] = False
    dist[start] = 0
    err_dist[start] = 0
    path[start]['path'] = [start]
    path[start]['cumulative_error'] = [np.zeros((2,))]
    path[start]['pre_modify_error'] = [np.zeros((2,))]
    pq = []
    heapq.heappush(pq, [dist[start], start])

    while len(pq):
        error_distance, v = heapq.heappop(pq)
        if visited[v]:
            continue
        visited[v] = True
        p = copy.deepcopy(path[v])
        temp_e = p['cumulative_error'][-1].copy()
        for n in g.adj[v]:
            # for k in g.adj[v]:
            #     print(v, k, g[v][k]['distance'])
            current_dis = g[v][n]['distance']
            new_distance = dist[v] + current_dis
            temp_e += (g[v][n]['edge_error'], g[v][n]['edge_error'])
            pre_modified_e = temp_e.copy()
            p['pre_modify_error'].append(pre_modified_e)
            # print(new_distance < dist[n], not visited[n], examine_error(temp_e, theta=theta))
            error_distance = pre_modified_e[0] * (50 + punishment_factor * 0.05) + \
                             pre_modified_e[1] * (50 + punishment_factor * 0.05) + \
                             new_distance + \
                             punishment_factor * len(path[n]['path'])

            if error_distance < err_dist[n] and (not visited[n]) and examine_error(temp_e, theta=theta):
                # print(new_distance < dist[n], not visited[n], examine_error(temp_e, theta=theta))
                err_dist[n] = error_distance
                dist[n] = new_distance
                if g.nodes[n]['type'] == 0:
                    temp_e = modify_horizontal_error(temp_e, beta1, beta2)
                if g.nodes[n]['type'] == 1:
                    temp_e = modify_vertical_error(temp_e, alpha1, alpha2)

                heapq.heappush(pq, (err_dist[n], n))
                temp = p.copy()
                temp['path'].append(n)
                temp['cumulative_error'].append(temp_e)
                path[n] = temp
            p = copy.deepcopy(path[v])
            temp_e = p['cumulative_error'][-1].copy()
    end_time = time.time()
    cost_time = end_time - start_time
    # return path
    return path[destination], cost_time


if __name__ == '__main__':
    main()
