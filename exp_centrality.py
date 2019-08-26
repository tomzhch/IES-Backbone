import centrality_layout as cenLay
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import tool
import os
import time
import random
from scipy import interpolate

def all_graph():
    pass

def print_graph(dir, rate_distance, rate_edge_number):
    name = dir.split("/")[-1]
    graph = tool.readGraph(dir)
    graph = nx.subgraph(graph, max(nx.connected_components(graph), key=len))
    # color2 = tool.render(dir, graph)
    print("node:", graph.number_of_nodes())
    print("edges:", graph.number_of_edges())

    # 检查文件夹
    savePath = dir + '/eval_result/'  # 实验结果文件夹
    EBCTime_name = name + '_EBCTime.txt'
    firstTime_bsm_name = name + '_firstTime_bsm.txt'
    if not os.path.exists(savePath):
        os.mkdir(savePath)

    # 布局
    # 读EBC
    sub_edge_name = savePath + name + '_sub_EBC.txt'
    graph_name = savePath + name + '_EBC.txt'
    graph, sub_edge, time_EBC = tool.get_every_epoch_EBC(
        graph_name, sub_edge_name, graph)

    # 判断是否需要一次布局
    pos_bsm_name = savePath + name + '_pos_bsm.txt'

    bsm_in, \
        pos_bsm, \
        pos_bsm_in, \
        time_first_bsm, \
        time_insert_bsm, \
        time_second_bsm, \
        out_of_range_bsm, \
        save_sub_bsm = tool.get_every_epoch_layout(pos_bsm_name,
                                                   sub_edge,
                                                   graph,
                                                   rate_distance,
                                                   rate_edge_number,
                                                   'bsm')

    # 保存结果
    nodeColor, nodeClassInCD = tool.render_csv(dir, sub_edge)
    # 时间
    if time_EBC > 0:
        tool.save_graph_time(savePath + EBCTime_name, time_EBC)
    if time_first_bsm > 0:
        tool.save_graph_time(savePath + firstTime_bsm_name, time_first_bsm)

    # 文件名
    savePath = dir + '/'
    bsm_in_name = name + '_bsm_in_' + \
        str(rate_distance) + '_' + str(rate_edge_number) + \
        '_' + str(bsm_in.number_of_edges())

    # 图片 
    draw(bsm_in, pos_bsm_in, nodeColor, sub_edge, graph,
         savePath + bsm_in_name + '.pdf')
    draw(bsm_in, pos_bsm, nodeColor, sub_edge,graph,
         savePath + bsm_in_name + '_edge' + '.pdf')
    draw(sub_edge, pos_bsm, nodeColor, sub_edge,graph,
             savePath + name + '.pdf')

    print(homo(graph, nodeClassInCD))
    print(homo(sub_edge, nodeClassInCD))
    print(homo(bsm_in, nodeClassInCD))



def every_graph(dir):
    # 检查文件夹
    name = dir.split("/")[-1]
    savePath = dir + '/eval_result/'  # 实验结果文件夹
    if not os.path.exists(savePath):
        os.mkdir(savePath)
    
    # 文件名
    diffhomo_spring_name = name + '_diffhomo_spring.txt'
    diffhomo_spring_pic_name = name + '_diffhomo_spring.pdf'
    diffhomo_bsm_name = name + '_diffhomo_bsm.txt'
    diffhomo_bsm_pic_name = name + '_diffhomo_bsm.pdf'
    homo_spring_name = name + '_homo_spring.txt'
    homo_spring_pic_name = name + '_homo_spring.pdf'
    homo_bsm_name = name + '_homo_bsm.txt'
    homo_bsm_pic_name = name + '_homo_bsm.pdf'
    EBCTime_name = name + '_EBCTime.txt'
    firstTime_spring_name = name + '_firstTime_spring.txt'
    firstTime_bsm_name = name + '_firstTime_bsm.txt'
    insertTime_spring_name = name + '_insertTime_spring.txt'
    insertTime_bsm_name = name + '_insertTime_bsm.txt'
    secondTime_spring_name = name + '_secondTime_spring.txt'
    secondTime_bsm_name = name + '_secondTime_bsm.txt'

    # homo文件
    diffhomo_CD_spring = tool.read_every_graph_result(savePath + diffhomo_spring_name)
    diffhomo_CD_bsm = tool.read_every_graph_result(savePath + diffhomo_bsm_name)
    homo_CD_spring = tool.read_every_graph_result(savePath + homo_spring_name)
    homo_CD_bsm = tool.read_every_graph_result(savePath + homo_bsm_name)
    # 时间文件
    insertTime_spring = tool.read_every_graph_result(savePath + insertTime_spring_name)
    insertTime_bsm = tool.read_every_graph_result(savePath + insertTime_bsm_name)
    secondTime_spring = tool.read_every_graph_result(savePath + secondTime_spring_name)
    secondTime_bsm = tool.read_every_graph_result(savePath + secondTime_bsm_name)

    epoch_num = 0
    for i in np.arange(0.05, 0.4, 0.05):
        rate_distance = round(i, 2)
        for j in np.arange(0.05, 0.55, 0.05):
            rate_edge_number = round(j, 2)
            print('test: ', rate_distance, rate_edge_number)
            if (rate_distance, rate_edge_number) not in homo_CD_spring.keys():
                runtime, homophily, cc, out_of_range = every_epoch(
                    dir, rate_distance, rate_edge_number)
                if not out_of_range:
                    # 收集同质性
                    homo_CD_spring.update(
                        {(rate_distance, rate_edge_number): homophily['CD_spring']})
                    homo_CD_bsm.update(
                        {(rate_distance, rate_edge_number): homophily['CD_bsm']})
                    diffhomo_CD_spring.update(
                        {(rate_distance, rate_edge_number): homophily['CD_increase_spring']})
                    diffhomo_CD_bsm.update(
                        {(rate_distance, rate_edge_number): homophily['CD_increase_bsm']})

                    # 收集时间
                    insertTime_spring.update(
                        {(rate_distance, rate_edge_number): runtime['insert_spring']})
                    insertTime_bsm.update(
                        {(rate_distance, rate_edge_number): runtime['insert_bsm']})
                    secondTime_spring.update(
                        {(rate_distance, rate_edge_number): runtime['second_spring']})
                    secondTime_bsm.update(
                        {(rate_distance, rate_edge_number): runtime['second_bsm']})
                    
                    # 定时保存
                    if epoch_num % 1 == 0 and epoch_num != 0:  
                        tool.save_every_graph_result(
                            savePath + homo_spring_name, homo_CD_spring)
                        tool.save_every_graph_result(
                            savePath + homo_bsm_name, homo_CD_bsm)
                        tool.save_every_graph_result(
                            savePath + diffhomo_spring_name, diffhomo_CD_spring)
                        tool.save_every_graph_result(
                            savePath + diffhomo_bsm_name, diffhomo_CD_bsm)
                        tool.save_every_graph_result(
                            savePath + insertTime_spring_name, insertTime_spring)
                        tool.save_every_graph_result(
                            savePath + insertTime_bsm_name, insertTime_bsm)
                        tool.save_every_graph_result(
                            savePath + secondTime_spring_name, secondTime_spring)
                        tool.save_every_graph_result(
                            savePath + secondTime_bsm_name, secondTime_bsm)
                    epoch_num = epoch_num + 1
                else:
                    break

    # 保存结果
    # 同质性
    tool.save_every_graph_result(savePath + homo_spring_name, homo_CD_spring)
    tool.save_every_graph_result(savePath + homo_bsm_name, homo_CD_bsm)
    tool.save_every_graph_result(savePath + diffhomo_spring_name, diffhomo_CD_spring)
    tool.save_every_graph_result(savePath + diffhomo_bsm_name, diffhomo_CD_bsm)
    # 时间
    tool.save_every_graph_result(savePath + insertTime_spring_name, insertTime_spring)
    tool.save_every_graph_result(savePath + insertTime_bsm_name, insertTime_bsm)
    tool.save_every_graph_result(savePath + secondTime_spring_name, secondTime_spring)
    tool.save_every_graph_result(savePath + secondTime_bsm_name, secondTime_bsm)

    EBC_time = tool.read_graph_time(savePath + EBCTime_name)
    firstTime_spring = tool.read_graph_time(savePath + firstTime_spring_name)
    firstTime_bsm = tool.read_graph_time(savePath + firstTime_bsm_name)
    aver_insert_spring = tool.get_every_graph_aver_time(insertTime_spring)
    aver_insert_bsm = tool.get_every_graph_aver_time(insertTime_bsm)
    aver_second_spring = tool.get_every_graph_aver_time(secondTime_spring)
    aver_second_bsm = tool.get_every_graph_aver_time(secondTime_bsm)
    

    # 显示结果
    tool.save_every_graph_homo_pic(
        savePath + homo_spring_pic_name, homo_CD_spring)
    tool.save_every_graph_homo_pic(savePath + homo_bsm_pic_name, homo_CD_bsm)
    tool.save_every_graph_homo_pic(
        savePath + diffhomo_spring_pic_name, diffhomo_CD_spring)
    tool.save_every_graph_homo_pic(savePath + diffhomo_bsm_pic_name, diffhomo_CD_bsm)

    print(EBC_time, firstTime_spring, firstTime_bsm, aver_insert_spring, aver_insert_bsm, aver_second_spring, aver_second_bsm)
    name = dir.split("/")[-1]
    graph = tool.readGraph(dir)
    graph = nx.subgraph(graph, max(nx.connected_components(graph), key=len))
    # color2 = tool.render(dir, graph)
    print("node:", graph.number_of_nodes())
    print("edges:", graph.number_of_edges())

    return EBC_time, firstTime_spring, firstTime_bsm, aver_insert_spring, aver_insert_bsm, aver_second_spring, aver_second_bsm


def every_epoch(dir, rate_distance, rate_edge_number):  # 测试一个网络的一组参数
    name = dir.split("/")[-1]
    graph = tool.readGraph(dir)
    graph = nx.subgraph(graph, max(nx.connected_components(graph), key=len))
    # color2 = tool.render(dir, graph)
    print("node:", graph.number_of_nodes())
    print("edges:", graph.number_of_edges())

    # 检查文件夹
    savePath = dir + '/eval_result/'  # 实验结果文件夹
    EBCTime_name = name + '_EBCTime.txt'
    firstTime_spring_name = name + '_firstTime_spring.txt'
    firstTime_bsm_name = name + '_firstTime_bsm.txt'
    if not os.path.exists(savePath):
        os.mkdir(savePath)

    springPic_path = savePath + 'spring_pic/'
    bsmPic_path = savePath + 'bsm_pic/'
    if not os.path.exists(springPic_path):  # 力引导布局结果图
        os.mkdir(springPic_path)
    if not os.path.exists(bsmPic_path):  # bsm布局结果图
        os.mkdir(bsmPic_path)

    springPosIn_path = savePath + 'spring_pos_in/'
    bsmPosIn_path = savePath + 'bsm_pos_in/'
    if not os.path.exists(springPosIn_path):  # 力引导布局结果坐标
        os.mkdir(springPosIn_path)
    if not os.path.exists(bsmPosIn_path):  # bsm布局结果坐标
        os.mkdir(bsmPosIn_path)

    springGraphIn_path = savePath + 'spring_graph_in/'
    bsmGraphIn_path = savePath + 'bsm_graph_in/'
    if not os.path.exists(springGraphIn_path):  # 力引导插入结果网络结构
        os.mkdir(springGraphIn_path)
    if not os.path.exists(bsmGraphIn_path):  # bsm插入结果网络结构
        os.mkdir(bsmGraphIn_path)

    # 布局
    # 读EBC
    sub_edge_name = savePath + name + '_sub_EBC.txt'
    graph_name = savePath + name + '_EBC.txt'
    graph, sub_edge, time_EBC = tool.get_every_epoch_EBC(
        graph_name, sub_edge_name, graph)

    # 判断是否需要一次布局
    pos_spring_name = savePath + name + '_pos_spring.txt'
    pos_bsm_name = savePath + name + '_pos_bsm.txt'

    spring_in, \
        pos_spring, \
        pos_spring_in, \
        time_first_spring, \
        time_insert_spring, \
        time_second_spring, \
        out_of_range_spring, \
        save_sub_spring = tool.get_every_epoch_layout(pos_spring_name,
                                                      sub_edge,
                                                      graph,
                                                      rate_distance,
                                                      rate_edge_number,
                                                      'spring')

    bsm_in, \
        pos_bsm, \
        pos_bsm_in, \
        time_first_bsm, \
        time_insert_bsm, \
        time_second_bsm, \
        out_of_range_bsm, \
        save_sub_bsm = tool.get_every_epoch_layout(pos_bsm_name,
                                                   sub_edge,
                                                   graph,
                                                   rate_distance,
                                                   rate_edge_number,
                                                   'bsm')

    # 保存结果
    nodeColor, nodeClassInCD = tool.render_csv(dir, sub_edge)
    # 时间
    if time_EBC > 0:
        tool.save_graph_time(savePath + EBCTime_name, time_EBC)
    if time_first_spring > 0:
        tool.save_graph_time(savePath + firstTime_spring_name, time_first_spring)
    if time_first_bsm > 0:
        tool.save_graph_time(savePath + firstTime_bsm_name, time_first_bsm)

    # 文件名
    spring_in_name = name + '_spring_in_' + \
        str(rate_distance) + '_' + str(rate_edge_number) + \
        '_' + str(spring_in.number_of_edges())
    bsm_in_name = name + '_bsm_in_' + \
        str(rate_distance) + '_' + str(rate_edge_number) + \
        '_' + str(bsm_in.number_of_edges())
    spring_pic_name = name + '_sub_spring'
    bsm_pic_name = name + '_sub_bsm'

    # 图片 
    draw(spring_in, pos_spring_in, nodeColor, sub_edge, graph,
         springPic_path + spring_in_name + '.pdf')
    draw(bsm_in, pos_bsm_in, nodeColor, sub_edge, graph,
         bsmPic_path + bsm_in_name + '.pdf')
    if save_sub_spring:
        draw(sub_edge, pos_spring, nodeColor, sub_edge, graph,
             savePath + spring_pic_name + '.pdf')
    if save_sub_bsm:
        draw(sub_edge, pos_bsm, nodeColor, sub_edge, graph,
             savePath + bsm_pic_name + '.pdf')

    # 布局坐标
    tool.save_every_epoch_result(
        springPosIn_path + spring_in_name + '.txt', pos_spring_in)
    tool.save_every_epoch_result(
        bsmPosIn_path + bsm_in_name + '.txt', pos_bsm_in)

    # 输出的网络结构
    nx.write_edgelist(spring_in, springGraphIn_path +
                      spring_in_name + '.txt', data=False)
    nx.write_edgelist(spring_in, springGraphIn_path +
                      spring_in_name + '.csv', data=False)
    nx.write_edgelist(bsm_in, bsmGraphIn_path +
                      bsm_in_name + '.txt', data=False)
    nx.write_edgelist(bsm_in, bsmGraphIn_path +
                      bsm_in_name + '.csv', data=False)

    # 测试
    homo_CD_graph,_,_ = homo(graph, nodeClassInCD)
    homo_CD_sub, ho_sub, he_sub = homo(sub_edge, nodeClassInCD)
    homo_CD_spring, ho_CD_spring, he_CD_spring = homo(spring_in, nodeClassInCD)
    homo_CD_bsm, ho_CD_bsm, he_CD_bsm = homo(bsm_in, nodeClassInCD)
    increasehomo_spring = 0;
    increasehomo_bsm = 0;
    if ho_sub == ho_CD_spring and he_sub == he_CD_spring:
        increasehomo_spring = 0;
    else:
        increasehomo_spring = (ho_CD_spring - ho_sub)/(ho_CD_spring - ho_sub + he_CD_spring - he_sub)
    if ho_sub == ho_CD_bsm and he_sub == he_CD_bsm:
        increasehomo_bsm = 0;
    else:
        increasehomo_bsm = (ho_CD_bsm - ho_sub)/(ho_CD_bsm - ho_sub + he_CD_bsm - he_sub)
    # print(homo(graph, nodeClassInCD))
    # print(homo(sub_edge, nodeClassInCD))
    # print(homo(spring_in, nodeClassInCD))
    # print(homo(bsm_in, nodeClassInCD))

    # 返回结果
    # 时间
    runtime = dict()
    runtime.update({'EBC': time_EBC})
    runtime.update({'first_spring': time_first_spring})
    runtime.update({'insert_spring': time_insert_spring})
    runtime.update({'second_spring': time_second_spring})
    runtime.update({'first_bsm': time_first_bsm})
    runtime.update({'insert_bsm': time_insert_bsm})
    runtime.update({'second_bsm': time_second_bsm})

    # 同质性
    homophily = dict()
    homophily.update({'CD_graph': homo_CD_graph})
    homophily.update({'CD_sub': homo_CD_sub})
    homophily.update({'CD_spring': homo_CD_spring})
    homophily.update({'CD_bsm': homo_CD_bsm})
    homophily.update({'CD_increase_spring': increasehomo_spring})
    homophily.update({'CD_increase_bsm': increasehomo_bsm})

    # 聚集系数
    cc = dict()
    cc.update({'bsm': nx.average_clustering(bsm_in)})
    cc.update({'spring': nx.average_clustering(spring_in)})

    if out_of_range_bsm or out_of_range_spring:
        out_of_range = True
    else:
        out_of_range = False

    return runtime, homophily, cc, out_of_range


def homo(graph, classID):
    ho = 0
    he = 0
    for e in graph.edges():
        s = e[0]
        t = e[1]
        if classID[s] == -1 or classID[t] == -1:
            continue
        elif classID[s] == classID[t]:
            ho = ho + 1
        else:
            he = he + 1
    homophily = ho / (ho + he)
    # print(ho,he)
    return homophily, ho, he


def draw(g, pos, color, sub, graph, dir):
    plt.figure(figsize=(10, 10))
    
    # nx.draw_networkx_edges(graph, pos=pos,  edgelist=graph.edges(),
    #                         edge_color='gray')
    # nx.draw(g, pos=pos, node_color=color, nodelist=g.nodes(),
    #         node_size=10, edge_color='black', edge_size=1)
    nx.draw(g, pos=pos, node_color=color, nodelist=g.nodes(),
            node_size=10, edge_color='gray', edge_size=1)
    plt.savefig(dir, dpi=300)
    plt.close()


if __name__ == "__main__":
    random.seed(0)

    dir = "./data/facebook100/Rice31"
    # dir = "./data/vis"

    every_graph(dir)
    # print_graph(dir,0.15,0.25)
    # plt.subplot(121)
    # nx.draw(graph, node_color=color,  # with_labels=True, font_size=15,
    # node_size=10, edge_color='gray')
    print('Done.')
    # plt.show()
