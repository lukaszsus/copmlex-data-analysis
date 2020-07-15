import networkx as nx
import matplotlib.pyplot as plt

from utils.data_loader import create_graph, save_graph, load_graph, create_reports_graph

if __name__ == '__main__':
    # g = create_graph()
    # save_graph(g, "graph_mails.pkl")
    # g = load_graph("graph_mails.pkl")
    # print(g.nodes)
    # print(g.edges)

    g = create_reports_graph()
    nx.draw_networkx(g)
    plt.show()
    print(nx.is_tree(g))
    print(list(nx.topological_sort(g)))
