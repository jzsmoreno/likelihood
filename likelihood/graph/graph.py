from typing import List

import networkx as nx
from IPython.display import HTML, display
from pandas.core.frame import DataFrame
from pyvis.network import Network

from likelihood.tools import FeatureSelection


class DynamicGraph(FeatureSelection):
    """A class to represent a dynamic graph"""

    def __init__(self, df: DataFrame, n_importances: int, **kwargs):
        self.G = Network(
            notebook=True, cdn_resources="remote", directed=True
        )  # enable interactive visualization in Jupyter Notebooks
        self.df = df
        self.n_importances = n_importances
        super().__init__(**kwargs)
        self.labels: List[str] = []

    def fit(self, **kwargs) -> None:
        """Fit the model according to the given data and parameters."""
        self.get_digraph(self.df, self.n_importances)
        # create a dictionary with the indexes and names of the dataframe
        self.get_index = dict(zip(self.X.columns, range(len(self.X.columns))))
        self._make_network()

    def _make_network(self) -> None:
        """Create nodes and edges of the network based on feature importance scores"""
        self._add_nodes()
        for i in range(len(self.all_features_imp_graph)):
            node = self.all_features_imp_graph[i][0]
            edges = self.all_features_imp_graph[i][1]

            for label, weight in edges:
                self.G.add_edge(self.get_index[node], self.get_index[label], weight=weight)

    def _add_nodes(self) -> None:
        for i in range(len(self.all_features_imp_graph)):
            node = self.all_features_imp_graph[i][0]
            self.labels.append(node)
            self.G.add_node(n_id=i, label=node)

    def draw(self, name="graph.html", **kwargs) -> None:
        """Display the network using HTML format"""
        spring_length = kwargs.get("spring_length", 500)
        node_distance = kwargs.get("node_distance", 100)
        self.G.repulsion(node_distance=node_distance, spring_length=spring_length)
        self.G.show_buttons(filter_=["physics"])
        self.G.show(name)

        html_file_content = open(name, "r").read()
        display(HTML(html_file_content))

    def pyvis_to_networkx(self):
        nx_graph = nx.Graph()

        # Adding nodes
        nodes = [d["id"] for d in self.G.nodes]
        for node_dic in self.G.nodes:
            id = node_dic["label"]
            del node_dic["label"]
            nx_graph.add_nodes_from([(id, node_dic)])
        self.node_edge_dict = dict(zip(nodes, self.labels))
        del nodes

        # Adding edges
        for edge in self.G.edges:
            source, target = self.node_edge_dict[edge["from"]], self.node_edge_dict[edge["to"]]
            del edge["from"]
            del edge["to"]
            nx_graph.add_edges_from([(source, target, edge)])

        return nx_graph


# -------------------------------------------------------------------------
if __name__ == "__main__":
    import numpy as np
    import pandas as pd

    # Generate data
    x = np.random.rand(3, 100)
    y = 0.1 * x[0, :] + 0.4 * x[1, :] + 0.5 * x[2, :] + 0.1
    # Create a DataFrame
    df = pd.DataFrame(x.T, columns=["x1", "x2", "x3"])
    df["y"] = y
    # Instantiate DynamicGraph
    fs = DynamicGraph(df, n_importances=2)
    fs.fit()
    fs.draw()
