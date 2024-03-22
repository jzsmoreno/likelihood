from IPython.display import HTML, display
from pandas.core.frame import DataFrame
from pyvis.network import Network

from likelihood.tools import FeatureSelection


class DynamicGraph(FeatureSelection):
    """A class to represent a dynamic graph"""

    def __init__(self, df: DataFrame, n_importances: int):
        self.G = Network(
            notebook=True, cdn_resources="remote", directed=True
        )  # enable interactive visualization in Jupyter Notebooks
        self.df = df
        self.n_importances = n_importances
        super().__init__()

    def fit(self, **kwargs) -> None:
        """Fit the model according to the given data and parameters."""
        self.get_digraph(self.df, self.n_importances)
        # create a dictionary with the indexes and names of the dataframe
        self.get_index = dict(zip(self.df.columns, range(len(self.df.columns))))
        self._make_network()

    def _make_network(self) -> None:
        """Create nodes and edges of the network based on feature importance scores"""
        self._add_nodes()
        for i in range(len(self.all_features_imp_graph)):
            node = self.all_features_imp_graph[i][0]
            edges = self.all_features_imp_graph[i][1]

            for label, weight in edges:
                self.G.add_edge(i, self.get_index[label], weight=weight)

    def _add_nodes(self) -> None:
        for i in range(len(self.all_features_imp_graph)):
            node = self.all_features_imp_graph[i][0]
            self.G.add_node(n_id=i, label=node)

    def draw(self, name="graph.html", **kwargs) -> None:
        """Display the network using HTML format"""
        spring_length = kwargs["spring_length"] if "spring_length" in kwargs else 500
        node_distance = kwargs["node_distance"] if "node_distance" in kwargs else 100
        self.G.repulsion(node_distance=node_distance, spring_length=spring_length)
        self.G.show_buttons(filter_=["physics"])
        self.G.show(name)

        html_file_content = open(name, "r").read()
        display(HTML(html_file_content))
