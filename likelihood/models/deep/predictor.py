import random
import warnings
from typing import List

import matplotlib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import tensorflow as tf
from IPython.display import HTML, display
from matplotlib import cm
from matplotlib.colors import Normalize
from pandas.core.frame import DataFrame
from pandas.plotting import radviz
from sklearn.manifold import TSNE
from tensorflow.keras.layers import InputLayer

from likelihood.models.deep.autoencoders import AutoClassifier, sampling


class GetInsights:
    """
    A class to analyze the output of a neural network model, including visualizations
    of the weights, t-SNE representation, and feature statistics.

    Parameters
    ----------
    model : `AutoClassifier`
        The trained model to analyze.
    inputs : `np.ndarray`
        The input data for analysis.
    """

    def __init__(self, model: AutoClassifier, inputs: np.ndarray) -> None:
        """
        Initializes the GetInsights class.

        Parameters
        ----------
        model : `AutoClassifier`
            The trained model to analyze.
        inputs : `np.ndarray`
            The input data for analysis.
        """
        self.inputs = inputs
        self.model = model

        self.encoder_layer = (
            self.model.encoder.layers[1]
            if isinstance(self.model.encoder.layers[0], InputLayer)
            else self.model.encoder.layers[0]
        )
        self.decoder_layer = self.model.decoder.layers[0]

        self.encoder_weights = self.encoder_layer.get_weights()[0]
        self.decoder_weights = self.decoder_layer.get_weights()[0]

        self.sorted_names = self._generate_sorted_color_names()

    def _generate_sorted_color_names(self) -> list:
        """
        Generate sorted color names based on their HSV values.

        Parameters
        ----------
        `None`

        Returns
        -------
        `list` : Sorted color names.
        """
        colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
        by_hsv = sorted(
            (tuple(mcolors.rgb_to_hsv(mcolors.to_rgba(color)[:3])), name)
            for name, color in colors.items()
        )
        sorted_names = [name for hsv, name in by_hsv if hsv[1] > 0.4 and hsv[2] >= 0.4]
        random.shuffle(sorted_names)
        return sorted_names

    def render_html_report(
        self,
        frac: float = 0.2,
        top_k: int = 5,
        threshold_factor: float = 1.0,
        max_rows: int = 5,
        **kwargs,
    ) -> None:
        """
        Generate and display an embedded HTML report in a Jupyter Notebook cell.
        """
        display(HTML("<h2 style='margin-top:20px;'>📊 Predictor Analysis</h2>"))
        display(
            HTML(
                "<p>This section visualizes how the model predicts the data. "
                "You will see original inputs, reconstructed outputs, and analyses such as t-SNE "
                "that reduce dimensionality to visualize latent space clustering.</p>"
            )
        )
        stats_df = self.predictor_analyzer(frac=frac, **kwargs)

        display(HTML("<h2 style='margin-top:30px;'>🔁 Encoder-Decoder Graph</h2>"))
        display(
            HTML(
                "<p>This visualization displays the connections between layers in the encoder and decoder. "
                "Edges with the strongest weights are highlighted to emphasize influential features "
                "in the model's transformation.</p>"
            )
        )
        if not self.model.encoder.name.startswith("vae"):
            self.viz_encoder_decoder_graphs(threshold_factor=threshold_factor, top_k=top_k)

            display(HTML("<h2 style='margin-top:30px;'>🧠 Classifier Layer Graphs</h2>"))
            display(
                HTML(
                    "<p>This visualization shows how features propagate through each dense layer in the classifier. "
                    "Only the strongest weighted connections are shown to highlight influential paths through the network.</p>"
                )
            )
        self.viz_classifier_graphs(threshold_factor=threshold_factor, top_k=top_k)

        display(HTML("<h2 style='margin-top:30px;'>📈 Statistical Summary</h2>"))
        display(
            HTML(
                "<p>This table summarizes feature statistics grouped by predicted classes, "
                "including means, standard deviations, and modes, providing insight into "
                "feature distributions across different classes.</p>"
            )
        )

        if max_rows is not None and max_rows > 0:
            stats_to_display = stats_df.head(max_rows)
        else:
            stats_to_display = stats_df

        display(
            stats_to_display.style.set_table_attributes(
                "style='display:inline;border-collapse:collapse;'"
            )
            .set_caption("Feature Summary per Class")
            .set_properties(
                **{
                    "border": "1px solid #ddd",
                    "padding": "8px",
                    "text-align": "center",
                }
            )
        )

        display(
            HTML(
                "<p style='color: gray; margin-top:30px;'>Report generated with "
                "<code>GetInsights</code> class. For detailed customization, extend "
                "<code>render_html_report</code>.</p>"
            )
        )

    def viz_classifier_graphs(self, threshold_factor=1.0, top_k=5, save_path=None):
        """
        Visualize all Dense layers in self.model.classifier as a single directed graph,
        connecting each Dense layer to the next.
        """

        def get_top_k_edges(weights, src_prefix, dst_prefix, k):
            flat_weights = np.abs(weights.flatten())
            indices = np.argpartition(flat_weights, -k)[-k:]
            top_k_flat_indices = indices[np.argsort(-flat_weights[indices])]
            top_k_edges = []

            for flat_index in top_k_flat_indices:
                i, j = np.unravel_index(flat_index, weights.shape)
                top_k_edges.append((f"{src_prefix}_{i}", f"{dst_prefix}_{j}", weights[i, j]))
            return top_k_edges

        def add_dense_layer_edges(G, weights, layer_idx, threshold_factor, top_k):
            src_prefix = f"L{layer_idx}"
            dst_prefix = f"L{layer_idx + 1}"
            input_nodes = [f"{src_prefix}_{i}" for i in range(weights.shape[0])]
            output_nodes = [f"{dst_prefix}_{j}" for j in range(weights.shape[1])]

            G.add_nodes_from(input_nodes + output_nodes)

            abs_weights = np.abs(weights)
            threshold = threshold_factor * np.mean(abs_weights)
            top_k_edges = get_top_k_edges(weights, src_prefix, dst_prefix, top_k)
            top_k_set = set((u, v) for u, v, _ in top_k_edges)

            for i, src in enumerate(input_nodes):
                for j, dst in enumerate(output_nodes):
                    w = weights[i, j]
                    if abs(w) > threshold:
                        G.add_edge(src, dst, weight=w, highlight=(src, dst) in top_k_set)

        def compute_layout(G):
            pos = {}
            layer_nodes = {}

            for node in G.nodes():
                layer_idx = int(node.split("_")[0][1:])
                layer_nodes.setdefault(layer_idx, []).append(node)

            for layer_idx, nodes in sorted(layer_nodes.items()):
                y_positions = np.linspace(1, -1, len(nodes))
                for y, node in zip(y_positions, nodes):
                    pos[node] = (layer_idx * 2, y)

            return pos

        def draw_graph(G, pos, title, save_path=None):
            weights = [abs(G[u][v]["weight"]) for u, v in G.edges()]
            if not weights:
                print("No edges to draw.")
                return

            norm = Normalize(vmin=min(weights), vmax=max(weights))
            cmap = cm.get_cmap("coolwarm")

            edge_colors = [cmap(norm(G[u][v]["weight"])) for u, v in G.edges()]
            edge_widths = [1.0 + 2.0 * norm(abs(G[u][v]["weight"])) for u, v in G.edges()]

            fig, ax = plt.subplots(figsize=(12, 8))

            nx.draw(
                G,
                pos,
                ax=ax,
                with_labels=True,
                node_color="lightgray",
                node_size=1000,
                font_size=8,
                edge_color=edge_colors,
                width=edge_widths,
                arrows=True,
            )

            ax.set_title(title, fontsize=14)

            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            plt.colorbar(sm, ax=ax, orientation="vertical", label="Edge Weight")

            plt.tight_layout()
            if save_path:
                plt.savefig(save_path)
            plt.show()

        dense_layers = [
            layer
            for layer in self.model.classifier.layers
            if isinstance(layer, tf.keras.layers.Dense)
        ]

        if len(dense_layers) < 1:
            print("No Dense layers found in classifier.")
            return

        G = nx.DiGraph()
        for idx, layer in enumerate(dense_layers):
            weights = layer.get_weights()[0]
            add_dense_layer_edges(G, weights, idx, threshold_factor, top_k)

        pos = compute_layout(G)
        draw_graph(G, pos, "Classifier Dense Layers Graph", save_path)

    def viz_encoder_decoder_graphs(self, threshold_factor=1.0, top_k=5, save_path=None):
        """
        Visualize Dense layers in self.model.encoder and self.model.decoder as directed graphs.
        """

        def get_top_k_edges(weights, labels_src, labels_dst_prefix, k):
            flat_weights = np.abs(weights.flatten())
            indices = np.argpartition(flat_weights, -k)[-k:]
            top_k_flat_indices = indices[np.argsort(-flat_weights[indices])]
            top_k_edges = []
            for flat_index in top_k_flat_indices:
                i, j = np.unravel_index(flat_index, weights.shape)
                src_label = labels_src[i] if isinstance(labels_src, list) else f"{labels_src}_{i}"
                dst_label = f"{labels_dst_prefix}_{j}"
                top_k_edges.append((src_label, dst_label, weights[i, j]))
            return top_k_edges

        def add_layer_to_graph(
            G, weights, labels_src, labels_dst_prefix, x_offset, top_k_set, threshold
        ):
            output_nodes = [f"{labels_dst_prefix}_{j}" for j in range(weights.shape[1])]

            for node in labels_src + output_nodes:
                if node not in G:
                    G.add_node(node, x=x_offset if node in labels_src else x_offset + 1)

            for i, src in enumerate(labels_src):
                for j, dst in enumerate(output_nodes):
                    w = weights[i, j]
                    if abs(w) > threshold:
                        G.add_edge(src, dst, weight=w, highlight=(src, dst) in top_k_set)
            return output_nodes

        def layout_graph(G):
            pos = {}
            layers = {}
            for node, data in G.nodes(data=True):
                x = data["x"]
                layers.setdefault(x, []).append(node)

            for x in sorted(layers):
                nodes = layers[x]
                y_positions = np.linspace(1, -1, len(nodes))
                for y, node in zip(y_positions, nodes):
                    pos[node] = (x, y)
            return pos

        def draw_graph(G, title, ax):
            weights = [abs(G[u][v]["weight"]) for u, v in G.edges()]
            if not weights:
                return

            norm = Normalize(vmin=min(weights), vmax=max(weights))
            cmap = cm.get_cmap("coolwarm")

            edge_colors = [cmap(norm(G[u][v]["weight"])) for u, v in G.edges()]
            edge_widths = [1.0 + 2.0 * norm(abs(G[u][v]["weight"])) for u, v in G.edges()]

            pos = layout_graph(G)
            nx.draw(
                G,
                pos,
                ax=ax,
                with_labels=True,
                node_color="lightgray",
                node_size=1000,
                font_size=8,
                edge_color=edge_colors,
                width=edge_widths,
                arrows=True,
            )

            ax.set_title(title, fontsize=12)
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            plt.colorbar(sm, ax=ax, orientation="vertical", label="Edge Weight")

        def build_graph(layers, label_prefix, input_labels=None):
            G = nx.DiGraph()
            x_offset = 0
            prev_labels = input_labels or [
                f"{label_prefix}0_{i}" for i in range(layers[0].get_weights()[0].shape[0])
            ]

            for idx, layer in enumerate(layers):
                weights = layer.get_weights()[0]
                label = f"{label_prefix}{idx+1}"
                threshold = threshold_factor * np.mean(np.abs(weights))
                top_k_edges = get_top_k_edges(weights, prev_labels, label, top_k)
                top_k_set = set((src, dst) for src, dst, _ in top_k_edges)

                prev_labels = add_layer_to_graph(
                    G, weights, prev_labels, label, x_offset, top_k_set, threshold
                )
                x_offset += 2

            return G

        encoder_layers = [
            l for l in self.model.encoder.layers if isinstance(l, tf.keras.layers.Dense)
        ]
        decoder_layers = [
            l for l in self.model.decoder.layers if isinstance(l, tf.keras.layers.Dense)
        ]

        if not encoder_layers and not decoder_layers:
            print("No Dense layers found in encoder or decoder.")
            return

        n_graphs = int(bool(encoder_layers)) + int(bool(decoder_layers))
        fig, axes = plt.subplots(1, n_graphs, figsize=(7 * n_graphs, 6), squeeze=False)

        col = 0
        if encoder_layers:
            input_labels = (
                self.y_labels
                if self.y_labels
                and len(self.y_labels) == encoder_layers[0].get_weights()[0].shape[0]
                else None
            )
            encoder_graph = build_graph(encoder_layers, "E", input_labels)
            draw_graph(encoder_graph, "Encoder", axes[0][col])
            col += 1

        if decoder_layers:
            decoder_graph = build_graph(decoder_layers, "D")
            draw_graph(decoder_graph, "Decoder", axes[0][col])

        fig.suptitle("Encoder & Decoder Dense Layer Graphs", fontsize=15)
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        if save_path:
            plt.savefig(save_path)
        plt.show()

        if encoder_layers:
            weights = encoder_layers[0].get_weights()[0]
            importances = np.abs(weights).mean(axis=1)
            sorted_idx = np.argsort(-importances)
            xticks = [
                (
                    self.y_labels[i]
                    if self.y_labels and len(self.y_labels) == weights.shape[0]
                    else f"Input_{i}"
                )
                for i in sorted_idx
            ]

            plt.figure(figsize=(10, 4))
            plt.bar(range(len(importances)), importances[sorted_idx], color="skyblue")
            plt.xticks(range(len(importances)), xticks, rotation=45, ha="right")
            plt.title("Feature Importances (Encoder Input Layer)", fontsize=13)
            plt.ylabel("Mean |Weight|")
            plt.tight_layout()
            plt.show()

    def predictor_analyzer(
        self,
        frac: float = None,
        cmap: str = "viridis",
        aspect: str = "auto",
        highlight: bool = True,
        **kwargs,
    ) -> None:
        """
        Analyze the model's predictions and visualize data.

        Parameters
        ----------
        frac : `float`, optional
            Fraction of data to use for analysis (default is `None`).
        cmap : `str`, optional
            The colormap for visualization (default is `"viridis"`).
        aspect : `str`, optional
            Aspect ratio for the visualization (default is `"auto"`).
        highlight : `bool`, optional
            Whether to highlight the maximum weights (default is `True`).
        **kwargs : `dict`, optional
            Additional keyword arguments for customization.

        Returns
        -------
        `DataFrame` : The statistical summary of the input data.
        """
        self._viz_weights(cmap=cmap, aspect=aspect, highlight=highlight, **kwargs)
        inputs = self.inputs.copy()
        inputs = self._prepare_inputs(inputs, frac)
        self.y_labels = kwargs.get("y_labels", None)
        encoded, reconstructed = self._encode_decode(inputs)
        self._visualize_data(inputs, reconstructed, cmap, aspect)
        self._prepare_data_for_analysis(inputs, reconstructed, encoded, self.y_labels)

        try:
            self._get_tsne_repr(inputs, frac)
            self._viz_tsne_repr(c=self.classification)

            self._viz_radviz(self.data, "class", "Radviz Visualization of Latent Space")
            self._viz_radviz(self.data_input, "class", "Radviz Visualization of Input Data")
        except ValueError:
            warnings.warn(
                "Some functions or processes will not be executed for regression problems.",
                UserWarning,
            )

        return self._statistics(self.data_input)

    def _prepare_inputs(self, inputs: np.ndarray, frac: float) -> np.ndarray:
        """
        Prepare the input data, possibly selecting a fraction of it.

        Parameters
        ----------
        inputs : `np.ndarray`
            The input data.
        frac : `float`
            Fraction of data to use.

        Returns
        -------
        `np.ndarray` : The prepared input data.
        """
        if frac:
            n = int(frac * self.inputs.shape[0])
            indexes = np.random.choice(np.arange(inputs.shape[0]), n, replace=False)
            inputs = inputs[indexes]
        inputs[np.isnan(inputs)] = 0.0
        return inputs

    def _encode_decode(self, inputs: np.ndarray) -> tuple:
        """
        Perform encoding and decoding on the input data.

        Parameters
        ----------
        inputs : `np.ndarray`
            The input data.

        Returns
        -------
        `tuple` : The encoded and reconstructed data.
        """
        try:
            mean, log_var = self.model.encoder(inputs)
            encoded = sampling(mean, log_var)
        except:
            encoded = self.model.encoder(inputs)
        reconstructed = self.model.decoder(encoded)
        return encoded, reconstructed

    def _visualize_data(
        self, inputs: np.ndarray, reconstructed: np.ndarray, cmap: str, aspect: str
    ) -> None:
        """
        Visualize the original data and the reconstructed data.

        Parameters
        ----------
        inputs : `np.ndarray`
            The input data.
        reconstructed : `np.ndarray`
            The reconstructed data.
        cmap : `str`
            The colormap for visualization.
        aspect : `str`
            Aspect ratio for the visualization.

        Returns
        -------
        `None`
        """
        ax = plt.subplot(1, 2, 1)
        plt.imshow(inputs, cmap=cmap, aspect=aspect)
        plt.colorbar()
        plt.title("Original Data")

        plt.subplot(1, 2, 2, sharex=ax, sharey=ax)
        plt.imshow(reconstructed, cmap=cmap, aspect=aspect)
        plt.colorbar()
        plt.title("Decoder Layer Reconstruction")
        plt.show()

    def _prepare_data_for_analysis(
        self,
        inputs: np.ndarray,
        reconstructed: np.ndarray,
        encoded: np.ndarray,
        y_labels: List[str],
    ) -> None:
        """
        Prepare data for statistical analysis.

        Parameters
        ----------
        inputs : `np.ndarray`
            The input data.
        reconstructed : `np.ndarray`
            The reconstructed data.
        encoded : `np.ndarray`
            The encoded data.
        y_labels : `List[str]`
            The labels of features.

        Returns
        -------
        `None`
        """
        self.classification = (
            self.model.classifier(tf.concat([reconstructed, encoded], axis=1))
            .numpy()
            .argmax(axis=1)
        )

        self.data = pd.DataFrame(encoded, columns=[f"Feature {i}" for i in range(encoded.shape[1])])
        self.data_input = pd.DataFrame(
            inputs,
            columns=(
                [f"Feature {i}" for i in range(inputs.shape[1])] if y_labels is None else y_labels
            ),
        )

        self.data["class"] = self.classification
        self.data_input["class"] = self.classification

    def _get_tsne_repr(self, inputs: np.ndarray = None, frac: float = None) -> None:
        """
        Perform t-SNE dimensionality reduction on the input data.

        Parameters
        ----------
        inputs : `np.ndarray`
            The input data.
        frac : `float`
            Fraction of data to use.

        Returns
        -------
        `None`
        """
        if inputs is None:
            inputs = self.inputs.copy()
            if frac:
                n = int(frac * self.inputs.shape[0])
                indexes = np.random.choice(np.arange(inputs.shape[0]), n, replace=False)
                inputs = inputs[indexes]
            inputs[np.isnan(inputs)] = 0.0
        self.latent_representations = inputs @ self.encoder_weights

        tsne = TSNE(n_components=2)
        self.reduced_data_tsne = tsne.fit_transform(self.latent_representations)

    def _viz_tsne_repr(self, **kwargs) -> None:
        """
        Visualize the t-SNE representation of the latent space.

        Parameters
        ----------
        **kwargs : `dict`
            Additional keyword arguments for customization.

        Returns
        -------
        `None`
        """
        c = kwargs.get("c", None)
        self.colors = (
            kwargs.get("colors", self.sorted_names[: len(np.unique(c))]) if c is not None else None
        )

        plt.scatter(
            self.reduced_data_tsne[:, 0],
            self.reduced_data_tsne[:, 1],
            cmap=matplotlib.colors.ListedColormap(self.colors) if c is not None else None,
            c=c,
        )

        if c is not None:
            cb = plt.colorbar()
            loc = np.arange(0, max(c), max(c) / float(len(self.colors)))
            cb.set_ticks(loc)
            cb.set_ticklabels(np.unique(c))

        plt.title("t-SNE Visualization of Latent Space")
        plt.xlabel("t-SNE 1")
        plt.ylabel("t-SNE 2")
        plt.show()

    def _viz_radviz(self, data: pd.DataFrame, color_column: str, title: str) -> None:
        """
        Visualize the data using RadViz.

        Parameters
        ----------
        data : `pd.DataFrame`
            The data to visualize.
        color_column : `str`
            The column to use for coloring.
        title : `str`
            The title of the plot.

        Returns
        -------
        `None`
        """
        data_normalized = data.copy(deep=True)
        data_normalized.iloc[:, :-1] = (
            2.0
            * (data_normalized.iloc[:, :-1] - data_normalized.iloc[:, :-1].min())
            / (data_normalized.iloc[:, :-1].max() - data_normalized.iloc[:, :-1].min())
            - 1
        )
        radviz(data_normalized, color_column, color=self.colors)
        plt.title(title)
        plt.show()

    def _viz_weights(
        self, cmap: str = "viridis", aspect: str = "auto", highlight: bool = True, **kwargs
    ) -> None:
        """
        Visualize the encoder layer weights of the model.

        Parameters
        ----------
        cmap : `str`, optional
            The colormap for visualization (default is `"viridis"`).
        aspect : `str`, optional
            Aspect ratio for the visualization (default is `"auto"`).
        highlight : `bool`, optional
            Whether to highlight the maximum weights (default is `True`).
        **kwargs : `dict`, optional
            Additional keyword arguments for customization.

        Returns
        -------
        `None`
        """
        title = kwargs.get("title", "Encoder Layer Weights (Dense Layer)")
        y_labels = kwargs.get("y_labels", None)
        cmap_highlight = kwargs.get("cmap_highlight", "Pastel1")
        highlight_mask = np.zeros_like(self.encoder_weights, dtype=bool)

        plt.imshow(self.encoder_weights, cmap=cmap, aspect=aspect)
        plt.colorbar()
        plt.title(title)
        if y_labels is not None:
            plt.yticks(ticks=np.arange(self.encoder_weights.shape[0]), labels=y_labels)
        if highlight:
            for i, j in enumerate(self.encoder_weights.argmax(axis=1)):
                highlight_mask[i, j] = True
            plt.imshow(
                np.ma.masked_where(~highlight_mask, self.encoder_weights),
                cmap=cmap_highlight,
                alpha=0.5,
                aspect=aspect,
            )
        plt.show()

    def _statistics(self, data_input: DataFrame) -> DataFrame:
        """
        Compute statistical summaries of the input data.

        Parameters
        ----------
        data_input : `DataFrame`
            The data to compute statistics for.

        Returns
        -------
        `DataFrame` : The statistical summary of the input data.
        """
        data = data_input.copy(deep=True)

        if not pd.api.types.is_string_dtype(data["class"]):
            data["class"] = data["class"].astype(str)

        data.ffill(inplace=True)
        grouped_data = data.groupby("class")

        numerical_stats = grouped_data.agg(["mean", "min", "max", "std", "median"])
        numerical_stats.columns = ["_".join(col).strip() for col in numerical_stats.columns.values]

        def get_mode(x):
            mode_series = x.mode()
            return mode_series.iloc[0] if not mode_series.empty else None

        mode_stats = grouped_data.apply(get_mode, include_groups=False)
        mode_stats.columns = [f"{col}_mode" for col in mode_stats.columns]
        combined_stats = pd.concat([numerical_stats, mode_stats], axis=1)

        return combined_stats.T


########################################################################################

if __name__ == "__main__":
    # Example usage
    import pandas as pd
    from sklearn.datasets import load_iris
    from sklearn.preprocessing import OneHotEncoder

    # Load the dataset
    iris = load_iris()

    # Convert to a DataFrame for easy exploration
    iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    iris_df["species"] = iris.target

    X = iris_df.drop(columns="species")
    y_labels = X.columns
    X = X.values
    y = iris_df["species"].values

    X = np.asarray(X).astype(np.float32)

    encoder = OneHotEncoder()
    y = encoder.fit_transform(y.reshape(-1, 1)).toarray()
    y = np.asarray(y).astype(np.float32)

    model = AutoClassifier(
        input_shape_parm=X.shape[1],
        num_classes=3,
        units=27,
        activation="tanh",
        num_layers=2,
        dropout=0.2,
    )
    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=[tf.keras.metrics.F1Score(threshold=0.5)],
    )
    model.fit(X, y, epochs=50, validation_split=0.2)

    insights = GetInsights(model, X)
    summary = insights.predictor_analyzer(frac=1.0, y_labels=y_labels)
    insights._get_tsne_repr()
    insights._viz_tsne_repr()
    insights._viz_tsne_repr(c=iris_df["species"])
    insights._viz_weights()
    print(summary)
