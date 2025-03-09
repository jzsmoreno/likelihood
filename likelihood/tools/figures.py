import os
import warnings
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import AutoMinorLocator
from scipy import stats

plt.rcParams.update({"font.size": 14})


def act_pred(
    y_act: np.ndarray,
    y_pred: np.ndarray,
    name: str = "example",
    x_hist: bool = True,
    y_hist: bool = True,
    reg_line: bool = True,
    save_dir: Optional[str] = None,
) -> None:
    """
    Creates a scatter plot of actual vs predicted values along with histograms and a regression line.

    Parameters
    ----------
    y_act : `np.ndarray`
        The actual values (ground truth) as a 1D numpy array.
    y_pred : `np.ndarray`
        The predicted values as a 1D numpy array.
    name : `str`, optional
        The name for saving the plot. Default is "example".
    x_hist : `bool`, optional
        Whether to display the histogram for the actual values (y_act). Default is True.
    y_hist : `bool`, optional
        Whether to display the histogram for the predicted values (y_pred). Default is True.
    reg_line : `bool`, optional
        Whether to plot a regression line (best-fit line) in the scatter plot. Default is True.
    save_dir : `Optional[str]`, optional
        The directory to save the figure. If None, the figure will not be saved. Default is None.

    Returns
    -------
    `None` : The function doesn't return anything. It generates and optionally saves a plot.
    """

    y_pred, y_act = y_pred.flatten(), y_act.flatten()

    if not isinstance(y_act, np.ndarray) or not isinstance(y_pred, np.ndarray):
        raise ValueError("y_act and y_pred must be numpy arrays.")
    if y_act.shape != y_pred.shape:
        raise ValueError("y_act and y_pred must have the same shape.")

    mec = "#2F4F4F"
    mfc = "#C0C0C0"

    fig = plt.figure(figsize=(6, 6))

    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    bottom_h = left + width
    left_h = left + width + 0.05

    ax2 = fig.add_axes([left, bottom, width, height])
    ax2.tick_params(direction="in", length=7, top=True, right=True)
    ax2.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax2.yaxis.set_minor_locator(AutoMinorLocator(2))

    ax2.scatter(y_act, y_pred, color=mfc, edgecolor=mec, alpha=0.5, s=35, lw=1.2)
    ax2.plot(
        [y_act.min(), y_act.max()], [y_act.min(), y_act.max()], "k--", alpha=0.8, label="Ideal"
    )

    ax2.set_xlabel("Actual value")
    ax2.set_ylabel("Predicted value")
    ax2.set_xlim([y_act.min() * 1.05, y_act.max() * 1.05])
    ax2.set_ylim([y_act.min() * 1.05, y_act.max() * 1.05])

    ax1 = fig.add_axes([left, bottom_h, width, 0.15])
    ax1.hist(y_act, bins=31, density=True, color=mfc, edgecolor=mec, alpha=0.6)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_xlim(ax2.get_xlim())

    if x_hist:
        ax1.set_alpha(1.0)

    ax3 = fig.add_axes([left_h, bottom, 0.15, height])
    ax3.hist(
        y_pred, bins=31, density=True, color=mfc, edgecolor=mec, orientation="horizontal", alpha=0.6
    )
    ax3.set_xticks([])
    ax3.set_yticks([])
    ax3.set_ylim(ax2.get_ylim())

    if y_hist:
        ax3.set_alpha(1.0)

    if reg_line:
        polyfit = np.polyfit(y_act, y_pred, deg=1)
        reg_line_vals = np.poly1d(polyfit)(np.unique(y_act))
        ax2.plot(np.unique(y_act), reg_line_vals, "r-", label="Regression Line", alpha=0.8)

    ax2.legend(loc="upper left", framealpha=0.35, handlelength=1.5)

    plt.tight_layout()

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        fig_name = os.path.join(save_dir, f"{name}_act_pred.png")
        plt.savefig(fig_name, bbox_inches="tight", dpi=300)

    plt.show()
    plt.close(fig)


def residual(
    y_act: np.ndarray, y_pred: np.ndarray, name: str = "example", save_dir: str = None
) -> None:
    """
    Plots the residual errors between the actual and predicted values.

    This function generates a residual plot by calculating the difference between the
    actual values (y_act) and predicted values (y_pred). The plot shows the residuals
    (y_pred - y_act) against the actual values. Optionally, the plot can be saved to a file.

    Parameters
    ----------
    y_act : `np.ndarray`
        The actual values, typically the ground truth values.
    y_pred : `np.ndarray`
        The predicted values that are compared against the actual values.
    name : `str`, optional
        The name of the plot file (without extension) used when saving the plot. Default is "example".
    save_dir : `str`, optional
        The directory where the plot will be saved. If None, the plot is not saved. Default is None.

    Returns
    -------
    `None` : This function does not return any value. It generates and optionally saves a plot.

    Notes
    -----
    - The plot is shown with the residuals (y_pred - y_act) on the y-axis and the actual values (y_act)
      on the x-axis. The plot includes a horizontal line representing the ideal case where the residual
      is zero (i.e., perfect predictions).
    - The plot will be saved as a PNG image if a valid `save_dir` is provided.
    """

    mec = "#2F4F4F"
    mfc = "#C0C0C0"

    y_act = np.array(y_act)
    y_pred = np.array(y_pred)

    xmin = np.min([y_act]) * 0.9
    xmax = np.max([y_act]) / 0.9
    y_err = y_pred - y_act
    ymin = np.min([y_err]) * 0.9
    ymax = np.max([y_err]) / 0.9

    fig, ax = plt.subplots(figsize=(4, 4))

    ax.plot(y_act, y_err, "o", mec=mec, mfc=mfc, alpha=0.5, label=None, mew=1.2, ms=5.2)
    ax.plot([xmin, xmax], [0, 0], "k--", alpha=0.8, label="ideal")

    ax.set_ylabel("Residual error")
    ax.set_xlabel("Actual value")
    ax.legend(loc="lower right")

    minor_locator_x = AutoMinorLocator(2)
    minor_locator_y = AutoMinorLocator(2)
    ax.get_xaxis().set_minor_locator(minor_locator_x)
    ax.get_yaxis().set_minor_locator(minor_locator_y)

    ax.tick_params(right=True, top=True, direction="in", length=7)
    ax.tick_params(which="minor", right=True, top=True, direction="in", length=4)

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    if save_dir is not None:
        fig_name = f"{save_dir}/{name}_residual.png"
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(fig_name, bbox_inches="tight", dpi=300)

    plt.draw()
    plt.pause(0.001)

    plt.close()


def residual_hist(
    y_act: np.ndarray, y_pred: np.ndarray, name: str = "example", save_dir: Optional[str] = None
) -> None:
    """
    Generates a residual error histogram with kernel density estimate (KDE) for the given true and predicted values.
    Optionally saves the plot to a specified directory.

    Parameters
    ----------
    y_act : `np.ndarray`
        Array of true (actual) values.

    y_pred : `np.ndarray`
        Array of predicted values.

    name : `str`, optional, default="example"
        The name used for the saved plot filename.

    save_dir : `str`, optional, default=None
        Directory path to save the generated plot. If None, the plot is not saved.

    Returns
    --------
    `None` : This function generates and optionally saves a plot but does not return any value.

    Raises
    -------
    `UserWarning` : If the data has high correlation among variables, suggesting dimensionality reduction.
    """
    mec = "#2F4F4F"
    mfc = "#C0C0C0"
    y_pred, y_act = y_pred.flatten(), y_act.flatten()

    fig, ax = plt.subplots(figsize=(4, 4))
    y_err = y_pred - y_act
    x_range = np.linspace(min(y_err), max(y_err), 1000)

    try:
        kde_act = stats.gaussian_kde(y_err)
        ax.plot(x_range, kde_act(x_range), "-", lw=1.2, color="k", label="kde")
    except np.linalg.LinAlgError as e:
        warnings.warn(
            "The data has very high correlation among variables. Consider dimensionality reduction.",
            UserWarning,
        )

    ax.hist(y_err, color=mfc, bins=35, alpha=1, edgecolor=mec, density=True)

    ax.set_xlabel("Residual error")
    ax.set_ylabel("Relative frequency")
    plt.legend(loc=2, framealpha=0.35, handlelength=1.5)

    ax.tick_params(direction="in", length=7, top=True, right=True)

    minor_locator_x = AutoMinorLocator(2)
    minor_locator_y = AutoMinorLocator(2)
    ax.get_xaxis().set_minor_locator(minor_locator_x)
    ax.get_yaxis().set_minor_locator(minor_locator_y)
    plt.tick_params(which="minor", direction="in", length=4, right=True, top=True)

    if save_dir is not None:
        fig_name = f"{save_dir}/{name}_residual_hist.png"
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(fig_name, bbox_inches="tight", dpi=300)

    plt.draw()
    plt.pause(0.001)
    plt.close()


def loss_curve(
    x_data: np.ndarray,
    train_err: np.ndarray,
    val_err: np.ndarray,
    name: str = "example",
    save_dir: Optional[str] = None,
) -> None:
    """
    Plots the loss curve for both training and validation errors over epochs,
    and optionally saves the plot as an image.

    Parameters
    ----------
    x_data : `np.ndarray`
        Array of x-values (usually epochs) for the plot.
    train_err : `np.ndarray`
        Array of training error values.
    val_err : `np.ndarray`
        Array of validation error values.
    name : `str`, optional
        The name to use when saving the plot. Default is "example".
    save_dir : `Optional[str]`, optional
        Directory where the plot should be saved. If None, the plot is not saved. Default is None.

    Returns
    -------
    `None` : This function does not return any value. It generates and optionally saves a plot.
    """
    mec1 = "#2F4F4F"
    mfc1 = "#C0C0C0"
    mec2 = "maroon"
    mfc2 = "pink"

    fig, ax = plt.subplots(figsize=(4, 4))

    ax.plot(
        x_data,
        train_err,
        "-",
        color=mec1,
        marker="o",
        mec=mec1,
        mfc=mfc1,
        ms=4,
        alpha=0.5,
        label="train",
    )

    ax.plot(
        x_data,
        val_err,
        "--",
        color=mec2,
        marker="s",
        mec=mec2,
        mfc=mfc2,
        ms=4,
        alpha=0.5,
        label="validation",
    )

    max_val_err = max(val_err)
    ax.axhline(max_val_err, color="b", linestyle="--", alpha=0.3)

    ax.set_xlabel("Number of training epochs")
    ax.set_ylabel("Loss (Units)")
    ax.set_ylim(0, 2 * np.mean(val_err))

    ax.legend(loc=1, framealpha=0.35, handlelength=1.5)

    minor_locator_x = AutoMinorLocator(2)
    minor_locator_y = AutoMinorLocator(2)
    ax.get_xaxis().set_minor_locator(minor_locator_x)
    ax.get_yaxis().set_minor_locator(minor_locator_y)

    ax.tick_params(right=True, top=True, direction="in", length=7)
    ax.tick_params(which="minor", right=True, top=True, direction="in", length=4)

    if save_dir is not None:
        fig_name = f"{save_dir}/{name}_loss_curve.png"
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(fig_name, bbox_inches="tight", dpi=300)

    plt.draw()
    plt.pause(0.001)
    plt.close()
