"""
Likelihood: Initialize the Package
=====================================

This is the entry point of the Likelihood package. It initializes all necessary modules and provides a central hub for accessing various tools and functions.

Main Modules:
- likelihood.main: Provides access to core functionality, including data preprocessing, model training, and analysis.
- likelihood.models: Offers pre-built models for AutoEncoder-based classification and regression tasks.
- likelihood.tools: Contains utility functions for data manipulation, normalization, and visualization.

By importing the main modules directly or accessing them through this central entry point (i.e., `from likelihood import *`), you can leverage the full range of Likelihood's capabilities to streamline your data analysis workflow.

To get started with Likelihood, simply import the desired modules and start exploring!
"""

from likelihood.main import *
from likelihood.models import *
from likelihood.tools import *
