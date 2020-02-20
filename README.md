# Bilateral Trade Modelling with Graph Neural Networks

[Kobby Panford-Quainoo][panford], [Michaël Defferrard][mdeff]

[panford]: https://panford.github.io
[mdeff]: https://deff.ch

This repository contains all materials that accompanies our [paper].
Here we show how bilateral trade between countries can be framed as a problem of learning on graphs where we do classification of node (countries) into their various income levels (node classes).
We also show that the likeliness of any two countries to trade can be predicted (link prediction).
The data for our experiments were downloaded from <https://comtrade.un.org>.

[paper]: https://doi.org/10.13140/RG.2.2.16746.47047

## Installation

1. Clone this repository.
   ```sh
   $ git clone https://github.com/panford/BiTrade-Graphs.git
   $ cd BiTrade-Graphs
   ```

1. Install dependencies.
   Dependencies can be installed using either the `requirements.txt` or `environments.yml` files.
   Follow any of the steps that follows to set up the environment.

   ```sh
   $ pip install -r requirements.txt
   ```

   ```sh
   $ conda create -f environment.yml
   $ conda activate bitgraph_env
   ```

   Check out the [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html) installation guide for hints on how to set up [PyTorch](https://pytorch.org) and [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/index.html) with the right version of cuda.

## Running our experiments

1. Enter the `Bitrade-Graph/code` folder.
   ```sh
   $ cd /path/to/BiTradeGraph/code
   ```

1. Preprocess the data.
   ```sh
   $ python process_data.py
   ```
   This will create a `preprocessed.npy` file in the `data/processed` folder (or a path specified by `--outdir`).

1. Run code for node classification and link prediction.
   ```sh
   $ python run_classifier.py
   $ python run_linkpredictor.py
   ```
   Results will be saved in the `results` folder.

## Notebooks

Notebooks are included to show the followed steps from data preprocessing to their use in downstream tasks.

1. Start [jupyter](https://jupyter.org).
   ```sh
   $ jupyter notebook
   ```

1. Navigate to the `notebooks` folder.

1. First run all cells in the `preprocessing.ipynb` notebook to process data.
   Then `training_nb.ipynb`.

## License

This project is licensed under the [MIT License](LICENSE.txt).
