# BiTrade-Graphs : Modelling Bilateral Trade with Graph Neural Networks

[Kobby Panford-Quainoo][panford], [Michaël Defferrard][mdeff]
[panford]: https://panford.github.io/kobby
[mdeff]: http://deff.ch

This repository contains all materials that accompanies our paper on "Bilateral Trade Modelling with Graph Neural Networks".
Here we show how bilateral trade between countries can be framed as a problem of learning on graphs where we do classification of node (countries) into their various income levels (node classes).
We also show that the likeliness of any two countries to trade can be predicted (link prediction). 
Data used for this experiment was downloaded from the Comtrade website.

#Installation
1. Clone this repository.
   ```sh
   git https://github.com/panford/BiTrade-Graphs.git
   cd BiTrade-Graphs
   ```

2. Install the dependencies.
   ```sh
   pip install -r requirements.txt
   ```
   
3. Play with the Jupyter notebooks.
   ```sh
   jupyter notebook
   ```
