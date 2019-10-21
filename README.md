# BiTrade-Graphs : Modelling Bilateral Trade with Graph Neural Networks

[Kobby Panford-Quainoo][panford], [Michaël Defferrard][mdeff]

[panford]: https://panford.github.io/kobby
[mdeff]: http://deff.ch

This repository contains all materials that accompanies our paper on "Bilateral Trade Modelling with Graph Neural Networks".
Here we show how bilateral trade between countries can be framed as a problem of learning on graphs where we do classification of node (countries) into their various income levels (node classes).
We also show that the likeliness of any two countries to trade can be predicted (link prediction). 
Data used for this experiment was downloaded from the Comtrade website.

### Installation
1. Clone this repository.
   ```sh
   git https://github.com/panford/BiTrade-Graphs.git
   cd BiTrade-Graphs
   ```

2. Install dependencies.
Dependencies used in our experiments can be installed using either the requirements.txt or environments.yml files. Follow any of the steps that follows to set up the environment.
   
   ```sh
   pip install -r requirements.txt
   ```
 
   ```sh
   conda create -f environment.yml
   conda activate bitgraph_env
   ```
### Running codes
* cd into the Bitrade-Graph/code folder
 ```sh
   cd <path/BiTradeGraph/code>
  ```
* Preprocess data
 ```sh
   $ process_data.py 
  ```
  This will create a "preprocessed.npy" file in the default path "data/processed" or a specific path using the "--outdir" command 
* Node classification and Link Prediction
   The classifier code can then be run using the command
 ```sh
   $ run_classifier.py 
  ```
  Results will be saved in the default directory "results". In the same way, terminal command below will  
   ```sh
    $ run_linkpredictor.py
   ```
  

### Notebooks
Notebooks included show the steps followed from preprocessing of data to use on downstream tasks. Open [Jupyter Notebook](https://jupyter.org/):
```sh jupyter notebook ```
Navigate to "notebooks" folder. First run all cells in the preprocessing.ipynb to process data and then training_nb.ipynb 
## License
This project is licensed under the MIT License. The LICENSE.txt file has more details.

