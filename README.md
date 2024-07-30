# Learning Cluster Causal Graphs using Markov Chain Monte Carlo

### Install dependencies

This code has been tested with [Python 3.12.4](https://docs.python.org/3.12/index.html). We recommend using [Conda](https://docs.anaconda.com/free/miniconda/) to manage your Python environment for this code.

Install dependencies using the following command:

```sh
pip install -r requirements.txt
```

### Running the sampler

Training and evaluation can be done using the `run` bash script.

```sh
chmod u+x ./run
./run
```

#### Options

Edit `run` to modify experiment parameters

|Option|Description|Type|Default|
|-|-|-|-|
|N_DATA_SAMPLES|Number of random synthetic data samples generated for training|`int`|1000|
|N_MCMC_SAMPLES|Number of C-DAG samples generated in one EM iteration|`int`|1000|
|N_MCMC_WARMUP|Number of warmup samples during one EM iteration|`int`|250|
|MAX_EM_ITERS|Number of EM steps|`int`|5|
|MAX_MLE_ITERS|Number of optimizer iterations for parameter optimization during one EM step|`int`|500|
