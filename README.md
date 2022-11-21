# Hyper parameter tuning using NNI 
### Owners:
* Saumil Shah
* Jinam Shah


## Tuners Explored
* TPE tuner
* Evolution tuner
* SMAC tuner

## Search space configurations explored

* Search space configuration 1:
    ```json
    {"lr": {"_type": "loguniform", "_value": [0.0001, 0.1]},"dropout_rate": {"_type": "uniform", "_value": [0.1, 0.5]},"momentum": {"_type": "uniform", "_value": [0, 1]}}
    ```

* Search space configuration 2:
    ```json
    {"lr": {"_type": "loguniform", "_value": [0.0001, 0.1]},"dropout_rate": {"_type": "uniform", "_value": [0.1, 0.5]},"batch_size": {"_type": "choice", "_value": [16, 32, 64, 128]},"momentum": {"_type": "uniform", "_value": [0, 1]}
    ```

## Tuner configurations explored

For all the tuners, the first version of the tuners uses the default parameters. The details mentioned below are the second versions of the tuners.

* TPE tuner:
  ```python
  experiment.config.tuner.class_args = {
            'optimize_mode': 'maximize',
            'seed': 9999,
            'tpe_args': {
                'constant_liar_type': 'mean',
                'n_startup_jobs': 10,
                'n_ei_candidates': 20,
                'linear_forgetting': 10,
                'prior_weight': 0,
                'gamma': 0.6
            }
        }
    ```

* Evolution tuner:
  ```python
  experiment.config.tuner.class_args = {
            'optimize_mode': 'maximize',
            'population_size': 300
        }
    ```

* SMAC tuner:
    ```python
    experiment.config.tuner.class_args = {
            'optimize_mode': 'maximize',
            'config_dedup': True
    }
    ```


# Results

| Tuner used | Tuner config | Search space  | Drop out | Learning rate | Momentum | Batch size | Trial Number | Time | Final Accuracy |
| --- | --- | ---  | --- | --- | --- | --- | --- | --- | --- |
| TPE | v1 | 1 | 0.1801 | 0.0243 | 0.9887 |  - | 32 | 8m 41s | 0.9195 |
| TPE | v1 | 2 | 0.3604 | 0.0329 | 0.9598 | 16 | 30 | 23m 12s | 0.9198 |
| TPE | v2 | 1 | 0.1023 | 0.0582 | 0.9484 |  - | 45 | 9m 9s | 0.9161 |
| TPE | v2 | 2 | 0.3638 | 0.0043 | 0.9363 | 16 | 43 | 21m 20s | 0.9168 |
| SMAC | v1 | 1 | 0.3982 | 0.0662 | 0.5686 |  - | 29 | 8m 47s | 0.9097 |
| SMAC | v1 | 2 | 0.3463 | 0.0288 | 0.1491 | 16 | 24 | 23 m 38s | 0.9128 |
| SMAC | v2 | 1 | 0.1097 | 0.0647 | 0.8293 |  - | 4 | 8m 21 | 0.9125 |
| SMAC | v2 | 2 | 0.1805 | 0.0035 | 0.9614 | 16 | 18 | 21m 51s | 0.9147 |
| Evolution | v1 | 1 | 0.2777 | 0.052 | 0.8171 | -  | 29 | 8m 16s | 0.9143 |
| Evolution | v1 | 2 | 0.3138 | 0.0059 | 0.9748 | 16 | 12 | 22m 24s | 0.9191 |
| Evolution | v2 | 1 | 0.3747 | 0.04737 | 0.5623 |  - | 17 | 8m 47s | 0.9082 |
| Evolution | v2 | 2 | 0.2665 | 0.0467 | 0.7547 | 16 | 48 | 11m 41s | 0.9143 |

# Code execution

The code can be executed locally or on the bridges-2 cluster.
>  Note: For the bridges2 cluster, one would need to avail the cuda module 10.2 using `module load cuda/10.2` command before installing the dependencies.

Execution command:
* Local system: `./runner.sh`
* Bridges2 cluster: `sbatch runner.sh`
