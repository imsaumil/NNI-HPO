# Importing required libraries
import sys
import json
from nni.experiment import Experiment

config_version = sys.argv[2]
# Mentioning the config list files
if config_version == 'config_1':
    with open('./search_space_1.json', 'r') as file:
        data = file.read()
elif config_version == 'config_2':
    with open('./search_space_2.json', 'r') as file:
        data = file.read()
else:
    print("ERROR: Try other config param !")

# Importing the custom search spaces
search_space = json.loads(data)

# Setting up the NNI experiment on local machine
experiment = Experiment('local')

# Conducting NNI evaluation in trail mode
experiment.config.trial_command = 'python3 model.py'
experiment.config.trial_code_directory = '.'

# Configuring the search space
experiment.config.search_space = search_space

# Configuring the tuning algorithm
tuner_name = sys.argv[1]  # => ('TPE', 'Evolution', 'SMAC')

experiment.config.tuner.name = tuner_name

tuner_version = sys.argv[3]
# Trying with the default tuner configs -> Minimum effort approach with default tuner params
if tuner_version == 'tuner_version_1':
    experiment.config.tuner.class_args['optimize_mode'] = 'maximize'

    # Additional class argument required for Evolution tuner
    if tuner_name == 'Evolution':
        experiment.config.tuner.class_args['population_size'] = 100

# Second config list approach
elif tuner_version == 'tuner_version_2':
    # Additional class argument required for Evolution tuner
    if tuner_name == 'TPE':
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

    if tuner_name == 'Evolution':
        experiment.config.tuner.class_args = {
            'optimize_mode': 'maximize',
            'population_size': 300
        }

    if tuner_name == 'SMAC':
        experiment.config.tuner.class_args = {
            'optimize_mode': 'maximize',
            'config_dedup ': True
            }

else:
    print("ERROR: Try other tuner config param !")


experiment.config.debug = True

# Setting a name for the experiment
experiment.config.experiment_name = f'NinjaTurtles - {tuner_name} {config_version} {tuner_version}'

# Setting up number of trials to run -> Sets of hyperparameters and trial concurrency
experiment.config.max_trial_number = 50  # Change to a higher number -> 50
experiment.config.trial_concurrency = 10

# Running the experiment on portal
experiment.run(8059)

# Stopping the experiment
experiment.stop()

# For still viewing the experiment
# nni.experiment.Experiment.view(port=8057)
