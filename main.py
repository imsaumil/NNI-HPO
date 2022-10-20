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
    print("Try other config param !")

# Importing the custom search spaces
search_space = json.loads(data)

# Setting up the NNI experiment on local machine
experiment = Experiment('local')

# Conducting NNI evaluation in trail mode
experiment.config.trial_command = 'python model.py'
experiment.config.trial_code_directory = '.'

# Configuring the search space
experiment.config.search_space = search_space

# Configuring the tuning algorithm
tuner_name = sys.argv[1]  # => ('TPE', 'Evolution', 'SMAC')

experiment.config.tuner.name = tuner_name
experiment.config.tuner.class_args['optimize_mode'] = 'maximize'

# Additional class argument required for Evolution tuner
if tuner_name == 'Evolution':
    experiment.config.tuner.class_args['population'] = 100

experiment.config.debug = True

# Setting a name for the experiment
experiment.config.experiment_name = f'Ninja Turtles {tuner_name} {config_version}'

# Setting up number of trials to run -> Sets of hyperparameters and trial concurrency
experiment.config.max_trial_number = 50  # Change to a higher number -> 50
experiment.config.trial_concurrency = 10

# Running the experiment on portal
experiment.run(8059)

# Stopping the experiment
experiment.stop()

# For still viewing the experiment
# nni.experiment.Experiment.view(port=8057)
