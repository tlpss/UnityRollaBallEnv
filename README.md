# RollABallEnv
Unity3D Environment for Goal-Based RL with Randomization. 

## Environment Features 

## Resources
https://github.com/Unity-Technologies/ml-agents/tree/release_12/docs
environment heavily inspired by tutorial on creating new environments.

## Configuration
ML-Agents Release 12, Unity 2020.2 
- conda environment available - `conda env create -f environment.yml` with correct versions of mlagents and mlagents_env
- Unity Project packages contain correct version (1.17.2) of the ML-Agents package


## Training
### Training with ML-agents
- provide a configuration file (see docs for how to) in the `config` folder (see example file)
- activate the environment
- from the root folder: `mlagents-learn config/rollaball_ppo.yaml --run-id=RollerBalltest --force` )
- when asked for in the terminal, press play in unity with the TrainingScene opened
- a results folder will be created and the network will be saved here
- visualise training on tensorboard: `tensorboard --logdir results --port 6006`

NB. to train with a prebuilt environment: build the env first and then add `--env= <relativepath-w/o-extension-to-build-file>` to the command
### Training with Python



## Inference
### Heuristic
The heuristic is defined in the `RollerAgent.css` script and uses the keyboard arrows to control the Agent. To use the heuristic, set the Behavior Type to "Heuristic Only" in the Behavior Parameters. Press.
### Trained Model
Trained models can be used for inference (using the ONNX standard for storing them). Trained models should be moved to the Assets/Models folder from where they can be selected by the Model field in the Behavior parameters. Don't forget to put behavior type to default to avoid using the heuristic. Press Play and watch how the agent behaves under the given policy. 