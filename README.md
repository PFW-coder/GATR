# GATR

This is a deep reinforcement learning-based method that solves routing problems with the ad hoc team. We provide codes for three CO (combinatorial optimization) problems:
- Capacitated Vehicle Routing Problem (CVRP), with the min-max arrival time objective, used for relief transportation. Team parameters include team size, member speed, and member capacity.
- Team Orienteering Problem (TOP), with the maximum collected prize objective, used for disaster data collection. Team parameters include team size, member speed, and member endurance.
- Pickup and Delivery Problem (PDP), with the minimum average arrival time objective, used for medical delivery. Team parameters include team size, member speed, and member endurance.

## Basic Usage

Before reading our code, we recommend reading the code of [POMO](https://github.com/yd-kwon/POMO/tree/master/NEW_py_ver), which is the basis of our code. 

- To train a model, run train_vrp.py/train_op.py/train_pdp.py.
  
In the train_vrp.py/train_op.py/train_pdp.py files, environment, model, and training parameters can be modified. The parameters of environments include the scales of problems and teams. If you want to adjust specific team settings, please modify CVRPProblemDef.py/OPProblemDef.py/PDPProblemDef.py. When setting model parameters, 'context_decoder' and 'random_choice' are two optional modules; the former represents the adaptive information sharing process, and the latter represents the multi-starting-member technique.

- To test a model, run test_vrp.py/test_op.py/test_pdp.py.
  
In the test_vrp.py/test_op.py/test_pdp.py files, environment, model, and test parameters can be modified. You can use 'test_data_load' to decide whether to test existing file data or randomly generated data. 

- To test multiple models, run test_all.py. It can help you test all saved models and record results.

## Data

- For the data used in CVRP and TOP, we provide the raw training data, which comes from the [dataset](https://figshare.com/articles/dataset/Enhanced_Dam_Failure_Loss_Estimation_Method_Using_Popula-tion_Heat_Map_and_Land_Use_Data_in_Water_Resources_Sector/25706562/1?file=45904947), and the processed training data is saved in training_data.csv.
  
- For the data used in PDP, the coordinates come from the article ["Cooperative Learning-Based Joint UAV and Human Courier Scheduling for Emergency Medical Delivery Service"](https://ieeexplore.ieee.org/abstract/document/10745907), and the processed training data is saved in training_data.csv.

- For the execution data, they are all saved in data folders for various problems.


## Trained models

The trained models for all related routing problems under different module choices are shown in the result folders. They can be directly used for decision-making or as pre-trained models.

## Dependencies

* Python>=3.9
* NumPy
* SciPy
* [PyTorch](http://pytorch.org/)>=1.7
* tqdm
* tensorboard_logger
* Matplotlib (optional, only for plotting)
