# Ad-hoc-team

This is a deep reinforcement learning-based method that solve routing problem with the ad hoc team. We provide codes for three CO (combinatorial optimization) problems:
- Capacitated Vehicle Routing Problem (CVRP), with the min-max arrival time objective, used for truck relief delivery.
- Team Orienteering Problem (TOP), with the maximum collected prize objective, used for UAV data collection.
- Pickup and Delivery Problem (PDP), with the minimum average arrival time objective,  used for UAV medical delivery.

## Basic Usage

Before reading our code, we strongly recommend reading the code of [POMO](https://github.com/yd-kwon/POMO/tree/master/NEW_py_ver) which is base of our code. 

To train a model, run train_vrp.py/train_op.py/train_pdp.py.


'context_decoder' and 'random_choice' are two optional modules, the former means the information sharing process, and the later represents the multi-starting-member technique.

For the data, we provide the raw training data, which comes from the [dataset](https://figshare.com/articles/dataset/Enhanced_Dam_Failure_Loss_Estimation_Method_Using_Popula-tion_Heat_Map_and_Land_Use_Data_in_Water_Resources_Sector/25706562/1?file=45904947), and the related data processing methods. Besides, we provide the execution data for each scenario.

The trained models for all related routing problems under different module choices are shown in the result folders.

## Dependencies

* Python>=3.9
* NumPy
* SciPy
* [PyTorch](http://pytorch.org/)>=1.7
* tqdm
* tensorboard_logger
* Matplotlib (optional, only for plotting)
