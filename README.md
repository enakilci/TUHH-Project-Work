# APPLYİNG DEEP LEARNİNG WİTH LSTM ON PX4 DRONE PARAMETERS

In this project, LSTM algorithm has been applied on the PX4 drone parameters that has been obtained from the Gazebo simulation environment.

Simulation provides a ULog file containing the changes of the drome parameters over time during the simulation. 
In order to make use of the parameters that are coded in the ULog file, a python script called ulog2csv.py that converts ULog files to csv files has been written. This script is located in the "pre-processing" folder. Necessary explanations to understand the way the script operates have been included in the script

**In the PX4Generator.py, these csv files are processed and trained through LSTM layers. As a result of the training session, desired val_loss and val_acc (appr. 0.94) values have been achieved.(please refer to ModelCheckpoints)**

The drone parameters that have trained in this project have been obtained from only a single simulation. 
Considering the fact that it is more convenient to train with more parameters, the script has been written to be adjustable to work 
with the parameters from several simulations. However, the expansion of the dataset by using parameters from several different simulation and 
combining them together has not been put into practise in our training due to GPU limitations I faced with laptop. 
I have tried to train the dataset with the GPU of the TUHH server. 
However, the virtual environment I was working in doesn't have the necessary GPU configurations to use cuDNNLSTM layer of Keras. 

Though the number of parameters that have been used in the training are indeed large enough to decide whether our algorithm is working properly, 
I have also trained the neural network with even larger dataset to challenge our learning algorithm and obtain similar results (appr. 0.94 val_acc)). 
I will add the result of this training to the repository as well. 

*P.S.: All the necessary explanations has been included in the corresponding scripts.*

