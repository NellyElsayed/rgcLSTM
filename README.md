# rgcLSTM

reduced-gate convolutional LSTM model. This model is applied on PredNet which implemented by Lotter 2016.

This code is containing the implementation of the paper:

Reduced-Gate Convolutional LSTM Using Predictive Coding for Spatiotemporal Prediction

You can find the paper on arxiv:
https://arxiv.org/abs/1810.07251

## Description:
We introduced a novel reduced-gated convolutional LSTM (rgcLSTM) which has the same performance as convolutional LSTM (cLSTM) in RGB videos and outperforms the convolutional LSTM on gray-scale videos.


To test our model, we used the model introduced by Lotter et al. 2016 (PredNet). We modified the code of Lotter et al. 2016 (PredNet) by replacing the convolution LSTM (cLSTM) by our novel reduced-gated convolutional LSTM (rgcLSTM).


The original code of the Prednet using cLSTM is in the following link:
https://github.com/coxlab/prednet


As we changed the (cLSTM)  in the Prednet to our (rgcLSTM), we name our model as Pred-regLSTM to destinguish between the both models.


Please read the comments within the code to run it.

# How to run:

Our model requires the same files as the model implemented by Lotter et al. 2016 (PredNet) that you can find in the following link:
https://github.com/coxlab/prednet


## Except,

replace the following file by our file in this page:



## For Kitti Experiment
**Replace:**

kitti_train.py **by** kitti_train_rgcLSTM.py

prednet.py **by**  pred_rgcLSTM.py

keras.util.py  **by** keras.util.rgcLSTM

kitti_evaluate.py **by** kitti_evaluate_rgcLSTM.py

The rest is the same requirements and description as in the Original PredNet , which you can find in the following Link:

https://github.com/coxlab/prednet

# obtaining datasets

For Moving MNIST, you can download the dataset from:

http://www.cs.toronto.edu/~nitish/unsupervised_video/

For the KITTI dataset the description and installing is in Lotter implementation:

https://github.com/coxlab/prednet

You can use any other gray-scale video or RGB video. 

**Note** changing the dimension of the video requires to modify the width and the hight of the video in the code.






# New Files added:
## These files to help in replicating the MNIST experiment

**These are new file that are uploaded into this directory**

> numpy_to_hkl.py
This file is a straight forward converter .npy file into .hkl 
This file could be used to prepare the mnist dataset training, testing and validation sets. 
In this file you should adjust the size of each of the datasets individually as this file builds only one file per time.
It is very simple and easy to use. Please follow the comments within the code script.

> mnist_kitti_train_rgcLSTM.py
The training file for MNIST dataset. All paramentes are adjusted for the experiment

> mnist_pred_rgcLSTM.py
The Pred-rgcLSTM file that adjusted for MNIST dataset size. All paramentes are adjusted for the experiment 

> mnist_evaluate_rgcLSTM.py
The evaluate file of rgcLSTM model for MNIST dataset size. All paramentes are adjusted for the experiment.



# Note: 
You still need to replace 
keras.util.py  **by** keras.util.rgcLSTM
as mentioned above in the KITTI dataset experiment. 


# Reminder
Please if you will use our code, do not forget to **cite both** our paper and the original Prednet:

@article{elsayed2018reduced,

  title={Reduced-Gate Convolutional LSTM Using Predictive Coding for Spatiotemporal Prediction},
  
  author={Elsayed, Nelly and Maida, Anthony S and Bayoumi, Magdy},
  
  journal={arXiv preprint arXiv:1810.07251},
  
  year={2018}
  
}

**AND**

@article{lotter2016deep,

  title={Deep predictive coding networks for video prediction and unsupervised learning
  
  author={Lotter, William and Kreiman, Gabriel and Cox, David},
  
  journal={arXiv preprint arXiv:1605.08104},
  
  year={2016}
  
}
