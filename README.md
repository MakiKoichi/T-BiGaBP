## A From-Scratch Approach to Deep Unfolding-Aided Bilinear Gaussian Belief Propagation for Correlated Large MIMO

Repository for “A From-Scratch Approach to Deep Unfolding-Aided Bilinear Gaussian Belief Propagation for Correlated Large MIMO”

Koichi Maki

Electrical Engineering Program, Graduate School of Science and Technology, Meiji University

News:

- 2023/05/6 : MATLAB source code and the theoretical equations for backpropagation have been provided.

If you have any questions or concerns, please feel free to contact the author directly.

Email : ce221065 at meiji.ac.jp

I apologize for any inconvenience, as I am still learning how to use GitHub. Due to my unfamiliarity with the platform, I may not be able to respond effectively to issues raised for some time. I appreciate your understanding.

------

#### How to use MATLAB source code

To use the provided MATLAB source code, follow these steps:

1. Add the directory “MATLAB source code/Add this directory to your MATLAB path” to your MATLAB path.
2. Run “Execute_DeepUnfolding_of_BiGaBP_QPSK.m” to start the deep unfolding process.

Upon completion of each mini-batch, the updated parameters will be stored in a .mat file named “ExpoDecay_epoch＿*(current epoch)*＿minibatch＿*(current minibatch)*＿batchsize＿*(batch size)*.mat” , and the progress of the loss function will be saved in “Losslist.mat”. A graph showing the temporary changes in the loss function will also be  displayed. If the learning process is interrupted due to any issues, you can simply rerun "Execute_DeepUnfolding_of_BiGaBP_QPSK.m" to resume  learning from where you left off, without any additional steps.

------
