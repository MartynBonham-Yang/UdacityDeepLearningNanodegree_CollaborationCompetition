# PROJECT: COLLABORATION & COMPETITION - REPORT 

------

AIM OF PROJECT: TRAIN A PAIR OF AGENTS TO PLAY TABLE TENNIS. AN AGENT RECEIVES A REWARD OF +0.1 WHEN HITTING THE BALL OVER THE NET AND A REWARD OF -0.1 IF IT ALLOWS THE BALL TO FALL. GOAL IS TO TRAIN AGENTS TO ACHIEVE AN AVERAGE SCORE OF +0.5 OVER 100 CONSECUTIVE EPISODES.

------

# First Actions 

------

Let us review the action & state spaces. The state space consists of 8 variables corresponding to the position & velocity of both agent-rackets and of the ball, which gives us an 8 x 8 matrix for observations. Each agent receives an observation. The action space consists of 2 separate continuous actions corresponding to movements forwards or backwards for each agent-racket. 

To begin with, we will train 2 random action agents, let's call them Abigail and Andrew, and let them play table tennis. We do this by simply copying-and-pasting the Udacity-provided "take random actions in the environment" code. After 100 episodes, their performance looks like this: 

`Episode 100:     Average Score: 0.004`

Oh dear. This is not very good. However, as this is random, further episodes will not lead to greater performance. We will obviously have to make some changes. It is however surprising that they managed to achieve even a small positive score as it suggests some fluke episodes where the pair of agents kept the ball going for more than 2 hits. 

### Explanation of learning algorithm

Unlike the previous project, in this project Udacity has provided us with a lot of code - a *lot* - which we can build upon, but we are given less advice on what to change. They recommend using a **Multi-Agent DDPG** implementation, and provide the code necessary to do this, and so we shall begin with that. MADDPG is an algorithm which does not train different actors separately but trains them using all independent observations. As our agents will each receive independent observations, this will in essence allow them to be trained using information that they both "know", whille retaining the ability to act independently. 

We can combine this with the Actor-Critic method outlined in Project 2, using a `ddpg_agent.py` (actually should be *ma*ddpg) file which calls to an `Actor()` and `Critic()` which we create in `model.py`. We can make further modifications that we have used in previous projects. For example, the Ornstein-Uhlenbeck process, the replay buffer & experience replay, and gradient clipping from project 2 all should be usable for this projects, and I have no reason *not* to use them. For this reason, to begin with we will copy the `model.py` file from my 2nd project. 

In order to get this code to run and fulfil the requirements of the project, we need to have 2 Agents. However, our existing code uses only 1 `Agent` class, and therefore we need a convenient way to create a 2nd Agent with minimal changes to the code. We can do this by creating a new Class called a `MADDPG_DualAgent`, which will create 2 agents for us, and will inherit attributes from `Agent` (the Udacity-provided code includes a `super()` line allowing this to be done more easily). We will also use a shared experience buffer, which was recommended in the previous project for the multi-agent variant (but I completed that project as a single-agent submission, therefore I will try it here).

-------

# Initial run 

100 episodes is almost certainly too short. As such I increased the number of episodes that the agents can train for to 10000. This made the process much slower (of course) but gave room for our gradient clipping and replay buffer to have more of an impact. After 10000 episodes of my initial run, we got the following:

`Episode 10000:     Average Score: 0.156`

Well, this was not as bad as it could have been. Including the replay buffer and gradient clipping seems to have worked in our favour. However, we can see that 0.156 is still very far short of the 0.5 which we need to obtain. 

------

# Making changes

We need to consider what changes could be made to make our models learn faster (so the process does not take many thousands of episodes) and to help with convergence (so that our models do not spend thousands of episodes fluctuating around). 2 methods which are recommended in the Udacity forums are batch normalisation (which I have used before, in my previous project) and a modification of the noise process, to remove noist after a certain length of time holding it constant. This is a very clever idea which I had not thought up myself and so I must credit the Udacity forums with providing inspiration for this choice. This is done by adding a variable called `t_stop_noise` to the code. I chose to set this at a value of 20000, which (given that each episode includes a few dozen timesteps) works out at about 500 episodes having noise before the noise is turned off. 

Things which I did not edit include the values of tau (soft target update parameter) and gamma (discount factor) from their "default" values of `1e-03` and `0.99`. These are values used by Udacity in the previous project and in their sample code for that project. I saw no reason to modify these. I also left `batch_size` at 128 and the `buffer_size` at 10000, just as I had used for my previous project. I set `learn_every` to a value of 10. All of these values are available for change if this run does not work out! 

## Batch normalisation 

Batch normalisation is a method that helped enormously in my previous project. I do not understand precisely why it made such a difference but I felt that it may help with this project too, and so I added a batch normalisation layer to my `Actor()` and `Critic()` neural networks. I chose to use only one layer, rather than 3 layers which I had tried last project, because in that project adding 3 layers did not improve performance but instead slowed down the model and gave no better results. 

## Result 

`Episode 10000: 	Average Score over past episodes: 0.225`

------
 
Unfortunately, our initial changes do not seem to have been enough. We therefore increased the learning frequency - reducing the parameter from 10 to 2 - and increased the minibatch size of samples which we draw from the buffer, from 128 to 256. Also, I made the decision to reduce the weighting factor applied to the noise - in my previous run there seemed to be a "crash" after about 1200 episodes, from which the model did not recover.

Following these changes, we obtained: 

`Environment solved in 9298 episodes!	Average Score: 0.510`

Fantastic! We have solved the environment!

-------

## Analysis of performance & possible improvements 

I attach here a plot of this models' output: 

![PLOT](https://user-images.githubusercontent.com/57990075/183413456-9f28598a-cdd8-487b-a213-4a6f4a49c8e0.png)

### Analysis 

I feel like my model performed about as well as I expected - I have come to see throughout this Nanodegree that my models take a very long time to reach the required threshold, and so when I initially developed this model I knew that I ought to drastically revise upwards the number of episodes that I should expect my model to need. As such I decided that 10000 episodes was a good starting value, leaving a lot of room for slow performance, rather than 2000 which I had initially intended to use; this was a good choice as my model in the end required 9298 episodes to complete! Training using the GPU was fast, each run taking around 50 minutes to complete. This was faster than I had expected, and I had prepared for perhaps 90 minutes to be required. This drastically improved the speed at which I could go through the hyperparameter choice process!

For the successful model, we can see that the model does not go through an ever-increasing performance curve. Starting progress is very very slow, and indeed watching the outputs for every episode showed that many early episodes had a zero or even negative score result (this can not be seen in the output in my notebook as printing every single episode result would lead to a very large file). My results are characterised by a very long period of low performance - see for example how the score does not reach 0.1 within the first 4700 episodes - and then a very slow but unstable increase up to around a score of 0.2, followed by a relatively very short increase up to the target score of 0.5. To first reach a score of 0.2 took more than 6000 episodes, but the final 500 episodes saw an increase from 0.21 up to 0.5. 

### Explanation of model architecture

For this project, I used the same sort of neural network as was used for the previous project. That is, I used 3 fully-connected layers with ReLu activation and a single Batch Normalisation layer for both the `Actor()` and `Critic()` networks. I used an input dimension of 256; I had intended to change this if it ever became necessary but as my model reached a successful conclusion before I got to this point I did not modify these values. 256 was chosen because it was the value I had intended to use last project (but in that project I reduced to 128). With the random seed value, I initially thought to use the date the same as I have done in previous work (that is, `seed = 20220808` or something like this) but I overlooked this and ran my code without modifying the value from 1. It did not have much effect on my model performance of course! 

The Actor and Critic learning rates remain the same as I used for my previous project, with values of $10e^-04$ and $10e^-03$ respectively. I did not need to modify these.  As mentioned above I also did not modify the discount rate, instead choosing to keep the Udacity-recommended default value. 

### Possible future improvements 

There is a lot of scope for improvement in the speed of my model training. In particular I feel that there would be great promise in looking into stabilisation - my model performance was very erratic. Other pathways for future investigation include looking into new model types; a paper I read by Xihuai Wang, Zhicheng Zhang, and Weinan Zhang of Shanghai Jiaotong University - available on arxiv.org [here](https://arxiv.org/abs/2203.10603) - suggests the tantalising possibility of using my now-successfully trained model as an appendage itself to a new model, where the existing model that performs "okay" can be used to provide a baseline. The new model can then spend less time on things which the old model has already ruled out. If we could do such a thing with a large collection of trained models then I believe that a very high level of performance could be achieved in a very short period of time.  






