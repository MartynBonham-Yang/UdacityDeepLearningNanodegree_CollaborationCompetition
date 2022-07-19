# PROJECT: COLLABORATION & COMPETITION - REPORT 

------

AIM OF PROJECT: TRAIN A PAIR OF AGENTS TO PLAY TABLE TENNIS. AN AGENT RECEIVES A REWARD OF +0.1 WHEN HITTING THE BALL OVER THE NET AND A REWARD OF -0.1 IF IT ALLOWS THE BALL TO FALL. GOAL IS TO TRAIN AGENTS TO ACHIEVE AN AVERAGE SCORE OF +0.5 OVER 100 CONSECUTIVE EPISODES.

------

# First Actions 

------

Let us review the action & state spaces. The state space consists of 8 variables corresponding to the position & velocity of both agent-rackets and of the ball, which gives us an 8 x 8 matrix for observations. Each agent receives an observation. The action space consists of 2 separate continuous actions corresponding to movements forwards or backwards for each agent-racket. 

To begin with, we will train 2 random action agents, let's call them Abigail and Andrew, and let them play table tennis. After 100 episodes, their performance looks like this: 

`Total score (averaged over agents) this episode: -0.004999999888241291`

Oh dear. This is not very good. We will obviously have to make some changes. 

Unlike the previous project, in this project Udacity has provided us with a lot of code - a *lot* - which we can build upon, but we are given less advice on what to change. They recommend using a Multi-Agent DDPG implementation, and provide the code necessary to do this, and so we shall begin with that. MADDPG is an algorithm which does not train different actors separately but trains them using all independent observations. As our agents will each receive independent observations, this will in essence allow them to be trained using information that they both "know", whille retaining the ability to act independently. 

We can combine this with the Actor-Critic method outlined in Project 2, using a `ddpg_agent.py` (actually should be *ma*ddpg) file which calls to an `Actor()` and `Critic()` which we create in `model.py`. We can make further modifications that we have used in previous projects. For example, the Ornstein-Uhlenbeck process, the replay buffer & experience replay, and gradient clipping from project 2 all should be usable for this projects, and I have no reason *not* to use them. For this reason, to begin with we will copy the `model.py` file from my 2nd project. 

-------

# Initial run 





 
