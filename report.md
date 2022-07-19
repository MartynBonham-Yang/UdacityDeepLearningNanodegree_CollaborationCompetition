# PROJECT: COLLABORATION & COMPETITION - REPORT 

------

AIM OF PROJECT: TRAIN A PAIR OF AGENTS TO PLAY TABLE TENNIS. AN AGENT RECEIVES A REWARD OF +0.1 WHEN HITTING THE BALL OVER THE NET AND A REWARD OF -0.1 IF IT ALLOWS THE BALL TO FALL. GOAL IS TO TRAIN AGENTS TO ACHIEVE AN AVERAGE SCORE OF +0.5 OVER 100 CONSECUTIVE EPISODES.

------

# First Actions 

------

Let us review the action & state spaces. The state space consists of 8 variables corresponding to the position & velocity of both agent-rackets and of the ball. Each agent receives an observation. The action space consists of 2 separate continuous actions corresponding to movements forwards or backwards for each agent-racket. 

Unlike the previous project, in this project Udacity has provided us with a lot of code - a *lot* - which we can build upon, but we are given less advice on what to change. They recommend using a Multi-Agent DDPG implementation, and provide the code necessary to do this, and so we shall begin with that. 
