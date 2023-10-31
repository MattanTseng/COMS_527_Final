# COMS_527_Final
COMS_527_Final_Project

**Objective**
	The objective of the project is to take an existing RL agent for the T-Rex Chrome Dino game and reduce the necessary training time without reducing the performance of the agent. 

 **Implementation Outline**
 
**Language:**
- Python
  
**Libraries:** 
- Pytorch [2]
- Ray RLlib [3]
- Selenium [4]
- OpenCV [5]

**RL Model:**
- Deep Q-Learning

**Reference Repositories and Sources:**
- https://github.com/aome510/chrome-dino-game-rl.git
- https://github.com/jeffasante/chrome-dino-RL.git
- https://github.com/e3oroush/dino_run_rl_pytorch.git

**Agent Success Metric:**
- Score agent achieves during game iteration
- Reward given to agent during game iteration

**Objectives:**
- Match the success of existing agents
- Reduce the training time required to reach the same level of success

**Risks:**
- Implementing an RL agent poses a number of challenges with regards to being able to get the agent to converge
- The complexity of an RL agent may obstruct the implementation of parallelization strategies
