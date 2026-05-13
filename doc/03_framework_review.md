# III. [General Framework Review]
## 1. Dataset
*   Data representation and storage
## 2. Policy Learning
### 2.1 ACT (Action Chunking with Transformers)
### 2.2 Diffusion Policy
#### 2.2.1 Policy Interface
- The Diffusion Policy can be designed to address two challenges consisting of `Hardware-specific latency` and `Embodiment-specific proprioception`. The solutions are proposed by UMI original paper.
##### 2.2.1.1 Observation latency matching
- The latency of each data sources have been measured
- At inference time, all observations are aligned with the highest latency
- Down-sample the RGB camera observations to the desired frequency.
- Use the capture timestamp of each image $`t_{obs}`$ to linearly interpolate gripper and robot proprioceptions.
##### 2.2.1.2 Action latency matching
- There is an execution latency in robotics system.
- To make sure the robots and grippers reaching the desired pose at desired time, the commands should be sent ahead of time to compensate the latency.
- The prediction comes after the last step of observation $`t_{obs}`$
- Due to the observation latency $`t_{input}-t_{obs}`$, policy inference latency $`t_{output}-t_{input}`$, and execution latency $`t_{act}-t_{output}`$ -> the first few actions are outdated.
- The solution is discarding the outdated actions and executing only the actions with the desired timestamp after $`t_{act}`$
### 2.3 VLA (Vision-Language-Action)
### 2.4 World Model
## 3. Inference
*   Deployment and runtime considerations
## 4. References
[1] Chi, C., Xu, Z., Pan, C., Cousineau, E., Burchfiel, B., Feng, S., Tedrake, R., & Song, S. (2024). Universal manipulation interface: In-the-wild robot teaching without in-the-wild robots. Proceedings of Robotics: Science and Systems (RSS).
