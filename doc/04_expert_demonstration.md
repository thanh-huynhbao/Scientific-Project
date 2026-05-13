# IV. [Expert Demonstration]
## 1. Data Quality
*   Precision and noise in expert trajectories
## 2. Data Quantity

### 2.1 How to select the number of environments and objects:
- Collecting multiple objects per env might improve performance.
- The recommendatation for number of envs and objs is collect data in as many diverse envs as possible with only one unique object in each environment
### 2.2 How to select the number of demonstrations:
- Increasing the number of demonstrations beyond a certain point yields minimal benefits.
- When the number of **env-obj** pairs is smaller, fewer total demonstrations are needed to reach saturation (graph going down).
- The recommendation for tasks with the similar difficulty to pour water, etc is `**50 demonstrations per env-obj**`
### 2.3 Experience of using UMI to collect data
##### 2.3.1 Random initial pose is crucial
- It is essential to randomize the initial pose of the hand-held gripper, including its height and orientation.
- The initial position range of objects should be as extensive as possible.
##### 2.3.2 Select an environment with rich visual features
- UMI relies on SLAM for camera pose tracking, sufficient visual features-such as dark areas or blank walls- can lead to tracking failures.
- This problem can be addressed by using the visualization tool Pangolin to make sure that the env having enough features.
- By introducing more distractor objects or adding textures to surfaces, such as tabletops, visual features can be increased and served as a form of data augmentation -> `**the policy can learn to disregard irrelevant changes in the env**`
- Performing multiple mapping rounds and using batch SLAM processing can enhance the number of valid demonstrations
##### 2.3.3 Use appropriately sized manipulation objects
- Large objects obstructing the camera's view cause tracking failure -> opening drawers becoming difficult tasks to perform. This is one of the drawbacks of UMI method
- Integrating off-the-shelf pose tracking hardware (IPhone Pro or VIVE Ultimate Tracker) could improve UMI's performance.
##### 2.3.4 Additional tips
- Bahavior patterns and task completion times should be standardized -> minimize multimodal behavior in the dataset. (`**!QUESTION: How to standardize time and behavior patterns**`)
- Avoid non-manipulation objects (distractors) and other moving entities entering the camera's FoV.
- Apply slight force when closing the gripper to introduce minor deformation
## 3. Data Collection Methods
*   **3.1 Teleoperation**
*   **3.2 Hand-held Gripper**
## 4. References
[1] Xie, C., Wang, Z., Li, S., Chi, C., Song, S., & Levine, S. (2025). *Data scaling laws in imitation learning for robotic manipulation*. International Conference on Learning Representations (ICLR). https://doi.org/10.48550/arXiv.2410.18647
