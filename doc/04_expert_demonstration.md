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
## 3. Data Collection Methods
### 3.1 Teleoperation
1. VR controllers
- **Read**: VR_01/02/03
- **Questions**: Setup, Data Collection, Data synchronization, Latency, Filtering Methods? ,....
#### VR_01
- How to build an inexpensive teleoperation system that allows intuitive robotic manipulation and collection of high-quality demonstrations suitable for learning ?
- This paper built a system uses consumer-grade Virtual Reality (VR) devices to teleoperate a PR2 robot:
    * VR headset to perceive the environment through the robot's sensor space.
    * Motion-tracked VR controllers to control the robot
    * The setup ensures that human and robot share exactly the same observation and action space.

**Hardware**: 
1. Head-mounted display and two hand controllers with 6 DoF pose tracking at precision at 90 Hz
2. Primesense Carmine mounted on the robot end-effector provides color and depth images at 30 Hz
3. Teleoperation System is written in Unity
**Visual Interface**:
1. Use RGB-D camera -> no need to use stereo camearas.
**Control Interface**:
1. Vive hand controllers for controlling the robot's arms
2. Use the trigger button on the controller to signal the robot gripper -> *`this methods may not suitable for capture the analog action of the gripper because it can only fully close or open`*
3. Collect target pose of the gripper at 10 Hz
4. The robot arm is controlled at *`torque level`* by using low-level Jacobian-transpose
5. **!NOTICE: The pretrained policy may be test on the digital twin of the real robotic arm berfore implement on the system. This process is for safety**

------------------------------
2. 3D spacemouse
3. Smartphones
4. Puppeting devices
5. Exoskeletons


### 3.2 Hand-held Gripper
#### 3.1 Experience of using UMI to collect data
##### 3.1.1 Random initial pose is crucial
- It is essential to randomize the initial pose of the hand-held gripper, including its height and orientation.
- The initial position range of objects should be as extensive as possible.
##### 3.1.2 Select an environment with rich visual features
- UMI relies on SLAM for camera pose tracking, sufficient visual features-such as dark areas or blank walls- can lead to tracking failures.
- This problem can be addressed by using the visualization tool Pangolin to make sure that the env having enough features.
- By introducing more distractor objects or adding textures to surfaces, such as tabletops, visual features can be increased and served as a form of data augmentation -> `**the policy can learn to disregard irrelevant changes in the env**`
- Performing multiple mapping rounds and using batch SLAM processing can enhance the number of valid demonstrations
##### 3.1.3 Use appropriately sized manipulation objects
- Large objects obstructing the camera's view cause tracking failure -> opening drawers becoming difficult tasks to perform. This is one of the drawbacks of UMI method
- Integrating off-the-shelf pose tracking hardware (IPhone Pro or VIVE Ultimate Tracker) could improve UMI's performance.
##### 3.1.4 Additional tips
- Bahavior patterns and task completion times should be standardized -> minimize multimodal behavior in the dataset. (`**!QUESTION: How to standardize time and behavior patterns**`)
- Avoid non-manipulation objects (distractors) and other moving entities entering the camera's FoV.
- Apply slight force when closing the gripper to introduce minor deformation
- In UMI original paper, the number of demonstration is recommended by 200 for a single task in a fixed environment
## 4. References
[1] Xie, C., Wang, Z., Li, S., Chi, C., Song, S., & Levine, S. (2025). *Data scaling laws in imitation learning for robotic manipulation*. International Conference on Learning Representations (ICLR). https://doi.org/10.48550/arXiv.2410.18647

[2] Chi, C., Xu, Z., Pan, C., Cousineau, E., Burchfiel, B., Feng, S., Tedrake, R., & Song, S. (2024). *Universal manipulation interface: In-the-wild robot teaching without in-the-wild robots. Proceedings of Robotics: Science and Systems* (RSS).

[3] Zhang, T., McCarthy, Z., Jow, O., Lee, D., Chen, X., Goldberg, K., & Abbeel, P. (2018). *Deep imitation learning for complex manipulation tasks from virtual reality teleoperation*. 2018 IEEE International Conference on Robotics and Automation (ICRA), 5628–5635. https://doi.org/10.1109/ICRA.2018.8461249