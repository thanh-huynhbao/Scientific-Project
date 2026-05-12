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
*   **3.1 Teleoperation**
*   **3.2 Hand-held Gripper**
## 4. References
[1] Xie, C., Wang, Z., Li, S., Chi, C., Song, S., & Levine, S. (2025). *Data scaling laws in imitation learning for robotic manipulation*. International Conference on Learning Representations (ICLR). https://doi.org/10.48550/arXiv.2410.18647
