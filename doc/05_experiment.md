# V. [Experiment]
## 1. Experiement #1
- **Title**: Data collection and policy training in AIC simulation
- **Method**:
   1. Collect `**220 demonstration**` single task with only one env-obj
   2. Training 2 policies (with 50 demos and 200 demos) using `**Diffusion model**`
   3. Evaluation on set 20 demonstration (rate of success)
- **Data Preparation**:
   1. 220 demonstrations are splitted into training set and test set. The training set contains 200 episodes, the other 20 episodes are included in the test set.
   2. Two diffusion policies are trained respectedly with the first 50 episodes and the whole of the training set.
   3. The evaluation stage has two phases. The first phase is run two policies on the test set to check the matching between the predictions and the ground truth of the test set. In the second phase, the policies are run on the simulation to check the success rate.

- **Expected Outcome**: The model should reach 90% success. The end-effector only need to reach the desired position, it is not necessary to complete full task.
- **Results**:
