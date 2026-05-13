# V. [Experiment](./05_experiment.md)
## 1. Experiement #1
- **Title**: Data collection and policy training in AIC simulation
- **Method**:
   1. Collect 220 demonstration single task with only one env-obj
   2. Training 2 policys (with 50 demos and 200 demos) using Diffusion model
   3. Evaluation on set 20 demonstration (rate of success)
- **Expected Outcome**: the model should reach 90% success. The end-effector only need to reach the desired position, it is not necessary to complete full task.
- **Results**:
