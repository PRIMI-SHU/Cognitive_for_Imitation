# Cognitive_for_Imitation
Codes for the paper: A multimodal framework for robot visual imitation from others

## Datasets
Example data can be found at: https://drive.google.com/drive/folders/1j-eLQt_qdH-ODAaCzA6Ws3kFzYM0o7OU?usp=sharing.
Should put the dataset in the data folder and modify the root path in /model/util/helper.py/Hparams class.

## Install 
The training codes can be directly used in our docker image:
```bash
docker pull ruidong14/ros_torch:latest
```
## Usage:
1. /model/util/helper.py: All the dataset loader functions are stored in this script.  Please modify the data, model paths and the hyperparameters settings in the Hparams class.
2. /model/networks/metric.py and mvae.py: contains the Multimodal Varational AutoEncoder and Triplet Network models written in pytorch.
3. In the /model/training folder.
   - First, train the MVAE internal model:   
       ```bash
       python3 trainer_mvae.py
       ```
   - Save the trained visual embeddings:
      ```bash
      python3 save_embedding.py
      ```
    - Train the metric model:
      ```bash
      python3 train_metric.py
      ```
   - Train the dynamic planner:
      ```bash
      python3 trainer_NDP.py
      ```
 4. You can test out our mental simulation function without real robots in /model/test_mvae.ipynb:
    - Can see the mental simulation result in m_s_result_ours and ms_result_AIF folders.
 6. real-robot.py is the example code to control a real sawyer robot via ROS.  For the real-world experiment, you will need to install the Sawyer ROS package: 
    https://support.rethinkrobotics.com/support/solutions/folders/80000686868.


