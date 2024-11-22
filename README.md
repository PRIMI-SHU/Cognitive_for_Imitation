# Cognitive_for_Imitation
Companion Codes for the paper: A multimodal robot internal model for visual imitation from others。

## Install 
The training codes can be directly used in out docker image:
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
