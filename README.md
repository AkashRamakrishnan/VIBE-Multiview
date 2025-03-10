# VIBE-Multiview

This repository contains the implementation of the [VIBE Pipeline](https://github.com/mkocabas/VIBE) on the [Comprehensive Kinetic and EMG Dataset](https://zenodo.org/records/6457662). The synchronous front and side view video data can be found [here](https://zenodo.org/records/6644593). This repository utilized the windows installation of the VIBE pipeline as shown in [vibe_win_install](https://github.com/carlosedubarreto/vibe_win_install/tree/main).   

The main script for inferencing VIBE on two camera views and fusing the generated 3D pose is in [main.py](main.py).  

To evaluate the generated SMPL parameters, use[evaluate_smpl.py](evaluate_smpl.py).  

[extractData.m](extractData.m) contains the matlab script for extracting the joint angle measurements of all trials for a particular subject. Measurements for Subject04 is present in [Subj04_all_trials_joint_angles.mat](Subj04_all_trials_joint_angles.mat).  