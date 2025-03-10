#### Akash Ramakrishnan
#### Program to evaluate the SMPL model using the joint angles from the biomechanics data

import joblib
import numpy as np
import os
import cv2
from tqdm import tqdm  # Import tqdm
from lib.utils.renderer import Renderer #Ensure this works
import re
import scipy.io
from scipy.spatial.transform import Rotation as R

def axis_angle_to_rotation_matrix(axis_angle):
    """Converts axis-angle representation to rotation matrix."""
    try:
        axis_angle = np.asarray(axis_angle).reshape(3)  # Ensure it's a numpy array and has the correct shape
        rotation_matrix, _ = cv2.Rodrigues(axis_angle)
        return rotation_matrix
    except Exception as e:
        print(f"Error in axis_angle_to_rotation_matrix: {e}")
        return np.eye(3)  # Return identity matrix as a fallback

def euler_to_rotation_matrix(euler_angles, sequence='XYZ'):
    try:
        r = R.from_euler(sequence, euler_angles, degrees=True)  # Assuming degrees
        return r.as_matrix()
    except Exception as e:
        print(f"Error in euler_to_rotation_matrix: {e}")
        return np.eye(3)  # Return identity matrix as a fallback

def rotation_matrix_to_angle(R):
    angle = np.arccos(((np.trace(R) - 1) / 2))
    return angle

def compare_joint_angles(smpl_rot, qualisys_rot):
    # R is rotation matrix
    R = np.dot(smpl_rot.T, qualisys_rot) # Relative rotation
    angle = rotation_matrix_to_angle(R)
    return angle


def evaluate_merged_smpl(joint_angles_data, merged_smpl_output_path, trial_name):

    
    mat = scipy.io.loadmat(joint_angles_data)

    ik_ang_data = mat['allTrialsData'][trial_name][0, 0]['joint_angles'][0, 0]
    print(ik_ang_data.shape)

    merged_results = joblib.load(merged_smpl_output_path)
    merged_results = merged_results[list(merged_results.keys())[0]]
    # print(merged_results.keys())
    pose = merged_results["pose"]
    frames = merged_results["frame_ids"]

    # Joint mapping (adjust as needed)
    joint_mapping = {
        "hip_flexion_r": 45,  
        "knee_angle_r": 5,  
        "ankle_angle_r": 8, 
        "hip_flexion_l": 46,  
        "knee_angle_l": 4,   
        "ankle_angle_l": 7 
    }

    # Prepare lists to store metrics
    joint_angle_differences = []
    error_counter = 0 

    for i in tqdm(range((ik_ang_data.shape[0]//100)*25), desc="Comparing frames"):
        # Since videos have fps of 25 and data is at 100 Hz
        frame_num = frames[i] * 4

        try:
            smpl_pose = pose[i]
            # print(f"Frame {i}: smpl_pose = {smpl_pose}")
            # Get the angles based on the data of qualisys
            mean_qualisys_data = ik_ang_data[frame_num]

            # Iterate through specified joints
            for joint_name, joint_index in joint_mapping.items():
                smpl_axis_angle = smpl_pose[joint_index * 3:joint_index * 3 + 3]

                # Convert Qualisys Euler angles to rotation matrix
                qualisys_rot_matrix = euler_to_rotation_matrix(
                    mean_qualisys_data[:3])  # Using first 3 angles - adjust as needed

                # Convert SMPL axis-angle to rotation matrix
                smpl_rot_matrix = axis_angle_to_rotation_matrix(smpl_axis_angle)

                # Compare joint angles (using angle between rotation matrices)
                angle_difference = compare_joint_angles(smpl_rot_matrix, qualisys_rot_matrix)
                joint_angle_differences.append(angle_difference)

        except Exception as e:
            print(f"The frame id {frame_num} has error {e}")
            error_counter += 1 
            continue 

    # Calculate average metrics
    print(len(joint_angle_differences))
    avg_angle_difference = np.mean(joint_angle_differences)

    print(f"Average Joint Angle Difference: {avg_angle_difference:.4f} radians")
    print(f"{error_counter} errors occured during the evaluation.")
    return {"average_joint_angle_difference": avg_angle_difference}

    
# Define paths

joint_angles_data = os.path.join("D:\weamai", "biomech_data", "Subj04_all_trials_joint_angles.mat")
merged_smpl_output_path = os.path.join("D:\weamai", "VIBE", "output", "Subj04_jump", "front_output.pkl")
trial_name = 'jump'

# Evaluate the merged SMPL model
evaluation_metrics = evaluate_merged_smpl(joint_angles_data, merged_smpl_output_path, trial_name)

print("Evaluation metrics:", evaluation_metrics)