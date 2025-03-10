#### Akash Ramakrishnan
#### Program to run VIBE pipeline on two camera views, merge predictions, and evaluate results. 

## Imports
import os
#os.environ['PYOPENGL_PLATFORM'] = 'egl'

import cv2
import time
import torch
import joblib
import shutil
import colorsys
import argparse
import numpy as np
from tqdm import tqdm
from multi_person_tracker import MPT
from torch.utils.data import DataLoader

from lib.models.vibe import VIBE_Demo
from lib.utils.renderer import Renderer
from lib.dataset.inference import Inference
from lib.utils.smooth_pose import smooth_pose
from lib.data_utils.kp_utils import convert_kps
from lib.utils.pose_tracker import run_posetracker

from lib.utils.demo_utils import (
    download_youtube_clip,
    smplify_runner,
    convert_crop_cam_to_orig_img,
    prepare_rendering_results,
    video_to_images,
    images_to_video,
    download_ckpt,
)

## Variables
MIN_NUM_FRAMES = 25
SUB = 'Subj04'
ACTIVITY = 'static'                                             # jump, land, walk_09, walk_18, run_81, run_99, squat, etc.  
RUN_VIBE = False                                                 # Set True the first time to generate SMPL pickle files for front and side views. 
DATA_DIR = os.path.join('D:\weamai', 'Videos_anonymized')
OUTPUT_FOLDER = 'output'
RENDER_FRONT_AND_SIDE = False                                    # To render the front and side views separately.
TRACKING_METHOD = 'bbox'                                        # either bbox or pose. 
DETECTOR = 'yolo'           
YOLO_IMG_SIZE = 416
TRACKER_BATCH_SIZE = 3
VIBE_BATCH_SIZE = 16
DISPLAY = False
SMOOTH = True
SMOOTH_MIN_CUTOFF = 0.004
SMOOTH_BETA = 0.7
RUN_SMPLIFY = False
STAF_DIR = None
SIDEVIEW = False

## FUNCTIONS
def merge_smpl(front_results, side_results, w1 = 0.5, w2 = 0.5):
    merged_results = {}
    front_person_id = list(front_results.keys())[0]
    side_person_id = list(side_results.keys())[0]

    front_pose = front_results[front_person_id]["pose"]
    front_betas = front_results[front_person_id]["betas"]
    front_pose_frames = front_results[front_person_id]['frame_ids']  # Frame IDs for front view
    side_pose = side_results[side_person_id]["pose"]
    side_betas = side_results[side_person_id]["betas"]
    side_pose_frames = side_results[side_person_id]['frame_ids']  # Frame IDs for side view

    # Extract joints and bounding box data
    front_joints2d = front_results[front_person_id]["joints2d"]
    side_joints2d = side_results[side_person_id]["joints2d"]
    front_bboxes = front_results[front_person_id]["bboxes"]
    side_bboxes = side_results[side_person_id]["bboxes"]
    front_joints3d = front_results[front_person_id]["joints3d"]
    side_joints3d = side_results[side_person_id]["joints3d"]
    front_verts = front_results[front_person_id]["verts"]
    side_verts = side_results[side_person_id]["verts"]
    front_pred_cam = front_results[front_person_id]["pred_cam"]
    side_pred_cam = side_results[side_person_id]["pred_cam"]
    front_orig_cam = front_results[front_person_id]["orig_cam"]
    side_orig_cam = side_results[side_person_id]["orig_cam"]

    merged_poses = []
    merged_betas = []
    merged_frame_ids = []
    merged_joints3d = []
    merged_bboxes = []
    merged_verts = []
    merged_pred_cam = []
    merged_orig_cam = []

    all_frame_ids = np.union1d(front_pose_frames, side_pose_frames)

    for frame_idx in tqdm(all_frame_ids, desc="Processing frames"):
        # Find corresponding poses in front and side views
        front_index = np.where(front_pose_frames == frame_idx)[0]
        side_index = np.where(side_pose_frames == frame_idx)[0]

        # Check if poses are available in both views
        if len(front_index) > 0 and len(side_index) > 0:

            front_3d_joints = front_joints3d[front_index[0]] # Get 3d joints
            side_3d_joints = side_joints3d[side_index[0]] # Get Side 3d joints

            #Calculate joint distance as 3d Error
            joint_distance = np.mean(np.linalg.norm(front_3d_joints - side_3d_joints, axis = 1))
            # print(joint_distance)
            if joint_distance > 10: #If joint distance is very high, disregard.
                merged_pose = front_pose[front_index[0]]
                merged_betas = front_betas[front_index[0]]

            else:
                # w1_local = w1 - (joint_distance/20) 
                # w2_local = w2 + (joint_distance/20)
                w1_local = 0.5
                w2_local = 0.5
                # Weighted averaging of pose and betas
                merged_pose = (w1_local * front_pose[front_index[0]]) + (w2_local * side_pose[side_index[0]])
                merged_beta = (w1_local * front_betas[front_index[0]]) + (w2_local * side_betas[side_index[0]])
            merged_joints3d_val = (w1_local * front_3d_joints) + (w2_local * side_3d_joints)

            merged_poses.append(merged_pose)
            merged_betas.append(merged_beta)
            merged_frame_ids.append(frame_idx)
            merged_joints3d.append(merged_joints3d_val)
            merged_bboxes.append(front_bboxes[front_index[0]]) 
            merged_verts.append(front_verts[front_index[0]]) 
            merged_pred_cam.append(front_pred_cam[front_index[0]]) 
            merged_orig_cam.append(front_orig_cam[front_index[0]]) 
            

        elif len(front_index) > 0:
            # Use front view pose if side view is missing
            merged_poses.append(front_pose[front_index[0]])
            merged_betas.append(front_betas[front_index[0]])
            merged_frame_ids.append(frame_idx)
            merged_joints3d.append(front_joints3d[front_index[0]])
            merged_bboxes.append(front_bboxes[front_index[0]]) 
            merged_verts.append(front_verts[front_index[0]]) 
            merged_pred_cam.append(front_pred_cam[front_index[0]]) 
            merged_orig_cam.append(front_orig_cam[front_index[0]]) 
            
            
            
        elif len(side_index) > 0:
            # Use side view pose if front view is missing
            merged_poses.append(side_pose[side_index[0]])
            merged_betas.append(side_betas[side_index[0]])
            merged_frame_ids.append(frame_idx)
            merged_joints3d.append(side_joints3d[side_index[0]])
            merged_bboxes.append(front_bboxes[side_index[0]]) 
            merged_verts.append(front_verts[side_index[0]]) 
            merged_pred_cam.append(front_pred_cam[side_index[0]]) 
            merged_orig_cam.append(front_orig_cam[side_index[0]]) 
             
        else:
            # No pose available for this frame, skip rendering
            continue  # Skip to the next frame

    
    merged_output_dict = {
        'pose': np.array(merged_poses),
        'betas': np.array(merged_betas),
        'joints3d': np.array(merged_joints3d),
        'frame_ids': np.array(merged_frame_ids),
        'bboxes': np.array(merged_bboxes),
        'verts': np.array(merged_verts),
        'pred_cam': np.array(merged_pred_cam), 
        'orig_cam': np.array(merged_orig_cam)
    }
    # joblib.dump(merged_output_dict, output_pkl_path)
    merged_results[0] = merged_output_dict
    return merged_results


def run_vibe(video_file, image_folder, orig_width, orig_height, num_frames, device):
    # ========= Run tracking ========= #
    print('=================Running Tracker=================')
    bbox_scale = 1.1
    if TRACKING_METHOD == 'pose':
        if not os.path.isabs(video_file):
            video_file = os.path.join(os.getcwd(), video_file)

        tracking_results = run_posetracker(video_file, staf_folder=STAF_DIR, display=DISPLAY)
        

    else:
        # run multi object tracker
        mot = MPT(
            device=device,
            batch_size=TRACKER_BATCH_SIZE,
            display=DISPLAY,
            detector_type=DETECTOR,
            output_format='dict',
            yolo_img_size=YOLO_IMG_SIZE,
        )
        tracking_results = mot(image_folder)
        
    print(tracking_results[list(tracking_results.keys())[0]]['bbox'].shape)
    # remove tracklets if num_frames is less than MIN_NUM_FRAMES
    for person_id in list(tracking_results.keys()):
        if tracking_results[person_id]['frames'].shape[0] < MIN_NUM_FRAMES:
            del tracking_results[person_id]

    # ========= Define VIBE model ========= #
    model = VIBE_Demo(
        seqlen=16,
        n_layers=2,
        hidden_size=1024,
        add_linear=True,
        use_residual=True,
    ).to(device)

    # ========= Load pretrained weights ========= #
    pretrained_file = download_ckpt(use_3dpw=False)
    ckpt = torch.load(pretrained_file)
    print(f'Performance of pretrained model on 3DPW: {ckpt["performance"]}')
    ckpt = ckpt['gen_state_dict']
    model.load_state_dict(ckpt, strict=False)
    model.eval()
    print(f'Loaded pretrained weights from \"{pretrained_file}\"')
    
    # VIBE on Front view

    print('==============Running VIBE==============')
    # ========= Run VIBE on each person ========= #
    print(f'Running VIBE on each tracklet...')
    results = {}
    for person_id in tqdm(list(tracking_results.keys())):
        bboxes = joints2d = None

        if TRACKING_METHOD == 'bbox':
            bboxes = tracking_results[person_id]['bbox']
        elif TRACKING_METHOD == 'pose':
            joints2d = tracking_results[person_id]['joints2d']

        frames = tracking_results[person_id]['frames']

        dataset = Inference(
            image_folder=image_folder,
            frames=frames,
            bboxes=bboxes,
            joints2d=joints2d,
            scale=bbox_scale
        )

        bboxes = dataset.bboxes
        frames = dataset.frames
        has_keypoints = True if joints2d is not None else False

        dataloader = DataLoader(dataset, batch_size=VIBE_BATCH_SIZE, num_workers=16)

        with torch.no_grad():

            pred_cam, pred_verts, pred_pose, pred_betas, pred_joints3d, norm_joints2d = [], [], [], [], [], []

            for batch in dataloader:
                if has_keypoints:
                    batch, nj2d = batch
                    norm_joints2d.append(nj2d.numpy().reshape(-1, 21, 3))

                batch = batch.unsqueeze(0)
                batch = batch.to(device)

                batch_size, seqlen = batch.shape[:2]
                output = model(batch)[-1]

                pred_cam.append(output['theta'][:, :, :3].reshape(batch_size * seqlen, -1))
                pred_verts.append(output['verts'].reshape(batch_size * seqlen, -1, 3))
                pred_pose.append(output['theta'][:,:,3:75].reshape(batch_size * seqlen, -1))
                pred_betas.append(output['theta'][:, :,75:].reshape(batch_size * seqlen, -1))
                pred_joints3d.append(output['kp_3d'].reshape(batch_size * seqlen, -1, 3))


            pred_cam = torch.cat(pred_cam, dim=0)
            pred_verts = torch.cat(pred_verts, dim=0)
            pred_pose = torch.cat(pred_pose, dim=0)
            pred_betas = torch.cat(pred_betas, dim=0)
            pred_joints3d = torch.cat(pred_joints3d, dim=0)

            del batch

        # ========= [Optional] run Temporal SMPLify to refine the results ========= #
        if RUN_SMPLIFY and TRACKING_METHOD == 'pose':
            norm_joints2d = np.concatenate(norm_joints2d, axis=0)
            norm_joints2d = convert_kps(norm_joints2d, src='staf', dst='spin')
            norm_joints2d = torch.from_numpy(norm_joints2d).float().to(device)

            # Run Temporal SMPLify
            update, new_opt_vertices, new_opt_cam, new_opt_pose, new_opt_betas, \
            new_opt_joints3d, new_opt_joint_loss, opt_joint_loss = smplify_runner(
                pred_rotmat=pred_pose, 
                pred_betas=pred_betas,
                pred_cam=pred_cam,
                j2d=norm_joints2d,
                device=device,
                batch_size=norm_joints2d.shape[0],
                pose2aa=False,
            )

            # update the parameters after refinement
            print(f'Update ratio after Temporal SMPLify: {update.sum()} / {norm_joints2d.shape[0]}')
            pred_verts = pred_verts.cpu()
            pred_cam = pred_cam.cpu()
            pred_pose = pred_pose.cpu()
            pred_betas = pred_betas.cpu()
            pred_joints3d = pred_joints3d.cpu()
            pred_verts[update] = new_opt_vertices[update]
            pred_cam[update] = new_opt_cam[update]
            pred_pose[update] = new_opt_pose[update]
            pred_betas[update] = new_opt_betas[update]
            pred_joints3d[update] = new_opt_joints3d[update]

        elif RUN_SMPLIFY and TRACKING_METHOD == 'bbox':
            print('[WARNING] You need to enable pose tracking to run Temporal SMPLify algorithm!')
            print('[WARNING] Continuing without running Temporal SMPLify!..')

        # ========= Save results to a pickle file ========= #
        pred_cam = pred_cam.cpu().numpy()
        pred_verts = pred_verts.cpu().numpy()
        pred_pose = pred_pose.cpu().numpy()
        pred_betas = pred_betas.cpu().numpy()
        pred_joints3d = pred_joints3d.cpu().numpy()

        # Runs 1 Euro Filter to smooth out the results
        if SMOOTH:
            min_cutoff = SMOOTH_MIN_CUTOFF # 0.004
            beta = SMOOTH_BETA # 1.5
            print(f'Running smoothing on person {person_id}, min_cutoff: {min_cutoff}, beta: {beta}')
            pred_verts, pred_pose, pred_joints3d = smooth_pose(pred_pose, pred_betas,
                                                               min_cutoff=min_cutoff, beta=beta)

        orig_cam = convert_crop_cam_to_orig_img(
            cam=pred_cam,
            bbox=bboxes,
            img_width=orig_width,
            img_height=orig_height
        )

        output_dict = {
            'pred_cam': pred_cam,
            'orig_cam': orig_cam,
            'verts': pred_verts,
            'pose': pred_pose,
            'betas': pred_betas,
            'joints3d': pred_joints3d,
            'joints2d': joints2d,
            'bboxes': bboxes,
            'frame_ids': frames,
        }

        del model
        results[person_id] = output_dict


    return results

def render_results(results, image_folder, orig_height, orig_width, num_frames, output_path, view):
    renderer = Renderer(resolution=(orig_width, orig_height), orig_img=True, wireframe=False)
    output_folder = f'{image_folder}_output'
    os.makedirs(output_folder, exist_ok=True)
    print(f'Rendering output video, writing frames to {output_folder}')


    # prepare results for rendering
    frame_results = prepare_rendering_results(results, num_frames)
    mesh_color = {k: colorsys.hsv_to_rgb(np.random.rand(), 0.5, 1.0) for k in results.keys()}

    image_file_names = sorted([
        os.path.join(image_folder, x)
        for x in os.listdir(image_folder)
        if x.endswith('.png') or x.endswith('.jpg')
    ])

    for frame_idx in tqdm(range(len(image_file_names))):
        img_fname = image_file_names[frame_idx]
        img = cv2.imread(img_fname)

        if SIDEVIEW:
            side_img = np.zeros_like(img)

        for person_id, person_data in frame_results[frame_idx].items():
            frame_verts = person_data['verts']
            frame_cam = person_data['cam']

            mc = mesh_color[person_id]

            mesh_filename = None

            img = renderer.render(
                img,
                frame_verts,
                cam=frame_cam,
                color=mc,
                mesh_filename=mesh_filename,
            )

            if SIDEVIEW:
                side_img = renderer.render(
                    side_img,
                    frame_verts,
                    cam=frame_cam,
                    color=mc,
                    angle=270,
                    axis=[0,1,0],
                )

        if SIDEVIEW:
            img = np.concatenate([img, side_img], axis=1)

        cv2.imwrite(os.path.join(output_folder, f'{frame_idx:06d}.png'), img)

        if DISPLAY:
            cv2.imshow('Video', img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    if DISPLAY:
        cv2.destroyAllWindows()

    # ========= Save rendered video ========= #
    save_name = os.path.join(output_path, f"{view}_result.mp4")
    images_to_video(img_folder=output_folder, output_vid_file=save_name)
    shutil.rmtree(output_folder)


def main():
    print(f'Running VIBE on {SUB} {ACTIVITY}')
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Running on device: {device}")
    front_video_file = os.path.join(DATA_DIR, F'{SUB}', f'{ACTIVITY}_front_anonymized.avi')
    side_video_file = os.path.join(DATA_DIR, F'{SUB}', f'{ACTIVITY}_side_anonymized.avi')

    if not os.path.isfile(front_video_file):
        exit(f'Input video \"{front_video_file}\" does not exist!')

    if not os.path.isfile(side_video_file):
        exit(f'Input video \"{side_video_file}\" does not exist!')

    output_path = os.path.join(OUTPUT_FOLDER, f'{SUB}_{ACTIVITY}')
    os.makedirs(output_path, exist_ok=True)

    f_image_folder, f_num_frames, f_img_shape = video_to_images(front_video_file, return_info=True)
    s_image_folder, s_num_frames, s_img_shape = video_to_images(side_video_file, return_info=True)

    print(f'Front video number of frames {f_num_frames}')
    print(f'Side video number of frames {s_num_frames}')
    orig_height, orig_width = f_img_shape[:2]

    # Run VIBE on both views
    if RUN_VIBE:
        print('=================Running VIBE on Front view=================')
        front_results = run_vibe(front_video_file, f_image_folder, orig_width, orig_height, f_num_frames, device)
        print(f'Saving output results to \"{os.path.join(output_path, "front_output.pkl")}\".')
        joblib.dump(front_results, os.path.join(output_path, "front_output.pkl"))

        print('=================Running VIBE on Side view=================')
        side_results = run_vibe(side_video_file, s_image_folder, orig_width, orig_height, s_num_frames, device)
        print(f'Saving output results to \"{os.path.join(output_path, "side_output.pkl")}\".')
        joblib.dump(side_results, os.path.join(output_path, "side_output.pkl"))

    else:
        front_results = joblib.load(os.path.join(output_path, "front_output.pkl"))
        side_results = joblib.load(os.path.join(output_path, "side_output.pkl"))

    # Merge the results
    merged_results = merge_smpl(front_results, side_results)

    print(f'Saving output results to \"{os.path.join(output_path, "merged_output.pkl")}\".')

    joblib.dump(side_results, os.path.join(output_path, "merged_output.pkl"))

    # Render the results
    if RENDER_FRONT_AND_SIDE:
        render_results(front_results, f_image_folder, orig_height, orig_width, f_num_frames, output_path, 'front')
        render_results(side_results, s_image_folder, orig_height, orig_width, s_num_frames, output_path, 'side')

    render_results(merged_results, f_image_folder, orig_height, orig_width, f_num_frames, output_path, 'merged')

    shutil.rmtree(f_image_folder)
    shutil.rmtree(s_image_folder)


if __name__ == '__main__':
    main()




    