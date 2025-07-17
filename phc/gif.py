import cv2
import numpy as np
import torch
from PIL import Image
import tensorflow as tf
import tensorflow_hub as hub
import requests
import os
from urllib.parse import urlparse
import tempfile
import shutil
from ultralytics import YOLO
import sys
sys.path.append(os.getcwd())
from scipy.spatial.transform import Rotation as sRot
from smpl_sim.smpllib.smpl_local_robot import SMPL_Robot as LocalRobot
from poselib.poselib.skeleton.skeleton3d import SkeletonTree, SkeletonMotion, SkeletonState
from smpl_sim.smpllib.smpl_joint_names import SMPL_BONE_ORDER_NAMES, SMPL_MUJOCO_NAMES

class GIFMotionImitator:
    def __init__(self, gif_path):
        self.gif_path = gif_path
        self.local_path = None
        self.metrabs_model = hub.load('https://bit.ly/metrabs_s')
        self.skeleton = 'smpl_24'
        self.motion_data = None
        
        # Initialize YOLO model for person detection
        try:
            self.yolo_model = YOLO("yolov8s.pt")
        except:
            print("Warning: YOLO model not found. Person detection will be skipped.")
            self.yolo_model = None

        # Initialize SMPL robot for skeleton structure
        robot_cfg = {
            "mesh": False,
            "model": "smpl",
            "upright_start": True,
            "body_params": {},
            "joint_params": {},
            "geom_params": {},
            "actuator_params": {},
        }
        
        self.smpl_local_robot = LocalRobot(
            robot_cfg,
            data_dir="data/smpl",
        )
    
    def download_gif(self, url, output_path):
        """Download GIF from URL to local path"""
        try:
            print(f"[download_gif] Downloading from {url}")
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            print(f"[download_gif] GIF saved to {output_path}")
            return output_path
        except Exception as e:
            print(f"[download_gif] Error downloading GIF: {e}")
            return None
    
    def get_local_path(self):
        """Get local path for the GIF, downloading if necessary"""
        if self.local_path and os.path.exists(self.local_path):
            return self.local_path
            
        if self.gif_path.startswith(('http://', 'https://')):
            # URL - need to download
            parsed_url = urlparse(self.gif_path)
            filename = os.path.basename(parsed_url.path) or "downloaded.gif"
            
            # Ensure it has .gif extension
            if not filename.lower().endswith('.gif'):
                if '.gif' in self.gif_path.lower():
                    filename += '.gif'
                else:
                    filename = filename.split('.')[0] + '.gif'
            
            # Create temp directory
            temp_dir = tempfile.mkdtemp(prefix='gif_motion_')
            output_path = os.path.join(temp_dir, filename)
            
            downloaded_path = self.download_gif(self.gif_path, output_path)
            if downloaded_path:
                self.local_path = downloaded_path
                return downloaded_path
            else:
                return None
        else:
            # Local file
            if os.path.exists(self.gif_path):
                self.local_path = self.gif_path
                return self.gif_path
            else:
                print(f"Error: Local file {self.gif_path} not found")
                return None
        
    def extract_frames(self):
        """Extract frames from GIF (local or downloaded)"""
        local_path = self.get_local_path()
        if not local_path:
            return []
            
        frames = []
        
        try:
            # Try PIL first for better GIF handling
            gif = Image.open(local_path)
            
            # Check if animated
            is_animated = True
            try:
                gif.seek(1)
                gif.seek(0)
            except EOFError:
                is_animated = False
            
            if not is_animated:
                # Single frame
                frame = np.array(gif.convert('RGB'))
                frames.append(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            else:
                # Multiple frames
                frame_idx = 0
                try:
                    while True:
                        gif.seek(frame_idx)
                        frame = gif.copy()
                        
                        # Convert to RGB
                        if frame.mode == 'P':
                            frame = frame.convert('RGBA')
                            background = Image.new('RGB', frame.size, (255, 255, 255))
                            if 'transparency' in frame.info:
                                background.paste(frame, mask=frame.split()[-1])
                            else:
                                background.paste(frame)
                            frame = background
                        else:
                            frame = frame.convert('RGB')
                        
                        # Convert to numpy array and then to BGR for OpenCV
                        frame_array = np.array(frame)
                        frame_bgr = cv2.cvtColor(frame_array, cv2.COLOR_RGB2BGR)
                        frames.append(frame_bgr)
                        
                        frame_idx += 1
                        
                except EOFError:
                    pass
                    
            gif.close()
            
        except Exception as e:
            print(f"PIL extraction failed: {e}, trying OpenCV...")
            # Fallback to OpenCV
            cap = cv2.VideoCapture(local_path)
            if cap.isOpened():
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frames.append(frame)
                cap.release()
        
        print(f"Extracted {len(frames)} frames from GIF")
        return frames
    
    def detect_person(self, frame):
        """Detect person in frame using YOLO"""
        if self.yolo_model is None:
            # Return full frame as bounding box if YOLO not available
            h, w = frame.shape[:2]
            return [0, 0, w, h]
        
        try:
            # Convert BGR to RGB for YOLO
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Run YOLO detection
            results = self.yolo_model.predict(source=frame_rgb, show=False, classes=[0], verbose=False)
            
            if len(results[0].boxes) > 0:
                # Get the largest bounding box (most confident person)
                boxes = results[0].boxes.xyxy.cpu().numpy()
                scores = results[0].boxes.conf.cpu().numpy()
                
                # Find box with highest confidence
                max_idx = np.argmax(scores)
                bbox = boxes[max_idx]
                
                # Convert from [x1, y1, x2, y2] to [x, y, w, h]
                x1, y1, x2, y2 = bbox
                return [x1, y1, x2 - x1, y2 - y1]
            else:
                # No person detected, return full frame
                h, w = frame.shape[:2]
                return [0, 0, w, h]
                
        except Exception as e:
            print(f"Person detection failed: {e}")
            h, w = frame.shape[:2]
            return [0, 0, w, h]
    
    def positions_to_rotations(self, poses3d):
        """Convert 3D joint positions to rotation representations"""
        # This is a simplified conversion - for more accurate results, 
        # you might want to use proper inverse kinematics
        
        # For now, we'll create a basic pose using simple rotations
        # In a real implementation, you'd want to use proper IK
        
        # Create basic pose data (T-pose with slight variations)
        num_frames = len(poses3d)
        
        # Create simple angle-axis rotations for each joint
        # This is a placeholder - you'd want more sophisticated pose estimation
        pose_aa = np.zeros((num_frames, 24, 3))
        
        # Add some basic movement based on the 3D positions
        if num_frames > 0:
            # Simple example: use root position for overall motion
            root_positions = poses3d[:, 0] if len(poses3d) > 0 else np.zeros((num_frames, 3))
            
            # Add some basic rotations based on position changes
            for i in range(1, num_frames):
                # Simple rotation based on position change
                if i < num_frames:
                    pos_diff = poses3d[i, 0] - poses3d[i-1, 0]
                    # Convert position change to simple rotation
                    pose_aa[i, 0, 1] = np.arctan2(pos_diff[0], pos_diff[2]) * 0.1  # Y rotation
                    pose_aa[i, 0, 0] = pos_diff[1] * 0.1  # X rotation
        
        return pose_aa.reshape(num_frames, -1)
    
    def process_frames(self):
        """Extract 3D poses from frames and convert to PHC format"""
        frames = self.extract_frames()
        if not frames:
            return []
            
        poses3d = []
        
        for i, frame in enumerate(frames):
            try:
                # Detect person in frame
                bbox = self.detect_person(frame)
                if bbox is None:
                    continue
                
                # Convert frame to RGB for Metrabs
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Metrabs expects bbox in format [x, y, w, h]
                bbox_tensor = tf.constant([bbox], dtype=tf.float32)
                
                # Run pose estimation
                pred = self.metrabs_model.estimate_poses(
                    frame_rgb, 
                    bbox_tensor,
                    skeleton=self.skeleton
                )
                
                # Extract 3D pose
                pose3d = pred['poses3d'].numpy()[0] / 1000  # Convert to meters
                poses3d.append(pose3d)
                
                if i % 10 == 0:  # Print progress every 10 frames
                    print(f"Processed frame {i+1}/{len(frames)}")
                    
            except Exception as e:
                print(f"Error processing frame {i+1}: {e}")
                continue
        
        print(f"Successfully processed {len(poses3d)} poses from {len(frames)} frames")
        return poses3d
    
    def save_motion_data(self, output_path):
        """Save motion data in PHC-compatible format"""
        poses3d = self.process_frames()
        
        if len(poses3d) == 0:
            print("No poses extracted from GIF")
            return None
        
        # Create output directory
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Convert 3D positions to rotations
        poses3d_array = np.array(poses3d)
        pose_aa = self.positions_to_rotations(poses3d_array)
        
        N = pose_aa.shape[0]
        if N < 1:
            print("No valid poses found")
            return None
        
        # Create basic motion data structure
        root_trans = poses3d_array[:, 0] if len(poses3d_array) > 0 else np.zeros((N, 3))
        
        # Create SMPL joint mapping
        smpl_2_mujoco = [SMPL_BONE_ORDER_NAMES.index(q) for q in SMPL_MUJOCO_NAMES if q in SMPL_BONE_ORDER_NAMES]
        pose_aa_mj = pose_aa.reshape(N, 24, 3)[:, smpl_2_mujoco]
        pose_quat = sRot.from_rotvec(pose_aa_mj.reshape(-1, 3)).as_quat().reshape(N, 24, 4)
        
        # Create SMPL skeleton
        beta = np.zeros((16))
        gender_number, gender = [0], "neutral"
        
        # Create skeleton tree
        self.smpl_local_robot.load_from_skeleton(betas=torch.from_numpy(beta[None,]), gender=gender_number, objs_info=None)
        
        # Create temporary XML file for skeleton
        temp_xml = f"phc/data/assets/mjcf/temp_humanoid.xml"
        os.makedirs(os.path.dirname(temp_xml), exist_ok=True)
        self.smpl_local_robot.write_xml(temp_xml)
        skeleton_tree = SkeletonTree.from_mjcf(temp_xml)
        
        # Clean up temp file
        if os.path.exists(temp_xml):
            os.remove(temp_xml)
        
        root_trans_offset = torch.from_numpy(root_trans) + skeleton_tree.local_translation[0]
        
        # Create skeleton state
        new_sk_state = SkeletonState.from_rotation_and_root_translation(
            skeleton_tree,
            torch.from_numpy(pose_quat),
            root_trans_offset,
            is_local=True
        )
        
        # Get global and local rotations
        pose_quat_global = new_sk_state.global_rotation.numpy()
        pose_quat_local = new_sk_state.local_rotation.numpy()
        
        # Create motion data in PHC format
        motion_data = {
            'pose_quat_global': pose_quat_global,
            'pose_quat': pose_quat_local,
            'trans_orig': root_trans,
            'root_trans_offset': root_trans_offset.numpy(),
            'pose_aa': pose_aa,
            'beta': beta,
            'gender': gender,
            'fps': 30.0,
            'source_url': self.gif_path
        }
        
        # Save as single motion entry (PHC expects dict format)
        motion_dict = {
            'motion_0': motion_data
        }
        
        import joblib
        joblib.dump(motion_dict, output_path)
        print(f"Motion data saved to {output_path}")
        return output_path
    
    def cleanup(self):
        """Clean up temporary files"""
        if (self.local_path and 
            self.local_path != self.gif_path and 
            os.path.exists(self.local_path)):
            
            temp_dir = os.path.dirname(self.local_path)
            if temp_dir.startswith('/tmp/') or 'gif_motion_' in temp_dir:
                try:
                    shutil.rmtree(temp_dir)
                    print(f"Cleaned up temporary directory: {temp_dir}")
                except:
                    pass

def run_imitation(
    gif_path: str = "https://i.pinimg.com/originals/58/f2/3c/58f23c62733b75711785a822cdb052c6.gif",
    output_dir: str = "output",
):
    """Main execution function"""
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Extract motion data from GIF
    imitator = GIFMotionImitator(gif_path)
    
    motion_file = imitator.save_motion_data(f"{output_dir}/motion_data.pkl")
    
    if motion_file is None:
        print("Failed to extract motion data from GIF")
        return None
    
    # 2. Generate PHC training command
    cmd = f"""python phc/run_hydra.py \\
    learning=im_mcp \\
    exp_name=gif_imitation \\
    env=env_im_getup_mcp \\
    env.task=HumanoidImMCPDemo \\
    robot=smpl_humanoid \\
    robot.freeze_hand=True \\
    env.motion_file={motion_file} \\
    env.num_envs=1 \\
    env.obs_v=7 \\
    headless=False \\
    test=False"""
        
    os.system(cmd)        



if __name__ == "__main__":
    import fire
    fire.Fire(run_imitation)