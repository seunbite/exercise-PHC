#!/usr/bin/env python3
"""
Test script to verify the GIF motion data format is correct for PHC
"""

import os
import sys
import joblib
import numpy as np
import torch

# Add PHC to path
sys.path.append(os.getcwd())

def test_motion_data_format(motion_file):
    """Test if the motion data file has the correct format for PHC"""
    print(f"Testing motion data format: {motion_file}")
    
    if not os.path.exists(motion_file):
        print(f"ERROR: Motion file {motion_file} does not exist")
        return False
    
    try:
        # Load motion data
        motion_data = joblib.load(motion_file)
        print(f"Successfully loaded motion data")
        
        # Check if it's a dictionary (PHC expects dict format)
        if not isinstance(motion_data, dict):
            print(f"ERROR: Motion data should be a dictionary, got {type(motion_data)}")
            return False
        
        print(f"Motion data keys: {list(motion_data.keys())}")
        
        # Check each motion entry
        for key, data in motion_data.items():
            print(f"\nChecking motion entry: {key}")
            
            # Check required keys
            required_keys = ['pose_quat_global', 'pose_quat', 'trans_orig', 'root_trans_offset', 'pose_aa', 'beta', 'gender', 'fps']
            
            for req_key in required_keys:
                if req_key not in data:
                    print(f"ERROR: Missing required key: {req_key}")
                    return False
                else:
                    print(f"✓ Found key: {req_key}")
            
            # Check data shapes
            pose_quat_global = data['pose_quat_global']
            pose_quat = data['pose_quat']
            pose_aa = data['pose_aa']
            
            print(f"  pose_quat_global shape: {pose_quat_global.shape}")
            print(f"  pose_quat shape: {pose_quat.shape}")
            print(f"  pose_aa shape: {pose_aa.shape}")
            print(f"  fps: {data['fps']}")
            print(f"  gender: {data['gender']}")
            print(f"  beta shape: {data['beta'].shape}")
            
            # Verify shapes are compatible
            if len(pose_quat_global.shape) != 3 or pose_quat_global.shape[2] != 4:
                print(f"ERROR: pose_quat_global should be [N, 24, 4], got {pose_quat_global.shape}")
                return False
            
            if len(pose_quat.shape) != 3 or pose_quat.shape[2] != 4:
                print(f"ERROR: pose_quat should be [N, 24, 4], got {pose_quat.shape}")
                return False
            
            if len(pose_aa.shape) != 2 or pose_aa.shape[1] != 72:
                print(f"ERROR: pose_aa should be [N, 72], got {pose_aa.shape}")
                return False
            
            print("✓ All shape checks passed")
        
        print("\n✓ Motion data format is correct for PHC!")
        return True
        
    except Exception as e:
        print(f"ERROR: Failed to load/verify motion data: {e}")
        return False

def create_test_gif_motion():
    """Create a test motion data file using the GIF motion extraction"""
    print("Creating test motion data from GIF...")
    
    # Import the fixed gif module
    from phc.gif import GIFMotionImitator
    
    # Test with a simple GIF URL
    gif_url = "https://i.pinimg.com/originals/58/f2/3c/58f23c62733b75711785a822cdb052c6.gif"
    
    try:
        # Create output directory
        output_dir = "test_output"
        os.makedirs(output_dir, exist_ok=True)
        
        # Create GIF motion imitator
        imitator = GIFMotionImitator(gif_url)
        
        # Extract motion data
        motion_file = imitator.save_motion_data(f"{output_dir}/test_motion_data.pkl")
        
        if motion_file:
            print(f"✓ Successfully created motion data: {motion_file}")
            return motion_file
        else:
            print("ERROR: Failed to create motion data")
            return None
            
    except Exception as e:
        print(f"ERROR: Failed to create test motion data: {e}")
        return None

if __name__ == "__main__":
    # Test the motion data format
    motion_file = create_test_gif_motion()
    
    if motion_file:
        test_motion_data_format(motion_file)
    else:
        print("Skipping format test due to motion data creation failure") 