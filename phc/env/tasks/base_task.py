# Copyright (c) 2018-2023, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import math
import numpy as np
import os
import os.path as osp
import random
import time
import torch
import imageio
import matplotlib.pyplot as plt
import joblib
import os
import os.path as osp
import functools
import gc
import sys
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import shutil
import glob
import cv2
import xml.etree.ElementTree as ET
import random
import numpy as np
from collections import deque
from PIL import Image
from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym.torch_utils import *
from poselib.poselib.skeleton.skeleton3d import SkeletonTree, SkeletonMotion, SkeletonState
from phc.utils.draw_utils import agt_color
from phc.utils.torch_utils import to_torch
from phc.utils.running_mean_std import RunningMeanStd
from phc.utils.flags import flags
from phc.utils.pytorch3d_transforms import *
from phc.utils.motion_lib_base import FixHeightMode
from phc.utils.torch_humanoid_batch import Humanoid_Batch
from phc.utils.torch_utils import *
from phc.learning.replay_buffer import ReplayBuffer
from phc.utils.torch_utils import to_torch
from datetime import datetime
from phc.utils.flags import flags
from collections import defaultdict
import aiohttp, cv2, asyncio
import json
from collections import deque
import threading
from tqdm import tqdm

# MuJoCo imports
import mujoco
import mujoco_viewer

# Base class for RL tasks
class BaseTask():

    def __init__(self, cfg, enable_camera_sensors=False):
        self.headless = cfg["headless"]
        if self.headless == False and not flags.no_virtual_display:
            from pyvirtualdisplay.smartdisplay import SmartDisplay
            self.virtual_display = SmartDisplay(size=(1800, 990), visible=True)
            self.virtual_display.start()

        self.gym = gymapi.acquire_gym()
        self.paused = False
        self.device_type = cfg.get("device_type", "cuda")
        self.device_id = cfg.get("device_id", 0)
        self.state_record = defaultdict(list)

        self.device = "cpu"
        if self.device_type == "cuda" or self.device_type == "GPU":
            self.device = "cuda" + ":" + str(self.device_id)

        # double check!
        self.graphics_device_id = self.device_id
        if enable_camera_sensors == False and self.headless == True:
            self.graphics_device_id = -1

        self.num_envs = cfg["env"]["num_envs"]
        self.num_obs = cfg["env"]["numObservations"]
        self.num_states = cfg["env"].get("numStates", 0)
        self.num_actions = cfg["env"]["numActions"]
        self.is_discrete = cfg["env"].get("is_discrete", False)

        self.control_freq_inv = cfg["control"].get("decimation", 2)

        # optimization flags for pytorch JIT
        torch._C._jit_set_profiling_mode(False)
        torch._C._jit_set_profiling_executor(False)

        # allocate buffers
        self.obs_buf = torch.zeros((self.num_envs, self.num_obs), device=self.device, dtype=torch.float)
        self.states_buf = torch.zeros((self.num_envs, self.num_states), device=self.device, dtype=torch.float)
        self.rew_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.reset_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.long)
        self.progress_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.randomize_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.extras = {}

        self.original_props = {}
        self.dr_randomizations = {}
        self.first_randomization = True
        self.actor_params_generator = None
        self.extern_actor_params = {}
        for env_id in range(self.num_envs):
            self.extern_actor_params[env_id] = None

        self.last_step = -1
        self.last_rand_step = -1

        # create envs, sim and viewer
        self.create_sim()
        self.gym.prepare_sim(self.sim)

        # todo: read from config
        self.enable_viewer_sync = True
        self.viewer = None

        # if running with a viewer, set up keyboard shortcuts and camera
        self.create_viewer()
        if flags.server_mode:
            # bgsk = threading.Thread(target=self.setup_video_client, daemon=True).start()
            bgsk = threading.Thread(target=self.setup_talk_client, daemon=False).start()

    def create_viewer(self):
        if self.headless == False:
            # Create MuJoCo model for visualization
            self.mujoco_model = self._create_mujoco_model()
            self.mujoco_data = mujoco.MjData(self.mujoco_model)
            self.viewer = mujoco_viewer.MujocoViewer(self.mujoco_model, self.mujoco_data)
            
            # Set up keyboard callbacks for MuJoCo viewer
            # self.viewer.add_key_callback("escape", self._handle_escape_key)
            # self.viewer.add_key_callback("v", self._handle_v_key)
            # self.viewer.add_key_callback("l", self._handle_l_key)
            # self.viewer.add_key_callback("r", self._handle_r_key)
            # self.viewer.add_key_callback("f", self._handle_f_key)
            # self.viewer.add_key_callback("g", self._handle_g_key)
            # self.viewer.add_key_callback("h", self._handle_h_key)
            # self.viewer.add_key_callback("c", self._handle_c_key)
            # self.viewer.add_key_callback("m", self._handle_m_key)
            # self.viewer.add_key_callback("b", self._handle_b_key)
            # self.viewer.add_key_callback("n", self._handle_n_key)
            # self.viewer.add_key_callback("k", self._handle_k_key)
            # self.viewer.add_key_callback("j", self._handle_j_key)
            # self.viewer.add_key_callback("left", self._handle_left_key)
            # self.viewer.add_key_callback("right", self._handle_right_key)
            # self.viewer.add_key_callback("t", self._handle_t_key)
            # self.viewer.add_key_callback("y", self._handle_y_key)
            # self.viewer.add_key_callback("i", self._handle_i_key)
            # self.viewer.add_key_callback("p", self._handle_p_key)
            # self.viewer.add_key_callback("o", self._handle_o_key)
            # self.viewer.add_key_callback("space", self._handle_space_key)
            
            # Set initial camera position
            self.viewer.cam.distance = 10.0
            self.viewer.cam.azimuth = 45.0
            self.viewer.cam.elevation = -20.0
            
            print("MuJoCo viewer initialized successfully")
        else:
            self.viewer = None
            self.mujoco_model = None
            self.mujoco_data = None

        ###### Custom Camera Sensors ######
        self.recorder_camera_handles = []
        self.max_num_camera = 10
        self.viewing_env_idx = 0
        for idx, env in enumerate(self.envs):
            self.recorder_camera_handles.append(self.gym.create_camera_sensor(env, gymapi.CameraProperties()))
            if idx > self.max_num_camera:
                break

        self.recorder_camera_handle = self.recorder_camera_handles[0]
        self.recording, self.recording_state_change = False, False
        self.max_video_queue_size = 100000
        self._video_queue = deque(maxlen=self.max_video_queue_size)
        rendering_out = osp.join("output", "renderings")
        states_out = osp.join("output", "states")
        os.makedirs(rendering_out, exist_ok=True)
        os.makedirs(states_out, exist_ok=True)
        self.cfg_name = self.cfg.exp_name
        self._video_path = osp.join(rendering_out, f"{self.cfg_name}-%s.mp4")
        self._states_path = osp.join(states_out, f"{self.cfg_name}-%s.pkl")
        # self.gym.draw_env_rigid_contacts(self.viewer, self.envs[1], gymapi.Vec3(0.9, 0.3, 0.3), 1.0, True)
        
    def _create_mujoco_model(self):
        """Create a simple MuJoCo model for visualization"""
        model_xml = """
        <mujoco model="humanoid_vis">
            <compiler coordinate="local"/>
            <statistic extent="2" center="0 0 1"/>
            <option timestep="0.00555"/>
            
            <default>
                <geom type="capsule" condim="1" friction="1.0 0.05 0.05" rgba="0.8 0.6 0.4 1"/>
                <joint type="hinge" damping="0.1" stiffness="5" limited="true"/>
            </default>
            
            <asset>
                <texture type="skybox" builtin="gradient" rgb1=".4 .5 .6" rgb2="0 0 0" width="100" height="100"/>
                <texture builtin="flat" height="1278" mark="cross" markrgb="1 1 1" name="texgeom" random="0.01" rgb1="0.8 0.6 0.4" rgb2="0.8 0.6 0.4" type="cube" width="127"/>
                <material name="geom" texture="texgeom" texuniform="true"/>
            </asset>
            
            <worldbody>
                <light diffuse=".5 .5 .5" pos="0 0 3" dir="0 0 -1"/>
                <geom type="plane" size="1 1 0.1" rgba=".9 0 0 1"/>
                
                <body name="torso" pos="0 0 1.4">
                    <geom type="capsule" size="0.07" fromto="0 -.07 0 0 .07 0" rgba="0.8 0.6 0.4 1"/>
                    <joint name="root_x" type="slide" axis="1 0 0" limited="false"/>
                    <joint name="root_y" type="slide" axis="0 1 0" limited="false"/>
                    <joint name="root_z" type="slide" axis="0 0 1" limited="false"/>
                    <joint name="root_rot_x" type="hinge" axis="1 0 0" limited="false"/>
                    <joint name="root_rot_y" type="hinge" axis="0 1 0" limited="false"/>
                    <joint name="root_rot_z" type="hinge" axis="0 0 1" limited="false"/>
                    
                    <body name="lwaist" pos="0 0 0.2">
                        <geom type="capsule" size="0.06" fromto="0 0 0 0 0 0.3" rgba="0.8 0.6 0.4 1"/>
                        <joint name="abdomen_y" type="hinge" axis="0 1 0" range="-45 45"/>
                        <joint name="abdomen_z" type="hinge" axis="0 0 1" range="-45 45"/>
                        <joint name="abdomen_x" type="hinge" axis="1 0 0" range="-45 45"/>
                        
                        <body name="pelvis" pos="0 0 -0.5">
                            <geom type="capsule" size="0.08" fromto="-.02 -.07 0 .02 .07 0" rgba="0.8 0.6 0.4 1"/>
                            
                            <body name="right_thigh" pos="0 -0.1 0">
                                <geom type="capsule" size="0.05" fromto="0 0 0 0 0 -0.4" rgba="0.8 0.6 0.4 1"/>
                                <joint name="right_hip_x" type="hinge" axis="1 0 0" range="-30 80"/>
                                <joint name="right_hip_y" type="hinge" axis="0 1 0" range="-30 30"/>
                                <joint name="right_hip_z" type="hinge" axis="0 0 1" range="-60 35"/>
                                
                                <body name="right_shin" pos="0 0 -0.4">
                                    <geom type="capsule" size="0.04" fromto="0 0 0 0 0 -0.4" rgba="0.8 0.6 0.4 1"/>
                                    <joint name="right_knee" type="hinge" axis="0 1 0" range="-160 -2"/>
                                    
                                    <body name="right_foot" pos="0 0 -0.4">
                                        <geom type="capsule" size="0.04" fromto="-.05 0 0 .05 0 0" rgba="0.8 0.6 0.4 1"/>
                                        <joint name="right_ankle_y" type="hinge" axis="0 1 0" range="-50 50"/>
                                        <joint name="right_ankle_x" type="hinge" axis="1 0 0" range="-50 50"/>
                                    </body>
                                </body>
                            </body>
                            
                            <body name="left_thigh" pos="0 0.1 0">
                                <geom type="capsule" size="0.05" fromto="0 0 0 0 0 -0.4" rgba="0.8 0.6 0.4 1"/>
                                <joint name="left_hip_x" type="hinge" axis="1 0 0" range="-30 80"/>
                                <joint name="left_hip_y" type="hinge" axis="0 1 0" range="-30 30"/>
                                <joint name="left_hip_z" type="hinge" axis="0 0 1" range="-35 60"/>
                                
                                <body name="left_shin" pos="0 0 -0.4">
                                    <geom type="capsule" size="0.04" fromto="0 0 0 0 0 -0.4" rgba="0.8 0.6 0.4 1"/>
                                    <joint name="left_knee" type="hinge" axis="0 1 0" range="-160 -2"/>
                                    
                                    <body name="left_foot" pos="0 0 -0.4">
                                        <geom type="capsule" size="0.04" fromto="-.05 0 0 .05 0 0" rgba="0.8 0.6 0.4 1"/>
                                        <joint name="left_ankle_y" type="hinge" axis="0 1 0" range="-50 50"/>
                                        <joint name="left_ankle_x" type="hinge" axis="1 0 0" range="-50 50"/>
                                    </body>
                                </body>
                            </body>
                        </body>
                    </body>
                </body>
            </worldbody>
            
            <actuator>
                <motor name="abdomen_y" joint="abdomen_y" gear="40"/>
                <motor name="abdomen_z" joint="abdomen_z" gear="40"/>
                <motor name="abdomen_x" joint="abdomen_x" gear="40"/>
                <motor name="right_hip_x" joint="right_hip_x" gear="40"/>
                <motor name="right_hip_y" joint="right_hip_y" gear="40"/>
                <motor name="right_hip_z" joint="right_hip_z" gear="40"/>
                <motor name="right_knee" joint="right_knee" gear="40"/>
                <motor name="right_ankle_y" joint="right_ankle_y" gear="40"/>
                <motor name="right_ankle_x" joint="right_ankle_x" gear="40"/>
                <motor name="left_hip_x" joint="left_hip_x" gear="40"/>
                <motor name="left_hip_y" joint="left_hip_y" gear="40"/>
                <motor name="left_hip_z" joint="left_hip_z" gear="40"/>
                <motor name="left_knee" joint="left_knee" gear="40"/>
                <motor name="left_ankle_y" joint="left_ankle_y" gear="40"/>
                <motor name="left_ankle_x" joint="left_ankle_x" gear="40"/>
            </actuator>
        </mujoco>
        """
        return mujoco.MjModel.from_xml_string(model_xml)

    # Keyboard callback methods for MuJoCo viewer
    def _handle_escape_key(self):
        """Handle ESC key - quit"""
        print("ESC pressed - quitting")
        self.viewer.close()
        
    def _handle_v_key(self):
        """Handle V key - toggle viewer sync"""
        self.enable_viewer_sync = not self.enable_viewer_sync
        print(f"Viewer sync: {self.enable_viewer_sync}")
        
    def _handle_l_key(self):
        """Handle L key - toggle video record"""
        self.recording = not self.recording
        if self.recording:
            print("Recording started")
        else:
            print("Recording stopped")
        
    def _handle_r_key(self):
        """Handle R key - reset"""
        print("Reset requested")
        
    def _handle_f_key(self):
        """Handle F key - follow"""
        flags.follow = not flags.follow
        print(f"Follow mode: {flags.follow}")
        
    def _handle_g_key(self):
        """Handle G key - fixed"""
        flags.fixed = not flags.fixed
        print(f"Fixed mode: {flags.fixed}")
        
    def _handle_h_key(self):
        """Handle H key - divide group"""
        flags.divide_group = not flags.divide_group
        print(f"Divide group: {flags.divide_group}")
        
    def _handle_c_key(self):
        """Handle C key - print camera"""
        print("Print camera info")
        
    def _handle_m_key(self):
        """Handle M key - disable collision reset"""
        flags.no_collision_check = not flags.no_collision_check
        print(f"No collision check: {flags.no_collision_check}")
        
    def _handle_b_key(self):
        """Handle B key - fixed path"""
        flags.fixed_path = not flags.fixed_path
        print(f"Fixed path: {flags.fixed_path}")
        
    def _handle_n_key(self):
        """Handle N key - real path"""
        flags.real_path = not flags.real_path
        print(f"Real path: {flags.real_path}")
        
    def _handle_k_key(self):
        """Handle K key - show trajectory"""
        flags.show_traj = not flags.show_traj
        print(f"Show trajectory: {flags.show_traj}")
        
    def _handle_j_key(self):
        """Handle J key - apply force"""
        print("Apply force")
        
    def _handle_left_key(self):
        """Handle LEFT key - previous env"""
        if self.viewing_env_idx > 0:
            self.viewing_env_idx -= 1
        print(f"Viewing environment: {self.viewing_env_idx}")
        
    def _handle_right_key(self):
        """Handle RIGHT key - next env"""
        if self.viewing_env_idx < len(self.envs) - 1:
            self.viewing_env_idx += 1
        print(f"Viewing environment: {self.viewing_env_idx}")
        
    def _handle_t_key(self):
        """Handle T key - resample motion"""
        print("Resample motion")
        
    def _handle_y_key(self):
        """Handle Y key - slow trajectory"""
        flags.slow = not flags.slow
        print(f"Slow trajectory: {flags.slow}")
        
    def _handle_i_key(self):
        """Handle I key - trigger input"""
        flags.trigger_input = not flags.trigger_input
        print(f"Trigger input: {flags.trigger_input}")
        
    def _handle_p_key(self):
        """Handle P key - show progress"""
        print("Show progress")
        
    def _handle_o_key(self):
        """Handle O key - change color"""
        print("Change color")
        
    def _handle_space_key(self):
        """Handle SPACE key - pause"""
        self.paused = not self.paused
        print(f"Paused: {self.paused}")
        
    def _sync_isaac_to_mujoco(self):
        """Synchronize Isaac Gym state to MuJoCo model for visualization"""
        if self.viewer is None or self.mujoco_model is None:
            return
            
        # Get Isaac Gym state
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        
        # For now, just animate the first environment
        if hasattr(self, '_rigid_body_pos') and hasattr(self, '_rigid_body_rot'):
            # Update MuJoCo model based on Isaac Gym state
            # This is a simplified mapping - you may need to adjust based on your specific model
            if len(self._rigid_body_pos) > 0:
                root_pos = self._rigid_body_pos[0, 0]  # First environment, root body
                root_quat = self._rigid_body_rot[0, 0]  # First environment, root body
                
                # Set root position
                self.mujoco_data.qpos[0] = root_pos[0].item()  # x
                self.mujoco_data.qpos[1] = root_pos[1].item()  # y
                self.mujoco_data.qpos[2] = root_pos[2].item()  # z
                
                # Set root orientation (convert from quaternion to euler if needed)
                self.mujoco_data.qpos[3] = root_quat[0].item()  # qw
                self.mujoco_data.qpos[4] = root_quat[1].item()  # qx
                self.mujoco_data.qpos[5] = root_quat[2].item()  # qy
                self.mujoco_data.qpos[6] = root_quat[3].item()  # qz

    # set gravity based on up axis and return axis index
    def set_sim_params_up_axis(self, sim_params, axis):
        if axis == 'z':
            sim_params.up_axis = gymapi.UP_AXIS_Z
            sim_params.gravity.x = 0
            sim_params.gravity.y = 0
            sim_params.gravity.z = -9.81
            return 2
        return 1

    def create_sim(self, compute_device, graphics_device, physics_engine, sim_params):
        sim = self.gym.create_sim(compute_device, graphics_device, physics_engine, sim_params)
        if sim is None:
            print("*** Failed to create sim")
            quit()

        return sim

    def step(self, actions):
        if self.dr_randomizations.get('actions', None):
            actions = self.dr_randomizations['actions']['noise_lambda'](actions)
        # apply actions
        self.pre_physics_step(actions)

        # step physics and render each frame
        self._physics_step()

        # to fix!
        if self.device == 'cpu':
            self.gym.fetch_results(self.sim, True)

        # compute observations, rewards, resets, ...
        self.post_physics_step()
        

        if self.dr_randomizations.get('observations', None):
            self.obs_buf = self.dr_randomizations['observations']['noise_lambda'](self.obs_buf)

    def get_states(self):
        return self.states_buf

    def _clear_recorded_states(self):
        pass

    def _record_states(self):
        pass

    def _write_states_to_file(self, file_name):
        pass

    def setup_video_client(self):
        loop = asyncio.new_event_loop()  # <-- create new loop in this thread here
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self.video_stream())
        loop.run_forever()

    def setup_talk_client(self):
        loop = asyncio.new_event_loop()  # <-- create new loop in this thread here
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self.talk())
        loop.run_forever()

    #print(URL)
    async def talk(self):
        URL = 'http://klab-cereal.pc.cs.cmu.edu:8080/ws'
        print("Starting websocket client")
        session = aiohttp.ClientSession()
        async with session.ws_connect(URL) as ws:
            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    if msg.data == 'close cmd':
                        await ws.close()
                        break
                    else:
                        print(msg.data)
                        try:
                            msg = json.loads(msg.data)
                            if msg['action'] == 'reset':
                                self.reset()
                            elif msg['action'] == 'start_record':
                                if self.recording:
                                    print("Already recording")
                                else:
                                    self.recording = True
                                    self.recording_state_change = True
                            elif msg['action'] == 'end_record':
                                if not self.recording:
                                    print("Not recording")
                                else:
                                    self.recording = False
                                    self.recording_state_change = True
                            elif msg['action'] == 'set_env':
                                query = msg['query']
                                env_id = query['env']
                                self.viewing_env_idx = int(env_id)
                                print("view env idx: ", self.viewing_env_idx)
                        except:
                            import ipdb
                            ipdb.set_trace()
                            print("error parsing server message")
                elif msg.type == aiohttp.WSMsgType.CLOSED:
                    break
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    break

    #print(URL)
    async def video_stream(self):
        URL = 'http://klab-cereal.pc.cs.cmu.edu:8080/ws'
        print("Starting websocket client")
        session = aiohttp.ClientSession()
        async with session.ws_connect(URL) as ws:
            await ws.send_str("Start")
            while True:
                if "color_image" in self.__dict__ and not self.color_image is None and len(self.color_image.shape) == 3:
                    image = cv2.resize(self.color_image, (800, 450), interpolation=cv2.INTER_AREA)
                    await ws.send_bytes(image.tobytes())
                else:
                    print("no image yet")
                    await asyncio.sleep(1)

    def render(self, sync_frame_time=False):
        if self.viewer:
            # Check if MuJoCo viewer is still open
            if not self.viewer.is_alive:
                print("MuJoCo viewer closed - exiting")
                sys.exit()

            # Synchronize Isaac Gym state to MuJoCo for visualization
            self._sync_isaac_to_mujoco()

            # Handle recording
            if self.recording_state_change:
                if not self.recording:
                    if not flags.server_mode:
                        self.writer.close()
                        del self.writer
                        
                    self._write_states_to_file(self.curr_states_file_name)
                    print(f"============ Video finished writing {self.curr_states_file_name}============")

                else:
                    print(f"============ Writing video ============")
                self.recording_state_change = False

            if self.recording:
                if not flags.server_mode:
                    if flags.no_virtual_display:
                        self.gym.render_all_camera_sensors(self.sim)
                        color_image = self.gym.get_camera_image(self.sim, self.envs[self.viewing_env_idx], self.recorder_camera_handles[self.viewing_env_idx], gymapi.IMAGE_COLOR)
                        self.color_image = color_image.reshape(color_image.shape[0], -1, 4)
                    else:
                        img = self.virtual_display.grab()
                        self.color_image = np.array(img)
                        H, W, C = self.color_image.shape
                        self.color_image = self.color_image[:(H - H % 2), :(W - W % 2), :]

                if not flags.server_mode:
                    if not "writer" in self.__dict__:
                        curr_date_time = datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
                        self.curr_video_file_name = self._video_path % curr_date_time
                        self.curr_states_file_name = self._states_path % curr_date_time
                        if not flags.server_mode:
                            self.writer = imageio.get_writer(self.curr_video_file_name, fps=int(1/self.dt), macro_block_size=None)
                    self.writer.append_data(self.color_image)
                    
                self._record_states()

            # fetch results from Isaac Gym
            if self.device != 'cpu':
                self.gym.fetch_results(self.sim, True)

            # step graphics and render with MuJoCo viewer
            if self.enable_viewer_sync:
                self.gym.step_graphics(self.sim)
                
                # Update MuJoCo visualization
                mujoco.mj_step(self.mujoco_model, self.mujoco_data)
                self.viewer.render()
                
            else:
                # For MuJoCo viewer, we still need to render but without sync
                self.viewer.render()
                
        # Handle headless server mode rendering (Isaac Gym cameras)
        else:
            if flags.server_mode:
                # headless server model only support rendering from one env
                self.gym.fetch_results(self.sim, True)
                self.gym.step_graphics(self.sim)
                self.gym.render_all_camera_sensors(self.sim)
                self.gym.start_access_image_tensors(self.sim)

                # self.gym.get_viewer_camera_handle(self.viewer)
                color_image = self.gym.get_camera_image(self.sim, self.envs[self.viewing_env_idx], self.recorder_camera_handles[self.viewing_env_idx], gymapi.IMAGE_COLOR)

                self.color_image = color_image.reshape(color_image.shape[0], -1, 4)[..., :3]

                if self.recording:
                    self._video_queue.append(self.color_image)
                    self._record_states()

    def get_actor_params_info(self, dr_params, env):
        """Returns a flat array of actor params, their names and ranges."""
        if "actor_params" not in dr_params:
            return None
        params = []
        names = []
        lows = []
        highs = []
        param_getters_map = get_property_getter_map(self.gym)
        for actor, actor_properties in dr_params["actor_params"].items():
            handle = self.gym.find_actor_handle(env, actor)
            for prop_name, prop_attrs in actor_properties.items():
                if prop_name == 'color':
                    continue  # this is set randomly
                props = param_getters_map[prop_name](env, handle)
                if not isinstance(props, list):
                    props = [props]
                for prop_idx, prop in enumerate(props):
                    for attr, attr_randomization_params in prop_attrs.items():
                        name = prop_name + '_' + str(prop_idx) + '_' + attr
                        lo_hi = attr_randomization_params['range']
                        distr = attr_randomization_params['distribution']
                        if 'uniform' not in distr:
                            lo_hi = (-1.0 * float('Inf'), float('Inf'))
                        if isinstance(prop, np.ndarray):
                            for attr_idx in range(prop[attr].shape[0]):
                                params.append(prop[attr][attr_idx])
                                names.append(name + '_' + str(attr_idx))
                                lows.append(lo_hi[0])
                                highs.append(lo_hi[1])
                        else:
                            params.append(getattr(prop, attr))
                            names.append(name)
                            lows.append(lo_hi[0])
                            highs.append(lo_hi[1])
        return params, names, lows, highs

    # Apply randomizations only on resets, due to current PhysX limitations
    def apply_randomizations(self, dr_params):
        # If we don't have a randomization frequency, randomize every step
        rand_freq = dr_params.get("frequency", 1)

        # First, determine what to randomize:
        #   - non-environment parameters when > frequency steps have passed since the last non-environment
        #   - physical environments in the reset buffer, which have exceeded the randomization frequency threshold
        #   - on the first call, randomize everything
        self.last_step = self.gym.get_frame_count(self.sim)
        if self.first_randomization:
            do_nonenv_randomize = True
            env_ids = list(range(self.num_envs))
        else:
            do_nonenv_randomize = (self.last_step - self.last_rand_step) >= rand_freq
            rand_envs = torch.where(self.randomize_buf >= rand_freq, torch.ones_like(self.randomize_buf), torch.zeros_like(self.randomize_buf))
            rand_envs = torch.logical_and(rand_envs, self.reset_buf)
            env_ids = torch.nonzero(rand_envs, as_tuple=False).squeeze(-1).tolist()
            self.randomize_buf[rand_envs] = 0

        if do_nonenv_randomize:
            self.last_rand_step = self.last_step

        param_setters_map = get_property_setter_map(self.gym)
        param_setter_defaults_map = get_default_setter_args(self.gym)
        param_getters_map = get_property_getter_map(self.gym)

        # On first iteration, check the number of buckets
        if self.first_randomization:
            check_buckets(self.gym, self.envs, dr_params)

        for nonphysical_param in ["observations", "actions"]:
            if nonphysical_param in dr_params and do_nonenv_randomize:
                dist = dr_params[nonphysical_param]["distribution"]
                op_type = dr_params[nonphysical_param]["operation"]
                sched_type = dr_params[nonphysical_param]["schedule"] if "schedule" in dr_params[nonphysical_param] else None
                sched_step = dr_params[nonphysical_param]["schedule_steps"] if "schedule" in dr_params[nonphysical_param] else None
                op = operator.add if op_type == 'additive' else operator.mul

                if sched_type == 'linear':
                    sched_scaling = 1.0 / sched_step * \
                        min(self.last_step, sched_step)
                elif sched_type == 'constant':
                    sched_scaling = 0 if self.last_step < sched_step else 1
                else:
                    sched_scaling = 1

                if dist == 'gaussian':
                    mu, var = dr_params[nonphysical_param]["range"]
                    mu_corr, var_corr = dr_params[nonphysical_param].get("range_correlated", [0., 0.])

                    if op_type == 'additive':
                        mu *= sched_scaling
                        var *= sched_scaling
                        mu_corr *= sched_scaling
                        var_corr *= sched_scaling
                    elif op_type == 'scaling':
                        var = var * sched_scaling  # scale up var over time
                        mu = mu * sched_scaling + 1.0 * \
                            (1.0 - sched_scaling)  # linearly interpolate

                        var_corr = var_corr * sched_scaling  # scale up var over time
                        mu_corr = mu_corr * sched_scaling + 1.0 * \
                            (1.0 - sched_scaling)  # linearly interpolate

                    def noise_lambda(tensor, param_name=nonphysical_param):
                        params = self.dr_randomizations[param_name]
                        corr = params.get('corr', None)
                        if corr is None:
                            corr = torch.randn_like(tensor)
                            params['corr'] = corr
                        corr = corr * params['var_corr'] + params['mu_corr']
                        return op(tensor, corr + torch.randn_like(tensor) * params['var'] + params['mu'])

                    self.dr_randomizations[nonphysical_param] = {'mu': mu, 'var': var, 'mu_corr': mu_corr, 'var_corr': var_corr, 'noise_lambda': noise_lambda}

                elif dist == 'uniform':
                    lo, hi = dr_params[nonphysical_param]["range"]
                    lo_corr, hi_corr = dr_params[nonphysical_param].get("range_correlated", [0., 0.])

                    if op_type == 'additive':
                        lo *= sched_scaling
                        hi *= sched_scaling
                        lo_corr *= sched_scaling
                        hi_corr *= sched_scaling
                    elif op_type == 'scaling':
                        lo = lo * sched_scaling + 1.0 * (1.0 - sched_scaling)
                        hi = hi * sched_scaling + 1.0 * (1.0 - sched_scaling)
                        lo_corr = lo_corr * sched_scaling + 1.0 * (1.0 - sched_scaling)
                        hi_corr = hi_corr * sched_scaling + 1.0 * (1.0 - sched_scaling)

                    def noise_lambda(tensor, param_name=nonphysical_param):
                        params = self.dr_randomizations[param_name]
                        corr = params.get('corr', None)
                        if corr is None:
                            corr = torch.randn_like(tensor)
                            params['corr'] = corr
                        corr = corr * (params['hi_corr'] - params['lo_corr']) + params['lo_corr']
                        return op(tensor, corr + torch.rand_like(tensor) * (params['hi'] - params['lo']) + params['lo'])

                    self.dr_randomizations[nonphysical_param] = {'lo': lo, 'hi': hi, 'lo_corr': lo_corr, 'hi_corr': hi_corr, 'noise_lambda': noise_lambda}

        if "sim_params" in dr_params and do_nonenv_randomize:
            prop_attrs = dr_params["sim_params"]
            prop = self.gym.get_sim_params(self.sim)

            if self.first_randomization:
                self.original_props["sim_params"] = {attr: getattr(prop, attr) for attr in dir(prop)}

            for attr, attr_randomization_params in prop_attrs.items():
                apply_random_samples(prop, self.original_props["sim_params"], attr, attr_randomization_params, self.last_step)

            self.gym.set_sim_params(self.sim, prop)

        # If self.actor_params_generator is initialized: use it to
        # sample actor simulation params. This gives users the
        # freedom to generate samples from arbitrary distributions,
        # e.g. use full-covariance distributions instead of the DR's
        # default of treating each simulation parameter independently.
        extern_offsets = {}
        if self.actor_params_generator is not None:
            for env_id in env_ids:
                self.extern_actor_params[env_id] = \
                    self.actor_params_generator.sample()
                extern_offsets[env_id] = 0

        for actor, actor_properties in dr_params["actor_params"].items():
            for env_id in env_ids:
                env = self.envs[env_id]
                handle = self.gym.find_actor_handle(env, actor)
                extern_sample = self.extern_actor_params[env_id]

                for prop_name, prop_attrs in actor_properties.items():
                    if prop_name == 'color':
                        num_bodies = self.gym.get_actor_rigid_body_count(env, handle)
                        for n in range(num_bodies):
                            self.gym.set_rigid_body_color(env, handle, n, gymapi.MESH_VISUAL, gymapi.Vec3(random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)))
                        continue
                    if prop_name == 'scale':
                        attr_randomization_params = prop_attrs
                        sample = generate_random_samples(attr_randomization_params, 1, self.last_step, None)
                        og_scale = 1
                        if attr_randomization_params['operation'] == 'scaling':
                            new_scale = og_scale * sample
                        elif attr_randomization_params['operation'] == 'additive':
                            new_scale = og_scale + sample
                        self.gym.set_actor_scale(env, handle, new_scale)
                        continue

                    prop = param_getters_map[prop_name](env, handle)
                    if isinstance(prop, list):
                        if self.first_randomization:
                            self.original_props[prop_name] = [{attr: getattr(p, attr) for attr in dir(p)} for p in prop]
                        for p, og_p in zip(prop, self.original_props[prop_name]):
                            for attr, attr_randomization_params in prop_attrs.items():
                                smpl = None
                                if self.actor_params_generator is not None:
                                    smpl, extern_offsets[env_id] = get_attr_val_from_sample(extern_sample, extern_offsets[env_id], p, attr)
                                apply_random_samples(p, og_p, attr, attr_randomization_params, self.last_step, smpl)
                    else:
                        if self.first_randomization:
                            self.original_props[prop_name] = deepcopy(prop)
                        for attr, attr_randomization_params in prop_attrs.items():
                            smpl = None
                            if self.actor_params_generator is not None:
                                smpl, extern_offsets[env_id] = get_attr_val_from_sample(extern_sample, extern_offsets[env_id], prop, attr)
                            apply_random_samples(prop, self.original_props[prop_name], attr, attr_randomization_params, self.last_step, smpl)

                    setter = param_setters_map[prop_name]
                    default_args = param_setter_defaults_map[prop_name]
                    setter(env, handle, prop, *default_args)

        if self.actor_params_generator is not None:
            for env_id in env_ids:  # check that we used all dims in sample
                if extern_offsets[env_id] > 0:
                    extern_sample = self.extern_actor_params[env_id]
                    if extern_offsets[env_id] != extern_sample.shape[0]:
                        print('env_id', env_id, 'extern_offset', extern_offsets[env_id], 'vs extern_sample.shape', extern_sample.shape)
                        raise Exception("Invalid extern_sample size")

        self.first_randomization = False

    def pre_physics_step(self, actions):
        raise NotImplementedError

    def _physics_step(self):
        for i in range(self.control_freq_inv):
            self.render()

            if not self.paused and self.enable_viewer_sync:
                self.gym.simulate(self.sim)
        return

    def post_physics_step(self):
        raise NotImplementedError


def get_attr_val_from_sample(sample, offset, prop, attr):
    """Retrieves param value for the given prop and attr from the sample."""
    if sample is None:
        return None, 0
    if isinstance(prop, np.ndarray):
        smpl = sample[offset:offset + prop[attr].shape[0]]
        return smpl, offset + prop[attr].shape[0]
    else:
        return sample[offset], offset + 1
