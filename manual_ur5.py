import mujoco
import mujoco.viewer
import numpy as np
import ikpy.chain
from simple_pid import PID
import time

class MJ_Controller_Modern:
    def __init__(self, xml_path, urdf_path):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        
        # Discover Joint Mapping
        self.gripper_actuator_id = 6 # Default fallback
        for i in range(self.model.nu):
            joint_id = self.model.actuator_trnid[i, 0]
            joint_name = self.model.joint(joint_id).name
            if 'finger' in joint_name or 'robotiq' in joint_name:
                self.gripper_actuator_id = i
                print(f"Confirmed: Actuator {i} is the Gripper ({joint_name})")
    
        self.ee_chain = ikpy.chain.Chain.from_urdf_file(urdf_path, 
                            active_links_mask=[False, True, True, True, True, True, True, False])
        
        # Increased P-gain for Gripper (Index 6) to ensure it can hold the object
        p_scale, d_scale, sample_time = 3.0, 0.1, 0.0001
        self.controllers = [
            PID(21.0, 0, 0.3, setpoint=0), # Arm 1
            PID(30.0, 0, 0.3, setpoint=-1.57), # Arm 2
            PID(15.0, 0, 0.15, setpoint=1.57), # Arm 3
            PID(21.0, 0, 0.03, setpoint=-1.57), # Arm 4
            PID(15.0, 0, 0.03, setpoint=-1.57), # Arm 5
            PID(15.0, 0, 0.03, setpoint=0), # Arm 6
            PID(40.0, 0, 1.0, setpoint=0)   # Gripper (Index 6)
        ]
        # Add output_limits to the gripper PID (Index 6)
        # Inside your MJ_Controller_Modern.__init__
        self.controllers[6] = PID(7.5, 0.0, 0.0, setpoint=0.0, output_limits=(-1, 1), sample_time=0.0001)
        
        # Set a legal starting pose
        self.data.qpos[1] = -1.57
        mujoco.mj_forward(self.model, self.data)
        self.current_target_joint_values = np.array([c.setpoint for c in self.controllers])
    def move_group_to_joint_target(self, target, tolerance=0.05, max_steps=5000, viewer=None):
        for i in range(6):
            self.current_target_joint_values[i] = target[i]
            self.controllers[i].setpoint = target[i]
        steps = 0
        while steps < max_steps:
            for i in range(7):
                self.data.ctrl[i] = self.controllers[i](self.data.qpos[i])
            mujoco.mj_step(self.model, self.data)
            time.sleep(self.model.opt.timestep)
            if viewer and steps % 20 == 0: viewer.sync()
            
            deltas = np.abs(self.current_target_joint_values[:6] - self.data.qpos[:6])
            if np.max(deltas) < tolerance:
                break
            steps += 1
            
    def grasp(self, viewer=None):
        """Attempts a grasp with the exact MJ_Controller timing."""
        # Set target to -0.4
        self.controllers[6].setpoint = -0.4
        # Execute for roughly 300 steps (at 2ms timestep, this is ~600ms)
        self.stay(600, viewer=viewer)
        
    def ik(self, ee_position):
        # Subtract base position (Replicating original logic)
        base_pos = self.data.body('base_link').xpos
        ee_position_base = ee_position - base_pos
        
        # The original offset to find the grasp center relative to ee_link
        gripper_center_target = ee_position_base + [0, -0.005, 0.16]
        
        initial_position = [0] + list(self.data.qpos[:6]) + [0]
        joint_angles = self.ee_chain.inverse_kinematics(
            gripper_center_target, [0, 0, -1], orientation_mode='X', 
            initial_position=initial_position
        )
        return joint_angles[1:7]
    def move_ee(self, ee_position, viewer=None):
        joint_angles = self.ik(ee_position)
        if joint_angles is not None:
            self.move_group_to_joint_target(joint_angles, viewer=viewer)
            
    def stay(self, duration, viewer=None):
        start = time.time()
        while (time.time() - start) * 1000 < duration:
            # Use self.model.nu to ensure ALL actuators (including gripper) are updated
            for i in range(self.model.nu):
                self.data.ctrl[i] = self.controllers[i](self.data.qpos[i])
            mujoco.mj_step(self.model, self.data)
            if viewer: viewer.sync()
            
    def open_gripper(self, viewer=None):
    # Mimicking MJ_Controller.open_gripper(half=False)
        print("Opening gripper to 0.4 rad...")
        self.controllers[6].setpoint = 0.9 
        # The provided script uses move_group_to_joint_target which iterates 
        # until tolerance is met. For your script, 'stay' approximates this.
        self.stay(1000, viewer=viewer) 

    def close_gripper(self, viewer=None):
        print("Executing Tight Close (-0.1 rad)...")
        # 1. Set the target to -0.4 as per the reference script
        self.controllers[6].setpoint = -1.5
        
        # 2. Instead of a raw signal loop, let the PID work over time
        # This allows the fingers to settle without exploding
        for _ in range(500):
            # Calculate PID output
            current_qpos = self.data.qpos[self.model.actuator_trnid[6, 0]]
            ctrl_signal = self.controllers[6](current_qpos)
            self.data.ctrl[6] = ctrl_signal
            
            mujoco.mj_step(self.model, self.data)
            if viewer:
                viewer.sync() # Give it time to physically close
# --- RUN LOGIC ---
def run_pick_and_place():
    XML = "UR5+gripper/UR5gripper_v3.xml" # Use your actual filename
    URDF = "UR5+gripper/ur5_gripper.urdf"
    ctrl = MJ_Controller_Modern(XML, URDF)
    
    with mujoco.viewer.launch_passive(ctrl.model, ctrl.data) as viewer:
        # Give time for the objects to fall/settle
        ctrl.stay(2000, viewer=viewer)
        # Target object_10 (the first box)
        try:
            target_pos = ctrl.data.body('box_1').xpos.copy()
        except:
            target_pos = np.array([0.0, -0.6, 0.95])
        # 1. Approach from above
        ctrl.move_ee([target_pos[0], target_pos[1], target_pos[2] + 0.1], viewer=viewer)
        ctrl.open_gripper(viewer=viewer)
        
        # 2. Descend (The 3-finger gripper is large, so stay a bit higher)
        grasp_height = target_pos[2] - 0.01
        ctrl.move_ee([target_pos[0], target_pos[1], grasp_height], viewer=viewer)
        
        # 3. Grasp
        ctrl.close_gripper(viewer=viewer)
        
        # 4. Lift
        ctrl.move_ee([target_pos[0], target_pos[1], target_pos[2] + 0.4], viewer=viewer)
        
if __name__ == "__main__":
    run_pick_and_place()