import time
import numpy as np
import pybullet as p
import pybullet_data


class Joint(object):
	"""Represents a robot joint with position control capabilities."""

	def __init__(self, robot_id, joint_id, limits):
		self.robot_id = robot_id
		self.joint_id = joint_id
		self.limits = limits

	def get_position(self):
		"""Get current joint position."""
		joint_state = p.getJointState(self.robot_id, self.joint_id)
		return joint_state[0]

	def set_position(self, position, max_force=100.):
		"""Set joint to target position."""
		p.setJointMotorControl2(
			self.robot_id,
			self.joint_id,
			controlMode=p.POSITION_CONTROL,
			targetPosition=position,
			force=max_force,
			positionGain=0.3,
			velocityGain=1.0
		)


class RobotController:
	"""Controls robot in PyBullet simulation environment."""

	def __init__(self, model_path, scale=1, initial_base_pos=[0, 0, 0],
				 use_fixed_base=True, table_size=[1, 1]):
		self.robot_id = p.loadURDF(
			model_path,
			initial_base_pos,
			globalScaling=scale,
			useFixedBase=use_fixed_base
		)
	
		self._time_step = p.getPhysicsEngineParameters()['fixedTimeStep']
		self._gripper_joint_id = 6
		self._left_finger_joint_id = 9
		self._right_finger_joint_id = 10
		self._end_effector_link_id = 11
		
		dis_from_table = abs(initial_base_pos[0]) - table_size[0] / 2
		self.workspace_x_limits = [
			initial_base_pos[0], 
			table_size[0] / 2 + dis_from_table
		]
		self.workspace_y_limits = [
			-table_size[1] / 2 - dis_from_table,
			table_size[1] / 2 + dis_from_table
		]

		# Speed settings (m/s or rad/s)
		self.arm_speed = 0.1 # m/s for position control
		self.gripper_speed = 0.5 # rad/s for position control
		self.ee_velocity = 0.2 # m/s for velocity control
		self.gripper_rot_velocity = 0.5 # rad/s for velocity control

		# Predefined duration for each waypoint in a movement
		self.waypoint_duration = 0.005
		
		# Internal state for holding objects
		self._grasped_object_id = None
		self._grasp_constraint_id = None
		
		self.initial_poses = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]
		self.load_model()

	def load_model(self):
		"""Load robot model and initialize joint objects."""
		joints = {}
		for i in range(p.getNumJoints(self.robot_id)):
			joint_info = p.getJointInfo(self.robot_id, i)
			joint_limits = {
				'lower': joint_info[8], 
				'upper': joint_info[9],
				'force': joint_info[10]
			}
			joints[i] = Joint(self.robot_id, i, joint_limits)
			# print(joint_info)
		
		# self.rtb_model.qlim = np.array([[joints[i].limits['lower'], joints[i].limits['upper']] for i in range(7)]).T
		self.joints = joints
		self._left_finger = self.joints[self._left_finger_joint_id]
		self._right_finger = self.joints[self._right_finger_joint_id]

		self.reset_joints()

		self.wrist_neutral = self.joints[self._gripper_joint_id].get_position()   # ~ 0.785 rad

	def simulate_step(self, duration=0):
		"""Run simulation for specified duration."""
		for _ in range(int(duration / self._time_step)):
			p.stepSimulation()

	def reset_joints(self):
		"""Reset all joints to initial positions."""
		for joint_id in range(len(self.initial_poses)):
			p.resetJointState(self.robot_id, joint_id, self.initial_poses[joint_id])

	def get_pos(self):
		"""Get current joint positions."""
		return [self.joints[i].get_position() for i in range(7)]

	def get_ee_pose(self):
		"""Get end-effector position and orientation."""
		ee_pos, ee_orn = p.getLinkState(self.robot_id, self._end_effector_link_id)[:2]
		return ee_pos, ee_orn

	def open_gripper(self, max_limit=True, duration=0.5, steps=20):
		"""Open gripper smoothly to specified limit."""
		# Get current finger positions
		current_left = self._left_finger.get_position()
		current_right = self._right_finger.get_position()
		
		# Determine target positions
		if max_limit:
			target_left = self._left_finger.limits['upper']
			target_right = self._right_finger.limits['upper']
		else:
			target_left = self._left_finger.limits['upper'] / 2
			target_right = self._right_finger.limits['upper'] / 2
		
		# Create smooth trajectory
		left_trajectory = np.linspace(current_left, target_left, steps)
		right_trajectory = np.linspace(current_right, target_right, steps)
		
		# Execute smooth movement
		step_duration = duration / steps
		for i in range(steps):
			self._left_finger.set_position(left_trajectory[i])
			self._right_finger.set_position(right_trajectory[i])
			self.simulate_step(step_duration)

	def close_gripper(self, duration=0.5, steps=20):
		"""Close gripper smoothly."""
		# Get current finger positions
		current_left = self._left_finger.get_position()
		current_right = self._right_finger.get_position()
		
		# Target positions (fully closed)
		target_left = self._left_finger.limits['lower']
		target_right = self._right_finger.limits['lower']
		
		# Create smooth trajectory
		left_trajectory = np.linspace(current_left, target_left, steps)
		right_trajectory = np.linspace(current_right, target_right, steps)
		
		# Execute smooth movement
		step_duration = duration / steps
		for i in range(steps):
			self._left_finger.set_position(left_trajectory[i])
			self._right_finger.set_position(right_trajectory[i])
			self.simulate_step(step_duration)

	def rotate_gripper_yaw(self, yaw_angle, duration=0.5):
		"""Rotates the gripper to a target yaw angle in the world frame."""
		# Convert yaw angle to radians
		yaw_angle = np.deg2rad(yaw_angle)

		# Get the end-effector (EE) rotation matrix from its quaternion
		q = self.get_ee_pose()[1]
		R = np.array(p.getMatrixFromQuaternion(q)).reshape(3, 3)

		# Determine the EE's local Z-axis direction in the world frame
		z_world = R[:, 2]

		# Check if the EE is pointing mostly down (z < 0) to determine the mapping sign
		sign = 1.0 if z_world[2] >= 0 else -1.0

		# Get the current angle of the gripper joint relative to its neutral position
		q_raw = self.joints[self._gripper_joint_id].get_position()
		rel0 = q_raw - self.wrist_neutral

		# Calculate the shortest angular distance (delta) to the target yaw
		_, _, glob_yaw = p.getEulerFromQuaternion(q)
		dglob = (yaw_angle - glob_yaw + np.pi) % (2 * np.pi) - np.pi

		# Map the global yaw delta to the required joint-space delta
		djoint = sign * dglob

		# Calculate duration and number of steps for a smooth motion
		if abs(dglob) < 1e-6: # No rotation needed
			return
		duration = abs(dglob) / self.gripper_speed
		num_steps = max(int(duration / self.waypoint_duration), 1)
		step_duration = duration / num_steps

		# Get current positions of all arm joints to hold them steady
		current_arm_joints = self.get_pos()

		# Execute the rotation by interpolating the joint angle
		for i in range(1, num_steps + 1):
			# Hold other arm joints at their current positions
			self.set_joint_positions(current_arm_joints)

			# Calculate the target joint angle for this step
			rel = rel0 + djoint * (i / num_steps)
			q_cmd = rel + self.wrist_neutral
			
			# Clip the command to respect the joint's physical limits
			q_cmd = np.clip(q_cmd,
							self.joints[self._gripper_joint_id].limits['lower'],
							self.joints[self._gripper_joint_id].limits['upper'])
			
			# Send the command for the gripper joint and step the simulation
			self.joints[self._gripper_joint_id].set_position(q_cmd)
			self.simulate_step(step_duration)

	def inverse_kinematics(self, target_pose, orientation: list, max_iterations: int=100) -> np.ndarray:
		"""Calculate joint angles for target pose using inverse kinematics."""
		joint_indices = list(range(7))
		lower_limits = [self.joints[i].limits['lower'] for i in joint_indices]
		upper_limits = [self.joints[i].limits['upper'] for i in joint_indices]
		joint_ranges = [upper - lower for upper, lower in zip(upper_limits, lower_limits)]

		q = p.calculateInverseKinematics(
			self.robot_id,
			self._end_effector_link_id,
			targetPosition=target_pose,
			targetOrientation=p.getQuaternionFromEuler(orientation),
			jointDamping=[0.1]*len(joint_indices),
			lowerLimits=lower_limits,
			upperLimits=upper_limits,
			restPoses=self.initial_poses,
			jointRanges=joint_ranges,
			maxNumIterations= max_iterations,
		)
		return q[:7]

	def generate_linear_path(self, start_pose, end_pose, num_waypoints: int, orientation: list) -> np.ndarray:
		"""Generate linear path between two poses using Cartesian interpolation and IK."""
		# Extract start/end positions
		start_pos = np.array(start_pose).reshape(3)
		end_pos = np.array(end_pose).reshape(3)

		# Linear interpolation in Cartesian space
		positions = np.linspace(start_pos, end_pos, num_waypoints)

		path = np.zeros((num_waypoints, 7))
		for i in range(num_waypoints):
			try:
				path[i] = self.inverse_kinematics(positions[i], orientation)
			except Exception as e:
				print(f"IK failed at point {i+1}/{num_waypoints}: {e}")
				# truncate path to successful points
				path = path[:i]
				break

		return path

	def move_to_position(self, target_position):
		"""Move end-effector to target position with appropriate orientation."""
		# grab current EE quaternion and extract yaw
		position, ee_quat = self.get_ee_pose()
		_, _, current_yaw = p.getEulerFromQuaternion(ee_quat)
		
		# do a straight-line move keeping that same yaw
		target_orientation = [np.pi, 0, current_yaw]
		
		# Calculate total duration and number of waypoints
		distance = np.linalg.norm(np.array(target_position) - np.array(position))
		if distance == 0:
			return
		duration = distance / self.arm_speed
		num_waypoints = max(int(duration / self.waypoint_duration), 2)

		# Generate Cartesian straight-line path and convert each waypoint via IK
		path = self.generate_linear_path(
			position, 
			target_position, 
			num_waypoints=num_waypoints, 
			orientation=target_orientation
		)
		for q in path:
			# If generate_linear_path returned a truncated path, ensure shape matches
			if q.shape[0] != 7:
				continue
			self.set_joint_positions(q)
			self.simulate_step(self.waypoint_duration)

	def set_joint_positions(self, joint_angles):
		"""Set all joint positions simultaneously."""
		for i in range(7):
			self.joints[i].set_position(joint_angles[i])

	def pick_object(self, obj_id, target_yaw=None, approach_height=0.3, grasp_height=0.1):
		"""Pick up object by grasping and creating constraint."""
		# Check if already holding an object
		if self._grasped_object_id is not None:
			print(f"Robot is already holding object {self._grasped_object_id}")
			return False

		# self.open_gripper()
		obj_pos, obj_orn = p.getBasePositionAndOrientation(obj_id)
		
		# Approach object from above
		self.move_to_position([obj_pos[0], obj_pos[1], obj_pos[2] + approach_height])

		# Rotate gripper to target yaw if specified
		if target_yaw is not None:
			self.rotate_gripper_yaw(target_yaw)

		# Move down to grasp height
		self.move_to_position([obj_pos[0], obj_pos[1], obj_pos[2] + grasp_height])

		# Create the fixed constraint with relative pose
		ee_pos, ee_orn = p.getLinkState(self.robot_id, self._end_effector_link_id)[:2]
		obj_pos, obj_orn = p.getBasePositionAndOrientation(obj_id)

		# Compute object pose in end-effector frame: T_obj_ee = inv(T_ee_w) * T_obj_w
		inv_ee_pos, inv_ee_orn = p.invertTransform(ee_pos, ee_orn)
		rel_pos, rel_orn  = p.multiplyTransforms(inv_ee_pos, inv_ee_orn, obj_pos, obj_orn)

		constraint_id = p.createConstraint(
			parentBodyUniqueId    = self.robot_id,
			parentLinkIndex       = self._end_effector_link_id,
			childBodyUniqueId     = obj_id,
			childLinkIndex        = -1,
			jointType             = p.JOINT_FIXED,
			jointAxis             = [0, 0, 0],
			parentFramePosition   = rel_pos,
			parentFrameOrientation= rel_orn,
			childFramePosition    = [0, 0, 0],
			childFrameOrientation = [0, 0, 0, 1]      # No extra rotation in object’s own frame
		)
		
		# Store internal state
		self._grasped_object_id = obj_id
		self._grasp_constraint_id = constraint_id
		
		# self.close_gripper()
		self.move_to_position([obj_pos[0], obj_pos[1], obj_pos[2] + approach_height])
		
		return True

	def place_object(self, target_position, target_yaw=None, approach_height=0.3, place_height=0.1):
		"""Place held object at target position."""
		# Check if holding an object
		if self._grasped_object_id is None:
			print("Robot is not holding any object to place")
			return False
		
		# Approach target position from above
		self.move_to_position([target_position[0], target_position[1], target_position[2] + approach_height])

		# Rotate gripper to target yaw if specified
		if target_yaw is not None:
			self.rotate_gripper_yaw(target_yaw)
		
		# Move to place height with target yaw orientation for the EE
		self.move_to_position([target_position[0], target_position[1], target_position[2] + place_height])
		
		# self.open_gripper()
		
		# Remove constraint and clear internal state
		if self._grasp_constraint_id is not None:
			p.removeConstraint(self._grasp_constraint_id)
			
		self._grasped_object_id = None
		self._grasp_constraint_id = None

		self.move_to_position([target_position[0], target_position[1], target_position[2] + approach_height])

		# Reset gripper to neutral position
		# self.rotate_gripper_yaw(0)
		
		return True

	def move_ee_velocity(self, linear_velocity):
		"""
		Control the end-effector using a target linear velocity, while keeping
		orientation constant.
		"""
		# Get the state for all movable joints
		num_joints = p.getNumJoints(self.robot_id)
		movable_joint_indices = [i for i in range(num_joints) if p.getJointInfo(self.robot_id, i)[2] != p.JOINT_FIXED]
		joint_states = p.getJointStates(self.robot_id, movable_joint_indices)
		joint_positions = [state[0] for state in joint_states]
		zero_vec = [0] * len(joint_positions)

		# Calculate the full 6D Jacobian (linear and angular)
		jac_t, jac_r = p.calculateJacobian(
			self.robot_id,
			self._end_effector_link_id,
			[0, 0, 0], # Point on link to measure, in link frame
			joint_positions,
			zero_vec,
			zero_vec
		)

		# We only want to control the 7 arm joints
		jac_t_arm = np.array(jac_t)[:, :7]
		jac_r_arm = np.array(jac_r)[:, :7]
		
		# Stack the Jacobians to create a 6x7 matrix for 6D control
		full_jacobian = np.vstack((jac_t_arm, jac_r_arm))

		# Define the 6D target velocity: [vx, vy, vz, wx, wy, wz]
		# We want the commanded linear velocity and ZERO angular velocity
		target_velocity_6d = np.hstack((linear_velocity, [0, 0, 0]))

		# Use the pseudo-inverse of the full Jacobian to find joint velocities
		try:
			j_inv = np.linalg.pinv(full_jacobian)
			joint_velocities = j_inv.dot(target_velocity_6d)

			# Set joint motor control to VELOCITY_CONTROL for the 7 arm joints
			for i in range(7):
				p.setJointMotorControl2(
					self.robot_id,
					i,
					controlMode=p.VELOCITY_CONTROL,
					targetVelocity=joint_velocities[i],
					force=self.joints[i].limits['force']
				)
		except np.linalg.LinAlgError:
			# This can happen if the robot is in a singular configuration
			for i in range(7):
				p.setJointMotorControl2(
					self.robot_id,
					i,
					controlMode=p.VELOCITY_CONTROL,
					targetVelocity=0,
					force=self.joints[i].limits['force']
				)

	def rotate_gripper_velocity(self, velocity):
		"""Control the gripper rotation using a target velocity."""
		p.setJointMotorControl2(
			self.robot_id,
			self._gripper_joint_id,
			controlMode=p.VELOCITY_CONTROL,
			targetVelocity=velocity,
			force=self.joints[self._gripper_joint_id].limits['force']
		)
