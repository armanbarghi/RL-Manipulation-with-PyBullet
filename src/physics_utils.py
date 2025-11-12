import time
import numpy as np
import pybullet as p
import pybullet_data
import matplotlib.pyplot as plt


class PyBulletSim:
	"""PyBullet physics simulation manager."""
	
	def __init__(self, mode=p.DIRECT):
		"""Initialize the PyBullet simulation."""
		try:
			p.disconnect()
		except:
			pass
		
		if mode == p.DIRECT:
			self.client = p.connect(p.DIRECT, options="--egl")
		else:
			self.client = p.connect(mode)
			
		p.setPhysicsEngineParameter(enableFileCaching=0)
		p.setAdditionalSearchPath(pybullet_data.getDataPath())
		p.resetSimulation()
		p.setGravity(0, 0, -9.8)
		p.setRealTimeSimulation(0)
		p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)
		p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
		p.loadURDF("plane.urdf")

		self.time_step = p.getPhysicsEngineParameters()['fixedTimeStep']

	def step(self, duration=0):
		"""Step the simulation for a given duration."""
		for _ in range(int(duration / self.time_step)):
			p.stepSimulation()

	def run(self):
		"""Keep the GUI simulation running indefinitely."""
		if p.getConnectionInfo(self.client)['connectionMethod'] == p.GUI:
			try:
				while p.isConnected():
					p.stepSimulation()
					time.sleep(self.time_step)
			except KeyboardInterrupt:
				print("Simulation stopped by user.")
		else:
			print("Warning: GUI simulation can only run in GUI mode.")

	def close(self):
		"""Close the simulation."""
		try:
			p.disconnect()
		except:
			pass


class CameraManager:
	"""A camera model for PyBullet simulations and computer vision tasks."""

	def __init__(self, target_pos=[0, 0, 0.5], 
			distance=1, yaw=90, pitch=-50, roll=0, 
			width=640, height=480, fov=60, cam_offset=0.75, cam_height=1):
		"""
		Initializes the camera and computes its matrices.
		"""
		self.width, self.height = width, height
		self.fov = fov

		self.cam_offset = cam_offset
		self.cam_height = cam_height

		# Set the initial view
		self.set_view(target_pos, distance, yaw, pitch, roll)

	def set_view(self, target_pos, distance, yaw, pitch, roll=0):
		"""Sets the camera's view and updates its matrices."""
		self.target_pos = target_pos
		self.distance = distance
		self.yaw = yaw
		self.pitch = pitch
		self.roll = roll

		# Configure the GUI camera visuals
		p.resetDebugVisualizerCamera(
			cameraDistance=self.distance,
			cameraTargetPosition=self.target_pos,
			cameraYaw=self.yaw, cameraPitch=self.pitch
		)
		
		# Store the raw PyBullet matrices (GL convention)
		self.view_matrix = p.computeViewMatrixFromYawPitchRoll(
			self.target_pos, self.distance, self.yaw, self.pitch, self.roll, upAxisIndex=2)
		self.proj_matrix = p.computeProjectionMatrixFOV(
			self.fov, self.width/self.height, 0.01, 100)
		
		## Set all internal matrix attributes (K, V, R, t)
		# Intrinsic Matrix K (from FOV)
		o_x, o_y = self.width / 2, self.height / 2
		f_y = (self.height / 2) / np.tan(np.deg2rad(self.fov) / 2)
		self.K = np.array([[f_y, 0, o_x], [0, f_y, o_y], [0, 0, 1]])

		# View Matrix V (World -> GL Camera)
		self.V = np.array(self.view_matrix).reshape(4, 4, order='F')

		# Inverse Transformation (CV Camera -> World) for unprojection
		V_inv_gl = np.linalg.inv(self.V)
		R_gl_to_world = V_inv_gl[:3, :3]
		self.t = V_inv_gl[:3, 3].reshape(3, 1)
		
		# Adapter to create a CV-to-World rotation matrix for unprojection math
		self.R = R_gl_to_world @ np.diag([1, -1, -1])

	def z_c_calculator(self, u, v, n, p0, h=0.0):
		"""Calculates a pixel's depth by intersecting its ray with a plane."""
		K_inv = np.linalg.inv(self.K)
		ray_cam_cv = K_inv @ [u, v, 1.0]
		numerator = h + (n @ p0) - (n @ self.t.flatten())
		denominator = n @ self.R @ ray_cam_cv
		if abs(denominator) < 1e-6: return None
		return numerator / denominator

	def project_world_to_pixel(self, world_points):
		"""Projects 3D world points to 2D image pixels."""
		world_points_h = np.vstack([np.asanyarray(world_points).T, np.ones(len(world_points))])
		
		# World -> GL Camera -> CV Camera
		camera_points_gl = (self.V @ world_points_h)[:3, :]
		camera_points_cv = np.diag([1, -1, -1]) @ camera_points_gl
		
		# Project to image plane
		image_coords_h = self.K @ camera_points_cv
		
		# Dehomogenize
		u = image_coords_h[0, :] / image_coords_h[2, :]
		v = image_coords_h[1, :] / image_coords_h[2, :]
		return u, v

	def project_pixel_to_world(self, u, v, n, p0, h=0.0):
		"""Unprojects a 2D pixel to a 3D world point on an arbitrary plane."""
		z_c = self.z_c_calculator(u, v, n, p0, h)
		if z_c is None: return None
		
		# Point in CV Camera Frame
		K_inv = np.linalg.inv(self.K)
		point_in_cam_cv = z_c * (K_inv @ [u, v, 1.0])
		
		# CV Camera Frame -> World Frame
		world_point = self.R @ point_in_cam_cv + self.t.flatten()
		return world_point

	def capture_image(self):
		"""Captures and returns an RGB image from the camera's viewpoint."""
		_, _, rgb, _, _ = p.getCameraImage(
			self.width, self.height, 
			viewMatrix=self.view_matrix,
			projectionMatrix=self.proj_matrix, 
			renderer=p.ER_BULLET_HARDWARE_OPENGL,
			shadow=1,
		)

		# Convert to numpy array and ensure correct data type
		image = np.array(rgb, dtype=np.uint8)
		image = image.reshape((self.height, self.width, 4))[:, :, :3]  # Remove alpha channel

		return image

	def show_img(self, image, ax=None, title=''):
		"""Display image."""
		ax_none = False
		if ax is None:
			ax_none = True
			fig, ax = plt.subplots(figsize=(self.width / 100, self.height / 100))

		ax.imshow(image)
		ax.set_title(title)
		ax.axis('off')

		if ax_none:
			plt.show()
