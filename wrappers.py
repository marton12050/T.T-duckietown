import gym
from gym import spaces
import numpy as np
from PIL import Image
from gym_duckietown.simulator import Simulator
from gym_duckietown.simulator import NotInLane


class ResizeWrapper(gym.ObservationWrapper):

    def __init__(self, env=None, shape=(84, 84, 3)):
        super(ResizeWrapper, self).__init__(env)
        self.shape = shape
        self.observation_space = spaces.Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0],
            self.shape,
            dtype=self.observation_space.dtype
        )
        
    def observation(self, observation):
        return np.array(Image.fromarray(obj=observation).resize(size=self.shape[:2]))

	
# Crop the sky from camera for faster learning
class CropImageWrapper(gym.ObservationWrapper):
    def __init__(self, env=None, top_margin_divider=3):
        super(CropImageWrapper, self).__init__(env)
        img_height, img_width, depth = self.observation_space.shape
        top_margin = img_height // top_margin_divider
        img_height = img_height - top_margin
        self.roi = [0, top_margin, img_width, img_height]

        self.observation_space = spaces.Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0],
            (img_height, img_width, depth),
            dtype=self.observation_space.dtype)

    def observation(self, observation):
        r = self.roi
        observation = observation[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]
        return observation

# Normalize camera image to 0-1
class NormalizeWrapper(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(NormalizeWrapper, self).__init__(env)
        self.obs_lo = self.observation_space.low[0, 0, 0]
        self.obs_hi = self.observation_space.high[0, 0, 0]
        obs_shape = self.observation_space.shape
        self.observation_space = spaces.Box(0.0, 1.0, obs_shape, dtype=np.float32)

    def observation(self, obs):
        if self.obs_lo == 0.0 and self.obs_hi == 1.0:
            return obs
        else:
            return (obs - self.obs_lo) / (self.obs_hi - self.obs_lo)
            
# this is needed because at max speed the duckie can't turn anymore
class ActionWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super(ActionWrapper, self).__init__(env)

    def action(self, action):
        action_ = [action[0] * 0.8, action[1]]
        return action_
        
class ImgWrapper(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(ImgWrapper, self).__init__(env)
        obs_shape = self.observation_space.shape
        self.observation_space = spaces.Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0],
            [obs_shape[2], obs_shape[0], obs_shape[1]],
            dtype=self.observation_space.dtype,
        )

    def observation(self, observation):
        return observation.transpose(2, 0, 1)


class DtRewardWrapper(gym.RewardWrapper):
    '''
    
    '''
    def __init__(self, env):
        super(DtRewardWrapper, self).__init__(env)
        self.prev_pos = None

    def reward(self, reward):
        my_reward = 0
        
        #Gettin current parameters/positions
        pos = self.cur_pos
        angle = self.cur_angle
        prev_pos = self.prev_pos
        self.prev_pos = pos
        
        if prev_pos is None:
            return my_reward
        
        #Compute parameters
        curve_point, curve_tangent = self.closest_curve_point(pos, angle)
        prev_curve_point, prev_curve_tangent = self.closest_curve_point(prev_pos, angle)
        if curve_point is None or prev_curve_point is None:
        	return my_reward

        # Calculate the distance of the length of curves        
        diff = curve_point - prev_curve_point
        dist = np.linalg.norm(diff)    

        # Return if out off the lane
        lane_pos = self.get_lane_pos2(pos, self.cur_angle)
        if lane_pos is NotInLane:
                return my_reward
        
        #Not in lane 
        if lane_pos.dist < -0.05:
        	return my_reward
        # Bad turning/ wrong direction
        if np.dot(curve_tangent, curve_point - prev_curve_point) < 0:
            return my_reward

        # Normalize rewards from each domain
        lane_center_dist_reward = np.interp(np.abs(lane_pos.dist), (0, 0.04), (1, 0))
        lane_center_angle_reward = np.interp(np.abs(lane_pos.angle_deg), (0, 180), (1,-1))
        speed_reward = np.interp(np.abs(lane_pos.angle_deg), (0, 0.15), (0,1))

        # Calculate reward based on travel distance, 
        # distance from the middle of the lane
        # the curve to the middle of the lane
        # and the speed at which this happen
        my_reward = 100*dist + 10*lane_center_dist_reward + 10*lane_center_angle_reward + 5*speed_reward

        return my_reward
        

