import gym
from gym import spaces
import numpy as np
from PIL import Image
from gym_duckietown.simulator import NotInLane

class ResizeWrapper(gym.ObservationWrapper):
    """
    Downscale image to faster traing time from gym_duckietown
    """
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


class CropImageWrapper(gym.ObservationWrapper):
    """
    Crop the sky from camera for faster learning
    """
    def __init__(self, env=None, top_margin_divider=3):
        super(CropImageWrapper, self).__init__(env)
        #Save size image and how much to crop from top
        img_height, img_width, depth = self.observation_space.shape
        top_margin = img_height // top_margin_divider
        img_height = img_height - top_margin
        self.top_margin = top_margin
        self.img_width = img_width
        self.img_height = img_height

        self.observation_space = spaces.Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0],
            (img_height, img_width, depth),
            dtype=self.observation_space.dtype)

    #Crop from the observation
    def observation(self, observation):
        return observation[int(self.top_margin):int(self.top_margin + self.img_height),0:int(self.img_width)]

class NormalizeWrapper(gym.ObservationWrapper):
    """
    Normalize camera image to 0-1 from gym_duckietown
    """
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

class ActionWrapper(gym.ActionWrapper):
    """
    This is needed because at max speed the duckie can't turn anymore from gym_duckietown
    """
    def __init__(self, env):
        super(ActionWrapper, self).__init__(env)

    def action(self, action):
        action_ = [0.25, action[1]]
        return action_

class DiscreteWrapper(gym.ActionWrapper):
    """
    Duckietown environment with discrete actions (left, right, forward)
    instead of continuous control
    """

    def __init__(self, env):
        gym.ActionWrapper.__init__(self, env)
        self.action_space = spaces.Discrete(3)

    def action(self, action):
        # Go forward
        if action == 0:
            vels = [0.5, 0.0]
        # Turn right
        elif action == 1:
            vels = [0.4, -1.0]
        # Turn left
        elif action == 2:
            vels = [0.4, +1.0]
        else:
            assert False, "unknown action"
        return np.array(vels)

    
class DtRewardWrapper(gym.RewardWrapper):
    """
    Rewards calculator
    """

    def __init__(self, env):
        super(DtRewardWrapper, self).__init__(env)
        self.prev_pos = None

    def reward(self, reward):
        my_reward = -1000

        # Get current parameters/positions
        pos = self.cur_pos
        angle = self.cur_angle
        prev_pos = self.prev_pos
        self.prev_pos = pos

        #Init state no reward
        if prev_pos is None:
            return 0

        # Compute parameters
        curve_point, curve_tangent = self.closest_curve_point(pos, angle)
        prev_curve_point, prev_curve_tangent = self.closest_curve_point(prev_pos, angle)
        if curve_point is None or prev_curve_point is None:
            return 0

        # Calculate the distance of the length of curves        
        diff = curve_point - prev_curve_point
        dist = np.linalg.norm(diff)

        # Return if out off the lane
        lane_pos = self.get_lane_pos2(pos, self.cur_angle)
        if lane_pos is NotInLane:
            return my_reward

        # Not in the right lane
        if lane_pos.dist < -0.06:
            return my_reward
        # Bad turning/ wrong direction
        if np.dot(curve_tangent, curve_point - prev_curve_point) < 0:
            return my_reward
        #Reverse
        if abs(lane_pos.angle_deg) > 120:
            return my_reward 

        # Normalize rewards from each domain
        lane_center_dist_reward = np.interp(abs(lane_pos.dist), (0, 0.05), (1, 0))
        lane_center_angle_reward = np.interp(abs(lane_pos.angle_deg), (0, 180), (1, -1))
        #speed_reward = np.interp(np.abs(lane_pos.angle_deg), (0, 0.15), (0, 1))

        # Calculate reward based on travel distance, 
        # distance from the middle of the lane
        # the curve to the middle of the lane
        my_reward = 100 * dist + 1 * lane_center_dist_reward + 1 * lane_center_angle_reward
        #my_reward = 1 * lane_pos.dist - 10 * abs(lane_pos.angle_rad)

        return my_reward
