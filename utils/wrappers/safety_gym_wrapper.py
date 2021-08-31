from safety_gym.envs.engine import Engine
import numpy as np
from mujoco_py import MujocoException

class SafetyGymCustomDynamics(Engine):

    def initialize(self, dyn_callback):
        self.dyn_callback = dyn_callback

    def step(self, action):
        ''' Take a step and return observation, reward, done, and info '''
        action = np.array(action, copy=False)   # Cast to ndarray
        assert not self.done, 'Environment must be reset before stepping.'

        info = {}
        # Set action
        action_range = self.model.actuator_ctrlrange
        ctrl = np.clip(action, action_range[:, 0], action_range[:, 1])
        if self.action_noise:
            ctrl += self.action_noise * self.rs.randn(self.model.nu)

        # Simulate physics forward
        exception = False
        for _ in range(self.rs.binomial(self.frameskip_binom_n, self.frameskip_binom_p)):
            try:
                self.set_mocaps()
                # self.sim.step()  # Physics simulation step
                # TODO: implement dynamcis callback call and set_state

                # state = self.env.sim.get_state
                # qpos = state.qpos
                # qvel = state.qvel
                # qacc = state.qacc
                # dyn_callback(qpos, qvel, ctrl)

            except MujocoException as me:
                print('MujocoException', me)
                exception = True
                break
        if exception:
            self.done = True
            reward = self.reward_exception
            info['cost_exception'] = 1.0
        else:
            self.sim.forward()  # Needed to get sensor readings correct!
        # call dynamics callback with action


