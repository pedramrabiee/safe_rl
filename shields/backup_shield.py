from shields.base_shield import BaseSheild
import torch

class BackupShield(BaseSheild):
    """
    Soft-Minimum and Soft-Maximum Barrier Functions Based on https://arxiv.org/abs/2305.10620

    Steps to implementing BackupShield
    0. You need the backup set and safe set     (AMIR)
    00. Clean safe set class                    (PEDRAM) Done
    000. Backup set and safe sets are environment specific and should be implemented in the environment specific files (AMIR)
    1. Make custom controller for the environment (Torch backup control)
    2. Make torch dynamics for the environment
    3. import dynamics and controller in the initialize method
    4. Think about class initialization and where to pass the controller and dynamics to this class
    5. Make helper function for computing sensitivity coefficient matrix
    6. Make dynamics sensitivity vector by flattening and converting to numpy
    7. Forward propagate using scipy (Look for other method... maybe using a torch solver be faster and you don't need torch to numpy conversion)
    8. Compute softmin gradients
    9. Implement CBF-QP Shield class
    10. Use filter method from the CBF-QP shield class to filter the action
    11. Make BackupShield Trainer (just implement _train to pass)
    """

    def initialize(self, params, init_dict=None):
        raise NotImplementedError

    # Safe set h
    # backup set
    # backup control
    # dynamics

    def _compute_backup_barrier(self):
        # forward propagate: call _make_dynamics_sensitivity
        # use ode (TODO: find equivalent of ode45)
        # https://github.com/rtqichen/torchdiffeq
        raise NotImplementedError


    def _compute_grad_along_traj(self):
        raise NotImplementedError