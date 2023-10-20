def get_env_params(env):
    data = env.robot.sim.data
    com_loc = data.get_body_xipos('robot')[0]   # alternatively: env.robot.sim.model.body_ipos[1][0]
    cinert = data.cinert[1]     # the second body is the robot
    I = cinert[1]
    m = cinert[-1]
    damping = env.robot.sim.model.dof_damping
    cs = damping[0]
    cr = damping[-1]
    return dict(m=m, I=I, com_loc=com_loc, cs=cs, cr=cr)