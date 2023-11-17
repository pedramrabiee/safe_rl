from envs_utils.safety_gym.point.point_configs import env_config

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


def moment_of_inertia(density, r_sphere, l_cube, com_sphere, com_cube):
    import numpy as np
    v_sphere = (4 / 3) * np.pi * r_sphere ** 3
    m_sphere = v_sphere * density
    v_cube = l_cube ** 3
    m_cube = v_cube * density
    com_pos = (m_sphere * com_sphere + m_cube * com_cube) / (m_sphere + m_cube)

    I_sphere = (2 / 5) * m_sphere * r_sphere ** 2
    I_cube = (1 / 6) * m_cube * l_cube ** 2
    I_trans_sphere = m_sphere * (com_sphere - com_pos) ** 2
    I_trans_cube = m_cube * (com_cube - com_pos) ** 2
    I_sphere_total = I_sphere + I_trans_sphere
    I_cube_total = I_cube + I_trans_cube
    I_com = I_sphere_total + I_cube_total

    I_joint = I_sphere + I_cube + m_cube * (com_cube) ** 2

    volume = 4 / 3 * np.pi * r_sphere ** 3 + l_cube ** 3
    m = env_config.density * volume
    r = -com_pos
    return I_com, r, m

def populate_env_config():

    I, r, m = moment_of_inertia(
        env_config.density,
        env_config.r_sphere,
        env_config.l_cube,
        env_config.com_sphere,
        env_config.com_cube
    )

    env_config.update(zip(['I', 'r', 'm'], [I, r, m]))
    return env_config

def xml_wrapper():
    import xml.etree.ElementTree as ET

    xml_path = '/home/bizon/MyProjects/safety-gym/safety_gym/xmls/point_m.xml'
    tree = ET.parse(xml_path)
    root = tree.getroot()

    root.find('option').set('timestep', str(env_config.timestep / 10)) #Mujoco hold control for 10 timestep
    root.find('option').set('integrator', env_config.integrator)

    for default in root.iter('default'):
        default.find('geom').set('density', str(env_config.density))

    for body in root.iter('body'):
        if body.get('name') == 'robot':
            body.find('geom').set('size', str(env_config.r_sphere))
            pointarrow = body.find('geom[@name="pointarrow"]')
            pointarrow.set('size',
                           '{} {} {}'.format(env_config.l_cube / 2, env_config.l_cube / 2, env_config.l_cube / 2))

    tree.write(xml_path)
    return



def populate_engine_config():



    env_config.update(zip(['I', 'r', 'm'], [I, r, m]))
    return env_config
