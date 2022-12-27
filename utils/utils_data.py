import numpy as np


def normalize_pc(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc


def random_scale_pc(pc):
    scale = np.random.uniform(low=2. / 3., high=3. / 2., size=[3])
    scaled_pc = np.multiply(pc, scale).astype('float32')
    return scaled_pc


def translate_pc(pc):
    shift = np.random.uniform(low=-0.2, high=0.2, size=[3])
    translated_pc = np.add(pc, shift).astype('float32')
    return translated_pc


def jitter_pc(pc, sigma=0.01, clip=0.05):
    N, C = pc.shape
    assert (clip > 0)
    jitter_data = np.clip(sigma * np.random.randn(N, C), -1 * clip, clip)
    pc += jitter_data
    return pc


def so3_rotate(point_data, angles=None):
    if np.ndim(point_data) != 2:
        raise ValueError("np.ndim(point_data) != 2, must be (N, 3)")
    if point_data.shape[-1] != 3:
        raise ValueError("point_data.shape[-1] != 3, must be (x, y, z)")

    if angles is None:
        angles = np.random.uniform(low=0, high=360, size=[3])
    else:
        assert isinstance(angles, list), "Angles must be stored in a list!"
        angles = np.array(angles)

    angles = angles * np.pi / 180
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(angles[0]), -np.sin(angles[0])],
                   [0, np.sin(angles[0]), np.cos(angles[0])]])
    Ry = np.array([[np.cos(angles[1]), 0, np.sin(angles[1])],
                   [0, 1, 0],
                   [-np.sin(angles[1]), 0, np.cos(angles[1])]])
    Rz = np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0],
                   [np.sin(angles[2]), np.cos(angles[2]), 0],
                   [0, 0, 1]])
    R = np.dot(Rz, np.dot(Ry, Rx))
    rotated_data = np.dot(point_data, R).astype(np.float32)
    return rotated_data


def z_rotate(point_data, angles=None):
    if np.ndim(point_data) != 2:
        raise ValueError("np.ndim(point_data) != 2, must be (N, 3)")
    if point_data.shape[-1] != 3:
        raise ValueError("point_data.shape[-1] != 3, must be (x, y, z)")

    if angles is None:
        angles = np.random.uniform(low=0, high=360)
    angles = angles * np.pi / 180

    R = np.array([[np.cos(angles), -np.sin(angles), 0],
                  [np.sin(angles), np.cos(angles), 0],
                  [0, 0, 1]])
    rotated_data = np.dot(point_data, R).astype(np.float32)
    return rotated_data
