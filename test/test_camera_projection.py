from typing import List
from projectyl.utils.properties import COLOR, POSITION, SIZE, CAMERA
from projectyl.utils.camera_projection import get_intrinic_matrix, get_4D_homogeneous_vector, project_3D_point
from projectyl.dynamics.meshcat_viewer_wrapper import MeshcatVisualizer
from pinocchio import SE3
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def get_sample_scene():
    DEFAULT_SIZE = [.05, .05, .05]
    object_list = {
        "ball": {
            COLOR: [.5, .1, .1, 1.],
            POSITION: [0.2, 0., 0.25],
            SIZE: DEFAULT_SIZE
        },
        "arm": {
            COLOR: [.1, 1., .1, 1.],
            POSITION: [-0.5, 0., 1.2],
            SIZE: DEFAULT_SIZE
        },
        "tower": {
            COLOR: [.1, .1, 1., 1.],
            POSITION: [0., 100., 25.],
            SIZE: [10., 10., 10.]
        },
        CAMERA: {
            COLOR: [1., 0.5, 0.5, 1.],
            POSITION: [0., -2., 1.],
            SIZE: [0.05, 0.2, 0.05]
        }
    }
    return object_list


def make_a_scene_in_3D(object_list: List[dict], viz: MeshcatVisualizer = None) -> MeshcatVisualizer:
    """Make the 3D scene with the given objects in Meshcat

    Args:
        object_list (List[dict]): _description_
        viz (MeshcatVisualizer, optional): _description_. Defaults to None.
    """
    if viz is None:
        viz = MeshcatVisualizer()
    for name, obj in object_list.items():
        viz.addBox(
            name,
            obj[SIZE],
            obj[COLOR]
        )
        viz.applyConfiguration(name, SE3(np.eye(3), np.array(obj[POSITION])))
    return viz


def project_scene_sample(object_list: List[dict]):
    # Create an empty image
    img = 10*np.ones((720, 1080, 3))  # 720p frame
    # focal length in pixel
    h, w = img.shape[:2]
    intrinsic_matrix = get_intrinic_matrix((h, w), fpix=600.)
    extrinsic_matrix = np.zeros((3, 4))
    extrinsic_matrix[:3, :3] = np.eye(3)
    cam_pos = get_4D_homogeneous_vector(object_list[CAMERA][POSITION])
    extrinsic_matrix[:3, -1] = -cam_pos[:3, 0]
    for _idx, (name, obj) in enumerate(object_list.items()):
        if name == CAMERA:
            continue
        pos = obj[POSITION]
        # Project 3D point to the image coordinate system
        pos2d = project_3D_point(pos, intrinsic_matrix, extrinsic_matrix)

        # Plot the 2D point on the image
        color = ((np.array(obj[COLOR][:3])*255).astype(int)).tolist()
        cv.circle(img, (int(pos2d[0]), int(pos2d[1])), 5, tuple(color), -1)
        cv.putText(img, name, (int(pos2d[0])+5, int(pos2d[1])+5), cv.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv.LINE_AA)
    cv.line(img, (0, h//2), (w, h//2), (255, 255, 255), 2)
    cv.line(img, (w//2, 0), (w//2, h), (255, 255, 255), 2)
    cv.circle(img, (w//2, h//2), 5, (255, 0, 0), -1)
    img = img.clip(0, 255).astype(np.uint8)
    plt.imshow(img)
    plt.show()


def visual_camera_projection():
    object_list = get_sample_scene()
    make_a_scene_in_3D(object_list)
    project_scene_sample(object_list)


def test_camera_projection():
    """Numerical validation of the camera projection
    50 mm on a Canon 5D Mark III
    Eiffel tower is 324 m high
    At 6km away from my window

    Eiffer tower is upwards, 3D vector Y axis should be positive
    so the 2D vector should be negative (matrix notation and opencv notations)
    """
    #
    w, h = (6000, 4000)
    focal_length = 50.E-3
    pixel_pitch = 6.E-6
    fpix = focal_length/pixel_pitch
    intrinsic_matrix = get_intrinic_matrix((h, w), fpix=fpix)
    extrinsic_matrix = np.zeros((3, 4))
    extrinsic_matrix[:3, :3] = np.eye(3)
    distance = 6000.
    height = 324.
    eiffel_tower = np.array([0., distance, height])
    eiffel_tower_2d = project_3D_point(eiffel_tower, intrinsic_matrix, extrinsic_matrix)
    eiffel_tower_vector_2d = eiffel_tower_2d - np.array([w/2., h/2.])
    assert np.isclose(eiffel_tower_vector_2d, np.array([0., -height*fpix/distance])).all()


if __name__ == "__main__":
    test_camera_projection()
    visual_camera_projection()
    # test_visual_camera_projection()
