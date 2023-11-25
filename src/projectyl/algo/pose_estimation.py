import mediapipe as mp
from mediapipe.tasks import python as python_mp
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from projectyl.utils.io import Image
import numpy as np
from pathlib import Path
from typing import Optional
import cv2 as cv
# https://github.com/googlesamples/mediapipe/blob/main/examples/pose_landmarker/python/%5BMediaPipe_Python_Tasks%5D_Pose_Landmarker.ipynb


def draw_joint(annotated_image, pose_landmarks, current_joint_id, color=(0, 255, 0)):
    annotated_image = cv.circle(
        annotated_image,
        (int(pose_landmarks[current_joint_id].x*annotated_image.shape[1]),
            int(pose_landmarks[current_joint_id].y*annotated_image.shape[0])),
        20, color, -1
    )


def draw_landmarks_on_image(rgb_image, pose_landmarks_list, joint_id=-1):
    annotated_image = np.copy(rgb_image)
    # Loop through the detected poses to visualize.
    for idx in range(len(pose_landmarks_list)):
        pose_landmarks = pose_landmarks_list[idx]

        # Draw the pose landmarks.
        pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        if joint_id == -2:
            pose_landmarks_proto.landmark.extend([
                landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
            ])
            solutions.drawing_utils.draw_landmarks(
                annotated_image,
                pose_landmarks_proto,
                solutions.pose.POSE_CONNECTIONS,
                solutions.drawing_styles.get_default_pose_landmarks_style()
            )
        elif joint_id == -1:
            # 11 - 13 - 15
            # 11 - left shoulder
            # 13 - left elbow
            # 15 - left wrist

            # 12 - 14 - 16
            # 12 - right shoulder
            # 14 - right elbow
            # 16 - right wrist
            for current_joint_id in [11, 13, 15]:
                draw_joint(annotated_image, pose_landmarks, current_joint_id, color=(0, 255, 0))
            for current_joint_id in [12, 14, 16]:
                draw_joint(annotated_image, pose_landmarks, current_joint_id, color=(255, 0, 0))
        else:
            draw_joint(annotated_image, pose_landmarks, joint_id)

    return annotated_image


def get_detector():
    try:
        base_options = python_mp.BaseOptions(model_asset_path='pose_landmarker.task')
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            output_segmentation_masks=True
        )
        detector = vision.PoseLandmarker.create_from_options(options)
    except ImportError as exc:
        print(exc)
        print("Use wget -O pose_landmarker.task -q https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task")
    return detector


def get_pose(path: Path, detector=None, visualization_path: Optional[Path] = None):
    if detector is None:
        detector = get_detector()
    image = mp.Image.create_from_file(str(path))
    detection_result = detector.detect(image)

    # STEP 5: Process the detection result. In this case, visualize it.
    if visualization_path:
        annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result.pose_landmarks)
        Image.write(visualization_path, annotated_image)
    return detection_result, visualization_path
