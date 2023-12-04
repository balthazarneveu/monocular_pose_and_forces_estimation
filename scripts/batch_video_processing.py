"""
Process a batch of videos
Asking to trim any video manually
"""
import argparse
from batch_processing import Batch
import sys
import numpy as np
from pathlib import Path
from projectyl.utils.cli_parser_tool import add_video_parser_args
from projectyl.utils.interactive import live_view
from projectyl.video.props import THUMBS, PATH_LIST, SIZE, INTRINSIC_MATRIX, EXTRINSIC_MATRIX
from projectyl.utils.properties import LEFT, RIGHT
from projectyl.utils.arm import plot_ik_states
import logging
from projectyl.utils.io import Dump
from projectyl.algo.pose_estimation import get_pose, get_detector
from projectyl.utils.pose_overlay import interactive_visualize_pose
from projectyl.dynamics.inverse_kinematics import (
    coarse_inverse_kinematics_initialization, coarse_inverse_kinematics_visualization
)
from projectyl.utils.camera_calibration import camera_calibration
from projectyl import root_dir
from projectyl.utils import fit_camera_pose as fit_cam
from projectyl.utils.pose_overlay import get_4D_homogeneous_vector
from projectyl.dynamics.inverse_kinematics import build_arm_model
from projectyl.utils.interactive_demo import interactive_demo
from tqdm import tqdm
POSE = "pose"
CAMERA_CALIBRATION = "camera_calibration"
IK = "ik"
VIEW = "view"
DEMO = "demo"
FIT_CAMERA_POSE = "fit_cam"


def get_trim_config(config_trim_file: Path, input: Path, preload_ram=False) -> dict:
    if config_trim_file.exists():  # and skip_existing:
        config_trim = Dump.load_yaml(config_trim_file)
    else:
        config_trim = live_view(input, trimming=True, preload_ram=preload_ram)
        Dump.save_yaml(config_trim, config_trim_file)
    return config_trim


def get_sequence_config(input: Path, thumb_dir, config_trim: dict, config_file: Path,
                        skip_existing: bool = True, resize: float = None) -> dict:
    if config_file.exists() and skip_existing:
        logging.warning(f"Results already exist - skip processing  {thumb_dir}")
        config = Dump.load_yaml(config_file, safe_load=False)
    else:
        # Moviepy takes a while to load, load only on demand
        from projectyl.video.decoder import save_video_frames
        logging.warning(
            f"Overwriting results - use --skip-existing to skip processing  {thumb_dir}")
        thumb_dir.mkdir(parents=True, exist_ok=True)
        trim = config_trim["start_ratio"], config_trim["end_ratio"]
        config = save_video_frames(input, thumb_dir, trim=trim, resize=resize)
        config["start_ratio"] = config_trim["start_ratio"]
        config["end_ratio"] = config_trim["end_ratio"]
        Dump.save_yaml(config, config_file)
        try:
            config = Dump.load_yaml(config_file, safe_load=False)
        except Exception as e:
            raise NameError(f"Error loading config file {config_file} {e}")
    return config


def get_pose_sequences(pose_dir: Path, config: dict, skip_existing: bool = True) -> list:
    """Retreive pose sequences from a list of images (contained in config).
    Save in pos_dir if not already saved.
    """
    pose_sequence_path = pose_dir/"pose_sequence.pkl"
    if pose_sequence_path.exists() and skip_existing:
        pose_annotations = Dump.load_pickle(pose_sequence_path)
        return pose_annotations
    pose_dir.mkdir(parents=True, exist_ok=True)
    detector = None
    pose_annotations = []
    pose_annotation_img_list = []
    for path in config[THUMBS][PATH_LIST]:
        pose_annotation_img = pose_dir/(Path(path).name)
        pose_path = pose_annotation_img.with_suffix(".pkl")
        if pose_path.exists() and skip_existing:
            assert pose_path.exists()
            dic_annot = Dump.load_pickle(pose_path)
        else:
            if not detector:
                detector = get_detector()
            annotations, _ = get_pose(
                path,
                detector,
                visualization_path=pose_annotation_img
            )
            dic_annot = {
                "pose_landmarks": annotations.pose_landmarks,
                "pose_world_landmarks": annotations.pose_world_landmarks
            }
            Dump.save_pickle(dic_annot, pose_path)
        pose_annotations.append(dic_annot)
        pose_annotation_img_list.append(pose_annotation_img)
    if not pose_sequence_path.exists():
        Dump.save_pickle(pose_annotations, pose_sequence_path)
    return pose_annotations


def get_camera_pose_problem_data(data3d, data2d, config, arm_robot=None, arm_side=RIGHT):
    if arm_robot is None:
        global_params = {}
        build_arm_model(global_params, headless=True)
        arm_robot = global_params["arm"]
    h, w = int(config[THUMBS][SIZE][0]), int(config[THUMBS][SIZE][1])
    p3d, p2d, q_states = fit_cam.build_3d_2d_data_arrays(data3d, data2d, (h, w), arm_side=arm_side)
    p4d_data_in = fit_cam.config_states_to_4d_points(q_states, arm_robot)
    return p4d_data_in, p2d


def fit_camera_pose(
    data3d, data2d, config, intrinsic_matrix,
    arm_robot=None, arm_side=RIGHT, full_solution_flag=False,
    cam_smoothness=0.05
):
    if cam_smoothness is None:
        cam_smoothness = 0.05
    p4d_data_in, p2d = get_camera_pose_problem_data(data3d, data2d, config, arm_robot=arm_robot, arm_side=arm_side)
    solutions = []
    flat_solution = []
    window = 30
    for t in tqdm(range(window, len(p4d_data_in)+2*window+1, 2*window+1), desc="fitting camera pose per windows"):
        start = max(0, t-window)
        end = min(t+window+1, len(p4d_data_in))
        win_p4d = p4d_data_in[start:end, :]
        win_p2d = p2d[start:end, :]
        try:
            solution_ = fit_cam.optimize_camera_pose(win_p4d, win_p2d, intrinsic_matrix, cam_smoothness=cam_smoothness)
        except Exception as e:
            print(f"Error fitting camera pose {e}")
            solution = solution_
        flat_solution.append(solution_)
        solution = solution_.reshape(-1, 3)
        solutions.append(solution)
    solutions_seq = np.concatenate(solutions, axis=0)  # for visualization
    init_solution = np.concatenate(flat_solution)
    if full_solution_flag:
        # THIS CAN BE VERY SLOW!
        full_solution = fit_cam.optimize_camera_pose(
            p4d_data_in, p2d, intrinsic_matrix, init_var=init_solution, cam_smoothness=cam_smoothness)
        full_solution = full_solution.reshape(-1, 3)
    else:
        full_solution = solutions_seq
    full_solution_world = np.array([get_4D_homogeneous_vector(sol, reverse=True) for sol in full_solution])
    full_solution_world = full_solution_world[:p4d_data_in.shape[0], ...]
    return full_solution_world


def video_decoding(input: Path, output: Path, args: argparse.Namespace):
    skip_existing = not args.override
    preload_ram = not args.disable_preload_ram

    config_trim_file = output/"trim_configuration.yaml"
    config_file = output/"config.yaml"
    config_trim = get_trim_config(config_trim_file, input, preload_ram=preload_ram)

    thumb_dir = output/"thumbs"
    config = get_sequence_config(
        input, thumb_dir, config_trim,
        config_file,
        skip_existing=skip_existing,
        resize=args.resize
    )

    im_list = config[THUMBS][PATH_LIST]
    camera_config = {}
    if CAMERA_CALIBRATION in args.algo:
        camera_calibration_folder = output/CAMERA_CALIBRATION
        camera_calibration(im_list, output_folder=camera_calibration_folder)
        return  # Finish
    if VIEW in args.algo:
        live_view(im_list, trimming=False, preload_ram=preload_ram)

    if args.camera_calibration:
        assert Path(args.camera_calibration).exists()
        calib_dict = Dump.load_json(Path(args.camera_calibration))
        config[INTRINSIC_MATRIX] = np.array(calib_dict[INTRINSIC_MATRIX])
        camera_config[INTRINSIC_MATRIX] = config[INTRINSIC_MATRIX]

    h, w = int(config[THUMBS][SIZE][0]), int(config[THUMBS][SIZE][1])
    camera_config[SIZE] = (h, w)
    if POSE in args.algo or IK in args.algo or FIT_CAMERA_POSE in args.algo or DEMO in args.algo:
        pose_dir = output/"pose"
        pose_annotations = get_pose_sequences(pose_dir, config, skip_existing=skip_existing)

    if POSE in args.algo and not args.headless:
        interactive_visualize_pose(im_list, pose_annotations, camera_config=camera_config)

    if IK in args.algo or FIT_CAMERA_POSE in args.algo or DEMO in args.algo:
        ik_path = output/"coarse_ik.pkl"
        global_params = {}
        if ik_path.exists() and skip_existing:
            conf_list = Dump.load_pickle(ik_path)
        else:
            conf_list, global_params = coarse_inverse_kinematics_initialization(pose_annotations)
            Dump.save_pickle(conf_list, ik_path)
    if IK in args.algo:
        if not args.headless:
            coarse_inverse_kinematics_visualization(conf_list["q"], global_params)
    if IK in args.algo:
        # @TODO: Warning with invalid frames!
        plot_ik_states(conf_list)
    if FIT_CAMERA_POSE in args.algo or DEMO in args.algo:
        # To fit camera pose, you need to have valid IK
        camera_fit_path = output/"camera_fit.pkl"
        if (camera_fit_path.exists() and skip_existing) and args.cam_smoothness is None:
            extrinsic_params = Dump.load_pickle(camera_fit_path)
        else:
            extrinsic_params = fit_camera_pose(
                conf_list,
                pose_annotations,
                config,
                config[INTRINSIC_MATRIX],
                arm_robot=None, arm_side=RIGHT,
                cam_smoothness=args.cam_smoothness
            )
            Dump.save_pickle(extrinsic_params, camera_fit_path)
        camera_config[EXTRINSIC_MATRIX] = extrinsic_params
    else:
        camera_config[EXTRINSIC_MATRIX] = None
    if FIT_CAMERA_POSE in args.algo or DEMO in args.algo:
        extr_array = np.array([
            fit_cam.get_extrinsics_default(extrinsic_params[t_index, :]) for t_index in range(len(extrinsic_params))
        ])
        intrinsic_matrix = camera_config[INTRINSIC_MATRIX]
        p4d_data_in, p2d = get_camera_pose_problem_data(
            conf_list, pose_annotations, config, arm_robot=None, arm_side=args.arm_side)
        p2d_fit = fit_cam.project_4d_points_to_2d(p4d_data_in, intrinsic_matrix, extr_array)
        p2d_dict = {
            "2d pose estimation": p2d,
            "fit with moving camera": p2d_fit,
        }
    if FIT_CAMERA_POSE in args.algo and DEMO not in args.algo:
        if not args.headless:
            # WARNING: does not work with DEMO
            fit_cam.__plot_camera_pose(
                {
                    "full solution - camera pose": extrinsic_params,
                }
            )
            fit_cam.visualize_2d_trajectories(
                p2d_dict,
                (h, w),
                # t_end=100,
                title="2D image trajectories\n"
            )
            # fit_cam.project_4d_points_to_2d(p4d_data_in, intrinsic_matrix, extr_array)

    if DEMO in args.algo:
        interactive_demo(
            im_list,
            pose_annotations,
            p2d_dict=p2d_dict,
            states_sequences=conf_list,
            camera_config=camera_config
        )


def parse_command_line(batch: Batch) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Batch video processing',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_video_parser_args(parser, allow_global_trim=False)
    parser.add_argument("-mp", "--multi-processing", action="store_true",
                        help="Enable multiprocessing - Warning with GPU - use -j2")
    parser.add_argument("--override", action="store_true",
                        help="overwrite processed results")
    parser.add_argument("-tryfast", "--disable-preload-ram", action="store_true",
                        help="Preload video in RAM")
    parser.add_argument_group("algorithm")
    parser.add_argument("-A", "--algo", nargs="+",
                        choices=[POSE, VIEW, IK, CAMERA_CALIBRATION, FIT_CAMERA_POSE, DEMO], default=[])
    parser.add_argument("-side", "--arm-side", choices=[LEFT, RIGHT], default=RIGHT)
    parser.add_argument("-noviz", "--headless", action="store_true", help="Disable visualizations")
    default_camera_calib = root_dir/"calibration"/"camera_calibration_xiaomi_mi11_ultra_video_vertical.json"
    parser.add_argument("-calib", "--camera_calibration", type=str,
                        default=default_camera_calib, help="Camera calibration file")
    parser.add_argument("-smooth", "--cam-smoothness", type=float, default=None,
                        help="Camera smoothness")
    return batch.parse_args(parser)


def main(argv):
    batch = Batch(argv)
    batch.set_io_description(
        input_help='input video files', output_help='output directory')
    args = parse_command_line(batch)
    # Disable mp - Highly recommended!
    if not args.multi_processing:
        batch.set_multiprocessing_enabled(False)
    batch.run(video_decoding)


if __name__ == "__main__":
    main(sys.argv[1:])
