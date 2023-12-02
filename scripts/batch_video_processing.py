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
from projectyl.video.props import THUMBS, PATH_LIST
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
from projectyl.video.props import INTRINSIC_MATRIX
from projectyl import root_dir


def video_decoding(input: Path, output: Path, args: argparse.Namespace):
    skip_existing = not args.override
    config_file = output/"config.yaml"
    config_trim_file = output/"trim_configuration.yaml"
    preload_ram = not args.disable_preload_ram
    if config_trim_file.exists():  # and skip_existing:
        config_trim = Dump.load_yaml(config_trim_file)
    else:
        config_trim = live_view(input, trimming=True, preload_ram=preload_ram)
        Dump.save_yaml(config_trim, config_trim_file)
    thumb_dir = output/"thumbs"
    if thumb_dir.exists() and skip_existing:
        logging.warning(f"Results already exist - skip processing  {thumb_dir}")
        config = Dump.load_yaml(config_file, safe_load=False)
    else:
        # Moviepy takes a while to load, load only on demand
        from projectyl.video.decoder import save_video_frames
        logging.warning(
            f"Overwriting results - use --skip-existing to skip processing  {thumb_dir}")
        thumb_dir.mkdir(parents=True, exist_ok=True)
        trim = config_trim["start_ratio"], config_trim["end_ratio"]
        config = save_video_frames(input, thumb_dir, trim=trim, resize=args.resize)
        config["start_ratio"] = config_trim["start_ratio"]
        config["end_ratio"] = config_trim["end_ratio"]
        Dump.save_yaml(config, config_file)
        try:
            config = Dump.load_yaml(config_file, safe_load=False)
        except Exception as e:
            raise NameError(f"Error loading config file {config_file} {e}")
    im_list = config[THUMBS][PATH_LIST]
    camera_config = {}
    if "camera_calibration" in args.algo:
        camera_calibration_folder = output/"camera_calibration"
        camera_calibration(im_list, output_folder=camera_calibration_folder)
        return  # Finish
    if "view" in args.algo:
        live_view(im_list, trimming=False, preload_ram=preload_ram)
    if args.camera_calibration:
        assert Path(args.camera_calibration).exists()
        calib_dict = Dump.load_json(Path(args.camera_calibration))
        config[INTRINSIC_MATRIX] = np.array(calib_dict[INTRINSIC_MATRIX])
        camera_config[INTRINSIC_MATRIX] = config[INTRINSIC_MATRIX]
    if "pose" in args.algo or "ik" in args.algo:
        pose_dir = output/"pose"
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
    if "pose" in args.algo and not args.headless:
        interactive_visualize_pose(im_list, pose_annotations, camera_config=camera_config)
    if "ik" in args.algo:
        ik_path = output/"coarse_ik.pkl"
        global_params = {}
        if ik_path.exists() and skip_existing:
            conf_list = Dump.load_pickle(ik_path)
        else:
            conf_list, global_params = coarse_inverse_kinematics_initialization(pose_annotations)
            Dump.save_pickle(conf_list, ik_path)
        if not args.headless:
            coarse_inverse_kinematics_visualization(conf_list["q"], global_params)
        plot_ik_states(conf_list)


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
                        choices=["pose", "view", "ik", "camera_calibration"], default=[])
    parser.add_argument("-side", "--arm-side", choices=[LEFT, RIGHT], default=RIGHT)
    parser.add_argument("-noviz", "--headless", action="store_true", help="Disable visualizations")
    default_camera_calib = root_dir/"calibration"/"camera_calibration_xiaomi_mi11_ultra_video_vertical.json"
    parser.add_argument("-calib", "--camera_calibration", type=str,
                        default=default_camera_calib, help="Camera calibration file")
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
