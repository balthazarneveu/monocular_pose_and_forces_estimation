"""
Process a batch of videos
Asking to trim any video manually
"""
import argparse
from batch_processing import Batch
import sys
from pathlib import Path
from projectyl.utils.cli_parser_tool import add_video_parser_args
from projectyl.utils.interactive import live_view
from projectyl.video.props import FRAMES, THUMBS, FRAME_IDX, TS, FOLDER, PATH_LIST, SIZE
import logging
from projectyl.utils.io import Dump
from projectyl.algo.pose_estimation import get_pose, get_detector


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
    if "view" in args.algo:
        live_view(config[THUMBS][PATH_LIST], trimming=False, preload_ram=preload_ram)
    if "pose" in args.algo:
        pose_dir = output/"pose"
        pose_dir.mkdir(parents=True, exist_ok=True)
        detector = get_detector()
        pose_annotations = []
        for path in config[THUMBS][PATH_LIST]:
            pose_annotation = pose_dir/(Path(path).name)
            pose_path = pose_annotation.with_suffix(".pkl")
            if pose_annotation.exists() and skip_existing:
                assert pose_path.exists()
                annotations = Dump.load_pickle(pose_path)
            else:
                annotations, _ = get_pose(
                    path,
                    detector,
                    visualization_path=pose_annotation
                )
                Dump.save_pickle(annotations.pose_landmarks, pose_path)
            pose_annotations.append(pose_annotation)
        live_view(pose_annotations, trimming=False, preload_ram=preload_ram)


def parse_command_line(batch: Batch) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Batch video processing',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_video_parser_args(parser, allow_global_trim=False)
    parser.add_argument("-mp", "--multi-processing", action="store_true",
                        help="Enable multiprocessing - Warning with GPU - use -j2")
    parser.add_argument("--override", action="store_true",
                        help="overwrite processed results")
    parser.add_argument("-fast", "--disable-preload-ram", action="store_true",
                        help="Preload video in RAM")
    parser.add_argument_group("algorithm")
    parser.add_argument("-A", "--algo", nargs="+",
                        choices=["pose", "view"], default=[])
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
