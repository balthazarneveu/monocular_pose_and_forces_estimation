"""
Process a batch of videos
Asking to trim any video manually
"""
import argparse
from batch_processing import Batch
import sys
from pathlib import Path
from projectyl.utils.cli_parser_tool import add_video_parser_args, get_trim
from projectyl.utils.interactive import live_view
import logging
from projectyl.utils.io import Dump

# Vocabulary:
# frames
# thumbs
FOLDER = "folder"
PATH_LIST = "path_list"
FRAME_IDX = "frame_index"
TS = "timestamp"
FRAMES, THUMBS = "frames", "thumbs"
FPS = "fps"
SIZE = "size"
INTRISIC_MATRIX = "intrinsic_matrix"
sample_config_file = {
    "start_ratio": 0.1,
    "end_ratio": 0.8,
    "start_frame": 50,
    "end_frame": 200,
    "total_frames": 1464,
    "fps": 30,
    FRAMES: {
        FRAME_IDX: [50, 51, 52, 53, 54, 55, 56, 57, 58, 59],
        TS: [0.1, 0.2, 0.3, 0.4, 0.5],
        FOLDER: "preprocessed_frames",
        PATH_LIST: ["preprocessed_frames/frame1.png", "preprocessed_frames/frame2.png"],
        SIZE: [1920, 1080],
        INTRISIC_MATRIX: None  # Requires calibration
    },
    THUMBS: {
        FRAME_IDX: [50, 51, 52, 53, 54, 55, 56, 57, 58, 59],
        TS: [0.1, 0.2, 0.3, 0.4, 0.5],
        FOLDER: "preprocessed_frames",
        PATH_LIST: ["preprocessed_frames/frame1.png", "preprocessed_frames/frame2.png"],
        SIZE: [1280, 720],
        INTRISIC_MATRIX: None  # Need proper rescaling
    }
}


def video_decoding(input: Path, output: Path, args: argparse.Namespace):
    skip_existing = not args.override
    preprocessing_config_file = output/"input_configuration.yaml"
    if preprocessing_config_file.exists() and skip_existing:
        config = Dump.load_yaml(preprocessing_config_file)
    else:
        preload_ram = not args.disable_preload_ram
        config = live_view(input, trimming=True, preload_ram=preload_ram)
        Dump.save_yaml(config, preprocessing_config_file)


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
