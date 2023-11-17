# Shared parser arguments
import argparse
from typing import Tuple
VIDEO_EXT = ["mp4", "avi", "mp4", "mov"]


def add_video_parser_args(parser: argparse.Namespace) ->None:
    video_args = parser.add_argument_group("input video")
    video_args.add_argument("-t", "--trim", nargs="+", type=float, help="Trim in seconds like -t 4.8 5.3 or -t 0.5")
    video_args.add_argument("-r", "--resize", type=float, help="resize input video factor")

def get_trim(args) -> Tuple[float, float]:
    trim = args.trim
    if trim:
        if len(trim)==1:
            trim = (None,  trim[0])
        else:
            assert len(trim)==2, "trim shall have one or two elements"
        print(f"Video trimming: {trim}")
    return trim