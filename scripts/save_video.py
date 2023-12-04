from batch_processing import Batch
import argparse
import sys
from pathlib import Path
from projectyl.video.encoder import encode_debug_figures


def parse_command_line(batch: Batch) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Batch video processing save',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--fps', type=int, default=30, help='fps')
    parser.add_argument('--decimate', type=int, default=1, help='decimate')
    parser.add_argument('--resize', type=float, default=1., help='resize')
    return batch.parse_args(parser)


def video_encoding(input: Path, output: Path, args: argparse.Namespace):
    path_list = []
    for path in sorted(list(input.glob("*.*g"))):
        path_list.append(path)
    if args.decimate > 1:
        path_list = path_list[::args.decimate]
    encode_debug_figures(
        path_list,
        output_path=output.with_suffix(".gif"),
        fps=args.fps,
        resize=args.resize
    )


def main(argv):
    batch = Batch(argv)
    batch.set_io_description(
        input_help='input video files', output_help='output directory')
    parse_command_line(batch)
    batch.run(video_encoding)


if __name__ == "__main__":
    main(sys.argv[1:])


parser = argparse.ArgumentParser(description='Batch video processing')
