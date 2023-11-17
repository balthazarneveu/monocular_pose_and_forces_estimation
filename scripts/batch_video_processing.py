import argparse
import sys
from pathlib import Path
from projectyl.utils.cli_parser_tool import add_video_parser_args, get_trim
from projectyl.video.decoder import save_video_frames
from batch_processing import Batch
import logging



def parse_command_line(batch: Batch) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Batch video processing',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_video_parser_args(parser)
    parser.add_argument("-mp", "--multi-processing", action="store_true", help="Enable multiprocessing - Warning with GPU - use -j2")    
    parser.add_argument("-skip", "--skip-existing", action="store_true", help="skip existing processed folders")
    return batch.parse_args(parser)


def video_decoding(input: Path, output: Path, args: argparse.Namespace):
    trim = get_trim(args)
    if output.exists() and args.skip_existing:
        logging.warning(f"Results already exist - skip processing  {output}")
    else:
        logging.warning(f"Overwriting results - use --skip-existing to skip processing  {output}")
        output.mkdir(parents=True, exist_ok=True)
        save_video_frames(input, output, trim=trim, resize=args.resize)

def main(argv):
    batch = Batch(argv)
    batch.set_io_description(input_help='input video files', output_help='output directory')
    args = parse_command_line(batch)
    # Disable mp - Highly recommended!
    if not args.multi_processing:
        batch.set_multiprocessing_enabled(False)
    batch.run(video_decoding)



if __name__ == "__main__":
    main(sys.argv[1:])