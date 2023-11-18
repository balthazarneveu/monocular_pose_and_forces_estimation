import argparse
import sys
from pathlib import Path
from projectyl.utils.cli_parser_tool import add_video_parser_args, get_trim
from projectyl.algo.background_substraction import bg_substract
from batch_processing import Batch
import logging
from interactive_pipe.data_objects.image import Image
import numpy as np
from projectyl.algo.interactive_segmentation import interactive_sam
from projectyl.algo.segmentation import segment_frames
from projectyl.utils.io import Dump
from projectyl.utils.interactive import interactive_trimming

def parse_command_line(batch: Batch) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Batch video processing',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_video_parser_args(parser)
    parser.add_argument("-mp", "--multi-processing", action="store_true", help="Enable multiprocessing - Warning with GPU - use -j2")    
    parser.add_argument("-skip", "--skip-existing", action="store_true", help="skip existing processed folders")
    parser.add_argument_group("algorithm")
    parser.add_argument("-A", "--algo", nargs="+", choices=["bgsub", "sam", "trim"])
    return batch.parse_args(parser)

def video_decoding(input: Path, output: Path, args: argparse.Namespace):
    trim = get_trim(args)
    if output.exists() and args.skip_existing:
        logging.warning(f"Results already exist - skip processing  {output}")
    else:
        # Moviepy takes a while to load, load only on demand
        from projectyl.video.decoder import save_video_frames
        logging.warning(f"Overwriting results - use --skip-existing to skip processing  {output}")
        output.mkdir(parents=True, exist_ok=True)
        save_video_frames(input, output, trim=trim, resize=args.resize)
    algo_list = args.algo
    all_frames = sorted(list(output.glob("*.*g")))
    
    if "trim" in algo_list:
        trim_path=output/"trim.yaml"
        if trim_path.exists() and args.skip_existing:
            logging.info("Trim file already exists")
        else:
            sequence = np.array([Image.load_image(img).data for img in all_frames])
            interactive_trimming(sequence, trim_path=output/trim_path)
        trim_conf = Dump.load_yaml(trim_path)
        sequence = np.array([Image.load_image(img).data for img in all_frames[trim_conf["start"]: trim_conf["end"]]])
    else:
        sequence = np.array([Image.load_image(img).data for img in all_frames])
    if "bgsub" in algo_list:
        bg_substract(sequence, interactive=True)
    if "sam" in algo_list:
        mask_path = output/"sam_masks.pkl"
        if mask_path.exists() and args.skip_existing:
            masks = Dump.load_pickle(mask_path)
        else:
            masks = segment_frames(sequence)
            Dump.save_pickle(masks, mask_path)
        interactive_sam(sequence, masks)

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