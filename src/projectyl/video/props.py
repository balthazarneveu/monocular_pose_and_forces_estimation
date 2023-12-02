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
INTRINSIC_MATRIX = "intrinsic_matrix"
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
        INTRINSIC_MATRIX: None  # Requires calibration
    },
    THUMBS: {
        FRAME_IDX: [50, 51, 52, 53, 54, 55, 56, 57, 58, 59],
        TS: [0.1, 0.2, 0.3, 0.4, 0.5],
        FOLDER: "preprocessed_frames",
        PATH_LIST: ["preprocessed_frames/frame1.png", "preprocessed_frames/frame2.png"],
        SIZE: [1280, 720],
        INTRINSIC_MATRIX: None  # Need proper rescaling
    }
}
