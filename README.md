# Monocular joint pose and forces estimation
Review of the paper Estimating 3D Motion and Forces of Person-Object Interactions from Monocular Video
- [Original Paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Li_Estimating_3D_Motion_and_Forces_of_Person-Object_Interactions_From_Monocular_CVPR_2019_paper.pdf) 
- [Author's code](https://github.com/zongmianli/Estimating-3D-Motion-Forces)



### Context
 
- Program: [MVA Master's degree](https://www.master-mva.com/) class on [Robotics](https://scaron.info/robotics-mva/). ENS Paris-Saclay.
- Authors:
    - [Balthazar Neveu](https://github.com/balthazarneveu)
    - [Matthieu Dinot](https://github.com/mattx20)



## Setup
#### Setup projectyl
```bash
git clone git@github.com:balthazarneveu/monocular_pose_and_forces_estimation.git
cd monocular_pose_and_forces_estimation
pip install -e .
```

#### Tools
```bash
python3 scripts/batch_video_processing.py -i "data/*.mp4" -o __processed -t 3.2 3.4   --resize 0.2 --skip-existing
```
- `-i` regex or list of videos
- `-o` output folder (*automatically created if not present*)
- `--trim t1 t2`: trim the videos from t1 to t2 in seconds
- `--resize`: resize ratio


-----

## Extra
### CERES setup
```bash
git clone https://ceres-solver.googlesource.com/ceres-solver
cd ceres-solver
mkdir ceres-bin
cd ceres-bin
cmake ../../ceres-solver -DCMAKE_CUDA_ARCHITECTURES=native -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
make -j3
make install
```