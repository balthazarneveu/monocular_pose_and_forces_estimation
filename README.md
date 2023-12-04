# Monocular joint pose and forces estimation
Review of the paper Estimating 3D Motion and Forces of Person-Object Interactions from Monocular Video
- [Original Paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Li_Estimating_3D_Motion_and_Forces_of_Person-Object_Interactions_From_Monocular_CVPR_2019_paper.pdf) 
- [Author's code](https://github.com/zongmianli/Estimating-3D-Motion-Forces)



### Context
 
- Program: [MVA Master's degree](https://www.master-mva.com/) class on [Robotics](https://scaron.info/robotics-mva/). ENS Paris-Saclay.
- Authors:
    - [Balthazar Neveu](https://github.com/balthazarneveu)
    - [Matthieu Dinot](https://github.com/mattx20)


### Summary
The original paper estimates 3D poses and internal torques / external forces applied on a human interacting with a tool.
The goal behind this paper is to collect the dynamics of human menial activities from standard videos (including educational tutorials) to later enable behavior cloning on real robots. 

For simplification reasons, we studied a much simple problem making the following assumptions
- :hammer::x: : no external objects or tools
- :muscle: a single arm, not the full body.
- :camera: the camera is assumed calibrated and standing on a tripod.

The current code is able to batch process videos and performs:
- 2D pose and 3D pose estimation using Mediapipe (off the shelf)
  - RGB video :arrow_right: 3D & 2D joint positions
- inverse kinematics from the 3D points
  - 3D points :arrow_right: arm configuration states (so called $q$)
- fits a camera pose in order to minimize 2d reprojection error
  - 2D points & 3D points from forward kinematics :arrow_right: Sequence of extrinsics matrices (just a 3D translation)


There are still many missing points:
- We do not minimize the 2D reprojection error while performing Inverse Kinematics
- :muscle:  Many attempts have been made in order to reproduce the dynamics optimizer in order to recover the torques (shoulder and elbow). This is part of an extra notebook.


-----

## Setup
#### Setup projectyl
```bash
git clone git@github.com:balthazarneveu/monocular_pose_and_forces_estimation.git
cd monocular_pose_and_forces_estimation
pip install -e .
```

Download vision models
```bash
wget -O pose_landmarker.task -q https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task
```


#### Processing videos
```bash
python scripts/batch_video_processing.py -i "data/*.mp4" -o __out -A demo
```
- `-i` regex or list of videos
- `-o` output folder (*automatically created if not present*)
- `--resize`: resize ratio
- `-A` to pick up an algo
  - `-A view`  will simply lanch a frame by frame viewer with zoom-in capabilities
  - `-A pose` will run mediapipe pose estimator. (run once, store the results, next time reload)
  - `-A ik` will run inverse kinematics frame by frame (but keep previous frame as an initlization)
  - `-A fit_cam` will perform camera pose estimation.
- `-smoothness` allows to modify the smoothness used when fitting camera pose
- `-calib` points to the geometric calibration file, in case you're using a different camera.
> `--override` can be used to recompute images and overwrite previous results.




### Demo
![](/report/figures/arm_demo.png)

------

## Modelization

|                Pose estimation                 |                 Arm model                  |
| :--------------------------------------------: | :----------------------------------------: |
| ![](/report/figures/pose_estimation_gui_2.png) | ![](/report/figures/live_arm_vertical.png) |
| :statue_of_liberty:  pose estimated from video |    :wrench: Synchronized "Digital twin"    |


> color code for joints: :red_circle: shoulder , :green_circle: elbow, :large_blue_circle: wrist
> 
> limbs: :green_circle: upper-arm , :large_blue_circle: forearm
### Pose estimation
Using [Google Mediapipe](https://developers.google.com/mediapipe/solutions/vision/pose_landmarker), we're able to retrieve
- 2D projected points
- 3D coarse estimation


### Coarse initialization of dynamic system inverse kinematics
Based on a frame by frame estimation (initalized from previous estimation), we're able to retrieve a decent state arm.

:warning: As mediapipe does not provide 3D coordinates that grant fixed lenght arms, the system can't always be resolved correctly.

:bulb: An idea is to force the initialization of the inverse kinematics by forcing standardized length of the arms.

![](/report/figures/estimated_poses_from_ik.png)



### Projective camera geometry
- Pinhole model is used

|                                   Camera                                    |                          World                           |
| :-------------------------------------------------------------------------: | :------------------------------------------------------: |
|                ![](/report/figures/camera_referentials.png)                 | ![](/report/figures/world_camera_referentials_small.png) |
| OpenCV [convention](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html) |                   Pinocchio convention                   |

- [test_camera_projection.py](/test/test_camera_projection.py) has a numerical example showing how you project a 3D point in the pinocchio referential onto the sensor.
  - The top of the Eiffel Tower located $Z=324m$ above the ground (*$Z$  world coordinates*)
  - is at a distance $Y=6km$ away from the camera (*$Y$ in world coordinates*)
  - Camera is a Canon 5D MarkIII 24Mpix ($h=4000$, $w=6000$) camera 
  - The full frame sensor (24mm x 36mm) has a $\mu=6\text{µm}$ pixel pitch.
  - $f_{\text{pix}} = \frac{f=\text{focal length}}{\mu=\text{pixel pitch}}$
  - $y - \frac{h}{2}= \frac{Z.f}{Y} \approx \frac{-324*8333}{6000} \approx{-450pixels}$


--------

### Projecting the arm into the camera plane
|                     Camera                     |               World                |
| :--------------------------------------------: | :--------------------------------: |
| ![](/report/figures/arm_camera_projection.png) | ![](/report/figures/arm_world.png) |



#### Camera calibration
From a video, shoot a 7x10 checkerboard in multiple orientations.
Use the [Zhang method](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tr98-71.pdf)
```bash
python scripts/batch_video_processing.py -i "data/*cam*calib*.mp4" -o __out -A camera_calibration
```
![camera_calibration](/report/figures/camera_calibration.gif)


-----
#### Camera pose optimization
A least-square optimizer is used in order to estimate the camera translation with regard to the shoulder.
We ensure smoothness on camera/shoulder translation.


$min_{t_{\text{camera}}}||K [Q_{\text{camera}}, t_{\text{camera}}]. \vec{X} - \vec{x}||^2 + ||\frac{\Delta t_{\text{camera}}}{\Delta{t}}||^2$

In our simplified model, since the shoulder can fully rotated, we freeze camera the rotation $Q_{\text{camera}}=I_{3}$ and only optimize on translation.

Average 2D reprojection error on the whole sequence is in the order of 20 pixels on a FullHD video.
It is no totally unexpected to have such an error:
- As a matter of fact, when we standardize the arm length, we introduce an extra reprojection error in the 3D points.
- Then the arm state has been estimated to fit these 3D points simply using inverse kinematics. The IK solution does not minimize the 2D reprojection error in the current solution.
- To be fast for the demo, we compute the optimizer solution for each sliding windows of 30 frames. The optimization over the whole sequence takes quite a while (roughly 1 minute for a 1 minute video). 

![](/report/figures/camera_pose_fitting_first_100_frames.png)



### Next step
Estimate inverse dynamics of the system and regularize.
- velocities
- torque


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