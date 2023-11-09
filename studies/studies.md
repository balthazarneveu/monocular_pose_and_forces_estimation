# Physcap
[Physcap](https://vcai.mpi-inf.mpg.de/projects/PhysCap/) is a paper from 2020 from Max Planck institute 
which tackles physically plausible pose estimation (without the use of "tools").
- claims on real time application.

![](figures/physcap.png)

## :one: Pose estimation
- *Kinematic* : estimate joint angles and velocities (no consideration of forces or torque)
- :one: **Pose estimation** Built on [VNect](https://vcai.mpi-inf.mpg.de/projects/VNect/) from Max Planck Institute.  
  - :rocket: "Real time"
  - :dancer: `MPI-INF-3DHP` : augmented videos from green screen markerless captures.
  - Predicts 2D & 3D joints positions.
- :two: Refinement step by energy minimization.
## :two: Contact detection


## :three: Physics based pose optimization

# Extra
[Ranking of pose estimators methods](https://paperswithcode.com/sota/3d-human-pose-estimation-on-mpi-inf-3dhp)