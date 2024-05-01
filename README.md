# Bluerov2 Trajectory dataset


This opensource dataset contains trajectories obtained with the bluerov2 robot. 


## Description


The dataset is stored in CSV files containing different trajectories. The variables are: 

- x,y,z: position in world frame
- qx, qy, qz, qw: Quaternions
- Bu, Bv, Bw: the linear velocity in body frame
- Bq, Bp, Br: the angular velocity in body frame
- Ux, Uy, Uz, Vx, Vy, Vz: the normalized input force (in body frame)
- Fx, Fy, Fz, Tx, Ty, Tz: input forces in body frame
- pwm_1 -> pwm_8: the raw pwm inputs to the eight thrusters