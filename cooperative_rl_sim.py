"""
Simple simulation of cooperative relative localization (inspired by V. SIMULATIONS - A. Cooperative RL Simulation Results)
- Two agents: leader and follower.
- True trajectories generated deterministically.
- Odometry (v, omega) measured with noise for each agent.
- UWB range measurements between agents at lower rate.
- EKF that estimates relative pose of follower in leader frame using odometry integration for prediction and range for correction.

Assumptions / simplifications:
- UWB provides only range (no bearing).
- Agents share odometry (cooperative) so estimator uses both odometry signals for prediction.
- Process Jacobian is approximated; process noise Q is tuned as a constant.

Run: python cooperative_rl_sim.py
"""

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

# Utility functions
def wrap_angle(a):
    return (a + np.pi) % (2 * np.pi) - np.pi

def rotation(theta):
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([[c, -s], [s, c]])

# Integrate unicycle pose with control (v, omega)
def integrate_pose(pose, v, w, dt):
    x, y, th = pose
    if abs(w) < 1e-6:
        x += v * np.cos(th) * dt
        y += v * np.sin(th) * dt
        th += w * dt
    else:
        # exact integration for constant v, w
        x += (v / w) * (np.sin(th + w * dt) - np.sin(th))
        y += (v / w) * (-np.cos(th + w * dt) + np.cos(th))
        th += w * dt
    return np.array([x, y, wrap_angle(th)])

# Simulation parameters
dt = 0.02
T = 40.0
N = int(T / dt)
range_dt = 0.2  # UWB rate
range_steps = int(range_dt / dt)

# True trajectories: leader on circle, follower on offset circle
leader_pose = np.array([0.0, 0.0, 0.0])
follower_pose = np.array([ -1.5, 0.0, 0.0])

# Controls for leader and follower (simple periodic commands)
def leader_control(t):
    # constant forward speed, small angular velocity to make circular
    v = 0.8
    w = 0.1 * np.cos(0.05 * t)
    return v, w

def follower_control(t):
    # follower tries to follow leader path but with different pattern
    v = 0.9
    w = 0.12 * np.sin(0.04 * t + 0.5)
    return v, w

# Noise parameters
odom_v_std = 0.05  # m/s
odom_w_std = 0.02  # rad/s
range_std = 0.15   # m

# EKF state: relative pose r = [dx, dy, dtheta] where dx,dy are follower in leader frame
r_est = np.array([follower_pose[0] - leader_pose[0], follower_pose[1] - leader_pose[1], wrap_angle(follower_pose[2] - leader_pose[2])])
P = np.diag([0.5, 0.5, 0.2])
Q = np.diag([0.02, 0.02, 0.01])  # process noise (tuned)
R_meas = np.array([[range_std**2]])

# Also maintain odometry-only global pose estimates by integrating noisy odometry
leader_pose_est = leader_pose.copy()
follower_pose_est = follower_pose.copy()

# Logging
true_leader = np.zeros((N,3))
true_follower = np.zeros((N,3))
odom_leader = np.zeros((N,3))
odom_follower = np.zeros((N,3))
rel_estimates = np.zeros((N,3))
rel_covs = np.zeros((N,3))
range_meas_log = np.full(N, np.nan)

# Simulation loop
for k in range(N):
    t = k * dt
    # true controls
    v_l, w_l = leader_control(t)
    v_f, w_f = follower_control(t)

    # propagate true poses
    leader_pose = integrate_pose(leader_pose, v_l, w_l, dt)
    follower_pose = integrate_pose(follower_pose, v_f, w_f, dt)

    true_leader[k] = leader_pose
    true_follower[k] = follower_pose

    # odometry measurements (noisy controls) --- agents share them
    v_l_m = v_l + np.random.randn() * odom_v_std
    w_l_m = w_l + np.random.randn() * odom_w_std
    v_f_m = v_f + np.random.randn() * odom_v_std
    w_f_m = w_f + np.random.randn() * odom_w_std

    # integrate odometry-based global estimates
    leader_pose_est = integrate_pose(leader_pose_est, v_l_m, w_l_m, dt)
    follower_pose_est = integrate_pose(follower_pose_est, v_f_m, w_f_m, dt)

    odom_leader[k] = leader_pose_est
    odom_follower[k] = follower_pose_est

    # EKF prediction: compute relative pose from odom-based global estimates
    # r_pred = R(theta_l_est)^T * (p_f_est - p_l_est)
    Rl = rotation(leader_pose_est[2])
    p_diff = follower_pose_est[:2] - leader_pose_est[:2]
    dxdy = Rl.T @ p_diff
    dth = wrap_angle(follower_pose_est[2] - leader_pose_est[2])
    r_pred = np.array([dxdy[0], dxdy[1], dth])

    # For covariance, use simple model: P = P + Q*dt (approx)
    P = P + Q * dt

    # Measurement update when range available
    if k % range_steps == 0:
        # measure true range with noise
        true_rel = np.array([ (rotation(true_leader[k,2]).T @ (true_follower[k,:2] - true_leader[k,:2]))[0],
                              (rotation(true_leader[k,2]).T @ (true_follower[k,:2] - true_leader[k,:2]))[1],
                              wrap_angle(true_follower[k,2] - true_leader[k,2])])
        true_range = np.linalg.norm(true_rel[:2])
        z = true_range + np.random.randn() * range_std
        range_meas_log[k] = z

        # EKF update using r_pred as prior
        r = r_pred.copy()
        # measurement prediction
        r_norm = max(1e-6, np.linalg.norm(r[:2]))
        z_pred = r_norm
        # Jacobian H = [dx/r, dy/r, 0]
        H = np.array([[r[0] / r_norm, r[1] / r_norm, 0.0]])

        S = H @ P @ H.T + R_meas
        K = (P @ H.T) @ np.linalg.inv(S)
        y = z - z_pred
        r_upd = r + (K.flatten() * y)
        r_upd[2] = wrap_angle(r_upd[2])
        P = (np.eye(3) - K @ H) @ P

        r_est = r_upd
    else:
        # no measurement: keep predicted
        r_est = r_pred

    rel_estimates[k] = r_est
    rel_covs[k] = np.array([P[0,0], P[1,1], P[2,2]])

# Post-process: reconstruct global estimated follower pose from leader true pose + relative estimate rotated
# We'll compare three trajectories: true follower, odometry-only follower_est, EKF-corrected follower (leader true + est rel)
recon_follower_from_ekf = np.zeros((N,3))
for k in range(N):
    rl = rel_estimates[k]
    leader_th = true_leader[k,2]
    Rl = rotation(leader_th)
    p_f_global = true_leader[k,:2] + Rl @ rl[:2]
    th = wrap_angle(leader_th + rl[2])
    recon_follower_from_ekf[k] = np.array([p_f_global[0], p_f_global[1], th])

# Compute errors
pos_err_odom = np.linalg.norm(true_follower[:,:2] - odom_follower[:,:2], axis=1)
pos_err_ekf = np.linalg.norm(true_follower[:,:2] - recon_follower_from_ekf[:,:2], axis=1)
range_err = np.abs(np.linalg.norm(true_follower[:,:2] - true_leader[:,:2], axis=1) - np.nan_to_num(range_meas_log, np.nan))

# Plotting
plt.figure(figsize=(10,6))
tvec = np.arange(N) * dt
plt.subplot(2,2,1)
plt.title('Trajectories (global)')
plt.plot(true_leader[:,0], true_leader[:,1], 'k-', label='Leader true')
plt.plot(true_follower[:,0], true_follower[:,1], 'k--', label='Follower true')
plt.plot(odom_leader[:,0], odom_leader[:,1], 'r-', alpha=0.6, label='Leader odom')
plt.plot(odom_follower[:,0], odom_follower[:,1], 'r--', alpha=0.6, label='Follower odom')
plt.plot(recon_follower_from_ekf[:,0], recon_follower_from_ekf[:,1], 'b-', linewidth=1.5, label='Follower EKF')
plt.axis('equal')
plt.legend()

plt.subplot(2,2,2)
plt.title('Position error (m)')
plt.plot(np.arange(N)*dt, pos_err_odom, 'r-', label='Odom-only')
plt.plot(np.arange(N)*dt, pos_err_ekf, 'b-', label='EKF')
plt.xlabel('Time [s]')
plt.ylabel('Position error [m]')
plt.legend()

plt.subplot(2,2,3)
plt.title('Range measurements vs true')
plt.plot(tvec, np.linalg.norm(true_follower[:,:2] - true_leader[:,:2], axis=1), 'k-', label='True range')
meas_idx = ~np.isnan(range_meas_log)
plt.plot(tvec[meas_idx], range_meas_log[meas_idx], 'gx', label='UWB measurement')
plt.xlabel('Time [s]')
plt.ylabel('Range [m]')
plt.legend()

plt.subplot(2,2,4)
plt.title('EKF covariance (diag)')
plt.plot(np.arange(N)*dt, rel_covs[:,0], label='P_xx')
plt.plot(np.arange(N)*dt, rel_covs[:,1], label='P_yy')
plt.plot(np.arange(N)*dt, rel_covs[:,2], label='P_thth')
plt.xlabel('Time [s]')
plt.legend()

plt.tight_layout()
plt.show()

# Save figures for convenience
plt.savefig('cooperative_rl_results.png', dpi=200)

print('Simulation finished. Mean pos error odom: {:.3f} m, EKF: {:.3f} m'.format(np.mean(pos_err_odom), np.mean(pos_err_ekf)))
