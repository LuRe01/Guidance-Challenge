"""
user_guidance.py
----------------
This is the main file you will analyse and change for the SGN Guidance Challenge.

You are expected to work on two sections only:

  1. which_mission()  – Set the initial conditions for the scenario
                        (drone and target starting positions, velocities, target mode).

  2. my_guidance()    – Implement your guidance law for the seeker drone.
                        This is the core of the challenge.

Do NOT modify any other function. The target behaviour and the rest of the
software pipeline are managed by the organisers.

Available sensor readings for development (import core_data_access and call the functions below):

  CDA.get_drone_navdata()    →  [time, pos_x, pos_y, vel_x, vel_y, acc_x, acc_y, yaw]
  CDA.get_target_navdata()   →  [time, pos_x, pos_y, vel_x, vel_y, acc_x, acc_y]
  CDA.get_seeker_data()      →  [time, r, sigma]
  CDA.get_seeker_ext_data()  →  [time, r, sigma, r_dot, sigma_dot]

  where:
    r         – range between seeker and target [m]
    sigma     – Line-Of-Sight (LOS) angle [rad]
    r_dot     – range rate [m/s]
    sigma_dot – LOS angle rate [rad/s]
    yaw       – drone heading angle [rad]

Available sensor readings as feedback for the Challenge:    

  CDA.get_seeker_data()      →  [time, r, sigma]

Coordinate frame: Vicon Navigation Frame (planar, z = 0 for both drone and target).
"""

import math
import core_data_access as CDA
from logging_script import *
import numpy as np
import scipy.io as sio
import atexit


# ---------------------------------------------------------------------------
# Logger setup — do not modify unless 10 values are not enough 
# ---------------------------------------------------------------------------

# Number of custom variables you want to log (max 10)
private_guidance_data_length = 29


# Lista globale per immagazzinare la storia dei dati
log_history = []

def export_to_matlab():
    """Funzione che verrà eseguita automaticamente alla fine della simulazione"""
    if log_history:
        # np.hstack unisce tutti i vettori colonna (29x1).
        # .T traspone la matrice in modo da avere: Righe = Timestep, Colonne = Variabili (Nx29)
        data_matrix = np.hstack(log_history).T 
        
        # Salva un vero e proprio file MATLAB .mat
        sio.savemat('TeamPonio_log.mat', {'guidance_data': data_matrix})
        print("Log esportato con successo in TeamPonio_log.mat!")

# Il framework chiamerà questa funzione automaticamente quando lo script termina
atexit.register(export_to_matlab)



# Storage array and async logger for your private data
private_guidance_data = np.zeros((private_guidance_data_length, 1))
private_logger = AsyncMatlabPrint(flag=9, num_data=private_guidance_data_length)


# ---------------------------------------------------------------------------
# SECTION 1 — Initial Conditions
# ---------------------------------------------------------------------------

def which_mission():
    """
    Define the initial conditions for the guidance scenario.

    Both drone and target move in the XY plane (z = 0) inside the Vicon room.
    The framework will automatically stop the simulation if the drone exits the
    allowed region or exceeds the velocity limit.

    Constraints
    -----------
    - Seeker (drone) speed  : ||vel|| ≤ 0.5 m/s
    - Tracking area (X)     : [-1.4,  1.3] m
    - Tracking area (Y)     : [-1.4,  1.7] m

    Target modes
    ------------
    Set `target_mode` to one of the following integers:

      0 – Stationary / uniformly moving target (no lateral acceleration)
      1 – Switching acceleration  →  edit acceleration_switcher() below
      2 – Intelligent target      →  edit target_guidance_law() below

    What to modify
    --------------
    - init_pos    : drone starting position [x, y] in metres
    - heading_error : initial heading offset from the desired angle [degrees]
    - v_norm      : drone axial speed magnitude [m/s]  (must stay ≤ 0.5)
    - in_p        : target initial position [x, y, 0] in metres
    - v_angle_t   : target initial heading [rad]
    - v_norm_t    : target axial speed magnitude [m/s]
    - target_mode : integer 0 / 1 / 2  (see above)

    Returns
    -------
    tuple
        (init_pos, init_vel, in_p, in_v, target_mode)
    """

    # --- Drone initial conditions -------------------------------------------

    init_pos = [-1.0, -1.0]         # Starting position [x, y]  (metres)

    heading_error = 0               # Offset from the nominal heading angle [degrees]
   

    v_angle = ((90 - heading_error) * math.pi / 180.0)   # Heading angle [rad] - EXAMPLE
    v_norm  = 0.5                                          # Speed [m/s] — must be ≤ 0.5

    init_vel = np.array([v_norm * math.cos(v_angle),
                         v_norm * math.sin(v_angle)])

    # --- Target initial conditions -------------------------------------------

    in_p = np.array([1.0, 0.0, 0])                        # Position [x, y, 0]  (metres)

    v_angle_t = (225 * math.pi / 180.0)                     # Target heading [rad]
    v_norm_t  = 0.25                                        # Target speed [m/s]

    in_v = np.array([v_norm_t * math.cos(v_angle_t),
                     v_norm_t * math.sin(v_angle_t), 0])

    # --- Target mode ---------------------------------------------------------
    # 0: constant / no lateral acceleration
    # 1: switching acceleration  →  see acceleration_switcher()
    # 2: intelligent pursuit     →  see target_guidance_law()

    target_mode = 1                 # ← CHANGE THIS to select the target behaviour

    return init_pos, init_vel, in_p, in_v, target_mode


# ---------------------------------------------------------------------------
# SECTION 2 — Seeker Guidance Law
# ---------------------------------------------------------------------------

def my_guidance(data: list):
    """
    Compute the lateral acceleration command for the pursuer drone.

    This function is called at every simulation timestep by the framework.
    Your task is to implement a guidance law that drives the drone to intercept
    the target.

    Input argument
    --------------
    data : list
        Fading-filter estimates  [r, sigma, r_dot, sigma_dot]
        These are the same quantities available through get_seeker_ext_data(),
        but pre-filtered. You may use either source .

    Sensor readings available
    -------------------------
    Use the CDA functions at the top of this file.  Quick reference:

      ddata = CDA.get_drone_navdata()    # index  →  0:time  1:px  2:py  3:vx  4:vy  5:ax  6:ay  7:yaw
      tdata = CDA.get_target_navdata()   # index  →  0:time  1:px  2:py  3:vx  4:vy  5:ax  6:ay
      sdata = CDA.get_seeker_data()      # index  →  0:time  1:r   2:sigma
      edata = CDA.get_seeker_ext_data()  # index  →  0:time  1:r   2:sigma  3:r_dot  4:sigma_dot

    Acceleration limits
    -------------------
    The commanded acceleration must stay within [-0.75, +0.75] m/s².
    Values outside this range will be saturated.  ← CHANGE the saturation
    limits below if you want a tighter constraint for your strategy.

    Sign convention
    ---------------
    Positive acceleration  →  turn LEFT  (counter-clockwise)
    Negative acceleration  →  turn RIGHT (clockwise)

    Logger (optional)
    -----------------
    You can log up to 10 custom variables for post-run plotting.
    Example:

        global private_guidance_data, private_logger
        private_guidance_data[0] = CDA.get_drone_navdata()[0]   # time
        private_guidance_data[1] = CDA.get_seeker_data()[1]     # range r
        private_logger.append(private_guidance_data)

    Returns
    -------
    float
        Lateral acceleration command perpendicular to the drone velocity [m/s²]
    """

    # Declare globals to enable logging and persistent filter states
    global private_logger, private_guidance_data, log_history
    global ff_initialized, ff_time_old
    global ff_r, ff_r_dot, ff_sigma, ff_sigma_dot
    global ff_r_dot_raw, ff_sigma_dot_raw
    global ff_tan_acc_est, ff_tan_acc_old

    # ------------------------------------------------------------------
    # Sensor data available in the challenge
    # ------------------------------------------------------------------
    ddata = CDA.get_drone_navdata()
    sdata = CDA.get_seeker_data()

    time_now   = float(sdata[0])
    r_meas     = float(sdata[1])
    sigma_meas = wrap(float(sdata[2]))
    yaw        = wrap(float(ddata[7]))

    # ------------------------------------------------------------------
    # 2nd-order fading filter on r and sigma
    # States:
    #   r, r_dot
    #   sigma, sigma_dot
    #
    # Important:
    #   sigma residual is wrapped to avoid jumps near +/-pi.
    # ------------------------------------------------------------------
    beta_memory = 0.55

    G = 1.0 - beta_memory**2
    H = (1.0 - beta_memory)**2

    try:
        ff_initialized
    except NameError:
        ff_initialized = False

    if not ff_initialized:
        ff_r = max(r_meas, 1e-3)
        ff_r_dot = 0.0
        ff_sigma = sigma_meas
        ff_sigma_dot = 0.0
        ff_r_dot_raw = 0.0
        ff_sigma_dot_raw = 0.0
        ff_tan_acc_est = 0.0
        ff_tan_acc_old = 0.0
        ff_time_old = time_now
        ff_initialized = True

    dt = time_now - ff_time_old

    if dt > 1e-5:
        # --- r filter
        r_pred = ff_r + dt * ff_r_dot
        r_dot_pred = ff_r_dot
        e_r = r_meas - r_pred

        ff_r = max(r_pred + G * e_r, 1e-3)
        ff_r_dot = r_dot_pred + H * e_r / dt

        # --- sigma filter
        sigma_pred = wrap(ff_sigma + dt * ff_sigma_dot)
        sigma_dot_pred = ff_sigma_dot
        e_sigma = wrap(sigma_meas - sigma_pred)

        ff_sigma = wrap(sigma_pred + G * e_sigma)
        ff_sigma_dot = sigma_dot_pred + H * e_sigma / dt

        # Optional raw derivatives for logging/debug
        ff_r_dot_raw = (r_meas - ff_r) / dt
        ff_sigma_dot_raw = e_sigma / dt

        ff_time_old = time_now

    r = ff_r
    r_dot = ff_r_dot
    sigma = ff_sigma
    sigma_dot = ff_sigma_dot

    # Closing speed, positive during approach.
    Vc = max(-r_dot, 0.0)

    # ------------------------------------------------------------------
    # Proportional Navigation
    # ------------------------------------------------------------------
    N = 4.2

    # Use a small pursuit fallback when not closing, otherwise PNG alone may
    # command almost zero acceleration.
    sigma_err = wrap(yaw - sigma)
    pursuit_fallback = -0.35 * sigma_err

    acc_png = N * Vc * sigma_dot 

    if Vc < 0.03:
        acc = pursuit_fallback
    else:
        acc = acc_png + pursuit_fallback

    acc_sat = 0.70
    if acc > acc_sat:
        acc = acc_sat
    elif acc < -acc_sat:
        acc = -acc_sat

    private_guidance_data[0] = time_now
    private_guidance_data[1] = r_meas
    private_guidance_data[2] = r
    private_guidance_data[3] = r_dot
    private_guidance_data[4] = sigma_meas
    private_guidance_data[5] = sigma
    private_guidance_data[6] = sigma_dot
    private_guidance_data[7] = Vc
    private_guidance_data[8] = acc
    private_guidance_data[9] = CDA.get_seeker_ext_data()[4]   # Sigma_dot ext.
    private_guidance_data[10] = CDA.get_seeker_ext_data()[3]  # r_dot ext.
    private_guidance_data[11:18] = CDA.get_target_navdata() # target data 
    private_guidance_data[18:26] = CDA.get_drone_navdata()
    private_guidance_data[26:29] = CDA.get_seeker_data()   # seeker data
    

    # ---------------- NUOVA RIGA DA AGGIUNGERE ----------------
    # È FONDAMENTALE usare .copy() altrimenti salverai N volte sempre lo stesso valore!
    log_history.append(private_guidance_data.copy())
    # ----------------------------------------------------------

    private_logger.append(private_guidance_data)

    return acc


# ---------------------------------------------------------------------------
# SECTION 3 — Target Behaviour  (modes 1 and 2)
# !! READ ONLY — provided for reference. Do NOT modify. !!
# ---------------------------------------------------------------------------

def target_guidance_function(mode, t_sim, yaw):
    """
    Dispatcher that selects the target acceleration based on the chosen mode.

    Parameters
    ----------
    mode  : int    Target mode (0 / 1 / 2) — set in which_mission()
    t_sim : float  Current simulation time [s]
    yaw   : float  Current target heading angle [rad]

    Returns
    -------
    float
        Target lateral acceleration command [m/s²]
    """
    if mode == 0:
        # Mode 0: no lateral acceleration — target moves straight
        acc = 0
        return acc

    elif mode == 1:
        # Mode 1: pre-programmed switching acceleration — see acceleration_switcher()
        acc = acceleration_switcher(t_sim)
        return acc

    elif mode == 2:
        # Mode 2: intelligent target using a guidance law — see target_guidance_law()
        acc = target_guidance_law(t_sim, yaw)
        return acc


def acceleration_switcher(t_sim):
    """
    Define a time-based switching lateral acceleration profile for the target (mode 1).

    The target applies different constant accelerations across three time windows.

    Modify this function to test different combinations of maneuvers.

    Parameters
    ----------
    t_sim : float   Current simulation time [s]

    Returns
    -------
    float
        Target lateral acceleration [m/s²]
    """

    t1_switch = 50    # End of first phase [s]
    t2_switch = 100   # End of second phase [s]

    if 0 < t_sim <= t1_switch:
        acc = -0.075          # Phase 1 acceleration
    elif t1_switch < t_sim <= t2_switch:
        acc =  0.2            # Phase 2 acceleration
    else:
        acc = -0.25           # Phase 3 acceleration

    return acc


def target_guidance_law(t_sim, yaw):
    """
    Intelligent target guidance law (mode 2).

    The target actively tries to evade the seeker using a pursuit-based strategy
    combined with an Artificial Potential Field (APF) component that keeps the
    target away from the room boundaries.

    Parameters
    ----------
    t_sim : float   Current simulation time [s]
    yaw   : float   Current target heading angle [rad]

    Returns
    -------
    float
        Target lateral acceleration command [m/s²]  (saturated to [-0.5, +0.5])
    """

    # Read navigation data
    ddata = CDA.get_drone_navdata()
    tdata = CDA.get_target_navdata()
    sdata = CDA.get_seeker_data()

    # Target and drone positions
    t_pose_x   = tdata[1]
    t_pose_y   = tdata[2]
    drone_pose_x = ddata[1]
    drone_pose_y = ddata[2]

    # --- Pursuit component --------------------------------------------------
    # The target steers away from the seeker by reversing a pursuit law:
    # sigma (from target's perspective) is shifted by π to point away from drone.

    yaw   = wrap(yaw)
    sigma = wrap(math.pi + sdata[2])   # LOS from target toward drone [rad]
    sigma_r = wrap(yaw - sigma)        # Target heading error w.r.t. anti-LOS

    kp  = 1.5
    acc = kp * sigma_r

    # --- Last-ditch maneuver ------------------------------------------------
    # If the drone gets very close, apply a fixed evasive jink.

    target_pos = np.array([t_pose_x, t_pose_y])
    drone_pos  = np.array([drone_pose_x, drone_pose_y])
    rel_pos    = np.linalg.norm(drone_pos - target_pos, 2)   # Relative distance [m]

    if rel_pos < 0.2:
        acc = 0.3

    # Acceleration saturation
    if acc >  0.5:
        acc =  0.5
    elif acc < -0.5:
        acc = -0.5

    # --- Artificial Potential Field (APF) component -------------------------
    # If the target is close to the room boundary, override with a fixed
    # repulsive acceleration to prevent it from exiting the tracking area.

    x_max = 1.4   # Half-width of tracking area  [m]
    y_max = 1.4   # Half-height of tracking area [m]

    threshold = 0.7   # APF activates when |pos| > threshold * x/y_max
                      # (e.g. 0.7 means the APF kicks in at 70 % of the boundary)

    if (math.fabs(t_pose_x) > x_max * threshold or
            math.fabs(t_pose_y) > y_max * threshold):
        acc = 0.35

    return acc


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def wrap(angle):
    """
    Wrap an angle to the interval (-π, π].

    Parameters
    ----------
    angle : float   Input angle [rad]

    Returns
    -------
    float
        Equivalent angle in (-π, π]  [rad]
    """
    while angle >  math.pi:
        angle -= 2 * math.pi
    while angle < -math.pi:
        angle += 2 * math.pi
    return angle
