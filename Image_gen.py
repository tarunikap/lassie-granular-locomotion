import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Arc

def rotation_matrix(deg):
    rad = np.radians(deg)
    return np.array([
        [np.cos(rad), -np.sin(rad)],
        [np.sin(rad),  np.cos(rad)]
    ])

def draw_angle_diagram(ax, beta, theta, v_direction, fig_index):
    mirror_fy = False
    if abs(beta) > 90:
        mirror_fy = True

    ax.plot([-2, 2], [0, 0], 'k--', lw=1)  # Fx axis dotted
    ax.plot([0, 0], [-2, 2], 'k--', lw=1)  # Fy axis dotted

    # Fx and Fy arrows
    fx_arrow_end = (-1.9, 0) if mirror_fy else (1.9, 0)
    ax.annotate('', xy=fx_arrow_end, xytext=(0, 0),
                arrowprops=dict(facecolor='forestgreen', edgecolor='forestgreen', linewidth=3, arrowstyle='-|>', mutation_scale=20))
    ax.annotate('', xy=(0, 1.9), xytext=(0, 0),
                arrowprops=dict(facecolor='dodgerblue', edgecolor='dodgerblue', linewidth=3, arrowstyle='-|>', mutation_scale=20))
    ax.text(1.6 * (-1 if mirror_fy else 1), 0.05, r'$F_x$', color='forestgreen', fontsize=15)
    ax.text(0.05, 1.7, r'$F_y$', color='dodgerblue', fontsize=15)

    # Plate (rectangular block)
    plate_angle = -beta
    rect_length = 2.0
    rect_width = 0.2
    rect = np.array([
        [-rect_length/2, -rect_width/2],
        [ rect_length/2, -rect_width/2],
        [ rect_length/2,  rect_width/2],
        [-rect_length/2,  rect_width/2],
        [-rect_length/2, -rect_width/2]
    ])
    R_plate = rotation_matrix(plate_angle)
    rect_pts = (R_plate @ rect.T).T
    if mirror_fy:
        rect_pts[:, 0] *= -1
    ax.fill(rect_pts[:,0], rect_pts[:,1], 'k')

    # Centerline of plate
    extended_length = 2.5
    centerline_pts = np.array([[-extended_length/2, 0], [extended_length/2, 0]])
    centerline_rot = (R_plate @ centerline_pts.T).T
    if mirror_fy:
        centerline_rot[:, 0] *= -1
    ax.plot(centerline_rot[:,0], centerline_rot[:,1], color='crimson', lw=2, linestyle='--')

    # Plate direction
    plate_dir_angle = np.degrees(np.arctan2(centerline_rot[0,1], centerline_rot[0,0])) % 360
    d_angle = (plate_dir_angle - theta) % 360
    d_vec = rotation_matrix(d_angle) @ np.array([1.2, 0])
    if mirror_fy:
        d_vec[0] *= -1

    if fig_index != 2:
        ax.plot([0, d_vec[0]], [0, d_vec[1]], linestyle=':', color='royalblue', lw=2)

    # Velocity vector
    sign = 1 if v_direction == 'up' else -1
    v_vec = sign * d_vec / np.linalg.norm(d_vec) * 1.5
    if mirror_fy:
        v_vec[0] *= -1
    ax.arrow(0, 0, v_vec[0], v_vec[1], head_width=0.09, head_length=0.13,
             fc='teal', ec='teal', lw=2, length_includes_head=True)
    ax.text(v_vec[0]*1.12, v_vec[1]*1.12, r'$v$', fontsize=15, color='teal', ha='center')

    # ANGLE ARCS
    v_angle = np.degrees(np.arctan2(v_vec[1], v_vec[0])) % 360
    gamma = (v_angle - 180) % 360
    if gamma > 180:
        gamma -= 360

    # Mirror if gamma out of range too
    if abs(gamma) > 90 and not mirror_fy:
        mirror_fy = True

    gamma_radius = 1.5
    beta_radius = 0.9
    theta_radius = 1.15

    # Beta arc
    if fig_index == 2:
        beta_start = (360 - 180) % 360
        beta_end = (360 - (180 - beta)) % 360
        arc_beta = Arc((0, 0), beta_radius, beta_radius, theta1=beta_start, theta2=beta_end, color='deepskyblue', lw=2)
        ax.add_patch(arc_beta)
        mid_beta = (beta_start + (beta_end - beta_start) / 2) % 360
        beta_label = rotation_matrix(mid_beta) @ np.array([beta_radius/2, 0])
        ax.text(beta_label[0], beta_label[1]+0.1, rf'$\beta$ = -{abs(beta):.0f}°', fontsize=13, color='deepskyblue', ha='center')
    else:
        beta_start = 180 - beta
        beta_end = 180
        arc_beta = Arc((0, 0), beta_radius, beta_radius, theta1=beta_start, theta2=beta_end, color='deepskyblue', lw=2)
        ax.add_patch(arc_beta)
        mid_beta = (beta_start + (beta_end - beta_start) / 2) % 360
        beta_label = rotation_matrix(mid_beta) @ np.array([beta_radius/2, 0])
        ax.text(beta_label[0]-0.5, beta_label[1]+0.12, rf'$\beta$ = {abs(beta):.0f}°', fontsize=13, color='deepskyblue')

    # Gamma arc
    gamma_start = 180
    gamma_end = v_angle
    arc_gamma = Arc((0,0), gamma_radius, gamma_radius, theta1=gamma_start, theta2=gamma_end, color='k', lw=1.5)
    ax.add_patch(arc_gamma)
    mid_gamma = (gamma_start + (gamma_end - gamma_start)/2) % 360
    label_gamma = rotation_matrix(mid_gamma) @ np.array([gamma_radius * 0.8, 0])
    sign_text = '-' if fig_index == 2 else ''
    ax.text(label_gamma[0], label_gamma[1]-0.15, rf'$\gamma$ = {sign_text}{abs(gamma):.0f}°', fontsize=13)

    # Theta arc
    angle_diff = (d_angle - plate_dir_angle + 360) % 360
    if angle_diff > 180:
        theta_arc_angle = 360 - angle_diff
        theta_start = d_angle
        theta_end = plate_dir_angle
    else:
        theta_arc_angle = angle_diff
        theta_start = plate_dir_angle
        theta_end = d_angle

    arc_theta = Arc((0,0), theta_radius, theta_radius, theta1=theta_start, theta2=theta_end, color='orange', lw=2)
    ax.add_patch(arc_theta)
    mid_theta = (theta_start + (theta_end - theta_start)/2) % 360
    label_theta = rotation_matrix(mid_theta) @ np.array([theta_radius/2, 0])
    ax.text(label_theta[0], label_theta[1]+0.08, rf'$\theta$ = {theta_arc_angle:.0f}°', fontsize=13, color='orange')

    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_aspect('equal')
    ax.axis('off')

def plot_diagram(beta, theta):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    draw_angle_diagram(ax=axes[0], beta=beta, theta=theta, v_direction='down', fig_index=0)
    axes[0].set_title("Push in")

    draw_angle_diagram(ax=axes[1], beta=beta, theta=theta, v_direction='up', fig_index=1)
    axes[1].set_title("Pull back")

    draw_angle_diagram(ax=axes[2], beta=beta, theta=theta, v_direction='down', fig_index=2)
    axes[2].set_title("Mirror along $F_y$")

    plt.tight_layout()
    plt.show()

# 🔽 CHANGE these values as needed
beta_input = 15
theta_input = 45

plot_diagram(beta_input, theta_input)
