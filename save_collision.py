import numpy as np

'''
Program that takes a relaxed initial planet, clones it, and adds velocities such that the planets collide.
'''

def save_collision(relaxed_data, output, separation_r, impact_speed, collision_axis=0):

    planet = np.loadtxt(relaxed_data)
    x, y, z, vx, vy, vz, m, rho, p = np.hsplit(planet, 9)

    positions = np.hstack((x, y, z))
    velocities = np.hstack((vx, vy, vz))
    m = m.flatten()

    N = len(m)

    total_mass = np.sum(m)
    com = np.sum(positions * m[:, None], axis=0) / total_mass

    positions -= com

    positions_clone = positions.copy()
    velocities_clone = velocities.copy()
    m_clone = m.copy()
    rho_clone = rho.copy()
    p_clone = p.copy()

    shift = np.zeros(3)
    shift[collision_axis] = separation_r / 2.0

    positions_1 = positions - shift
    positions_2 = positions_clone + shift
    positions_2[:, 1] += 5 * 10**9 # impact_parameter (offset)

    v_bulk = np.zeros(3)
    v_bulk[collision_axis] = impact_speed / 2.0

    velocities_1 = velocities + v_bulk
    velocities_2 = velocities_clone - v_bulk

    all_positions = np.vstack((positions_1, positions_2))
    all_velocities = np.vstack((velocities_1, velocities_2))
    all_m = np.concatenate((m, m_clone))
    all_rho = np.concatenate((rho.flatten(), rho_clone.flatten()))
    all_p = np.concatenate((p.flatten(), p_clone.flatten()))

    output_array = np.column_stack([all_positions, all_velocities, all_m, all_rho, all_p])

    np.savetxt(output, output_array)

    print(f"Saved {2*N} particles to {output}")

if __name__ == "__main__":
    save_collision(relaxed_data='./data/Planet300_relaxed.dat', output='./data/Planet300_collision.dat', separation_r=5e9, impact_speed=1300)