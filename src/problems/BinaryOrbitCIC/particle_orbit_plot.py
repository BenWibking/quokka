# note: requires yt>=4.3.0
import yt

def particle_dist(plotfiles):
    t_arr = []
    err_arr = []
    d0 = 2.0 * 3.125e12

    for pltfile in plotfiles:
        ds = yt.load(pltfile)
        Lx = ds.domain_right_edge[0] - ds.domain_left_edge[0]
        Nx = ds.domain_dimensions[0]
        cell_dx = Lx/Nx
        ad = ds.all_data()
        x = ad["CIC_particles", "particle_position_x"]
        y = ad["CIC_particles", "particle_position_y"]
        z = ad["CIC_particles", "particle_position_z"]
        dx = x[0] - x[1]
        dy = y[0] - y[1]
        dz = z[0] - z[1]
        from math import sqrt
        d = sqrt(dx*dx + dy*dy + dz*dz)
        #fractional_err = (d-d0)/d0
        grid_err = (d-d0)/cell_dx
        t_arr.append(float(ds.current_time) / 3.15e7)
        err_arr.append(grid_err)
    
    return t_arr, err_arr

import glob
files = glob.glob("plt*")
files = sorted(files)
t, err = particle_dist(files)

import matplotlib.pyplot as plt
plt.figure(figsize=(6,4))
plt.plot(t, err)
plt.xlabel("time (yr)")
plt.ylabel(r"$(d-d_0)/\Delta x$")
plt.tight_layout()
plt.savefig("orbit.png", dpi=150)
