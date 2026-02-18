from argparse import ArgumentParser
import numpy as np
from matplotlib import pyplot as plt
import re

from jorek_tools.jorek_dat_to_array import read_timestep_map

if __name__=='__main__':
    parser = ArgumentParser()

    parser.add_argument(
        "files", nargs='+', type=str, help="Sequence of poincare .txt files"
    )
    parser.add_argument(
        "-r", "--r-range", type=float, nargs=2, help="R-min, R-max region to plot (or rho)", 
        default=(None, None)
    )
    parser.add_argument(
        "-z", "--z-range", type=float, nargs=2, help="Z-min, Z-max region to plot (or theta)",
        default=(None, None)
    )
    parser.add_argument(
        "-ms", "--marker-size", type=float, help="Size of scatter markers", default=0.125
    )
    parser.add_argument(
        "-t", "--timestep-map", type=str, help="Path to timestep->time map", default="time_map.txt"
    )
    parser.add_argument(
        "-p", "--plot-interactive", action="store_true", help="Whether to plot interactively"
    )
    args = parser.parse_args()

    r_min, r_max = args.r_range
    z_min, z_max = args.z_range

    frames = [
        np.loadtxt(f) for f in args.files
    ]
    timesteps = [int(re.findall(r'\d+', s)[0]) for s in args.files]


    is_rz_plot = np.all(['R-Z' in f for f in args.files])

    if is_rz_plot:
        fig, ax = plt.subplots(1, figsize=(4,5))
        ax.set_aspect('equal')
        ax.set_xlabel("R (m)")
        ax.set_ylabel("Z (m)")
    else:
        fig, ax = plt.subplots(1, figsize=(5,4))
        ax.set_xlabel(r"$\rho/a$")
        ax.set_ylabel(r"$\theta$")

    if(np.all(args.r_range)):
        ax.set_xlim(args.r_range)
    if(np.all(args.z_range)):
        ax.set_ylim(args.z_range)

    
    tstep_map = None
    try:
        tstep_map = read_timestep_map(args.timestep_map)
        ax.set_title(f"Time {tstep_map.times[0]:.2g}s")
    except Exception as e:
        print(e)
        print("Unable to load timestep map. Plotting without...")

    sp = ax.scatter(
        frames[0][:,0], frames[0][:,1],
        marker=".",
        s=args.marker_size
    )


    if args.plot_interactive:
        from matplotlib.widgets import Slider
        def bar(pos):
            pos = int(pos)
            sp.set_offsets(frames[pos])

            if tstep_map:
                time = np.interp(
                    timesteps[pos],
                    tstep_map.time_steps,
                    tstep_map.times
                )
                ax.set_title(f"Time {time:.4g}s")

            fig.canvas.draw()

        barpos = plt.axes([0.18,0.0,0.65,0.03], facecolor="skyblue")
        slider = Slider(barpos, "Frame", 0, len(frames)-1, valinit=0, valstep=1)
        slider.on_changed(bar)
        fig.tight_layout()
        plt.show()

    else:
        fig.tight_layout()
        print("Saving animations...")
        from matplotlib.animation import FuncAnimation
        import os
        def animate(pos):
            pos = int(pos)
            sp.set_offsets(frames[pos])

            if tstep_map:
                time = np.interp(
                    timesteps[pos],
                    tstep_map.time_steps,
                    tstep_map.times
                )
                ax.set_title(f"Time {time:.4g}s")

            return (sp,)

        ani = FuncAnimation(fig, animate, frames=len(frames)-1)

        if(is_rz_plot):
            file_prefix = "poinc_R-Z"
        else:
            file_prefix = "poinc_rho-theta"

        folder = "poincare_output"
        try:
            os.mkdir(folder)
        except FileExistsError:
            print(f"{folder} already exists.")

        try:
            ani.save(f"{folder}/{file_prefix}.mp4", writer="ffmpeg",dpi=150)
        except:
            print("Couldn't save as .mp4. Missing ffmpeg. Trying .gif...")
            ani.save(f"{folder}/{file_prefix}.gif", writer="imagemagick",dpi=150)
        ani.save(f"{folder}/{file_prefix}.png", writer="imagemagick",dpi=300)




