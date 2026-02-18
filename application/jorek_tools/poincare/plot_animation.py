from argparse import ArgumentParser
import numpy as np

from matplotlib import pyplot as plt

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
        "-p", "--plot-interactive", action="store_true", help="Whether to plot interactively"
    )
    args = parser.parse_args()

    r_min, r_max = args.r_range
    z_min, z_max = args.z_range

    frames = [
        np.loadtxt(f) for f in args.files
    ]


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
            fig.canvas.draw()

        barpos = plt.axes([0.18,0.0,0.65,0.03], facecolor="skyblue")
        slider = Slider(barpos, "Frame", 0, len(frames), valinit=0, valstep=1)
        slider.on_changed(bar)
        fig.tight_layout()
        plt.show()

    else:
        print("Saving animations...")
        from matplotlib.animation import FuncAnimation
        def animate(pos):
            pos = int(pos)
            sp.set_offsets(frames[pos])
            return (sp,)

        ani = FuncAnimation(fig, animate, frames=len(frames))
        
        if(is_rz_plot):
            file_prefix = "poinc_R-Z"
        else:
            file_prefix = "poinc_rho-theta"

        try:
            ani.save(f"{file_prefix}.mp4", writer="ffmpeg",dpi=150)
        except:
            print("Couldn't save as .mp4. Missing ffmpeg. Trying .gif...")
            ani.save(f"{file_prefix}.gif", writer="imagemagick",dpi=150)
        ani.save(f"{file_prefix}.png", writer="imagemagick",dpi=300)




