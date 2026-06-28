from argparse import ArgumentParser
import numpy as np
from matplotlib import pyplot as plt
import re

from jorek_tools.jorek_dat_to_array import read_timestep_map, read_postproc_profiles

# Source - https://stackoverflow.com/a/51778313
# Posted by tmakaro, modified by community. See post 'Timeline' for change history
# Retrieved 2026-05-19, License - CC BY-SA 4.0

from matplotlib.animation import FileMovieWriter
from pathlib import Path

class BunchOFiles(FileMovieWriter):
    supported_formats = ['png', 'jpeg', 'bmp', 'svg', 'pdf']

    def __init__(self, *args, extra_args=None, **kwargs):
        # extra_args aren't used but we need to stop None from being passed
        super().__init__(*args, extra_args=(), **kwargs)

    def setup(self, fig, filename, dpi):
        super().setup(fig, filename, dpi=300.0)
        self.fname_format_str = '%s%%d.%s'
        print(self.outfile)
        self.temp_prefix, self.frame_format = self.outfile.split('.')

    def grab_frame(self, **savefig_kwargs):
        '''
        Grab the image information from the figure and save as a movie frame.
        All keyword arguments in savefig_kwargs are passed on to the 'savefig'
        command that saves the figure.
        '''
        # docstring inherited
        # Creates a filename for saving using basename and counter.
        #_validate_grabframe_kwargs(savefig_kwargs)
        path = Path(self._base_temp_name() % self._frame_counter)
        self._temp_paths.append(path)  # Record the filename for later use.
        self._frame_counter += 1  # Ensures each created name is unique.
        #_log.debug('FileMovieWriter.grab_frame: Grabbing frame %d to path=%s',
        #           self._frame_counter, path)
        with open(path, 'wb') as sink:  # Save figure to the sink.
            self.fig.savefig(sink, format=self.frame_format, dpi=self.dpi,
                             **savefig_kwargs)

    def finish(self):
        #self._frame_sink().close()
        return

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
        "-ms", "--marker-size", type=float, help="Size of scatter markers", default=0.1
    )
    parser.add_argument(
        "-t", "--timestep-map", type=str, help="Path to timestep->time map", default="time_map.txt"
    )
    parser.add_argument(
        "-p", "--plot-interactive", action="store_true", help="Whether to plot interactively"
    )
    parser.add_argument(
        '-i', '--surface-highlight-indices', nargs='+', help="Mark separate colour for flux surface indices", default=[], type=int
    )
    parser.add_argument(
        '-o', '--only-plot-highlighted', action='store_true', help='Only plot highlighted surfaces'
    )
    args = parser.parse_args()

    r_min, r_max = args.r_range
    z_min, z_max = args.z_range

    frames = [
        read_postproc_profiles(f) for f in args.files
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

    frame = frames[0]
    sps = []
    for i,surface in enumerate(frame):
        color='blue'
        alpha=0.1
        s=args.marker_size
        if args.only_plot_highlighted:
            alpha=0.0
        if i in args.surface_highlight_indices:
            color='red'
            alpha=1.0
            s=10.0*s
        sps.append(ax.scatter(
            surface.x_vals, surface.y_vals,
            marker="o",
            s=s,
            color=color,
            alpha=alpha
        ))


    if args.plot_interactive:
        from matplotlib.widgets import Slider
        def bar(pos):
            pos = int(pos)
            frame = frames[pos]
            for i,sp in enumerate(sps):
                sp.set_offsets(frame[i].x_vals, frame[i].y_vals)

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
            frame = frames[pos]
            for i,sp in enumerate(sps):
                sp.set_offsets(frame[i].x_vals, frame[i].y_vals)

            if tstep_map:
                time = np.interp(
                    timesteps[pos],
                    tstep_map.time_steps,
                    tstep_map.times
                )
                ax.set_title(f"Time {time:.4g}s")

            return (sps,)

        ani = FuncAnimation(fig, animate, frames=len(frames))

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
        ani.save(f"{folder}/{file_prefix}.png", writer=BunchOFiles(),dpi=300)




