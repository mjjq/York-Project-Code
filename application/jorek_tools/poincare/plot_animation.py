from argparse import ArgumentParser
import numpy as np
from matplotlib import pyplot as plt
import re
from typing import List, Optional, Tuple
from enum import Enum

from jorek_tools.jorek_dat_to_array import (
        read_timestep_map, read_postproc_profiles, PostprocProfile
)
from jorek_tools.poincare.read_poincare_vtk import read_poincare_vtk

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

def update_offsets(sps, 
                   frame: List[PostprocProfile], 
                   point_size: float,
                   highlight_indices: Optional[List[int]] = []):
    for i,sp in enumerate(sps):
        last_good_index = 0
        try:
            sp.set_offsets(np.column_stack((frame[i].x_vals, frame[i].y_vals)))
            if i in highlight_indices:
                sp.set_color('red')
                sp.set_sizes([50.0*point_size]*len(sp.get_sizes()))
            else:
                sp.set_color('blue')
                sp.set_sizes([point_size]*len(sp.get_sizes()))
            last_good_index = i
        except IndexError:
            sp.set_offsets(np.column_stack((
                frame[last_good_index].x_vals, frame[last_good_index].y_vals)
            ))
            if i in highlight_indices:
                sp.set_color('red')
                sp.set_sizes([50.0*point_size]*len(sp.get_sizes()))
            else:
                sp.set_color('blue')
                sp.set_sizes([point_size]*len(sp.get_sizes()))
            last_good_index = i


def closest_surface_to_pos(frame: List[PostprocProfile], 
                           pos: Tuple[float, float]) -> int:
    """
    Return closest surface index to a given position
    """
    x, y = pos

    min_dist_global = 1e6
    min_dist_global_surface = None
    for i,surface in enumerate(frame):
        distances_sq = (
            (surface.x_vals-x)**2 + (surface.y_vals-y)**2
        )
        min_dist_index = np.argmin(distances_sq)
        min_dist = distances_sq[min_dist_index]

        if min_dist < min_dist_global:
            min_dist_global = min_dist
            min_dist_global_surface = i

    return min_dist_global_surface

def intersect_linesegment_z0(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """
    Get R-coordinate of intersection between midplane and line segment p2-p1
    """
    p1x, p1y = p1
    p2x, p2y = p2

    return p1x - p1y*(p2x-p1x)/(p2y-p1y)


class MPLoc(Enum):
    inboard = 1
    outboard = 2
def get_midplane_intersections(surface: PostprocProfile, 
                               loc: MPLoc = MPLoc.outboard) -> float:
    """
    Get R-value of the intersection between a surface and the
    midplane.
    """
    centroid_R = np.mean(surface.x_values)
    centroid_Z = np.mean(surface.y_values)

    if loc==MPLoc.outboard:
        theta0 = 0.0
    elif loc==MPLoc.inboard:
        theta0 = np.pi/2.0
    else:
        raise ValueError("Must specify inboard or outboard")

    thetas = np.atan2(
        surface.y_values-centroid_Z,
        surface.x_values-centroid_R
    ) - theta0
    negative_branch = thetas[thetas < 0.0]
    positive_branch = thetas[thetas >= 0.0]

    closest_positive = np.argmin(positive_branch)
    closest_negative = np.argmin(negative_branch)
    
    p1 = (
        surface.x_values[closest_positive],
        surface.y_values[closest_positive]
    )
    p2 = (
        surface.x_values[closest_negative],
        surface.y_values[closest_negative]
    )
    r_midplane = intersect_linesegment_z0(p1, p2)

    return r_midplane

def get_minor_radius(frame: List[PostprocProfile]) -> float:
    max_r = -1e9
    min_r = 1e9
    for surface in frame:
        curr_max_r = np.max(surface.x_vals)
        if curr_max_r > max_r:
            max_r = curr_max_r

        curr_min_r = np.min(surface.x_vals)
        if curr_min_r < min_r:
            min_r = curr_min_r

    return 0.5*(max_r - min_r)

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
        '-i', '--surface-highlight-indices', type=int, nargs='+', help="Mark separate colour for flux surface indices",
        default=[]
    )
    args = parser.parse_args()

    r_min, r_max = args.r_range
    z_min, z_max = args.z_range

    if np.all([".vtk" in f for f in args.files]):
        frames = [
            read_poincare_vtk(f) for f in args.files
        ]
    else:
        frames = [
            read_postproc_profiles(f) for f in args.files
        ]

    timesteps = [int(re.findall(r'\d+', s)[0]) for s in args.files]
    timesteps,frames = zip(*sorted(zip(timesteps, frames)))

    use_psi_coord = np.all(['psi' in f for f in args.files])
    use_rho_coord = np.all(['rho' in f for f in args.files])
    use_theta_coord = np.all(['theta' in f for f in args.files])
    is_rz_plot = np.all(['R-Z' in f for f in args.files])

    if is_rz_plot:
        fig, ax = plt.subplots(1, figsize=(4,5))
        ax.set_aspect('equal')
        ax.set_xlabel("R (m)")
        ax.set_ylabel("Z (m)")
    else:
        fig, ax = plt.subplots(1, figsize=(5,4))
        if use_rho_coord:
            ax.set_xlabel(r"$\rho/a$")
        elif use_psi_coord:
            ax.set_xlabel(r"$\psi_N$")
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

    if use_psi_coord:
        for frame in frames:
            for surface in frame:
                # Convert psi_N -> s=sqrt(psi_N) co-ords
                surface.x_vals = np.sqrt(surface.x_vals)


    frame = frames[0]
    sps = []
    print(args.surface_highlight_indices, "highlight")
    for i,surface in enumerate(frame):
        color='blue'
        alpha=1.0
        ms = args.marker_size
        if args.surface_highlight_indices:
            if i in args.surface_highlight_indices:
                color='red'
                alpha=1.0
                ms = 2.0*args.marker_size

        sps.append(ax.scatter(
            surface.x_vals, surface.y_vals,
            marker=".",
            s=ms,
            color=color,
            alpha=alpha
        ))


    if args.plot_interactive:
        from matplotlib.widgets import Slider

        highlight_indices = args.surface_highlight_indices
        curr_time = np.interp(
            timesteps[0],
            tstep_map.time_steps,
            tstep_map.times
        )
        curr_minor_radius = get_minor_radius(frame)
        def bar(pos):
            global frame
            global curr_time
            global curr_minor_radius
            pos = int(pos)
            frame = frames[pos]
            update_offsets(
                sps, frame, args.marker_size, highlight_indices
            )
            
            if tstep_map:
                time = np.interp(
                    timesteps[pos],
                    tstep_map.time_steps,
                    tstep_map.times
                )
                curr_time = time
                ax.set_title(f"Time {time:.4g}s")
            curr_minor_radius = get_minor_radius(frame)
            fig.canvas.draw()
        barpos = plt.axes([0.18,0.0,0.65,0.03], facecolor="skyblue")
        slider = Slider(barpos, "Frame", 0, len(frames)-1, valinit=0, valstep=1)
        slider.on_changed(bar)

        mouse_pos = (0,0)
        def on_move(event):
            global mouse_pos
            if event.inaxes:
                mouse_pos = event.xdata, event.ydata
        fig.canvas.mpl_connect("motion_notify_event", on_move)

        r_measurements = []
        def onclick(event):
            global highlight_indices
            global r_measurements
            if event.key=='h':
                index = closest_surface_to_pos(frame, mouse_pos)
                if index is not None:
                    highlight_indices.append(index)
                    update_offsets(
                        sps, frame, args.marker_size, highlight_indices
                    )
                    fig.canvas.draw()
            elif event.key == 'r':
                if highlight_indices:
                    highlight_indices = []
                    update_offsets(
                        sps, frame, args.marker_size, highlight_indices
                    )
                    fig.canvas.draw()
            elif event.key == 'm':
                r_measurements.append(mouse_pos[0])
                if len(r_measurements)==2:
                    print(
                        curr_time, 
                        (r_measurements[1]-r_measurements[0])/curr_minor_radius
                    )
                    r_measurements = []
                
        fig.canvas.mpl_connect("key_press_event", onclick)


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

            update_offsets(sps, frame, args.marker_size)

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
            if use_rho_coord:
                file_prefix = "poinc_rho-theta"
            else:
                file_prefix = "poinc_psi-theta"

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




