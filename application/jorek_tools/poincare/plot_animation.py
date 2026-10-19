from argparse import ArgumentParser
import numpy as np
from matplotlib import pyplot as plt
import re
from typing import List, Optional, Tuple
from enum import Enum
from dataclasses import dataclass

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

def load_files(filenames: List[str]) -> List[PostprocProfile]:
    if np.all([".vtk" in f for f in args.files]):
        frames = [
            read_poincare_vtk(f) for f in args.files
        ]
    else:
        frames = [
            read_postproc_profiles(f) for f in args.files
        ]
    return frames

def get_timesteps_from_filenames(filenames: List[str]) -> List[int]:
    return [int(re.findall(r'\d+', s)[0]) for s in args.files]

@dataclass
class PlotType:
    use_psi_coord: bool
    use_rho_coord: bool
    use_theta_coord: bool
    is_rz_plot: bool

def get_plot_type(filenames: List[str]) -> PlotType:
    use_psi_coord = np.all(['psi' in f for f in args.files])
    use_rho_coord = np.all(['rho' in f for f in args.files])
    use_theta_coord = np.all(['theta' in f for f in args.files])
    is_rz_plot = np.all(['R-Z' in f for f in args.files])

    return PlotType(
        use_psi_coord,
        use_rho_coord,
        use_theta_coord,
        is_rz_plot
    )

def create_plot(plot_type: PlotType) -> Tuple[plt.Figure, plt.Axes]:
    if plot_type.is_rz_plot:
        fig, ax = plt.subplots(1, figsize=(4,5))
        ax.set_aspect('equal')
        ax.set_xlabel("R (m)")
        ax.set_ylabel("Z (m)")
    else:
        fig, ax = plt.subplots(1, figsize=(5,4))
        if plot_type.use_rho_coord:
            ax.set_xlabel(r"$\rho/a$")
        elif plot_type.use_psi_coord:
            ax.set_xlabel(r"$\psi_N$")
        ax.set_ylabel(r"$\theta$")

    return fig, ax


class AnimatedPoincare():
    def __init__(self,
                 filenames: List[str],
                 timestep_map_filename: str,
                 r_range: Tuple[float, float], 
                 z_range: Tuple[float, float]):
        self.r_min, self.r_max = r_range
        self.z_min, self.z_max = z_range

        frames = load_files(filenames)
        timesteps = get_timesteps_from_filenames(filenames)
        self.timesteps, self.frames = zip(*sorted(zip(timesteps, frames)))
        self.frame = self.frames[0]

        self.plot_type = get_plot_type(filenames)
        self.fig, self.ax = create_plot(self.plot_type)
        self.setup_coordinates()
        self.sps = []
        self.setup_sps()
        
        self.tstep_map = None
        self.setup_tstep_map_if_exists(timestep_map_filename)
        
    def setup_tstep_map_if_exists(self, timestep_map_filename: str):
        try:
            self.tstep_map = read_timestep_map(timestep_map_filename)
            self.ax.set_title(f"Time {self.tstep_map.times[0]:.2g}s")
        except Exception as e:
            print(e)
            print("Unable to load timestep map. Plotting without...")

    def setup_coordinates(self):
        if self.plot_type.use_psi_coord:
            for frame in self.frames:
                for surface in frame:
                    # Convert psi_N -> s=sqrt(psi_N) co-ords
                    surface.x_vals = np.sqrt(surface.x_vals)


    def setup_sps(self):
        for i,surface in enumerate(self.frame):
            color='blue'
            alpha=1.0
            ms = args.marker_size

            self.sps.append(self.ax.scatter(
                surface.x_vals, surface.y_vals,
                marker=".",
                s=ms,
                color=color,
                alpha=alpha
            ))

    def bar(self, pos):
        if pos < 0 or pos > len(self.frames)-1:
            return

        self.curr_pos = pos
        pos = int(pos)
        frame = self.frames[pos]
        highlight_indices = [
                closest_surface_to_pos(frame, pos)
                for pos in self.position_markers
        ]
        update_offsets(
            self.sps, frame, args.marker_size, highlight_indices
        )
        
        if self.tstep_map:
            time = np.interp(
                self.timesteps[pos],
                self.tstep_map.time_steps,
                self.tstep_map.times
            )
            self.curr_time = time
            self.ax.set_title(f"Time {time:.4g}s")
        self.curr_minor_radius = get_minor_radius(frame)
        self.fig.canvas.draw()

    def on_mouse_move(self, event):
        if event.inaxes:
            self.mouse_pos = event.xdata, event.ydata

    def on_button_press(self, event):
        if event.key=='h':
            self.position_markers.append(self.mouse_pos)
            highlight_indices = [
                closest_surface_to_pos(self.frame, pos)
                for pos in self.position_markers
            ]
            update_offsets(
                self.sps, self.frame, args.marker_size, highlight_indices
            )
            self.fig.canvas.draw()
        elif event.key == 'r':
            if self.position_markers:
                self.position_markers = []
                update_offsets(
                    self.sps, self.frame, args.marker_size, []
                )
                self.fig.canvas.draw()
        elif event.key == 'm':
            self.r_measurements.append(self.mouse_pos[0])
            if len(self.r_measurements)==2:
                print(
                    self.curr_time, 
                    (self.r_measurements[1]-self.r_measurements[0])/self.curr_minor_radius
                )
                self.r_measurements = []
        elif event.key == ',':
            if self.curr_pos > 0:
                self.slider.set_val(self.curr_pos - 1)
        elif event.key == '.':
            if self.curr_pos < len(self.frames)-1:
                self.slider.set_val(self.curr_pos + 1)

    def setup_interactive_plot(self):
        from matplotlib.widgets import Slider

        self.position_markers = []
        self.curr_pos = 0
        self.mouse_pos = (0,0)
        self.r_measurements = []   

        self.curr_time = 0.0
        if self.tstep_map:
            self.curr_time = np.interp(
                self.timesteps[0],
                self.tstep_map.time_steps,
                self.tstep_map.times
            )
        self.curr_minor_radius = get_minor_radius(self.frame)

            
        barpos = plt.axes([0.18,0.0,0.65,0.03], facecolor="skyblue")
        self.slider = Slider(barpos, "Frame", 0, len(self.frames)-1, valinit=0, valstep=1)
        
        self.slider.on_changed(self.bar)
        self.fig.canvas.mpl_connect("motion_notify_event", self.on_mouse_move)
        self.fig.canvas.mpl_connect("key_press_event", self.on_button_press)

        self.fig.tight_layout()
        plt.show()

    
    def animate(self, pos):
        pos = int(pos)
        frame = self.frames[pos]

        update_offsets(self.sps, frame, args.marker_size)

        if self.tstep_map:
            time = np.interp(
                self.timesteps[pos],
                self.tstep_map.time_steps,
                self.tstep_map.times
            )
            self.ax.set_title(f"Time {time:.4g}s")

        return (self.sps,)

    def save_animation(self):
        self.fig.tight_layout()
        print("Saving animations...")
        from matplotlib.animation import FuncAnimation
        import os

        ani = FuncAnimation(self.fig, self.animate, frames=len(self.frames))

        if(self.plot_type.is_rz_plot):
            file_prefix = "poinc_R-Z"
        else:
            if self.plot_type.use_rho_coord:
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

    plot = AnimatedPoincare(
        args.files,
        args.timestep_map,
        args.r_range,
        args.z_range
    )

    if args.plot_interactive:
        plot.setup_interactive_plot()
    else:
        plot.save_animation()