# Standard library imports
import os
import configparser
import shutil
import tkinter as tk
from tkinter import ttk
from pathlib import Path
import logging
import matplotlib as mpl
import ctypes
import matplotlib.pyplot as plt
import platform

# Third-party library imports
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.widgets import RectangleSelector
from RangeSlider.RangeSlider import RangeSliderV
from PIL import Image, ImageTk

# Custom module imports
from moove.utils import (
    get_display_data, get_directories, read_batch, get_file_data_by_index,
    save_seg_class_recfile, plot_data, select_event, edit_syllable,
    handle_keypress, zoom, unzoom, swipe_left, swipe_right, handle_playback,
    handle_delete, handle_crop, open_resegment_window, update,
    open_cluster_window, open_training_window, open_relabel_window, find_batch_files,
    create_batch_file
)

# Model imports
from moove.models.ConvMLP import ConvMLP
from moove.models.CNN import CNN

# Global config imports
from moove.app_state import AppState  # App state management

# Avoid key binding conflicts
mpl.rcParams['keymap.yscale'] = ''
mpl.rcParams['keymap.xscale'] = ''

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create / read config file
# Allow custom config directory via environment variable, fallback to ~/.moove
moove_config_dir = os.environ.get('MOOVE_CONFIG_DIR')
if moove_config_dir:
    home_config_dir = os.path.expanduser(moove_config_dir)
else:
    home_config_dir = os.path.join(Path.home(), ".moove")

config_file_path = os.path.join(home_config_dir, 'moove_config.ini')

if not os.path.exists(config_file_path):
    example_config_file_path = os.path.join(os.path.dirname(__file__), 'moove_config.ini.example')
    os.makedirs(home_config_dir, exist_ok=True)
    shutil.copy(example_config_file_path, config_file_path)
    logger.info(f"Created config file at: {config_file_path}")

config = configparser.ConfigParser()
config.read(config_file_path)
global_dir = config.get("GENERAL", "global_dir")
global_dir = os.path.expanduser(global_dir)

# Create subfolders needed for moove in global_dir
subdirs = ["rec_data", "trained_models", "training_data", "cluster_data", "playbacks"]
for subdir in subdirs:
    subdir_path = os.path.join(global_dir, subdir)
    os.makedirs(subdir_path, exist_ok=True)

# Load example file if missing
package_example_data = os.path.join(os.path.dirname(__file__), "example_data", "bird_x")
target_bird_x_dir = os.path.join(global_dir, "rec_data", "bird_x")
if not os.path.exists(target_bird_x_dir):
    shutil.copytree(package_example_data, target_bird_x_dir)

# Load example white noise playback if missing
package_example_data_WN = os.path.join(os.path.dirname(__file__), "example_data", "white_noise")
target_WN_dir = os.path.join(global_dir, "playbacks", "white_noise")
if not os.path.exists(target_WN_dir):
    shutil.copytree(package_example_data_WN, target_WN_dir)

# Create the main window
root = tk.Tk()
root.title("MooveGUI")

# Set window icon
try:
    # Get the absolute path to the moove package directory
    package_dir = os.path.dirname(os.path.abspath(__file__))

    # Always use logo_128.png in full resolution for sharpest quality
    icon_path = os.path.join(package_dir, "templates", "logo_128_white_bg_small.png")
    icon_path = os.path.abspath(icon_path)

    if os.name == 'nt':
        # Windows: set AppUserModelID for correct taskbar icon
        myappid = 'tkinter.python.test'
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)

    # Load icon (fallback logic)
    if os.path.exists(icon_path):
        # Load the 128x128 PNG in full resolution - no resizing for best quality
        icon_image = Image.open(icon_path)
    else:
        # Fallback to original logo.png if 128 version not found
        fallback_path = os.path.join(package_dir, "templates", "logo.png")
        fallback_path = os.path.abspath(fallback_path)
        if os.path.exists(fallback_path):
            icon_image = Image.open(fallback_path)
        else:
            icon_image = None

    if icon_image:
        icon_photo = ImageTk.PhotoImage(icon_image)
        root.iconphoto(True, icon_photo)
        logger.debug(f"Window icon set successfully ({icon_path if os.path.exists(icon_path) else fallback_path})")
    else:
        logger.warning("No icon files found in templates directory")
except Exception as e:
    logger.warning(f"Could not set window icon: {e}")

# Initialize app state
app_state = AppState(global_dir)
app_state.load_state()

app_state.config["global_dir"] = global_dir
app_state.config["rec_data"] = os.path.join(global_dir, "rec_data")
app_state.config["lower_spec_plot"] = int(config.get('GUI', 'lower_spec_plot'))
app_state.config["upper_spec_plot"] = int(config.get('GUI', 'upper_spec_plot'))
app_state.config["vmin_range_slider"] = float(config.get('GUI', 'vmin_range_slider'))
app_state.config["vmax_range_slider"] = float(config.get('GUI', 'vmax_range_slider'))
app_state.config["spec_nperseg"] = int(config.get('GUI', 'spec_nperseg'))
app_state.config["spec_noverlap"] = int(config.get('GUI', 'spec_noverlap'))
app_state.config["spec_nfft"] = int(config.get('GUI', 'spec_nfft'))
app_state.config["performance"] = str(config.get('GUI', 'performance'))


# Get background color for Tkinter
tkinter_bg_color = root.cget("bg")

# Function to determine text color based on background brightness
def get_text_color(bg_color):
    """Get text color based on background brightness."""
    rgb = root.winfo_rgb(bg_color)
    r, g, b = [x // 256 for x in rgb]
    brightness = (r * 299 + g * 587 + b * 114) / 1000
    return "#ffffff" if brightness < 128 else "#000000"

# Save text color and convert background color to hex
app_state.text_color = get_text_color(tkinter_bg_color)
app_state.bg_color = "#%02x%02x%02x" % tuple(c // 256 for c in root.winfo_rgb(tkinter_bg_color))


# Experiment and day combobox update functions
def update_experiment_combobox():
    """Update the experiment combobox based on selected bird."""
    selected_bird = bird_combobox.get()
    experiments = sorted(get_directories(os.path.join(app_state.config['rec_data'], selected_bird)))
    experiment_combobox['values'] = experiments
    experiment_combobox.current(0)
    update_day_combobox()


def update_day_combobox():
    """Update the day combobox based on selected bird and experiment."""
    selected_bird = bird_combobox.get()
    selected_experiment = experiment_combobox.get()
    days = sorted(get_directories(os.path.join(app_state.config['rec_data'], selected_bird, selected_experiment)))
    day_combobox['values'] = days
    day_combobox.current(0)
    update_selected_day(app_state)


def update_selected_day(app_state):
    """Update the data based on selected day."""
    if app_state.init_flag:
        selected_day = day_combobox.get()
        selected_day_path = os.path.join(app_state.config['rec_data'], bird_combobox.get(), experiment_combobox.get(),
                                         selected_day)
        app_state.logger.debug(f"Selected day path: {selected_day_path}")
        app_state.data_dir = selected_day_path
        app_state.current_file_index = 0

        # Update batch files dropdown
        batch_files = find_batch_files(selected_day_path)
        app_state.batch_combobox['values'] = batch_files
        app_state.batch_combobox.set("batch.txt")  # Reset to default batch
        if not batch_files:
            create_batch_file(selected_day_path)
        app_state.current_batch_file = "batch.txt"

        # Load song files from default batch
        app_state.song_files = read_batch(selected_day_path, app_state.current_batch_file)
        app_state.combobox.set(app_state.song_files[app_state.current_file_index])
        app_state.combobox['values'] = app_state.song_files
        plot_data(app_state)
        app_state.logger.debug("Data updated for the selected day")


# Redraw spectrogram functions
def update_spectrogram_from_slider(app_state):
    """Update the spectrogram based on slider values with a delay."""
    if app_state.update_timer is not None:
        root.after_cancel(app_state.update_timer)
    app_state.update_timer = root.after(100, lambda: redraw_spectrogram(app_state))


def redraw_spectrogram(app_state):
    """Redraw the spectrogram with the current slider values."""
    app_state.update_timer = None
    vmin = vBottom.get()
    vmax = vTop.get()
    app_state.current_vmin = vmin
    app_state.current_vmax = vmax
    app_state.redraw_spectrogram(vmin, vmax)


# Combobox selection handling
def combobox_selection(event):
    """Handle combobox selection."""
    selected_file = app_state.combobox.get()
    app_state.current_file_index = app_state.song_files.index(selected_file)
    plot_data(app_state)


def on_select(eclick, erelease):
    """Handle the selection event for the spectrogram axes."""
    
    # prevents zoom when the difference between mouse click and release is too small
    # to imply a zoom
    # only zooming in > 5 ms possible
    time_diff_ms = abs(eclick.xdata - erelease.xdata) * 1000
    if time_diff_ms < 5:
        return
    
    def set_axes_limits(axis, eclick, erelease):
        """Set x and y limits for the selected axis based on click and release coordinates."""
        if eclick.ydata > erelease.ydata:
            eclick.ydata, erelease.ydata = erelease.ydata, eclick.ydata
        if eclick.xdata > erelease.xdata:
            eclick.xdata, erelease.xdata = erelease.xdata, eclick.xdata
        axis.set_xlim(eclick.xdata, erelease.xdata)

    if eclick.inaxes == ax1:
        set_axes_limits(ax1, eclick, erelease)
    elif eclick.inaxes == ax3:
        set_axes_limits(ax3, eclick, erelease)

    canvas.draw()


# Hover and checkbox toggle events
def on_hover(event):
    """Change cursor to cross when hovering over canvas."""
    canvas.get_tk_widget().configure(cursor="tcross")


def on_leave(event):
    """Reset cursor when leaving the canvas."""
    canvas.get_tk_widget().configure(cursor="")


def on_checkbox_toggle():
    """Handle checkbox toggle for saving recfile."""
    file_path = get_file_data_by_index(app_state.data_dir, app_state.song_files, app_state.current_file_index, app_state)
    current_file = os.path.splitext(file_path["file_path"])[0]+".rec"
    segmented = app_state.segmented_var.get()
    classified = app_state.classified_var.get()
    save_seg_class_recfile(current_file, segmented, classified)


def on_edit_type_selected(value):
    """Handle the selection of edit type."""
    app_state.edit_type = value

    if value == "New Segment":
        canvas.get_tk_widget().configure(cursor="tcross")
    elif value == "Move Segment":
        canvas.get_tk_widget().configure(cursor="tcross")
    elif value == "None":
        canvas.get_tk_widget().configure(cursor="tcross")
    else:
        canvas.get_tk_widget().configure(cursor="")

    app_state.logger.debug("Edit type selected: %s", app_state.edit_type)


def update_batch_selection(app_state):
    """Update the song files list based on the selected batch file."""
    selected_batch = app_state.batch_combobox.get()
    app_state.current_batch_file = selected_batch
    app_state.song_files = read_batch(app_state.data_dir, selected_batch)
    app_state.current_file_index = 0 if app_state.song_files else None

    # Update the file combobox
    app_state.combobox['values'] = app_state.song_files
    if app_state.song_files:
        app_state.combobox.set(app_state.song_files[app_state.current_file_index])
        plot_data(app_state)
    else:
        app_state.combobox.set("")
        # Clear the canvas when no files are available
        for ax in [app_state.ax1, app_state.ax2, app_state.ax3]:
            ax.clear()
        app_state.canvas.draw()

    app_state.logger.debug(f"Updated batch selection to {selected_batch} with {len(app_state.song_files)} files")


# Create comboboxes
top_frame = tk.Frame(root)
top_frame.pack(side=tk.TOP, fill=tk.X)

if app_state.data_dir:
    selected_day_path = app_state.data_dir
    path_parts = Path(selected_day_path).parts
    # Check if config datadir has changed
    if path_parts[-4] != Path(app_state.config['rec_data']).name:
        app_state.data_dir = None

# Bird combobox
birds = sorted(get_directories(app_state.config['rec_data']))
bird_combobox = ttk.Combobox(top_frame, values=birds)
bird_combobox.pack(side=tk.LEFT)
bird_combobox.bind("<<ComboboxSelected>>", lambda _: update_experiment_combobox())
if app_state.data_dir:
    bird_name = path_parts[-3]
    # checks if bird folder is still there
    if bird_name in birds:
        bird_combobox.set(bird_name)
    else:
        # defaults to first bird if folder deleted
        bird_combobox.current(0)
        logger.info(f"'{bird_name}' not found in birds list — defaulting to first bird")
else:
    # default for first opening
    bird_combobox.current(0)

# Experiment combobox
experiments = sorted(get_directories(os.path.join(app_state.config['rec_data'], bird_combobox.get())))
experiment_combobox = ttk.Combobox(top_frame, values=experiments, width=30)
experiment_combobox.pack(side=tk.LEFT)
experiment_combobox.bind("<<ComboboxSelected>>", lambda _: update_day_combobox())
if app_state.data_dir:
    experiment_name = path_parts[-2]
    # checks if experiment folder is still there
    if experiment_name in experiments:
        experiment_combobox.set(experiment_name)
    else:
        # defaults to first experiment if folder deleted
        experiment_combobox.current(0)
        logger.info(f"'{experiment_name}' not found in birds list — defaulting to first experiment")
else:
    # default for first opening
    experiment_combobox.current(0)

# Day combobox
days = sorted(
    get_directories(os.path.join(app_state.config['rec_data'], bird_combobox.get(), experiment_combobox.get())))
day_combobox = ttk.Combobox(top_frame, values=days)
day_combobox.pack(side=tk.LEFT)
day_combobox.bind("<<ComboboxSelected>>", lambda _: update_selected_day(app_state))
if app_state.data_dir:
    day_name = path_parts[-1]
    if day_name in days:
        # checks if day folder still exists
        day_combobox.set(day_name)
    else:
        day_combobox.current(0)
        selected_day_path = os.path.join(app_state.config['rec_data'], bird_combobox.get(), experiment_combobox.get(),
                                         day_combobox.get())
        app_state.data_dir = selected_day_path
        logger.info(f"'{day_name}' not found in birds list — defaulting to first day")
    # print(path_parts) # debugging message to check existing folder structure
else:
    day_combobox.current(0)
    selected_day_path = os.path.join(app_state.config['rec_data'], bird_combobox.get(), experiment_combobox.get(),
                                     day_combobox.get())
    app_state.data_dir = selected_day_path


batch_files = find_batch_files(selected_day_path)
# load song files based on previous or default batch file
if app_state.current_batch_file in batch_files:
    app_state.song_files = read_batch(selected_day_path, app_state.current_batch_file)
else:
    # load default batch if previous batch not found
    app_state.current_batch_file = "batch.txt"
    app_state.song_files = read_batch(selected_day_path)

if app_state.current_file_index is None:
    # start with file 0 if no file index is saved
    app_state.current_file_index = 0

current_file_name = get_file_data_by_index(app_state.data_dir, app_state.song_files, app_state.current_file_index, app_state)[
    'file_name']
full_path = os.path.join(app_state.data_dir, current_file_name)

# update batch files with every GUI start
valid_files = sorted(
    f for f in os.listdir(app_state.data_dir) 
    if f.endswith('.wav') or f.endswith('.cbin')
)
for batch in batch_files:
    batch_path = os.path.join(app_state.data_dir, batch)
    if batch == 'batch.txt':
        with open(batch_path, 'w') as f:
            f.write('\n'.join(valid_files))
    else:
        with open(batch_path, 'r') as f:
            keep_files = f.read().splitlines()
        filtered_files = [f for f in keep_files if f in valid_files]
        # only keep existing files in batches
        with open(batch_path, 'w') as f:
            f.write('\n'.join(filtered_files))
app_state.logger.info(f"Batch files have been updated.")

app_state.song_files = read_batch(app_state.data_dir, app_state.current_batch_file)

file_path = get_file_data_by_index(app_state.data_dir, app_state.song_files, app_state.current_file_index, app_state)

app_state.display_dict = get_display_data(file_path, app_state.config)

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(9, 5.5), gridspec_kw={'height_ratios': [6, 1, 6]}, sharex=True) # added

# Dropdown menu for file selection
file_combobox = ttk.Combobox(top_frame, values=app_state.song_files, width=38)
app_state.combobox = file_combobox
app_state.combobox.pack(side=tk.LEFT)
app_state.combobox.bind('<<ComboboxSelected>>', combobox_selection)
app_state.combobox.set(app_state.song_files[app_state.current_file_index])

# Dropdown for batch file selection
batch_combobox = ttk.Combobox(top_frame, values=batch_files, width=20)
app_state.batch_combobox = batch_combobox
app_state.batch_combobox.pack(side=tk.LEFT)
app_state.batch_combobox.bind('<<ComboboxSelected>>', lambda event: update_batch_selection(app_state))
app_state.batch_combobox.set(app_state.current_batch_file)  # open previously used batch file

# Create checkboxes for "Segmented" and "Classified"
classified_var = tk.StringVar(value="0")
segmented_var = tk.StringVar(value="0")

classified_checkbox = ttk.Checkbutton(top_frame, text="Classified  ", variable=app_state.classified_var, onvalue="1",
                                      offvalue="0")  # space next to text is for padding
segmented_checkbox = ttk.Checkbutton(top_frame, text="Segmented  ", variable=app_state.segmented_var, onvalue="1",
                                     offvalue="0")  # space next to text is for padding

classified_checkbox.pack(side=tk.RIGHT)
segmented_checkbox.pack(side=tk.RIGHT, padx=(5, 0))

app_state.segmented_var.trace_add("write", lambda *args: on_checkbox_toggle())
app_state.classified_var.trace_add("write", lambda *args: on_checkbox_toggle())

axes = fig.get_axes()
ax1 = axes[0]
ax2 = axes[1]
ax3 = axes[2]
# Frame for canvas and slider
plot_frame = tk.Frame(root)
plot_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

# Canvas for the figure
canvas = FigureCanvasTkAgg(fig, master=plot_frame)
app_state.set_axes(ax1, ax2, ax3)
app_state.set_canvas(canvas)
app_state.display_dict = get_display_data(file_path, app_state.config)
# save background of axis 2 and 3 to restore 
app_state.ax2_background = app_state.canvas.copy_from_bbox(app_state.ax2.bbox)
app_state.ax3_background = app_state.canvas.copy_from_bbox(app_state.ax3.bbox)

canvas_widget = canvas.get_tk_widget()
canvas_widget.grid(row=0, column=0, sticky="nsew")  # Position canvas in the grid

# Configure grid
plot_frame.grid_rowconfigure(0, weight=1)
plot_frame.grid_columnconfigure(0, weight=1)
plot_frame.grid_columnconfigure(1, weight=0)

# ensures that previous slider settings are within the min and max range of the slider
if (app_state.current_vmin is not None and app_state.current_vmax is not None) and (
        app_state.current_vmin > app_state.config['vmin_range_slider'] and app_state.current_vmax < app_state.config[
    'vmax_range_slider']):
    vBottom = tk.DoubleVar(value=app_state.current_vmin)
    vTop = tk.DoubleVar(value=app_state.current_vmax)
else:
    distance = app_state.config['vmax_range_slider'] - app_state.config['vmin_range_slider']
    distance = distance / 4
    vBottom = tk.DoubleVar(value=app_state.config['vmin_range_slider'] + distance)
    vTop = tk.DoubleVar(value=app_state.config['vmax_range_slider'] - distance)

vBottom.trace_add("write", lambda name, index, mode: update_spectrogram_from_slider(app_state))
vTop.trace_add("write", lambda name, index, mode: update_spectrogram_from_slider(app_state))

# Create and position the slider
vSlider = RangeSliderV(plot_frame, [vBottom, vTop], min_val=app_state.config['vmin_range_slider'],
                       max_val=app_state.config['vmax_range_slider'],
                       font_family="Arial", font_size=10, padY=20, Width=128, bar_radius=9,
                       bgColor=app_state.bg_color, font_color=app_state.text_color)
vSlider.grid(row=0, column=1, sticky="ns")

plot_frame.grid_columnconfigure(1, weight=0)  # Make column 1 (for slider) non-adjustable

# Rectangle selectors
rect_selector_ax1 = RectangleSelector(ax1, on_select, useblit=True,
                                      button=[1],  # Left mouse button
                                      minspanx=30, minspany=30,
                                      spancoords='pixels',
                                      interactive=False)

rect_selector_ax3 = RectangleSelector(ax3, on_select, useblit=True,
                                      button=[1],  # Left mouse button
                                      minspanx=30, minspany=30,
                                      spancoords='pixels',
                                      interactive=False)

# Update button
center_frame = tk.Frame(root)
button_frame = tk.Frame(root)

# Platform-specific styling for refresh button to fix macOS display issue
if platform.system() == 'Darwin':  # macOS
    refresh_text = "↻"  # Alternative refresh symbol that renders better on macOS
else:  # Windows and other platforms
    refresh_text = "⟳"  # Original symbol works fine on Windows

btn_update = tk.Button(button_frame, text=refresh_text, command=lambda: update(app_state))
btn_update.pack(side=tk.LEFT, padx=(0, 20))

# Navigation buttons
btn_previous = tk.Button(button_frame, text="Previous",
                         command=lambda: (app_state.change_file(-1), plot_data(app_state)))
btn_previous.pack(side=tk.LEFT)
btn_next = tk.Button(button_frame, text="Next", command=lambda: (app_state.change_file(1), plot_data(app_state)))
btn_next.pack(side=tk.LEFT)
button_frame.pack(padx=5, pady=5)
center_frame.pack(fill=tk.X)

# Swiping buttons
btn_swipe_left = tk.Button(button_frame, text="  <  ", command=lambda: swipe_left(app_state))
btn_swipe_left.pack(side=tk.LEFT)
btn_swipe_right = tk.Button(button_frame, text="  >  ", command=lambda: swipe_right(app_state))
btn_swipe_right.pack(side=tk.LEFT)

# Zoom and Unzoom buttons
btn_zoom = tk.Button(button_frame, text="Zoom", command=lambda: zoom(app_state))
btn_zoom.pack(side=tk.LEFT)
btn_unzoom = tk.Button(button_frame, text="Unzoom", command=lambda: unzoom(app_state))
btn_unzoom.pack(side=tk.LEFT)

# Crop button
btn_crop = tk.Button(button_frame, text="Crop", command=lambda: handle_crop(app_state))
btn_crop.pack(side=tk.LEFT)
# Delete button
btn_delete = tk.Button(button_frame, text="Delete", command=lambda: handle_delete(app_state))
btn_delete.pack(side=tk.LEFT)
# Play button
btn_play = tk.Button(button_frame, text="Play", command=lambda: handle_playback(app_state))
btn_play.pack(side=tk.LEFT)
# Resegment button
btn_resegment = tk.Button(button_frame, text="Resegment",
                          command=lambda: open_resegment_window(root, app_state, bird_combobox, experiment_combobox,
                                                                day_combobox))
btn_resegment.pack(side=tk.LEFT)
# Relabel button
btn_relabel = tk.Button(button_frame, text="Relabel",
                        command=lambda: open_relabel_window(root, app_state, bird_combobox, experiment_combobox,
                                                            day_combobox))
btn_relabel.pack(side=tk.LEFT)
# Create Training Dataset button
btn_training = tk.Button(button_frame, text="Training",
                         command=lambda: open_training_window(root, app_state, bird_combobox, experiment_combobox,
                                                              day_combobox))
btn_training.pack(side=tk.LEFT)
# Cluster button
btn_cluster = tk.Button(button_frame, text="Cluster",
                        command=lambda: open_cluster_window(root, app_state, bird_combobox, experiment_combobox,
                                                            day_combobox))
btn_cluster.pack(side=tk.LEFT)

# Global state for syllable selection
selected_syllable_index = None
edit_mode = False

# Edit mode selection
radio_frame = tk.Frame(root)

v = tk.StringVar(root, "1")
options = {"None": "1",
           "New Segment": "2",
           "Delete Segment": "3",
           "Move Segment": "4",
           "Label Interactive": "5"}

# Add radio buttons to frame
for txt, val in options.items():
    tk.Radiobutton(radio_frame, text=txt, variable=v, value=val,
                   command=lambda txt=txt: on_edit_type_selected(txt)).pack(side=tk.LEFT)

# Center the frame in the root widget
radio_frame.pack(side=tk.TOP, anchor='center')

def reset_edit_type_selection():
    """Reset the edit type radio button selection to 'None'"""
    v.set("1")  # "1" corresponds to "None"

# Connect the GUI reset function to app_state
app_state.reset_edit_type_gui = reset_edit_type_selection

# Add events to canvas
canvas.mpl_connect('key_press_event', lambda event: handle_keypress(event, app_state, v))
canvas.mpl_connect('button_press_event', lambda event: select_event(event, app_state))
canvas.mpl_connect('key_press_event', lambda event: edit_syllable(event, app_state))

plot_data(app_state)
app_state.init_flag = True


def force_exit_after_timeout():
    """Force exit the application after a timeout"""
    import time
    import os
    import signal
    
    time.sleep(3.0)  # Wait 3 seconds
    
    try:
        logger.warning("Timeout reached - force killing application")
        os.kill(os.getpid(), signal.SIGKILL)
    except:
        try:
            os._exit(1)
        except:
            pass


def on_closing():
    """Handle application closing with graceful thread shutdown"""
    try:
        # Check for active threads
        with app_state.thread_lock:
            active_count = len(app_state.active_threads)
        
        if active_count > 0:
            # Ask user if they want to close with active threads (simplified to OK/Cancel)
            from tkinter import messagebox
            result = messagebox.askokcancel(
                "Active Threads", 
                f"There {'is' if active_count == 1 else 'are'} {active_count} active thread{'s' if active_count > 1 else ''} running.\n\n"
                f"Closing will terminate {'it' if active_count == 1 else 'them'}. Continue?"
            )
            
            if not result:  # User chose Cancel
                return  # Don't close the application
            
            logger.info(f"User confirmed closing with {active_count} active threads")
            logger.info("Shutting down threads...")
            app_state.shutdown_all_threads()
        
        # Start timeout timer as backup
        import threading
        timeout_thread = threading.Thread(target=force_exit_after_timeout, daemon=True)
        timeout_thread.start()
        
        # Save application state
        app_state.save_state()
        logger.info("Application state saved")
        
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")
    
    # Simplified but more reliable shutdown
    try:
        logger.info("Starting application shutdown...")
        
        # Stop the mainloop first
        root.quit()
        
        # Give a moment for cleanup
        root.update()
        
        # Destroy all widgets
        root.destroy()
        
        logger.info("Tkinter cleanup completed")
        
    except Exception as e:
        logger.error(f"Error during Tkinter cleanup: {e}")
    
    # Force exit using multiple methods
    try:
        import sys
        import os
        import signal
        
        logger.warning("Forcing application exit...")
        
        # Method 1: Standard exit
        sys.exit(0)
        
    except:
        try:
            # Method 2: OS exit
            os._exit(0)
        except:
            try:
                # Method 3: Kill signal (Unix/macOS)
                os.kill(os.getpid(), signal.SIGTERM)
            except:
                try:
                    # Method 4: Kill signal force (Unix/macOS)
                    os.kill(os.getpid(), signal.SIGKILL)
                except:
                    pass


def main():
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()


if __name__ == "__main__":
    main()
