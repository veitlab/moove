# state.py
import tkinter as tk
import logging
import os
import threading
import json
import re

class AppState:
    def __init__(self, global_dir):
        self.text_color = None
        self.bg_color = None
        self.dash_thread = None  
        self.server = None  # Flask server instance
        self.server_thread = None  # Server thread
        self.stop_event = threading.Event() 
        self.spec = None
        self.canvas = None
        self.ax1 = None
        self.ax2 = None
        self.ax3 = None
        self.last_ax3_marker = None
        self.display_dict = None
        self.edit_type = None
        self.moved_point = None 
        self.selected_syllable_index = None  
        self.edit_type = "None"
        self.data_dir = ""
        self.current_file_index = 0
        self.song_files = []
        self.current_batch_file = "batch"  # Default batch file
        self.original_x_range = None
        self.original_y_range_ax1 = None
        self.original_y_range_ax2 = None
        self.original_y_range_ax3 = None
        self.current_vmin = None
        self.current_vmax = None
        self.combobox = None
        self.config = {
            # as given in the config, by default your .moove folder
            'global_dir': global_dir,
            # folder where your data is saved
            'rec_data': tk.StringVar(value="rec_data"),
            # GUI settings
            'lower_spec_plot': tk.StringVar(value="500"),
            'upper_spec_plot': tk.StringVar(value="12500"),
            'vmin_range_slider': tk.StringVar(value="-100"),
            'vmax_range_slider': tk.StringVar(value="-10"),
            'spec_nfft': tk.StringVar(value="1024"),
            'spec_noverlap': tk.StringVar(value="896"),
            'spec_nperseg': tk.StringVar(value="1024"),
        }
        self.evfuncs_params = {
            'threshold': tk.StringVar(value="-50"),
            'min_syl_dur': tk.StringVar(value="0.03"),
            'min_silent_dur': tk.StringVar(value="0.005"),
            'freq_cutoffs': tk.StringVar(value="500,10000"),
            'smooth_window': tk.StringVar(value="2"),
        }
        self.mlseg_params = {
            'decision_threshold': tk.StringVar(value="0.5"),
            'onset_window_size': tk.StringVar(value="5"),
            'n_onset_true': tk.StringVar(value="3"),
            'offset_window_size': tk.StringVar(value="5"),
            'n_offset_false': tk.StringVar(value="4"),
            'min_syllable_length': tk.StringVar(value="0.03"),
            'min_silent_duration': tk.StringVar(value="0.005"),
        }
        self.spec_params = {
            'nperseg': tk.StringVar(value="64"),
            'noverlap': tk.StringVar(value="32"),
            'nfft': tk.StringVar(value="128"),
            'freq_cutoffs': tk.StringVar(value="0,22050"),
            'input_length': tk.StringVar(value="21,64"), # 21/64
        }
        self.umap_k_means_params = {
            'n_neighbors': tk.StringVar(value="15"),
            'min_dist': tk.StringVar(value="0.1"),
            'n_clusters': tk.StringVar(value="10"),
        }
        self.train_classification_params = {
            'epochs': tk.StringVar(value="1000"),
            'batch_size': tk.StringVar(value="64"),
            'learning_rate': tk.StringVar(value="0.001"),
            'early_stopping_patience': tk.StringVar(value="5"),
            'downsampling': tk.BooleanVar(value=True),
            'qat': tk.BooleanVar(value=False), 
        }
        self.train_segmentation_params = {
            'hist_size': tk.StringVar(value="3"),
            'chunk_size': tk.StringVar(value="64"),
            'overlap_chunks': tk.BooleanVar(value=False),
            'epochs': tk.StringVar(value="1000"),
            'batch_size': tk.StringVar(value="64"),
            'learning_rate': tk.StringVar(value="0.001"),
            'early_stopping_patience': tk.StringVar(value="5"),
            'downsampling': tk.BooleanVar(value=True),
            'qat': tk.BooleanVar(value=False)
        }
        # values for segmentation/ classification either 0 or 1
        # by default 0, will be changed in the metadata.json 
        # to 1 if you tick it in the GUI
        # metadata file is saved in the .moove folder
        self.metadata_file =  os.path.join(global_dir, "metadata.json")
        self.segmented_var = tk.StringVar(value="0")
        self.classified_var = tk.StringVar(value="0")
        self.resegment_window = None
        self.training_window = None
        self.cluster_window = None
        self.relabel_window = None
        self.current_segmentation_model = None
        self.current_classification_model = None
        self.init_flag = False
        self.update_timer = None

        # Set up logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def save_state(self, filename="app_state.json"):
        # saves the current state of the GUI in a .json file, 
        # so that it will return to this point when reopened
        global_dir = self.config['global_dir']
        filepath = os.path.join(global_dir, filename)

        # Ensure the directory exists (should be the .moove folder the data is stored in
        # and the config is in)
        try:
            os.makedirs(global_dir, exist_ok=True)
            # if there is no .json file yet (e.g. opening the GUI for the first time)
            # default values will be opened as defined in the moovegui.py, 
            # meaning first bird, first exp,...
            # self.logger.info(f"Ensured directory exists: {global_dir}") # debugging message
        except Exception as e:
            self.logger.error(f"Failed to create directory {global_dir}: {e}")
            return

        # Create the state dictionary saving the current settings of the gui
        try:
            state_dict = {
                'current_vmin': self.current_vmin,
                'current_vmax': self.current_vmax,
                'data_dir': self.data_dir, # directory to your current day in rec_data
                'song_files': self.song_files, # list of all song files in this day
                'current_file_index': self.current_file_index, # index of the song file last opened
                'current_batch_file': self.current_batch_file, # current batch file selected
                'evfuncs_params': {key: value.get() for key, value in self.evfuncs_params.items()},
                'mlseg_params': {key: value.get() for key, value in self.mlseg_params.items()},
                'spec_params': {key: value.get() for key, value in self.spec_params.items()},
                'umap_k_means_params': {key: value.get() for key, value in self.umap_k_means_params.items()},
                'train_segmentation_params': {key: value.get() for key, value in self.train_segmentation_params.items()},
                'train_classification_params': {key: value.get() for key, value in self.train_classification_params.items()},
            }

            # Saves a new .json file in the .moove folder or overwrites the one
            # already existing
            with open(filepath, 'w') as f:
                json.dump(state_dict, f)
            # self.logger.info(f"App state saved successfully to {filepath}") # debugging message

        except Exception as e:
            self.logger.error(f"Error saving app state to {filepath}: {e}")

    def load_state(self, filename="app_state.json"):
    # loads the previous state of the GUI saved in the app_state.json 
    # to reopen it at the same position
        global_dir = self.config['global_dir']
        filepath = os.path.join(global_dir, filename)
        # self.logger.info(f"Trying to load state from: {filepath}") # debugging message

        if not os.path.exists(filepath):
            # logs a warning when .json file doesnt't exist yet and 
            # GUI is opened with default values defined in moovegui.py,
            # thus first bird, first exp,...
            self.logger.warning(f"No state file found at: {filepath}")
            return

        try:
            with open(filepath, 'r') as f:
                state_dict = json.load(f)
        except json.JSONDecodeError as e:
            self.logger.error(f"Error reading state file: {e}")
            return

        self.current_vmin = state_dict.get('current_vmin')
        self.current_vmax = state_dict.get('current_vmax')
        self.data_dir = state_dict.get('data_dir')
        self.song_files = state_dict.get('song_files')
        self.current_file_index = state_dict.get('current_file_index')
        self.current_batch_file = state_dict.get('current_batch_file', 'batch')  # Default to 'batch' if not found
        
        for key, value in state_dict.get('evfuncs_params', {}).items():
            self.evfuncs_params[key].set(value)
        for key, value in state_dict.get('mlseg_params', {}).items():
            self.mlseg_params[key].set(value)
        for key, value in state_dict.get('spec_params', {}).items():
            self.spec_params[key].set(value)
        for key, value in state_dict.get('umap_k_means_params', {}).items():
            self.umap_k_means_params[key].set(value)
        for key, value in state_dict.get('train_segmentation_params', {}).items():
            self.train_segmentation_params[key].set(value)
        for key, value in state_dict.get('train_classification_params', {}).items():
            self.train_classification_params[key].set(value)

        # self.logger.info(f"Successfully loaded state from: {filepath}") # debugging message

    def set_canvas(self, canvas):
        self.canvas = canvas

    def draw_canvas(self):
        if self.canvas:
            self.canvas.draw()

    def redraw_spectrogram(self, vmin, vmax):
        if self.spec:
            self.spec.set_clim(vmin, vmax)
            self.draw_canvas()

    def set_axes(self, ax1, ax2, ax3):
        self.ax1 = ax1
        self.ax2 = ax2
        self.ax3 = ax3
        
        for ax in [self.ax1, self.ax2, self.ax3]:
            ax.set_clip_on(True)  
            for line in ax.lines:
                line.set_clip_on(True)

    def get_axes(self):
        return self.ax1, self.ax2, self.ax3

    def set_original_x_range(self, original_x_range):
        self.original_x_range = original_x_range
    
    def set_original_y_range_ax1(self, original_y_range_ax1, original_y_range_ax2, original_y_range_ax3):
        self.original_y_range_ax1 = original_y_range_ax1
        self.original_y_range_ax2 = original_y_range_ax2
        self.original_y_range_ax3 = original_y_range_ax3

    def get_data_dir(self):
        return self.data_dir
    
    def change_file(self, delta):
        current_file_index = self.current_file_index
        current_file_index += delta
        current_file_index = max(0, min(len(self.song_files) - 1, current_file_index))
        self.current_file_index = current_file_index
        self.combobox.set(self.song_files[current_file_index])
        self.selected_syllable_index = None

    def update_classification_datasets_combobox(self):
        training_data_folder_classification = os.path.join(self.config['global_dir'], "training_data")
        training_datasets_classification = [f for f in os.listdir(training_data_folder_classification) if f.endswith("_class.pkl")]
        self.training_window.training_dataset_combobox_classification['values'] = training_datasets_classification

    def update_segmentation_datasets_combobox(self):
        training_data_folder_segmentation = os.path.join(self.config['global_dir'], "training_data")
        training_datasets_segmentation = [f for f in os.listdir(training_data_folder_segmentation) if f.endswith("_seg.pkl")] # .npy
        self.training_window.training_dataset_combobox_segmentation['values'] = training_datasets_segmentation

    def update_cluster_datasets_combobox(self):
        cluster_data_folder = os.path.join(self.config['global_dir'], "cluster_data")
        cluster_datasets = [f for f in os.listdir(cluster_data_folder) if f.endswith(".pkl")]
        self.cluster_window.cluster_dataset_combobox['values'] = cluster_datasets
    
    def update_batch_select_combobox(self, select_path = "current_day"):
        birds = os.path.abspath(os.path.join(self.data_dir, "..", ".."))
        experiments = os.path.join(self.data_dir, "..")
        day = self.data_dir
        batch_files = [] 
        if select_path == "current_day":
            batch_files = [f for f in os.listdir(day) if re.match('.*batch.*', f)]
        if select_path == "current_experiment":
            batch_files = [f for f in os.listdir(experiments) if re.match('.*batch.*', f)]
        if select_path == "current_bird":
            batch_files = [f for f in os.listdir(birds) if re.match('.*batch.*', f)]
        batch_files = ["All Files"] + batch_files
        self.training_window.training_batch_combobox_classification['values'] = batch_files
        self.training_window.training_batch_combobox_classification.set("All Files")

    def update_batch_select_combobox_segment(self, select_path = "current_day"):
        birds = os.path.abspath(os.path.join(self.data_dir, "..", ".."))
        experiments = os.path.join(self.data_dir, "..")
        day = self.data_dir
        batch_files = [] 
        if select_path == "current_day":
            batch_files = [f for f in os.listdir(day) if re.match('.*batch.*', f)]
        if select_path == "current_experiment":
            batch_files = [f for f in os.listdir(experiments) if re.match('.*batch.*', f)]
        if select_path == "current_bird":
            batch_files = [f for f in os.listdir(birds) if re.match('.*batch.*', f)]
        batch_files = ["All Files"] + batch_files
        self.training_window.training_batch_combobox_segmentation['values'] = batch_files
        self.training_window.training_batch_combobox_segmentation.set("All Files")
