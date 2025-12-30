# utils/clustering_utils.py
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import threading
import tkinter as tk
import warnings
import evfuncs
import tkinter.font as tkFont
from matplotlib import cm
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
from scipy import interpolate
from scipy.signal import spectrogram
from sklearn.cluster import KMeans
from tkinter import ttk, messagebox
from umap import UMAP

# Disable system warnings
warnings.filterwarnings('ignore')


def start_create_cluster_dataset_thread(app_state, dataset_name, use_selected_files, selection, batch_file, bird_combobox, experiment_combobox, day_combobox, root):
    """Start a thread to create a cluster dataset based on selected files and criteria."""
    from moove.utils import get_files_for_day, get_files_for_experiment, get_files_for_bird, filter_segmented_files
    bird = bird_combobox.get()
    experiment = experiment_combobox.get()
    day = day_combobox.get()

    if selection == "current_day":
        files = get_files_for_day(app_state, bird, experiment, day, batch_file)
    elif selection == "current_experiment":
        files = get_files_for_experiment(app_state, bird, experiment, batch_file)
    elif selection == "current_bird":
        files = get_files_for_bird(app_state, bird, batch_file)

    if use_selected_files:
        files = filter_segmented_files(files)

    app_state.logger.debug(
        "Creating training dataset with parameters: Use selected files: %s, Selection: %s, Batch file: %s", use_selected_files, selection, batch_file
    )

    # check if dataset name is valid
    if len(dataset_name) < 1:
        messagebox.showinfo("Error", "Dataset name not valid! A dataset name needs to contain at least one character.")
    else:
        # create dataset with valid name
        max_value = len(files)
        progressbar = ttk.Progressbar(app_state.cluster_window, orient=tk.HORIZONTAL, length=200, mode='determinate', maximum=max_value)
        progressbar.grid(row=999, column=0, columnspan=2, pady=(10, 0), sticky=tk.W + tk.E)
        
        # Create wrapper function for thread management
        def thread_wrapper():
            current_thread = threading.current_thread()
            try:
                create_cluster_dataset(app_state, dataset_name, progressbar, max_value, files, root)
            finally:
                app_state.remove_thread(current_thread)
        
        thread = threading.Thread(target=thread_wrapper, name="CreateClusterDatasetThread")
        app_state.add_thread(thread)
        thread.start()


def create_cluster_dataset(app_state, dataset_name, progressbar, max_value, all_files, root):
    """Generate and save a cluster dataset, tracking progress with a progress bar."""
    from moove.utils import get_display_data, seconds_to_index, decibel, plot_data
    
    # Store the complete original file context to restore it later
    original_data_dir = app_state.data_dir
    original_song_files = app_state.song_files.copy() if app_state.song_files else []
    original_current_file_index = app_state.current_file_index
    
    if dataset_name:
        going_prod_df = pd.DataFrame(columns=['file', 'onset_no', 'cluster_flattend_spectrogram', 'label'])
        entry_no = 0

    progressbar.grid_remove()

    # Add running label to the GUI
    font_style = tkFont.Font(family="Arial", size=14)
    running_label = tk.Label(app_state.cluster_window, text="Looking for segments...", fg="green", font=font_style)
    running_label.grid(row=22, column=0, columnspan=2, pady=(10, 0), sticky=tk.W)
    root.update_idletasks() 

    def get_onset_offset_info(file_path):
        notmat_file = file_path + ".not.mat"
        if os.path.exists(notmat_file):
            notmat_dict = evfuncs.load_notmat(notmat_file)
            return {
                "onsets": notmat_dict.get("onsets", []),
                "offsets": notmat_dict.get("offsets", [])
            }
        else:
            return {"onsets": [], "offsets": []}
    
    # Count number of segments given for dataset    
    num_segs = 0
    for file_path in all_files:
        info = get_onset_offset_info(file_path)
        if len(info["onsets"]) > 0 and len(info["offsets"]) > 0:
            segs = min(len(info["onsets"]), len(info["offsets"]))
            num_segs += segs

    if num_segs < 10:
        running_label.destroy()
        root.update_idletasks() 
        messagebox.showinfo("Error", "Not enough segments given. Need at least 10 segments to form clusters.")
        return
    
    running_label.destroy()
    root.update_idletasks() 
    progressbar.grid()

    for i in range(max_value):
        progressbar['value'] = i
        file_i = all_files[i]
        file_path = {"file_name": os.path.basename(file_i), "file_path": os.path.join(os.getcwd(), file_i)}
        display_dict = get_display_data(file_path, app_state.config)
        app_state.data_dir = os.path.dirname(file_i)

        sampling_rate = int(display_dict["sampling_rate"])
        rawsong = display_dict["song_data"]

        freq_cutoffs = tuple(map(int, app_state.spec_params['freq_cutoffs'].get().split(',')))
        onsets, offsets = display_dict["onsets"], display_dict["offsets"]

        if dataset_name:
            nperseg = int(app_state.spec_params['nperseg'].get())
            noverlap = int(app_state.spec_params['noverlap'].get())
            nfft = int(app_state.spec_params['nfft'].get())
            
            for syllable_no, (onset, offset) in enumerate(zip(onsets, offsets)):
                entry_no += 1
                onset_index = int(seconds_to_index(onset, sampling_rate))
                offset_index = int(seconds_to_index(offset, sampling_rate))
                cutted_raw_song = rawsong[onset_index:offset_index]
                f, t, Sxx_cluster = spectrogram(cutted_raw_song, fs=sampling_rate, nperseg=nperseg, noverlap=noverlap, nfft=nfft)

                # Filter frequencies within defined range
                Sxx_cluster = Sxx_cluster[(f >= freq_cutoffs[0]) & (f <= freq_cutoffs[1]), :]

                # Linear interpolation
                original_shape = Sxx_cluster.shape
                x_old = np.linspace(0, 1, original_shape[1])
                x_new = np.linspace(0, 1, 40)
                f_i = interpolate.interp1d(x_old, Sxx_cluster, kind='linear', axis=1)
                Sxx_cluster = f_i(x_new)
                Sxx_cluster = decibel(Sxx_cluster)

                going_prod_df.loc[entry_no] = [file_i, syllable_no, Sxx_cluster.flatten(), "x"]

    if dataset_name:
        file_path = os.path.join(app_state.config['global_dir'], 'cluster_data', f'{dataset_name}_clus.pkl')
        going_prod_df.to_pickle(file_path)
        app_state.update_cluster_datasets_combobox()

    # Restore the complete original file context
    app_state.data_dir = original_data_dir
    app_state.song_files = original_song_files
    app_state.current_file_index = original_current_file_index

    progressbar['value'] = max_value
    progressbar.grid_forget()
    
    # Schedule Tkinter operations in the main thread
    def show_message():
        messagebox.showinfo("Info", f"Cluster dataset '{dataset_name}' created successfully!")
    
    root.after(0, show_message)


def start_clustering_thread(root, app_state, dataset_name_entry):
    """Start the clustering process in a separate thread."""
    dataset_name = dataset_name_entry
    if dataset_name == "Select Cluster Dataset":
        messagebox.showinfo("Error", "Selected cluster dataset not valid! Perhaps you forgot to pick a dataset?")
    else:
        messagebox.showinfo("Info", "Clustering started. This may take a while, please wait!")
    
    # Create wrapper function for thread management
    def thread_wrapper():
        current_thread = threading.current_thread()
        try:
            run_clustering(root, app_state, dataset_name)
        finally:
            app_state.remove_thread(current_thread)
    
    thread = threading.Thread(target=thread_wrapper, name="ClusteringThread")
    app_state.add_thread(thread)
    thread.start()


def run_clustering(root, app_state, dataset_name):
    """Run the clustering process using UMAP and KMeans."""
    dataset_name = f"{dataset_name}.pkl"
    n_syllables = int(app_state.umap_k_means_params['n_clusters'].get())
    n_neighbors = int(app_state.umap_k_means_params['n_neighbors'].get())
    min_dist = float(app_state.umap_k_means_params['min_dist'].get())

    font_style = tkFont.Font(family="Arial", size=14)
    running_label = tk.Label(app_state.cluster_window, text="Running...", fg="green", font=font_style)
    running_label.grid(row=999, column=0, columnspan=2, pady=(10, 0), sticky=tk.W)
    root.update_idletasks()

    # Load the dataset 
    dataset_path = os.path.join(os.path.join(app_state.config['global_dir'], 'cluster_data', f'{dataset_name}'))
    if not os.path.exists(dataset_path):
        app_state.logger.error("Dataset %s not found in cluster_data folder.", dataset_name)
        return

    df = pd.read_pickle(dataset_path)
    spectrogram_feature_array = np.array([np.array(x) for x in df['cluster_flattend_spectrogram'].values])

    # Run UMAP
    umap_model = UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=2, metric='euclidean', random_state=42)
    low_dimensional_data = umap_model.fit_transform(spectrogram_feature_array)

    # KMeans Clustering
    kmeans = KMeans(n_clusters=n_syllables, random_state=42)
    labels = kmeans.fit_predict(low_dimensional_data)

    # Map numerical labels to alphabetic characters
    label_mapping = {i: chr(97 + i) for i in range(n_syllables)}
    alphabet_labels = [label_mapping[label] for label in labels]
    df['clustered_label'] = alphabet_labels

    # Add UMAP coordinates to the DataFrame
    df['UMAP1'] = low_dimensional_data[:, 0]
    df['UMAP2'] = low_dimensional_data[:, 1]

    # Save the clustered data
    output_path = os.path.join(app_state.config['global_dir'], 'cluster_data', dataset_name)
    df.to_pickle(output_path)

    # Plot and save the clusters
    plot_clusters(root, app_state, low_dimensional_data, alphabet_labels, output_path)

    running_label.destroy()

    app_state.logger.debug("Clustering complete. Results saved to %s", output_path)
    
    # Schedule Tkinter operations in the main thread
    def show_message():
        messagebox.showinfo("Info", f"Clustering complete! Results saved to {output_path}")
    
    root.after(0, show_message)


def plot_clusters(root, app_state, low_dimensional_data, labels, output_path):
    """Plot and save the clustering results in a new window."""

    def plot():
        unique_labels = sorted(set(labels))
        label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
        numeric_labels = [label_mapping[label] for label in labels]

        # Create a new window for the plot
        plot_window = tk.Toplevel(root)
        plot_window.title("Cluster Plot")

        fig, ax = plt.subplots(figsize=(10, 8))
        scatter = ax.scatter(low_dimensional_data[:, 0], low_dimensional_data[:, 1], 
                             c=numeric_labels, s=5, cmap=cm.get_cmap('jet'))

        # Create a custom legend with alphabetic labels
        handles, _ = scatter.legend_elements()
        legend = ax.legend(handles, unique_labels, title="Labels")

        ax.add_artist(legend)
        ax.set_title("UMAP Clustering")
        ax.set_xlabel("UMAP1")
        ax.set_ylabel("UMAP2")

        # Save the plot
        plot_path = output_path.replace('_clus.pkl', '_clusters.png')
        plt.savefig(plot_path)
        app_state.logger.debug("Cluster plot saved to %s", plot_path)

        # Embed the plot in the new window
        canvas = FigureCanvasTkAgg(fig, master=plot_window)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        toolbar = NavigationToolbar2Tk(canvas, plot_window)
        toolbar.update()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    root.after(0, plot)


def replace_labels_from_df(app_state, dataset_name, root=None):
    """Replace labels in the dataset based on clustering results."""
    from moove.utils.file_utils import get_display_data
    from moove.utils.movefuncs_utils import save_notmat
    from moove.utils.plot_utils import plot_data

    if dataset_name == "Select Cluster Dataset":
        messagebox.showinfo("Error", "Selected cluster dataset not valid! Perhaps you forgot to pick a dataset?")
    else:
        messagebox.showinfo("Info", "Replacement of syllables started. This may take a while, please wait!")

    # Store the complete original file context to restore it later
    original_data_dir = app_state.data_dir
    original_song_files = app_state.song_files.copy() if app_state.song_files else []
    original_current_file_index = app_state.current_file_index

    dataset_path = os.path.join(app_state.config['global_dir'], 'cluster_data', f'{dataset_name}.pkl')
    df = pd.read_pickle(dataset_path)
    files = df['file'].unique()

    app_state.logger.debug("Starting replacement of syllables with dataset %s", dataset_name)

    max_value = len(files)
    progressbar = ttk.Progressbar(app_state.cluster_window, orient=tk.HORIZONTAL, length=200, mode='determinate', maximum=max_value)
    progressbar.grid(row=999, column=0, columnspan=2, pady=(10, 0), sticky=tk.W + tk.E)

    for i, file in enumerate(files):
        try:
            progressbar['value'] = i
            progressbar.update()
            # Concatenate all the clustered labels for the current file into a single string
            # Only use clustered_label (from clustering/Dash), not the original 'label' column
            if 'clustered_label' not in df.columns:
                raise KeyError("Dataset has not been clustered yet. Please cluster the dataset first before replacing labels.")
            labels = df.loc[df['file'] == file]['clustered_label'].astype(str).str.cat(sep='')

            # Get display data for the current file
            display_dict = get_display_data({"file_name": os.path.basename(file), "file_path": file}, app_state.config)
            display_dict["labels"] = labels

            app_state.data_dir = os.path.dirname(file)
            save_path = os.path.join(app_state.data_dir, f"{display_dict['file_name']}.not.mat")
            app_state.logger.debug("Saving labels to %s", save_path)

            # Save modified labels to the .not.mat file
            save_notmat(save_path, display_dict)
            
        except Exception as e:
            app_state.logger.error(f"File {file} could not be processed correctly: {e}. Check manually.")
            return

    # Restore the original data_dir so file navigation continues to work
    app_state.data_dir = original_data_dir
    app_state.song_files = original_song_files
    app_state.current_file_index = original_current_file_index
    progressbar['value'] = max_value
    
    # Final UI update
    progressbar.grid_forget()
    app_state.reset_edit_type()
    plot_data(app_state)
    
    # Schedule Tkinter operations in the main thread if root is available
    if root:
        def show_message():
            messagebox.showinfo("Info", "Replacement of syllables complete!")
        root.after(0, show_message)
    else:
        messagebox.showinfo("Info", "Replacement of syllables complete!")


