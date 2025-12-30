# utils/segment_utils.py
import evfuncs
import matplotlib
import numpy as np
import os
import threading
import tkinter as tk
import tkinter.font as tkFont
import torch
import re
from tkinter import ttk, messagebox

matplotlib.use('Agg')


def load_segmentation_checkmarks(all_files):
    """Check whether files have been manually checked as being segmented"""
    unsegment_files = []
    for file in all_files:
        recfile_path = os.path.splitext(file)[0] + ".rec"
        with open(recfile_path, "r") as f:
            content = f.read()

        hand_segmented_pattern = r"Hand Segmented = (\d+)"
        hand_segmented_match = re.search(hand_segmented_pattern, content)

        if hand_segmented_match.group(1) == '0':
            unsegment_files.append(file)

    return unsegment_files


def start_segment_evfuncs(app_state, selection, batch_file, bird_combobox, experiment_combobox, day_combobox):
    """Start the segmentation process using Evfuncs in a separate thread."""
    from moove.utils import get_files_for_day, get_files_for_experiment, get_files_for_bird, get_file_data_by_index

    # Retrieve file selection
    bird, experiment, day = bird_combobox.get(), experiment_combobox.get(), day_combobox.get()
    files = []
    if selection == "current_day":
        files = get_files_for_day(app_state, bird, experiment, day, batch_file)
    elif selection == "current_experiment":
        files = get_files_for_experiment(app_state, bird, experiment, batch_file)
    elif selection == "current_bird":
        files = get_files_for_bird(app_state, bird, batch_file)
    elif selection == "current_file":
        files = [get_file_data_by_index(app_state.data_dir, app_state.song_files, app_state.current_file_index, app_state)["file_path"]]

    # Initialize progress bar
    max_value = len(files)
    progressbar = ttk.Progressbar(app_state.resegment_window, orient=tk.HORIZONTAL, length=200, mode='determinate', maximum=max_value)
    progressbar.grid(row=22, column=0, columnspan=2, pady=(10, 0), sticky="ew")
    
    # Create wrapper function for thread management
    def thread_wrapper():
        current_thread = threading.current_thread()
        try:
            segment_evfuncs(app_state, progressbar, files)
        finally:
            app_state.remove_thread(current_thread)
    
    # Start the segmentation thread
    thread = threading.Thread(target=thread_wrapper, name="SegmentEvfuncsThread")
    app_state.add_thread(thread)
    thread.start()


def segment_evfuncs(app_state, progressbar, files):
    """Perform segmentation using Evfuncs for each selected file."""
    from moove.utils import get_display_data, plot_data, save_notmat, decibel

    # Store the complete original file context to restore it later
    original_data_dir = app_state.data_dir
    original_song_files = app_state.song_files.copy() if app_state.song_files else []
    original_current_file_index = app_state.current_file_index

    for i, file_i in enumerate(files):
        try:
            progressbar['value'] = i
            app_state.resegment_window.update_idletasks()
            
            file_data = get_display_data({"file_name": os.path.basename(file_i), "file_path": file_i}, app_state.config)
            app_state.data_dir = os.path.dirname(file_i)

            # Retrieve segmentation parameters from app_state
            params = app_state.evfuncs_params
            threshold, min_syl_dur = float(params['threshold'].get()), float(params['min_syl_dur'].get())
            min_silent_dur, freq_cutoffs = float(params['min_silent_dur'].get()), tuple(map(int, params['freq_cutoffs'].get().split(',')))
            smooth_window = int(params['smooth_window'].get())

            sampling_rate, rawsong = int(file_data["sampling_rate"]), file_data["song_data"]
            smooth = evfuncs.smooth_data(rawsong, sampling_rate, freq_cutoffs, smooth_window)
            db_smooth = decibel(smooth)
            onsets, offsets = evfuncs.segment_song(db_smooth, sampling_rate, threshold, min_syl_dur, min_silent_dur)

            if onsets is not None and offsets is not None:
                onsets, offsets = np.multiply(onsets, 1000), np.multiply(offsets, 1000)
                file_data.update({"onsets": onsets, "offsets": offsets, "labels": "x" * len(onsets)})
            else:
                file_data.update({"onsets": np.array([]), "offsets": np.array([]), "labels": ""})

            save_notmat(os.path.join(app_state.data_dir, file_data["file_name"] + ".not.mat"), file_data)
            
        except Exception as e:
            app_state.logger.error(f"File {file_i} could not be processed correctly: {e}. Check manually.")
            return

    # Restore the complete original file context
    app_state.data_dir = original_data_dir
    app_state.song_files = original_song_files
    app_state.current_file_index = original_current_file_index

    progressbar['value'] = len(files)
    plot_data(app_state)
    progressbar.grid_forget()
    
    # Schedule Tkinter operations in the main thread
    def show_message():
        messagebox.showinfo("Info", f"Segmentation with Evfuncs completed successfully!")
    
    app_state.resegment_window.after(0, show_message)


def segment_ml(
    model, metadata, device, raw_song_data, sampling_rate=44100,
    chunk_size=64, decision_threshold=0.5, hist_size=3, onset_window_size=5,
    n_onset_true=3, offset_window_size=5, n_offset_false=4, min_silent_duration=0.03,
    min_syllable_length=0.005):
    """Perform segmentation using a machine learning model."""
    from moove.utils.audio_utils import index_to_seconds, seconds_to_chunk_index

    mean_loaded, std_loaded = torch.tensor(metadata['mean']).to(device), torch.tensor(metadata['std']).to(device)
    seg_input_size = hist_size # number of audio chunks used for classification

    y_pred_list, onset_flag = [], False
    onset_idxs, offset_idxs, raw_audio_data_list = [], [], []

    for i in range(0, len(raw_song_data), chunk_size):
        # audio data is processed in chunks
        raw_audio_data = raw_song_data[i:i + chunk_size]
        raw_audio_data_list.append(raw_audio_data)

        # stop if not enough chunks available
        if len(raw_audio_data) < chunk_size:
            break

        if i >= hist_size * chunk_size:
            X = torch.tensor(np.concatenate(raw_audio_data_list[-seg_input_size:]).astype(np.float32)).unsqueeze(0).to(device)
            X = (X - mean_loaded) / std_loaded
            
            y_pred = torch.sigmoid(model(X)).item() # probability of the chunk being part of a syllable
            y_pred_list.append(1 if y_pred > decision_threshold else 0) # if prob > threshold becomes a 1

            sub_y_onset, sub_y_offset = y_pred_list[-onset_window_size:], y_pred_list[-offset_window_size:]
            # if enough 1's in sliding window detected, onset detected
            if not onset_flag and sub_y_onset.count(1) >= n_onset_true:
                # ensure silence was present before detected onset
                if 0 in sub_y_onset[::-1]:
                    onset_idxs.append(len(y_pred_list) - sub_y_onset[::-1].index(0) - 1)
                    onset_flag = True
                else:
                    pass # no valid onset - pass
            # if enough 0's in sliding window detected after onset, offset detected
            elif onset_flag and sub_y_offset.count(0) >= n_offset_false:
                # ensure syllable was present before detected offset
                if 1 in sub_y_offset[::-1]:
                    offset_idxs.append(len(y_pred_list) - sub_y_offset[::-1].index(1) - 1)
                    onset_flag = False
                else:
                    pass # no valid offset - pass

    # Filter segments based on duration thresholds
    segments_2_remove = []
    min_dist, min_dist2 = seconds_to_chunk_index(min_syllable_length, chunk_size, sampling_rate), seconds_to_chunk_index(min_silent_duration, chunk_size, sampling_rate)
    
    if len(onset_idxs) == len(offset_idxs) + 1:
        onset_idxs.pop()
    elif len(onset_idxs) != len(offset_idxs):
        raise ValueError("Mismatch in onset and offset counts; cannot proceed.")

    for i, (onset_idx, offset_idx) in enumerate(zip(onset_idxs, offset_idxs)):
        if offset_idx - onset_idx < min_dist or (i < len(onset_idxs) - 1 and onset_idxs[i + 1] - offset_idx < min_dist2):
            segments_2_remove.append(i)

    # Filter valid onsets and offsets based on duration checks
    onset_idxs = [onset_idxs[i] for i in range(len(onset_idxs)) if i not in segments_2_remove]
    offset_idxs = [offset_idxs[i] for i in range(len(offset_idxs)) if i not in segments_2_remove]

    onsets = [index_to_seconds(idx, chunk_size, sampling_rate) for idx in onset_idxs]
    offsets = [index_to_seconds(idx, chunk_size, sampling_rate) for idx in offset_idxs]

    return onsets, offsets


def create_segmentation_training_dataset(app_state, progressbar, dataset_name, all_files, root):
    """Create a segmentation training dataset from multiple files."""
    from moove.utils import get_display_data, save_features, plot_data, extract_raw_audio

    if len(all_files) == 0:
        messagebox.showinfo("Error", "Not enough files given! You need at least 1 file to create a dataset.")
        return

    chunk_size = int(app_state.train_segmentation_params['chunk_size'].get())
    hist_size = int(app_state.train_segmentation_params['hist_size'].get()) + 1  # Increment for overlapping chunks
    overlap_chunks = app_state.train_segmentation_params['overlap_chunks'].get()

    all_features = []  # List to collect all feature rows

    def generate_concatenated_chunks_with_labels(arr, hist_size, overlap_chunks=False):
        """Concatenate chunks with labels based on history size and overlap setting."""
        concatenated_chunks = []
        step_size = 1 if overlap_chunks else hist_size

        for i in range(0, len(arr) - (hist_size - 1), step_size):
            if np.all(arr[i:i + hist_size, 0] == arr[i, 0]):  # Ensure all rows are from the same file
                file_index = arr[i, 0]  # Preserve file index
                chunk_data = np.concatenate(arr[i:i + hist_size, 1:-1]).flatten()  # Concatenate chunk data
                label = arr[i, -1]  # Preserve label
                concatenated_chunks.append(np.concatenate(([file_index], chunk_data, [label])))

        return np.array(concatenated_chunks)
    
    progressbar.grid_remove()

    # Add running label to the GUI
    font_style = tkFont.Font(family="Arial", size=14)
    running_label = tk.Label(app_state.training_window, text="Looking for segments...", fg="green", font=font_style)
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
        
    num_segs = 0
    for file_path in all_files:
        info = get_onset_offset_info(file_path)
        if len(info["onsets"]) > 0 and len(info["offsets"]) > 0:
            segs = min(len(info["onsets"]), len(info["offsets"]))
            num_segs += segs

    if num_segs == 0:
        running_label.destroy()
        root.update_idletasks() 
        messagebox.showinfo("Error", "No segments found in the given files.")
        return
    
    running_label.destroy()
    root.update_idletasks() 
    progressbar.grid()

    for file_index, file_path in enumerate(all_files):
        display_data = get_display_data({"file_name": os.path.basename(file_path), "file_path": file_path}, app_state.config)
        sampling_rate, rawsong = int(display_data["sampling_rate"]), display_data["song_data"]
        onsets = (np.array(display_data["onsets"]) * sampling_rate / 1000).astype(int)
        offsets = (np.array(display_data["offsets"]) * sampling_rate / 1000).astype(int)
        progressbar['value'] += 1  # Update progress

        # Extract audio features in chunks
        audio_features = extract_raw_audio(rawsong, chunk_size)
        labels = np.zeros(len(audio_features[0]))  # Initialize labels

        # Assign labels based on overlaps with onsets and offsets
        for i, start_idx in enumerate(range(0, len(rawsong) - chunk_size + 1, chunk_size)):
            end_idx = start_idx + chunk_size
            if i < len(labels) and any(onset <= end_idx and start_idx <= offset for onset, offset in zip(onsets, offsets)):
                labels[i] = 1

        # Generate feature rows for this file
        file_features = []
        for i, features in enumerate(np.array(audio_features).T):
            row = np.insert(features, 0, file_index)  # Insert file index as first element
            row = np.append(row, labels[i])  # Append label
            file_features.append(row)

        # Apply concatenation function and extend to all features
        concatenated_features = generate_concatenated_chunks_with_labels(np.array(file_features), hist_size, overlap_chunks)
        if concatenated_features.size > 0:
            all_features.extend(concatenated_features)
    
    # Convert to NumPy array and save
    feature_array = np.array(all_features, dtype=object)
    save_features(app_state, dataset_name, feature_array, chunk_size=chunk_size, hist_size=hist_size, num_syls=num_segs)

    app_state.update_segmentation_datasets_combobox()
    progressbar['value'] = len(all_files)  # Complete progress
    plot_data(app_state)
    progressbar.grid_forget()
    
    # Schedule Tkinter operations in the main thread
    def show_message():
        messagebox.showinfo("Info", "The segmentation training dataset has been created successfully!")
        app_state.change_file(0)
    
    root.after(0, show_message)
    


def segment_files_ml(app_state, progressbar, all_files, model, metadata, device):
    """Segment files using a machine learning model in a threaded process."""
    from moove.utils import get_display_data, plot_data, save_notmat

    # Store the complete original file context to restore it later
    original_data_dir = app_state.data_dir
    original_song_files = app_state.song_files.copy() if app_state.song_files else []
    original_current_file_index = app_state.current_file_index

    # extract parameters used for segmentation
    hist_size, chunk_size = int(metadata['hist_size']), int(metadata['chunk_size'])

    for i, file_path in enumerate(all_files):
        try:
            progressbar['value'] = i  # Update progress
            display_data = get_display_data({"file_name": os.path.basename(file_path), "file_path": file_path}, app_state.config)
            app_state.data_dir = os.path.dirname(file_path)

            sampling_rate = int(display_data["sampling_rate"])
            rawsong = display_data["song_data"]

            # Retrieve segmentation parameters
            params = app_state.mlseg_params
            onsets, offsets = segment_ml(
                model, metadata, device, rawsong, sampling_rate, chunk_size,
                float(params['decision_threshold'].get()), hist_size,
                int(params['onset_window_size'].get()), int(params['n_onset_true'].get()),
                int(params['offset_window_size'].get()), int(params['n_offset_false'].get()),
                float(params['min_silent_duration'].get()), float(params['min_syllable_length'].get())
            )

            # Convert onsets and offsets to milliseconds
            display_data.update({
                "onsets": np.array(onsets) * 1000,
                "offsets": np.array(offsets) * 1000,
                "labels": "x" * len(onsets)  # Label all segments with 'x'
            })
            save_notmat(os.path.join(app_state.data_dir, display_data["file_name"] + ".not.mat"), display_data)
            
        except Exception as e:
            app_state.logger.error(f"File {file_path} could not be processed correctly: {e}. Check manually.")
            return

    # Restore the complete original file context
    app_state.data_dir = original_data_dir
    app_state.song_files = original_song_files
    app_state.current_file_index = original_current_file_index

    # Ensure we have a valid file context for navigation
    if not app_state.song_files or len(app_state.song_files) == 0:
        app_state.logger.debug("No song files found, trying to populate from restored directory")
        try:
            if os.path.exists(app_state.data_dir):
                files_in_dir = [f for f in os.listdir(app_state.data_dir) 
                               if f.lower().endswith('.wav') or f.lower().endswith('.cbin')]
                if files_in_dir:
                    app_state.song_files = files_in_dir
                    app_state.current_file_index = 0
                    app_state.logger.debug(f"Populated song_files with {len(files_in_dir)} files")
                else:
                    app_state.logger.debug("No audio files found in restored directory")
        except Exception as e:
            app_state.logger.debug(f"Could not populate song_files: {e}")

    # Final UI update
    app_state.reset_edit_type()
    plot_data(app_state)
    progressbar.grid_forget()
    
    # Schedule Tkinter operations in the main thread
    def show_message():
        messagebox.showinfo("Info", "Segmentation completed successfully!")
    
    app_state.resegment_window.after(0, show_message)


def start_segment_files_thread(app_state, segmentation_model_name, selection, checkbox_ow, batch_file, bird_combobox, experiment_combobox, day_combobox):
    """Start a threaded process to segment files based on a selected model and criteria."""
    from moove.utils import get_files_for_day, get_files_for_experiment, get_files_for_bird, get_file_data_by_index

    # Retrieve file selection based on user criteria
    bird = bird_combobox.get()
    experiment = experiment_combobox.get()
    day = day_combobox.get()
    files = []

    if selection == "current_day":
        files = get_files_for_day(app_state, bird, experiment, day, batch_file)
    elif selection == "current_experiment":
        files = get_files_for_experiment(app_state, bird, experiment, batch_file)
    elif selection == "current_bird":
        files = get_files_for_bird(app_state, bird, batch_file)
    elif selection == "current_file":
        files = [get_file_data_by_index(app_state.data_dir, app_state.song_files, app_state.current_file_index, app_state)["file_path"]]

    # Set up device and load model with metadata
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        checkpoint = torch.load(os.path.join(app_state.config['global_dir'], 'trained_models', f'{segmentation_model_name}.pth'), map_location=device)
    except: 
        messagebox.showinfo("Error", "Selected segmentation model doesn't exist or is not valid! Perhaps you forgot to pick a model?")
    model, metadata = checkpoint['model'], checkpoint['metadata']
    
    model.to(device)
    model.eval()
    
    if checkbox_ow:
        files = files
    else:
        # use non-manually segmented files only
        files = load_segmentation_checkmarks(files)

    # Initialize and display progress bar
    max_value = len(files)
    progressbar = ttk.Progressbar(app_state.resegment_window, orient=tk.HORIZONTAL, length=200, mode='determinate', maximum=max_value)
    progressbar.grid(row=22, column=0, columnspan=2, pady=(10, 0), sticky="ew")
    
    # Create wrapper function for thread management
    def thread_wrapper():
        current_thread = threading.current_thread()
        try:
            segment_files_ml(app_state, progressbar, files, model, metadata, device)
        finally:
            app_state.remove_thread(current_thread)
    
    thread = threading.Thread(target=thread_wrapper, name="SegmentMLThread")
    app_state.add_thread(thread)
    thread.start()


def start_create_segmentation_training_dataset(app_state, dataset_name, use_selected_files, selection, batch_file, bird_combobox, experiment_combobox, day_combobox, root):
    """Start a threaded process to create a segmentation training dataset."""
    from moove.utils import get_files_for_day, get_files_for_experiment, get_files_for_bird, filter_segmented_files

    font_style = tkFont.Font(family="Arial", size=14)
    running_label = tk.Label(app_state.training_window, text="Looking for files...", fg="green", font=font_style)
    running_label.grid(row=22, column=0, columnspan=2, pady=(10, 0), sticky=tk.W) 
    root.update_idletasks() 
    
    # Retrieve file selection based on user criteria
    bird = bird_combobox.get()
    experiment = experiment_combobox.get()
    day = day_combobox.get()
    files = []

    if selection == "current_day":
        files = get_files_for_day(app_state, bird, experiment, day, batch_file)
    elif selection == "current_experiment":
        files = get_files_for_experiment(app_state, bird, experiment, batch_file)
    elif selection == "current_bird":
        files = get_files_for_bird(app_state, bird, batch_file)

    if use_selected_files:
        files = filter_segmented_files(files)
        
    running_label.destroy()
    root.update_idletasks() 

    # Initialize and display progress bar
    dataset_name = str(dataset_name)
    # check if dataset name is valid
    if len(dataset_name) < 1:
        messagebox.showinfo("Error", "Dataset name not valid! A dataset name needs to contain at least one character.")
    else:
        # create dataset with valid name
        max_value = len(files)
        progressbar = ttk.Progressbar(app_state.training_window, orient=tk.HORIZONTAL, length=200, mode='determinate', maximum=max_value)
        progressbar.grid(row=22, column=0, columnspan=2, pady=(10, 0), sticky="ew")
        
        # Create wrapper function for thread management
        def thread_wrapper():
            current_thread = threading.current_thread()
            try:
                create_segmentation_training_dataset(app_state, progressbar, dataset_name, files, root)
            finally:
                app_state.remove_thread(current_thread)
        
        thread = threading.Thread(target=thread_wrapper, name="CreateSegDatasetThread")
        app_state.add_thread(thread)
        thread.start()
