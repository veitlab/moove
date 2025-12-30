# utils/label_utils.py
import os
import pandas as pd
import pickle
import threading
import tkinter as tk
import tkinter.font as tkFont
import torch
import torch.nn.functional as F
import evfuncs
import re
from scipy.signal import spectrogram
from tkinter import ttk, messagebox
import numpy as np


def load_classification_checkmarks(all_files):
    """Check whether files have been manually checked as being classified"""
    unclass_files = []
    for file in all_files:
        recfile_path = os.path.splitext(file)[0] + ".rec"
        with open(recfile_path, "r") as f:
            content = f.read()

        hand_segmented_pattern = r"Hand Classified = (\d+)"
        hand_segmented_match = re.search(hand_segmented_pattern, content)

        if hand_segmented_match.group(1) == '0':
            unclass_files.append(file)

    return unclass_files


def start_create_classification_training_dataset(app_state, dataset_name, use_selected_files, selection, batch_file, bird_combobox, experiment_combobox, day_combobox, root):
    """Initialize the creation of a classification training dataset in a new thread."""
    from moove.utils import get_files_for_day, get_files_for_experiment, get_files_for_bird, filter_classified_files
    
    font_style = tkFont.Font(family="Arial", size=14)
    running_label = tk.Label(app_state.training_window, text="Looking for files...", fg="green", font=font_style)
    running_label.grid(row=22, column=0, columnspan=2, pady=(10, 0), sticky=tk.W) 
    root.update_idletasks() 

    # choose files depending on the user selection
    bird, experiment, day = bird_combobox.get(), experiment_combobox.get(), day_combobox.get()
    if selection == "current_day":
        files = get_files_for_day(app_state, bird, experiment, day, batch_file)
    elif selection == "current_experiment":
        files = get_files_for_experiment(app_state, bird, experiment, batch_file)
    elif selection == "current_bird":
        files = get_files_for_bird(app_state, bird, batch_file)

    if use_selected_files:
        files = filter_classified_files(files)
        
    running_label.destroy()
    root.update_idletasks() 

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
                create_classification_training_dataset(app_state, progressbar, dataset_name, files, root)
            finally:
                app_state.remove_thread(current_thread)
        
        thread = threading.Thread(target=thread_wrapper, name="CreateClassDatasetThread")
        app_state.add_thread(thread)
        thread.start()


def create_classification_training_dataset(app_state, progressbar, dataset_name, files, root):
    """Create a classification training dataset based on selected files and parameters."""
    from moove.utils import get_display_data, seconds_to_index

    if len(files) == 0:
        messagebox.showinfo("Error", "Not enough files given! You need at least 1 file to create a dataset.")
        return

    input_length_str = app_state.spec_params['input_length'].get()
    input_length, chunk_size = map(int, input_length_str.split(','))
    nperseg, noverlap, nfft = int(app_state.spec_params['nperseg'].get()), int(app_state.spec_params['noverlap'].get()), int(app_state.spec_params['nfft'].get())
    freq_cutoffs = tuple(map(int, app_state.spec_params['freq_cutoffs'].get().split(',')))
    input_array_size = input_length * chunk_size
    
    going_prod_df = pd.DataFrame(columns=['file', 'onset_no', 'taf_unflattend_spectrogram', 'label'])
    entry_no = 0

    progressbar.grid_remove()

    # Add running label to the GUI
    font_style = tkFont.Font(family="Arial", size=14)
    running_label = tk.Label(app_state.training_window, text="Looking for syllables...", fg="green", font=font_style)
    running_label.grid(row=22, column=0, columnspan=2, pady=(10, 0), sticky=tk.W) 
    root.update_idletasks() 

    def get_onsets(file_path):
        notmat_file = file_path + ".not.mat"
        if os.path.exists(notmat_file):
            notmat_dict = evfuncs.load_notmat(notmat_file)
            return notmat_dict.get("onsets", [])
        else:
            return []
    
    # Count number of syllable onsets    
    num_onsets = 0
    for i, file_i in enumerate(files):
        working_dir = os.getcwd()
        file_path = os.path.join(working_dir, file_i)
        onsets = get_onsets(file_path)
        if len(onsets) > 0:
            num_onsets += len(onsets)

    if num_onsets == 0:
        running_label.destroy()
        root.update_idletasks() 
        messagebox.showinfo("Error", "No syllable onsets found in the given files.")
        return

    running_label.destroy()
    root.update_idletasks() 
    progressbar.grid()

    for i, file_i in enumerate(files):
        app_state.training_window.update_idletasks()
        working_dir = os.getcwd()
        file_path = os.path.join(working_dir, file_i)
        file_data = get_display_data({"file_name": os.path.basename(file_path), "file_path": file_path}, app_state.config)
        sampling_rate = int(file_data["sampling_rate"])
        rawsong, onsets, labels = file_data["song_data"], file_data["onsets"], file_data["labels"]

        if len(onsets) > 0:
            progressbar['value'] = i
            for syllable_no, onset in enumerate(onsets):
                entry_no += 1
                onset_index = int(seconds_to_index(onset, sampling_rate))
                cutted_raw_song = rawsong[onset_index:onset_index + input_array_size]

                # Ensure consistent shape by setting nperseg and noverlap
                f, t, Sxx_taf = spectrogram(cutted_raw_song, fs=sampling_rate, nperseg=nperseg, noverlap=noverlap, nfft=nfft)
                if Sxx_taf.ndim == 2:
                    Sxx_taf = Sxx_taf[(f >= freq_cutoffs[0]) & (f <= freq_cutoffs[1]), :]
                else:
                    app_state.logger.warning(f"Warning: Sxx_taf is {Sxx_taf.ndim}-dimensional for file {file_i}, skipping this entry.")
                    continue

                going_prod_df.loc[entry_no] = [file_i, syllable_no, Sxx_taf, labels[syllable_no]]

    metadata = {
        'input_length': input_length_str,
        'nperseg': nperseg,
        'noverlap': noverlap,
        'nfft': nfft,
        'lowcut': freq_cutoffs[0],
        'highcut': freq_cutoffs[1],
    }

    save_path = os.path.join(app_state.config['global_dir'], 'training_data', f'{dataset_name}_class.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump({'dataframe': going_prod_df, 'metadata': metadata}, f)

    app_state.update_classification_datasets_combobox()
    progressbar['value'] = len(files)
    progressbar.grid_forget()
    
    # Schedule Tkinter operations in the main thread
    def show_message():
        messagebox.showinfo("Info", "Classification training dataset has been created successfully!")
    
    root.after(0, show_message)

    first_index = going_prod_df.index[0]
    shape_first_entry = pd.DataFrame(going_prod_df.loc[first_index, 'taf_unflattend_spectrogram']).shape
    app_state.logger.debug(f"The shape of the first entry in 'taf_unflattend_spectrogram' is {shape_first_entry}")


def normalize_spectrogram(spectrogram):
    """Normalize the spectrogram to zero mean and unit variance."""
    mean, std = spectrogram.mean(), spectrogram.std()
    return (spectrogram - mean) / std if std != 0 else spectrogram


def start_classify_files_thread(app_state, model_name, selection, checkbox_ow, batch_file, bird_combobox, experiment_combobox, day_combobox):
    """Start the classification process for selected files in a new thread."""
    from moove.utils import get_files_for_day, get_files_for_experiment, get_files_for_bird, get_file_data_by_index

    bird, experiment, day = bird_combobox.get(), experiment_combobox.get(), day_combobox.get()
    # Choose files depending on the user selection
    if selection == "current_day":
        files = get_files_for_day(app_state, bird, experiment, day, batch_file)
    elif selection == "current_experiment":
        files = get_files_for_experiment(app_state, bird, experiment, batch_file)
    elif selection == "current_bird":
        files = get_files_for_bird(app_state, bird, batch_file)
    elif selection == "current_file":
        files = [get_file_data_by_index(app_state.data_dir, app_state.song_files, app_state.current_file_index, app_state)["file_path"]]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        checkpoint = torch.load(os.path.join(app_state.config['global_dir'], 'trained_models', f'{model_name}.pth'), map_location=device)
    except: 
        messagebox.showinfo("Error", "Selected classification model doesn't exist or is not valid! Perhaps you forgot to pick a model?")
    model, metadata = checkpoint['model'], checkpoint['metadata']

    model.to(device).eval()

    if checkbox_ow:
        files = files
    else:
        # use non-manually classified files only
        files = load_classification_checkmarks(files)

    max_value = len(files)
    progressbar = ttk.Progressbar(app_state.relabel_window, orient=tk.HORIZONTAL, length=200, mode='determinate', maximum=max_value)
    progressbar.grid(row=22, column=0, columnspan=2, pady=(10, 0), sticky="ew")
    
    # Create wrapper function for thread management
    def thread_wrapper():
        current_thread = threading.current_thread()
        try:
            ml_classify_file(app_state, progressbar, max_value, files, model, metadata, device)
        finally:
            app_state.remove_thread(current_thread)
    
    thread = threading.Thread(target=thread_wrapper, name="ClassifyFilesThread")
    app_state.add_thread(thread)
    thread.start()


def ml_classify_file(app_state, progressbar, max_value, all_files, model, metadata, device):
    """Perform classification on each file and update labels."""
    from moove.utils import get_display_data, plot_data, save_notmat, seconds_to_index

    # Store the complete original file context to restore it later
    original_data_dir = app_state.data_dir
    original_song_files = app_state.song_files.copy() if app_state.song_files else []
    original_current_file_index = app_state.current_file_index

    input_length, chunk_size = map(int, metadata['input_length'].split(','))
    nperseg, noverlap, nfft = int(metadata['nperseg']), int(metadata['noverlap']), int(metadata['nfft'])
    lowcut, highcut, int_to_label = int(metadata['lowcut']), int(metadata['highcut']), metadata['int_to_label']
    input_array_size = input_length * chunk_size

    for i, file_i in enumerate(all_files):
        try:
            progressbar['value'] = i
            app_state.relabel_window.update_idletasks()
            file_data = get_display_data({"file_name": os.path.basename(file_i), "file_path": file_i}, app_state.config)
            sampling_rate, rawsong, onsets = int(file_data["sampling_rate"]), file_data["song_data"], file_data["onsets"]
            app_state.data_dir = os.path.dirname(file_i)

            labels = []

            # Classify syllables in selected files
            for onset in onsets:
                onset_index = int(seconds_to_index(onset, sampling_rate))
                cutted_raw_song = rawsong[onset_index:onset_index + input_array_size]

                f, _, Sxx_taf = spectrogram(cutted_raw_song, fs=sampling_rate, nperseg=nperseg, noverlap=noverlap, nfft=nfft)
                Sxx_taf = Sxx_taf[(f >= lowcut) & (f <= highcut), :]
                Sxx_normalized = normalize_spectrogram(Sxx_taf)

                input_data = torch.tensor(Sxx_normalized).float().unsqueeze(0).unsqueeze(0).to(device)
                input_tensor = F.pad(input_data, (0, 1, 0, 1))

                with torch.no_grad():
                    output = model(input_tensor)
                    predicted_class = torch.argmax(output).item()
                labels.append(int_to_label[predicted_class])

            file_data["labels"] = ''.join(labels)
            save_notmat(os.path.join(app_state.data_dir, f"{file_data['file_name']}.not.mat"), file_data)
            
        except Exception as e:
            app_state.info(f"File {file_i} could not be processed correctly: {e}. Check manually.")
            return

    # Restore the complete original file context
    app_state.data_dir = original_data_dir
    app_state.song_files = original_song_files
    app_state.current_file_index = original_current_file_index

    progressbar['value'] = len(all_files)
    
    # Final UI update
    app_state.reset_edit_type()
    plot_data(app_state)
    progressbar.grid_forget()
    
    # Schedule Tkinter operations in the main thread
    def show_message():
        messagebox.showinfo("Info", f"Relabeling of files completed successfully!")
    
    app_state.relabel_window.after(0, show_message)
