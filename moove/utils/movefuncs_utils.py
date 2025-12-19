# utils/movefuncs_utils.py
import numpy as np
import os
import re
import scipy.io.wavfile as wav
import sounddevice as sd
import threading
import tkinter as tk
from jinja2 import Template
from pathlib import Path
from scipy.io import savemat


def save_cbin(filepath, data, sample_freq):
    """Writes data to a .cbin file."""
    filename = os.path.basename(filepath)
    
    filename = filename
    filename = Path(filename)
    
    # Convert data to big endian, 16-bit signed int
    data = data.astype('>i2')
    
    # Write data to the .cbin file
    data.tofile(filepath)


def save_notmat(filename, notmat_dict):
    """Saves dictionary as a .not.mat file with all numeric fields as float64, using empty arrays if fields are missing."""
    filename = Path(filename)

    if not str(filename).endswith(".not.mat"):
        raise ValueError(
            f"Filename should have extension .not.mat but extension was: {filename.suffix}"
        )

    # Convert 'onsets' and 'offsets' arrays to floats in ms
    onsets = notmat_dict['onsets'].astype(np.float64)
    offsets = notmat_dict['offsets'].astype(np.float64)

    # Create dictionary with keys and default values if needed
    save_dict = {
        '__header__': notmat_dict['__header__'],
        '__version__': notmat_dict['__version__'],
        '__globals__': notmat_dict['__globals__'],
        'Fs': np.float64(notmat_dict['Fs']),
        'fname': notmat_dict['file_name'],
        'labels': notmat_dict['labels'],
        'onsets': onsets,
        'offsets': offsets,
        'min_int': np.float64(notmat_dict['min_int']) if 'min_int' in notmat_dict and notmat_dict['min_int'] else np.array([], dtype=np.float64),
        'min_dur': np.float64(notmat_dict['min_dur']) if 'min_dur' in notmat_dict and notmat_dict['min_dur'] else np.array([], dtype=np.float64),
        'threshold': np.float64(notmat_dict['threshold']) if 'threshold' in notmat_dict and notmat_dict['threshold'] else np.array([], dtype=np.float64),
        'sm_win': np.float64(notmat_dict['sm_win']) if 'sm_win' in notmat_dict and notmat_dict['sm_win'] else np.array([], dtype=np.float64)
    }

    # Convert header into bytes
    save_dict['__header__'] = np.compat.asbytes(save_dict['__header__'])

    # Save file as .mat file
    savemat(filename, save_dict, do_compression=True)
 

def load_recfile(file_path):
    '''Loads a .rec file and returns its contents as a dictionary.'''
    with open(file_path, "r") as f:
        content = f.read()

    date_pattern = r"File created: (.+)"
    begin_rec_pattern = r"begin rec = (\d+) ms"
    trig_time_pattern = r"trig time  = (\d+(\.\d+)?) ms"
    rec_end_pattern = r"rec end = (\d+) ms"
    adfreq_pattern = r"ADFREQ =\s+(\d+)"
    chans_pattern = r"Chans = (\d+)"
    samples_pattern = r"Samples = (\d+)"
    catch_song_pattern = r"Catch Song = (\d+)"
    hand_segmented_pattern = r"Hand Segmented = (\d+)"
    hand_classified_pattern = r"Hand Classified = (\d+)"
    t_before_pattern = r"T Before = ([\d\.]+)"
    t_after_pattern = r"T After = ([\d\.]+)"
    feedback_pattern = r"([\d\.]+E\+?\d+) msec: (FB|catch) # ([A-Za-z0-9_\.\\/:]+) : Templ = (\d+)" # exchanged ([\w\.]+) for evtaf compatibility

    # search for the patterns in the content
    date_match = re.search(date_pattern, content)
    begin_rec_match = re.search(begin_rec_pattern, content)
    trig_time_match = re.search(trig_time_pattern, content)
    rec_end_match = re.search(rec_end_pattern, content)
    adfreq_match = re.search(adfreq_pattern, content)
    chans_match = re.search(chans_pattern, content)
    samples_match = re.search(samples_pattern, content)
    catch_song_match = re.search(catch_song_pattern, content)
    hand_segmented_match = re.search(hand_segmented_pattern, content)
    hand_classified_match = re.search(hand_classified_pattern, content)
    t_before_match = re.search(t_before_pattern, content)
    t_after_match = re.search(t_after_pattern, content)
    feedback_matches = re.findall(feedback_pattern, content)

    # extract feedback information
    feedback_info = []
    for match in feedback_matches:
        feedback_time = float(match[0])/1000
        trig_pulse = match[2] # can be string or int
        templ = int(match[3])
        feedback_info.append((feedback_time, trig_pulse, templ))

    recfile_dict = {
        "file_created": date_match.group(1),
        "begin_rec": int(begin_rec_match.group(1)),
        "trig_time": int(float(trig_time_match.group(1))),
        "rec_end": int(rec_end_match.group(1)),
        "adfreq": int(adfreq_match.group(1)),
        "chans": int(chans_match.group(1)),
        "samples": int(samples_match.group(1)),
        "catch_song": int(catch_song_match.group(1)),
        "hand_segmented": int(hand_segmented_match.group(1)),
        "hand_classified": int(hand_classified_match.group(1)),
        "t_before": float(t_before_match.group(1)),
        "t_after": float(t_after_match.group(1)),
        "feedback_info": feedback_info
    }

    return recfile_dict


def save_recfile(file_path, recfile_dict):
    '''Saves dictionary as a .rec file.'''
    template_string = """File created: {{ file_created }}

    begin rec = {{ begin_rec }} ms
    trig time  = {{ trig_time }} ms
    rec end = {{ rec_end }} ms

ADFREQ = {{ adfreq }}
Chans = {{ chans }}
Samples = {{ samples }}
Catch Song = {{ catch_song }}
Hand Segmented = {{ hand_segmented }}
Hand Classified = {{ hand_classified }}
T Before = {{ t_before }}
T After = {{ t_after }}
Feedback information:
{% for info in feedback_info %}
{{ info[0] }} msec: {{ info[1] }}
{%- endfor %}
"""
    template = Template(template_string)
    output = template.render(recfile_dict)

    with open(file_path, 'w') as f:
        f.write(output)
    return


def extract_raw_audio(full_audio_data, chunk_size):
    '''Extracts raw audio data from a full audio file.'''
    num_full_chunks = len(full_audio_data) // chunk_size

    audio_features = [[] for _ in range(chunk_size)]
    
    for i in range(num_full_chunks * chunk_size): 
        chunk_index = i % chunk_size
        audio_features[chunk_index].append(full_audio_data[i])

    return audio_features


def play_sound(display_dict, ax1):
    '''Plays the sound of the displayed data.'''
    x_start, x_end = ax1.get_xlim()

    x1_border = int(x_start * display_dict["sampling_rate"])
    x2_border = int(x_end * display_dict["sampling_rate"])

    sound = display_dict["song_data"][x1_border:x2_border]
    sampling_rate = display_dict["sampling_rate"]

    def play():
        try:
            sd.play(sound, samplerate=sampling_rate)
            sd.wait()
        except Exception as e:
            return

    play_sound_thread = threading.Thread(target=play)
    play_sound_thread.start()


def handle_playback(app_state):
    '''Function to handle the playback of the displayed data.'''
    display_dict = app_state.display_dict
    ax1 = app_state.ax1

    # Create wrapper function for thread management
    def thread_wrapper():
        current_thread = threading.current_thread()
        try:
            play_sound(display_dict, ax1)
        finally:
            app_state.remove_thread(current_thread)

    play_sound_thread = threading.Thread(target=thread_wrapper, name="PlaybackThread")
    app_state.add_thread(play_sound_thread)
    play_sound_thread.start()


def confirm_delete(app_state):
    ''' Confirm the deletion of the displayed file.'''
    from moove.utils.plot_utils import plot_data
    current_file = app_state.song_files[app_state.current_file_index]
    working_dir = os.getcwd()

    # Delete all instances of the file
    if current_file[-5:] == ".cbin":
        if os.path.exists(os.path.join(working_dir, app_state.data_dir, current_file + ".not.mat")):
            os.remove(os.path.join(working_dir, app_state.data_dir, current_file + ".not.mat"))
        if os.path.exists(os.path.join(working_dir, app_state.data_dir, current_file[:-5] + ".rec")):
            os.remove(os.path.join(working_dir, app_state.data_dir, current_file[:-5] + ".rec"))
        if os.path.exists(os.path.join(working_dir, app_state.data_dir, current_file)):
            os.remove(os.path.join(working_dir, app_state.data_dir, current_file))
    elif current_file[-4:] == ".wav":
        if os.path.exists(os.path.join(working_dir, app_state.data_dir, current_file + ".not.mat")):
            os.remove(os.path.join(working_dir, app_state.data_dir, current_file + ".not.mat"))
        if os.path.exists(os.path.join(working_dir, app_state.data_dir, current_file[:-4] + ".rec")):
            os.remove(os.path.join(working_dir, app_state.data_dir, current_file[:-4] + ".rec"))
        if os.path.exists(os.path.join(working_dir, app_state.data_dir, current_file)):
            os.remove(os.path.join(working_dir, app_state.data_dir, current_file))
    else:
        app_state.logger.warning("Not supported file format")
        return

    del app_state.song_files[app_state.current_file_index]

    # Use the current batch file instead of default "batch.txt"
    with open(os.path.join(app_state.data_dir, app_state.current_batch_file), "w") as file:
        for song in app_state.song_files:
            file.write(song + '\n')

    app_state.combobox['values'] = app_state.song_files

    if app_state.current_file_index >= len(app_state.song_files):
        app_state.change_file(-1)
        plot_data(app_state)
    else:
        app_state.change_file(0)
        plot_data(app_state)


def handle_delete(app_state):
    '''Function to handle the deletion of the displayed file.'''
    import tkinter as tk
    from tkinter import messagebox
    
    current_file = app_state.song_files[app_state.current_file_index]
    
    # Create a custom dialog with three options
    result = messagebox.askyesnocancel(
        "Delete Options", 
        f"How do you want to remove '{current_file}'?",
        detail="Yes = Delete file from disk\nNo = Remove from batch only\nCancel = Do nothing"
    )
    
    if result is True:  # Yes - Delete file completely
        confirm_delete(app_state)
    elif result is False:  # No - Remove from batch only
        remove_from_batch_only(app_state)
    # Result is None (Cancel) - Do nothing


def remove_from_batch_only(app_state):
    '''Remove file from current batch but keep it on disk.'''
    from moove.utils.plot_utils import plot_data
    
    # Remove from song_files list
    del app_state.song_files[app_state.current_file_index]
    
    # Update the current batch file
    with open(os.path.join(app_state.data_dir, app_state.current_batch_file), "w") as file:
        for song in app_state.song_files:
            file.write(song + '\n')
    
    # Update combobox
    app_state.combobox['values'] = app_state.song_files
    
    # Navigate to next/previous file
    if app_state.current_file_index >= len(app_state.song_files):
        app_state.change_file(-1)
        plot_data(app_state)
    else:
        app_state.change_file(0)
        plot_data(app_state)


def crop_not_mat(file_path, display_dict, x1_border, x2_border):
    '''Function to crop a .not.mat file.'''
    onsets = display_dict["onsets"]
    offsets = display_dict["offsets"]
    labels = display_dict["labels"]

    x1_border_ms = x1_border * 1000
    x2_border_ms = x2_border * 1000

    onset_index = next((i for i, onset in enumerate(onsets) if onset >= x1_border_ms), len(onsets))
    offset_index = next((i for i, offset in enumerate(offsets) if offset > x2_border_ms), len(offsets))

    cropped_onsets = onsets[onset_index:offset_index]
    cropped_offsets = offsets[onset_index:offset_index]
    cropped_labels = labels[onset_index:offset_index]

    display_dict["onsets"] = np.subtract(cropped_onsets, x1_border_ms)
    display_dict["offsets"] = np.subtract(cropped_offsets, x1_border_ms)
    display_dict["labels"] = cropped_labels

    save_notmat(file_path, display_dict)
    return


def crop_rec_file(file_path, display_dict, x1_border, x2_border, len_cropped_song):
    '''Function to crop a .rec file.'''
    from moove.utils.movefuncs_utils import save_recfile, load_recfile
    import datetime
    recfile_dict = load_recfile(file_path)

    recfile_dict["file_created"] = datetime.datetime.now().strftime("%a, %b %d, %Y, %H:%M:%S")
    recfile_dict["rec_end"] = int(np.round((x2_border - x1_border) * 1000))
    recfile_dict["samples"] = int(len_cropped_song)

    # Adjust all infos relative to the new cropped time
    for feedbackinfo_idx in range(len(recfile_dict["feedback_info"])):
        new_feedback_triggertime = (recfile_dict["feedback_info"][feedbackinfo_idx][0] - x1_border)*1000
        coeff, exp = "{:.6E}".format(new_feedback_triggertime).split("E")
        if recfile_dict["catch_song"] == 1:
            recfile_dict["feedback_info"][feedbackinfo_idx] = (f"{coeff}E{str(int(exp))}",
                                                               f"catch # catch_file.wav : Templ = {0}")
        elif recfile_dict["catch_song"] == 0:
            recfile_dict["feedback_info"][feedbackinfo_idx] = (f"{coeff}E{str(int(exp))}",
                                                               f"FB # {recfile_dict['feedback_info'][feedbackinfo_idx][1]} : Templ = {0}")

    save_recfile(file_path, recfile_dict)
    return


def confirm_crop(app_state):
    '''Function to confirm the cropping of the displayed data.'''
    from moove.utils.file_utils import get_file_data_by_index, get_display_data
    from moove.utils.plot_utils import plot_data
    from moove.utils.movefuncs_utils import save_cbin

    file_path = get_file_data_by_index(app_state.data_dir, app_state.song_files, app_state.current_file_index, app_state)
    display_dict = get_display_data(file_path, app_state.config)

    ax1 = app_state.ax1

    # Crop x-axis to the selected area
    x_start, x_end = ax1.get_xlim()
    x1_border = int(x_start * display_dict["sampling_rate"])
    x2_border = int(x_end * display_dict["sampling_rate"])

    cropped_song_data = display_dict["song_data"][x1_border:x2_border]

    file_name = display_dict["file_name"]
    file_extension = os.path.splitext(file_name)[1].lower()
    file_path = os.path.join(app_state.data_dir, file_name)

    if file_extension == ".cbin":
        save_cbin(file_path, cropped_song_data, display_dict["sampling_rate"])
        if os.path.exists(file_path + ".not.mat"):
            crop_not_mat(os.path.join(app_state.data_dir, file_name + ".not.mat"), display_dict, x_start, x_end)

        if os.path.exists(os.path.join(app_state.data_dir, file_name[:-5] + ".rec")):
            crop_rec_file(os.path.join(app_state.data_dir, file_name[:-5] + ".rec"), display_dict, x_start, x_end, len(cropped_song_data))

    elif file_extension == ".wav":
        wav.write(file_path, display_dict["sampling_rate"], cropped_song_data)
        if os.path.exists(file_path + ".not.mat"):
            crop_not_mat(os.path.join(app_state.data_dir, file_name + ".not.mat"), display_dict, x_start, x_end)
        
        if os.path.exists(os.path.join(app_state.data_dir, file_name[:-4] + ".rec")):
            crop_rec_file(os.path.join(app_state.data_dir, file_name[:-4] + ".rec"), display_dict, x_start, x_end, len(cropped_song_data))

    file_path = get_file_data_by_index(app_state.data_dir, app_state.song_files, app_state.current_file_index, app_state)
    display_dict = get_display_data(file_path, app_state.config)

    app_state.change_file(0)
    plot_data(app_state)


def handle_crop(app_state):
    '''Function to handle the cropping of the displayed data'''
    response = tk.messagebox.askokcancel("Confirm", "Are you sure you want to crop to the displayed area?")
    if response:
        confirm_crop(app_state)
