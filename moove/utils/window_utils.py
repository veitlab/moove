# utils/window_utils.py
import os
import tkinter as tk
from tkinter import ttk


def open_resegment_window(root, app_state, bird_combobox, experiment_combobox, day_combobox):
    """Open the resegmentation window for segmenting the current file using Evfuncs or the Segmentation Network."""
    from moove.utils import start_segment_evfuncs, start_segment_files_thread

    resegment_window = tk.Toplevel(root)
    resegment_window.title("Resegmentation")
    app_state.resegment_window = resegment_window

    resegment_window.geometry("400x450")
    resegment_window.grid_rowconfigure(0, weight=1)
    resegment_window.grid_columnconfigure(0, weight=1)
    resegment_window.grid_columnconfigure(1, weight=1)
    # ensure the window stays in front
    resegment_window.transient(root)

    container_frame = tk.Frame(resegment_window)
    container_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
    container_frame.grid_columnconfigure(0, weight=1)
    container_frame.grid_columnconfigure(1, weight=1)

    # Left frame for Evfuncs
    left_frame = tk.Frame(container_frame)
    left_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
    left_frame.grid_columnconfigure(0, weight=1)
    left_frame.grid_columnconfigure(1, weight=1)

    row = 0
    tk.Label(left_frame, text="Evfuncs", font=("Arial", 16)).grid(row=row, column=0, columnspan=2, pady=10, sticky="ew")
    row += 1

    def update_batch_combobox_resegment_ev():
        app_state.update_batch_select_combobox_resegment_ev(select_path=ev_selection_var.get())

    # Radio buttons for Evfuncs selection
    ev_selection_var = tk.StringVar(value="current_file")
    tk.Radiobutton(left_frame, text="Current File", variable=ev_selection_var, value="current_file").grid(row=row, column=0, columnspan=2, sticky="w")
    row += 1
    tk.Radiobutton(left_frame, text="Current Day", variable=ev_selection_var, value="current_day",
                   command=update_batch_combobox_resegment_ev).grid(row=row, column=0, columnspan=2, sticky="w")
    row += 1
    tk.Radiobutton(left_frame, text="Current Experiment", variable=ev_selection_var, value="current_experiment",
                   command=update_batch_combobox_resegment_ev).grid(row=row, column=0, columnspan=2, sticky="w")
    row += 1
    tk.Radiobutton(left_frame, text="Current Bird", variable=ev_selection_var, value="current_bird",
                   command=update_batch_combobox_resegment_ev).grid(row=row, column=0, columnspan=2, sticky="w")
    row += 1

    # Combobox for batch file selection
    ev_batch_file_var = tk.StringVar()
    resegment_batch_combobox_ev = ttk.Combobox(left_frame,textvariable=ev_batch_file_var)
    resegment_batch_combobox_ev.set("Select Batch File")
    app_state.resegment_window.resegment_batch_combobox_ev = resegment_batch_combobox_ev
    app_state.update_batch_select_combobox_resegment_ev(select_path = ev_selection_var.get())
    resegment_batch_combobox_ev.grid(row=2, column=1, sticky="ew")

    evfuncs_params = [
        ("Threshold:", 'threshold'),
        ("Min Syllable Length:", 'min_syl_dur'),
        ("Min Silent Duration:", 'min_silent_dur'),
        ("Frequency Cutoffs:", 'freq_cutoffs'),
        ("Smoothing Window:", 'smooth_window')
    ]

    for label_text, param_key in evfuncs_params:
        tk.Label(left_frame, text=label_text).grid(row=row, column=0, sticky="w")
        entry = tk.Entry(left_frame, textvariable=app_state.evfuncs_params[param_key])
        entry.grid(row=row, column=1, sticky="ew")
        row += 1

    segment_btn_evfuncs = tk.Button(
        left_frame, text="Segment",
        command=lambda: start_segment_evfuncs(app_state, ev_selection_var.get(), ev_batch_file_var.get(), bird_combobox, experiment_combobox, day_combobox)
    )
    segment_btn_evfuncs.grid(row=row, column=0, columnspan=2, pady=(10, 0), sticky="ew")

    # Right frame for Segmentation Network
    right_frame = tk.Frame(container_frame)
    right_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
    right_frame.grid_columnconfigure(0, weight=1)
    right_frame.grid_columnconfigure(1, weight=1)

    row = 0
    tk.Label(right_frame, text="Segmentation Network", font=("Arial", 16)).grid(row=row, column=0, columnspan=2, pady=10, sticky="ew")
    row += 1

    def update_batch_combobox_resegment():
        app_state.update_batch_select_combobox_resegment(select_path=sm_selection_var.get())

    # Radio buttons for Segmentation Network selection
    sm_selection_var = tk.StringVar(value="current_file")
    tk.Radiobutton(right_frame, text="Current File", variable=sm_selection_var, value="current_file").grid(row=row, column=0, columnspan=2, sticky="w")
    row += 1
    tk.Radiobutton(right_frame, text="Current Day", variable=sm_selection_var, value="current_day",
                   command=update_batch_combobox_resegment).grid(row=row, column=0, columnspan=2, sticky="w")
    row += 1
    tk.Radiobutton(right_frame, text="Current Experiment", variable=sm_selection_var, value="current_experiment",
                   command=update_batch_combobox_resegment).grid(row=row, column=0, columnspan=2, sticky="w")
    row += 1
    tk.Radiobutton(right_frame, text="Current Bird", variable=sm_selection_var, value="current_bird",
                   command=update_batch_combobox_resegment).grid(row=row, column=0, columnspan=2, sticky="w")
    row += 1

    # Combobox for batch file selection
    batch_file_var = tk.StringVar()
    resegment_batch_combobox = ttk.Combobox(right_frame,textvariable=batch_file_var)
    resegment_batch_combobox.set("Select Batch File")
    app_state.resegment_window.resegment_batch_combobox = resegment_batch_combobox
    app_state.update_batch_select_combobox_resegment(select_path = sm_selection_var.get())
    resegment_batch_combobox.grid(row=2, column=1, sticky="ew")

    checkbox_var = tk.BooleanVar()
    tk.Checkbutton(right_frame, text="Overwrite Already Segmented Files", variable=checkbox_var).grid(row=row, column=0, columnspan=2, sticky="w")
    row += 1

    # Model selection combobox
    trained_models_path = os.path.join(app_state.config["global_dir"], "trained_models")
    model_files = [f[:-4] for f in os.listdir(trained_models_path) if f.endswith("_seg_model.pth")]
    app_state.current_segmentation_model = tk.StringVar()
    model_combobox = ttk.Combobox(right_frame, textvariable=app_state.current_segmentation_model)
    model_combobox['values'] = model_files
    model_combobox.set("Select Trained Segmentation Model")
    model_combobox.grid(row=row, column=0, columnspan=2, sticky="ew")
    row += 1

    params = [
        ("Decision Threshold:", 'decision_threshold'),
        ("Onset Window Size:", 'onset_window_size'),
        ("N Onset True:", 'n_onset_true'),
        ("Offset Window Size:", 'offset_window_size'),
        ("N Offset False:", 'n_offset_false'),
        ("Min Syllable Length:", 'min_syllable_length'),
        ("Min Silent Duration:", 'min_silent_duration')
    ]

    for label_text, param_key in params:
        tk.Label(right_frame, text=label_text).grid(row=row, column=0, sticky="w")
        entry = tk.Entry(right_frame, textvariable=app_state.mlseg_params[param_key])
        entry.grid(row=row, column=1, sticky="ew")
        row += 1

        # Additional spacing for readability after certain parameters
        if param_key in ['chunk_size', 'decision_threshold', 'n_offset_false']:
            row += 1

    # Segment button for Segmentation Network
    segment_btn_ml = tk.Button(
        right_frame, text="Segment",
        command=lambda: start_segment_files_thread(app_state, app_state.current_segmentation_model.get(),
                                                    sm_selection_var.get(), checkbox_var.get(), batch_file_var.get(),
                                                      bird_combobox, experiment_combobox, day_combobox)
    )
    segment_btn_ml.grid(row=row, column=0, columnspan=2, pady=(10, 0), sticky="ew")

    # Adjust window size to content
    resegment_window.update_idletasks()
    resegment_window.minsize(resegment_window.winfo_reqwidth(), resegment_window.winfo_reqheight())


def open_relabel_window(root, app_state, bird_combobox, experiment_combobox, day_combobox):
    """Open the relabel window for relabeling the current file using the Classification Network."""
    from moove.utils import start_classify_files_thread

    relabel_window = tk.Toplevel(root)
    relabel_window.title("Relabel")
    app_state.relabel_window = relabel_window
    # ensure the window stays in front
    relabel_window.transient(root)

    relabel_window.geometry("400x280")
    relabel_window.grid_rowconfigure(0, weight=1)
    relabel_window.grid_columnconfigure(0, weight=1)

    # Container Frame for layout
    container_frame = tk.Frame(relabel_window)
    container_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
    container_frame.grid_rowconfigure(0, weight=1)
    container_frame.grid_columnconfigure(0, weight=1)

    # Frame for content
    content_frame = tk.Frame(container_frame)
    content_frame.grid(row=0, column=0, sticky="nsew")
    content_frame.grid_rowconfigure(99, weight=1)
    content_frame.grid_columnconfigure(0, weight=1)
    content_frame.grid_columnconfigure(1, weight=1)

    row = 0
    # Header Label
    tk.Label(content_frame, text="Classification Network", font=("Arial", 16)).grid(row=row, column=0, columnspan=2, pady=10, sticky="nsew")
    row += 1

    def update_batch_combobox_relabel():
        app_state.update_batch_select_combobox_relabel(select_path=selection_var.get())

    # Radio buttons for selection
    selection_var = tk.StringVar(value="current_file")
    tk.Radiobutton(content_frame, text="Current File", variable=selection_var, value="current_file").grid(row=row, column=0, columnspan=2, sticky="w")
    row += 1
    tk.Radiobutton(content_frame, text="Current Day", variable=selection_var, value="current_day",
                   command=update_batch_combobox_relabel).grid(row=row, column=0, columnspan=2, sticky="w")
    row += 1
    tk.Radiobutton(content_frame, text="Current Experiment", variable=selection_var, value="current_experiment",
                   command=update_batch_combobox_relabel).grid(row=row, column=0, columnspan=2, sticky="w")
    row += 1
    tk.Radiobutton(content_frame, text="Current Bird", variable=selection_var, value="current_bird",
                   command=update_batch_combobox_relabel).grid(row=row, column=0, columnspan=2, sticky="w")
    row += 1

    # Combobox for batch file selection
    batch_file_var = tk.StringVar()
    relabel_batch_combobox = ttk.Combobox(content_frame,textvariable=batch_file_var, width=5)
    relabel_batch_combobox.set("Select Batch File")
    app_state.relabel_window.relabel_batch_combobox = relabel_batch_combobox
    app_state.update_batch_select_combobox_relabel(select_path = selection_var.get())
    relabel_batch_combobox.grid(row=2, column=1, sticky="ew")

    checkbox_var = tk.BooleanVar()
    tk.Checkbutton(content_frame, text="Overwrite Already Classified Files", variable=checkbox_var).grid(row=row, column=0, columnspan=2, sticky="w")
    row += 1

    # ComboBox for selecting models
    trained_models_path = os.path.join(app_state.config["global_dir"], "trained_models")
    model_files = [f[:-4] for f in os.listdir(trained_models_path) if f.endswith("_class_model.pth")]
    app_state.current_classification_model = tk.StringVar()
    model_combobox = ttk.Combobox(content_frame, textvariable=app_state.current_classification_model)
    model_combobox['values'] = model_files
    model_combobox.set("Select Trained Classification Model")
    model_combobox.grid(row=row, column=0, columnspan=2, sticky="ew")
    row += 1

    # Relabel Button
    segment_btn_ml = tk.Button(content_frame, text="Relabel", command=lambda: start_classify_files_thread(app_state, app_state.current_classification_model.get(),
                                                                                                           selection_var.get(), checkbox_var.get(), batch_file_var.get(),
                                                                                                             bird_combobox, experiment_combobox, day_combobox))
    segment_btn_ml.grid(row=row, column=0, columnspan=2, sticky="ew")

    # Adjust window size to content
    relabel_window.update_idletasks()
    relabel_window.minsize(relabel_window.winfo_reqwidth(), relabel_window.winfo_reqheight())


def open_training_window(root, app_state, bird_combobox, experiment_combobox, day_combobox):
    """Open the training window for training the Segmentation and Classification Networks."""
    from moove.utils import (
        start_create_segmentation_training_dataset, start_segmentation_training,
        start_create_classification_training_dataset, start_classification_training, 
        find_batch_files
    )

    training_window = tk.Toplevel(root)
    training_window.title("Training")
    training_window.geometry("400x540")
    app_state.training_window = training_window
    # ensure the window stays in front
    training_window.transient(root)

    training_window.grid_rowconfigure(0, weight=1)
    training_window.grid_columnconfigure(0, weight=1)
    training_window.grid_columnconfigure(1, weight=1)

    container_frame = tk.Frame(training_window)
    container_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
    container_frame.grid_rowconfigure(0, weight=1)
    container_frame.grid_columnconfigure(0, weight=1)
    container_frame.grid_columnconfigure(1, weight=1)

    # Left frame for Segmentation Network
    left_frame = tk.Frame(container_frame)
    left_frame.grid(row=0, column=0, sticky="nsew", padx=10)
    left_frame.grid_rowconfigure(99, weight=1)
    left_frame.grid_columnconfigure(0, weight=1)
    left_frame.grid_columnconfigure(1, weight=1)

    row = 0
    tk.Label(left_frame, text="Segmentation Network", font=("Arial", 16)).grid(row=row, column=0, columnspan=2, pady=10, sticky="nsew")
    row += 1

    def update_batch_combobox_segment():
        app_state.update_batch_select_combobox_segment(select_path=selection_var_segmentation.get())

    use_selected_files_var = tk.BooleanVar()
    tk.Checkbutton(left_frame, text="Use segmented files only", variable=use_selected_files_var).grid(row=row, column=0, columnspan=2, sticky="w")
    row += 1

    # Radio buttons for selection
    selection_var_segmentation = tk.StringVar(value="current_day")
    tk.Radiobutton(left_frame, text="Current Day", variable=selection_var_segmentation, value="current_day",
                   command=update_batch_combobox_segment).grid(row=row, column=0, columnspan=2, sticky="w")
    row += 1
    tk.Radiobutton(left_frame, text="Current Experiment", variable=selection_var_segmentation, value="current_experiment",
                   command=update_batch_combobox_segment).grid(row=row, column=0, columnspan=2, sticky="w")
    row += 1
    tk.Radiobutton(left_frame, text="Current Bird", variable=selection_var_segmentation, value="current_bird",
                   command=update_batch_combobox_segment).grid(row=row, column=0, columnspan=2, sticky="w")
    row += 1
    
    # Combobox for batch file selection
    batch_file_var = tk.StringVar()
    training_batch_combobox_segmentation = ttk.Combobox(left_frame,textvariable=batch_file_var)
    training_batch_combobox_segmentation.set("Select Batch File")
    app_state.training_window.training_batch_combobox_segmentation = training_batch_combobox_segmentation
    app_state.update_batch_select_combobox_segment(select_path = selection_var_segmentation.get())
    training_batch_combobox_segmentation.grid(row=2, column=1, sticky="ew")

    tk.Label(left_frame, text="Training Dataset Name: ").grid(row=row, column=0, sticky="w")
    dataset_name_entry = tk.Entry(left_frame)
    dataset_name_entry.grid(row=row, column=1, sticky="ew")
    dataset_name_entry.insert(0, "edit_seg_dataset_name")
    row += 1

    tk.Label(left_frame, text="Chunk Size: ").grid(row=row, column=0, sticky="w")
    chunk_size_entry = tk.Entry(left_frame, textvariable=app_state.train_segmentation_params['chunk_size'])
    chunk_size_entry.grid(row=row, column=1, sticky="ew")
    row += 1

    tk.Label(left_frame, text="Hist Size: ").grid(row=row, column=0, sticky="w")
    hist_size_entry = tk.Entry(left_frame, textvariable=app_state.train_segmentation_params['hist_size'])
    hist_size_entry.grid(row=row, column=1, sticky="ew")
    row += 1

    tk.Checkbutton(left_frame, text="Overlap chunks", variable=app_state.train_segmentation_params['overlap_chunks']).grid(row=row, column=0, sticky="w")
    row += 1

    tk.Button(
        left_frame, text="Create Training Dataset",
        command=lambda: start_create_segmentation_training_dataset(
            app_state, dataset_name_entry.get(), use_selected_files_var.get(), selection_var_segmentation.get(), batch_file_var.get(), bird_combobox, experiment_combobox, day_combobox, root
        )
    ).grid(row=row, column=0, columnspan=2, sticky="ew")
    row += 1

    tk.Label(left_frame, text="").grid(row=row, column=0, columnspan=2)
    row += 1

    training_dataset_var = tk.StringVar()
    training_dataset_combobox = ttk.Combobox(left_frame, textvariable=training_dataset_var)
    training_dataset_combobox.set("Select Training Dataset")
    app_state.training_window.training_dataset_combobox_segmentation = training_dataset_combobox
    app_state.update_segmentation_datasets_combobox()
    training_dataset_combobox.grid(row=row, column=0, columnspan=2, sticky="ew")
    row += 1

    checkbox_frame_segmentation = tk.Frame(left_frame)
    checkbox_frame_segmentation.grid(row=row, column=0, columnspan=2, sticky="w")
    tk.Checkbutton(checkbox_frame_segmentation, text="Downsampling", variable=app_state.train_segmentation_params['downsampling']).grid(row=0, column=0, sticky="w")
    #tk.Checkbutton(checkbox_frame_segmentation, text="QAT", variable=app_state.train_segmentation_params['qat']).grid(row=0, column=2, sticky="w") # optional for further usage
    row += 1

    segmentation_params = [
        ("Epochs:", 'epochs'),
        ("Batch Size:", 'batch_size'),
        ("Learning Rate:", 'learning_rate'),
        ("Early Stopping Patience:", 'early_stopping_patience'),
    ]

    for label_text, param_key in segmentation_params:
        tk.Label(left_frame, text=label_text).grid(row=row, column=0, sticky="w")
        tk.Entry(left_frame, textvariable=app_state.train_segmentation_params[param_key]).grid(row=row, column=1, sticky="ew")
        row += 1

    tk.Button(
        left_frame, text="Start Training",
        command=lambda: start_segmentation_training(root, app_state, training_dataset_var.get())
    ).grid(row=row, column=0, columnspan=2, sticky="ew")

    # Right frame for Classification Network
    right_frame = tk.Frame(container_frame)
    right_frame.grid(row=0, column=1, sticky="nsew", padx=10)
    right_frame.grid_rowconfigure(99, weight=1)
    right_frame.grid_columnconfigure(0, weight=1)
    right_frame.grid_columnconfigure(1, weight=1)

    row = 0
    tk.Label(right_frame, text="Classification Network", font=("Arial", 16)).grid(row=row, column=0, columnspan=2, pady=10, sticky="nsew")
    row += 1

    def update_batch_combobox_class():
        app_state.update_batch_select_combobox_class(select_path=selection_var_classification.get())

    use_selected_files_var_classification = tk.BooleanVar()
    tk.Checkbutton(right_frame, text="Use classified files only", variable=use_selected_files_var_classification).grid(row=row, column=0, columnspan=2, sticky="w")
    row += 1

    # Radio buttons for selection
    selection_var_classification = tk.StringVar(value="current_day")
    tk.Radiobutton(right_frame, text="Current Day", variable=selection_var_classification, value="current_day", 
                   command=update_batch_combobox_class).grid(row=row, column=0, columnspan=2, sticky="w")
    row += 1
    tk.Radiobutton(right_frame, text="Current Experiment", variable=selection_var_classification, value="current_experiment",
                   command=update_batch_combobox_class).grid(row=row, column=0, columnspan=2, sticky="w")
    row += 1
    tk.Radiobutton(right_frame, text="Current Bird", variable=selection_var_classification, value="current_bird",
                   command=update_batch_combobox_class).grid(row=row, column=0, columnspan=2, sticky="w")
    row += 1

    # Combobox for batch file selection
    batch_file_var = tk.StringVar()
    training_batch_combobox_classification = ttk.Combobox(right_frame,textvariable=batch_file_var)
    training_batch_combobox_classification.set("Select Batch File")
    app_state.training_window.training_batch_combobox_classification = training_batch_combobox_classification
    app_state.update_batch_select_combobox_class(select_path = selection_var_classification.get())
    training_batch_combobox_classification.grid(row=2, column=1, sticky="ew")

    tk.Label(right_frame, text="Training Dataset Name:").grid(row=row, column=0, sticky="w")
    dataset_name_entry_classification = tk.Entry(right_frame)
    dataset_name_entry_classification.grid(row=row, column=1, sticky="ew")
    dataset_name_entry_classification.insert(0, "edit_class_dataset_name")
    row += 1

    spec_params = [
        ("N Input Chunks / Size:", 'input_length'),
        ("Nperseg:", 'nperseg'),
        ("Noverlap:", 'noverlap'),
        ("NFFT:", 'nfft'),
        ("Frequency Cutoffs:", 'freq_cutoffs'),
    ]

    for label_text, param_key in spec_params:
        tk.Label(right_frame, text=label_text).grid(row=row, column=0, sticky="w")
        tk.Entry(right_frame, textvariable=app_state.spec_params[param_key]).grid(row=row, column=1, sticky="ew")
        row += 1

    tk.Button(
        right_frame, text="Create Training Dataset",
        command=lambda: start_create_classification_training_dataset(
            app_state, dataset_name_entry_classification.get(), use_selected_files_var_classification.get(), selection_var_classification.get(), batch_file_var.get(), bird_combobox, experiment_combobox, day_combobox, root
        )
    ).grid(row=row, column=0, columnspan=2, sticky="ew")
    row += 1

    tk.Label(right_frame, text="").grid(row=row, column=0, columnspan=2)
    row += 1

    training_dataset_var_classification = tk.StringVar()
    training_dataset_combobox_classification = ttk.Combobox(right_frame, textvariable=training_dataset_var_classification)
    training_dataset_combobox_classification.set("Select Training Dataset")
    app_state.training_window.training_dataset_combobox_classification = training_dataset_combobox_classification
    app_state.update_classification_datasets_combobox()
    training_dataset_combobox_classification.grid(row=row, column=0, columnspan=2, sticky="ew")
    row += 1

    checkbox_frame_classification = tk.Frame(right_frame)
    checkbox_frame_classification.grid(row=row, column=0, columnspan=2, sticky="w")
    tk.Checkbutton(checkbox_frame_classification, text="Downsampling", variable=app_state.train_classification_params['downsampling']).grid(row=0, column=0, sticky="w")
    #tk.Checkbutton(checkbox_frame_classification, text="QAT", variable=app_state.train_classification_params['qat']).grid(row=0, column=2, sticky="w") # make invisible for now
    row += 1

    classification_params = [
        ("Epochs:", 'epochs'),
        ("Batch Size:", 'batch_size'),
        ("Learning Rate:", 'learning_rate'),
        ("Early Stopping Patience:", 'early_stopping_patience'),
    ]

    for label_text, param_key in classification_params:
        tk.Label(right_frame, text=label_text).grid(row=row, column=0, sticky="w")
        tk.Entry(right_frame, textvariable=app_state.train_classification_params[param_key]).grid(row=row, column=1, sticky="ew")
        row += 1

    tk.Button(
        right_frame, text="Start Training",
        command=lambda: start_classification_training(
            root,
            app_state,
            training_dataset_var_classification.get(),
            bird_combobox.get()
        )
    ).grid(row=row, column=0, columnspan=2, sticky="ew")

    # Adjust window size to content
    training_window.update_idletasks()
    training_window.minsize(training_window.winfo_reqwidth(), training_window.winfo_reqheight())


def open_cluster_window(root, app_state, bird_combobox, experiment_combobox, day_combobox):
    """Open the cluster window for obtaining labels via clustering."""
    from moove.utils import start_create_cluster_dataset_thread, start_clustering_thread, start_dash_app_thread, replace_labels_from_df, stop_dash_app_thread, remove_pkl_suffix

    cluster_window = tk.Toplevel(root)
    cluster_window.title("Cluster")

    cluster_window.geometry("400x565")
    cluster_window.grid_rowconfigure(0, weight=1)
    cluster_window.grid_columnconfigure(0, weight=1)
    # ensure the window stays in front
    cluster_window.transient(root)

    app_state.cluster_window = cluster_window

    container_frame = tk.Frame(cluster_window)
    container_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
    container_frame.grid_columnconfigure(0, weight=1)

    row_num = 0
    tk.Label(container_frame, text="Cluster Operations", font=("Arial", 16)).grid(row=row_num, column=0, columnspan=2, pady=10, sticky="ew")
    row_num += 1

    def update_batch_combobox_cluster():
        app_state.update_batch_select_combobox_cluster(select_path=selection_var.get())

    # CheckBox for "Use selected files only"
    use_selected_files_var = tk.BooleanVar()
    tk.Checkbutton(container_frame, text="Use segmented files only", variable=use_selected_files_var).grid(row=row_num, column=0, sticky="w")
    row_num += 1

    # Radio buttons for selection
    selection_var = tk.StringVar(value="current_day")
    tk.Radiobutton(container_frame, text="Current Day", variable=selection_var, value="current_day",
                   command=update_batch_combobox_cluster).grid(row=row_num, column=0, sticky="w")
    row_num += 1
    tk.Radiobutton(container_frame, text="Current Experiment", variable=selection_var, value="current_experiment",
                   command=update_batch_combobox_cluster).grid(row=row_num, column=0, sticky="w")
    row_num += 1
    tk.Radiobutton(container_frame, text="Current Bird", variable=selection_var, value="current_bird",
                   command=update_batch_combobox_cluster).grid(row=row_num, column=0, sticky="w")
    row_num += 1

    # Combobox for batch file selection
    batch_file_var = tk.StringVar()
    cluster_batch_combobox = ttk.Combobox(container_frame,textvariable=batch_file_var)
    cluster_batch_combobox.set("Select Batch File")
    app_state.cluster_window.cluster_batch_combobox = cluster_batch_combobox
    app_state.update_batch_select_combobox_cluster(select_path = selection_var.get())
    cluster_batch_combobox.grid(row=2, column=1, sticky="ew")

    # Entry for cluster dataset name
    tk.Label(container_frame, text="Cluster Dataset Name:").grid(row=row_num, column=0, sticky="w")
    dataset_name_entry = tk.Entry(container_frame)
    dataset_name_entry.grid(row=row_num, column=1, sticky="ew")
    dataset_name_entry.insert(0, "edit_cluster_dataset_name")
    row_num += 1

    spec_params = [
        ("Nperseg:", 'nperseg'),
        ("Noverlap:", 'noverlap'),
        ("NFFT:", 'nfft'),
        ("Frequency Cutoffs:", 'freq_cutoffs'),
    ]

    for label_text, param_key in spec_params:
        tk.Label(container_frame, text=label_text).grid(row=row_num, column=0, sticky="w")
        tk.Entry(container_frame, textvariable=app_state.spec_params[param_key]).grid(row=row_num, column=1, sticky="ew")
        row_num += 1

    # Button to create cluster dataset
    tk.Button(
        container_frame, text="Create Cluster Dataset",
        command=lambda: start_create_cluster_dataset_thread(app_state, dataset_name_entry.get(), use_selected_files_var.get(), selection_var.get(), batch_file_var.get(), bird_combobox, experiment_combobox, day_combobox, root)
    ).grid(row=row_num, column=0, columnspan=2, pady=(10, 0), sticky="ew")
    row_num += 1

    tk.Label(container_frame, text="").grid(row=row_num, column=0, columnspan=2)
    row_num += 1

    # ComboBox for selecting the cluster dataset
    cluster_dataset_var = tk.StringVar()
    cluster_dataset_combobox = ttk.Combobox(container_frame, textvariable=cluster_dataset_var)
    cluster_dataset_combobox.grid(row=row_num, column=0, columnspan=2, sticky="ew")
    cluster_dataset_combobox.set("Select Cluster Dataset")
    app_state.cluster_window.cluster_dataset_combobox = cluster_dataset_combobox
    app_state.update_cluster_datasets_combobox()
    row_num += 1

    umap_k_means_params = [
        ("N_neighbors:", 'n_neighbors'),
        ("Min_dist:", 'min_dist'),
        ("N Syllables:", 'n_clusters'),
    ]

    for label_text, param_key in umap_k_means_params:
        tk.Label(container_frame, text=label_text).grid(row=row_num, column=0, sticky="w")
        tk.Entry(container_frame, textvariable=app_state.umap_k_means_params[param_key]).grid(row=row_num, column=1, sticky="ew")
        row_num += 1

    # Button to start clustering
    tk.Button(
        container_frame, text="Cluster Syllables",
        command=lambda: start_clustering_thread(root, app_state, remove_pkl_suffix(cluster_dataset_var.get()))
    ).grid(row=row_num, column=0, columnspan=2, pady=(10, 0), sticky="ew")
    row_num += 1

    # Button to open Dash GUI
    tk.Button(
        container_frame, text="Open Dash GUI",
        command=lambda: start_dash_app_thread(app_state, remove_pkl_suffix(cluster_dataset_var.get()))
    ).grid(row=row_num, column=0, pady=(10, 0), sticky="ew")

    # Button to close Dash GUI
    tk.Button(
        container_frame, text="Close Dash GUI",
        command=lambda: stop_dash_app_thread(app_state)
    ).grid(row=row_num, column=1, pady=(10, 0), sticky="ew")

    row_num += 1

    # Button to replace labels
    tk.Button(
        container_frame, text="Replace Labels",
        command=lambda: replace_labels_from_df(app_state, remove_pkl_suffix(cluster_dataset_var.get()), root)
    ).grid(row=row_num, column=0, columnspan=2, pady=(10, 0), sticky="ew")

    # Adjust window size to content
    cluster_window.update_idletasks()
    cluster_window.minsize(cluster_window.winfo_reqwidth(), cluster_window.winfo_reqheight())
