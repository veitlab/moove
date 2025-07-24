# utils/training_utils.py
import copy
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from moove.models.CNN import CNN
from moove.models.ConvMLP import ConvMLP
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tkinter import ttk, messagebox
import tkinter.font as tkFont
import tkinter as tk
from torch.utils.data import DataLoader, TensorDataset


def start_segmentation_training(root, app_state, training_dataset_name):
    """Start training of segmentation model using provided dataset and parameters."""

    # checks whether a dataset has been picked
    if training_dataset_name == "Select Training Dataset":
        messagebox.showinfo("Error", "Selected training dataset not valid! Perhaps you forgot to pick a dataset?")
        return
    
    QAT = False

    # Add running label to the GUI
    font_style = tkFont.Font(family="Arial", size=14)
    running_label = tk.Label(app_state.training_window, text="Checking files...", fg="green", font=font_style)
    running_label.grid(row=999, column=0, columnspan=2, pady=(10, 0), sticky=tk.W)
    root.update_idletasks() 

    # Retrieve training parameters from app_state
    downsampling = app_state.train_segmentation_params['downsampling'].get()
    epochs = int(app_state.train_segmentation_params['epochs'].get())
    batch_size = int(app_state.train_segmentation_params['batch_size'].get())
    learning_rate = float(app_state.train_segmentation_params['learning_rate'].get())
    early_stopping_patience = int(
        app_state.train_segmentation_params['early_stopping_patience'].get()
    )

    dataset_path = os.path.join(app_state.config['global_dir'], 'training_data', f'{training_dataset_name}')

    # Load data and metadata with pickle
    with open(dataset_path, 'rb') as f:
        data_dict = pickle.load(f)

    # Extract features and metadata
    features = np.array(data_dict['features'])
    metadata = data_dict['metadata']
    num_segs = data_dict['syllables']

    # Free memory by deleting data_dict
    del data_dict

    if features.ndim == 2:
        file_indices = np.unique(features[:, 0])  
    else:
        running_label.destroy()
        messagebox.showinfo(
            "Error",
            "Given dataset is empty!"
        )
        return
    
    # check number of segments in the file(s)
    if num_segs <= 7:
        running_label.destroy()
        messagebox.showinfo("Error", f"Not enough segments given (n = {num_segs}), need at least 7 to train a network!\n You might want to adjust the threshold.")
        return

    if len(file_indices) >= 7:
        # Split data based on files
        def filter_data_by_files(data, file_set):
            """Filter data to include only specified files."""
            return data[np.isin(data[:, 0], file_set)]

        train_files, temp_files = train_test_split(file_indices, test_size=0.3, random_state=42)
        val_files, test_files = train_test_split(temp_files, test_size=0.5, random_state=42)
        messagebox.showinfo(
            "Info",
            "Training of segmentation model started. This may take a while, please wait!"
        )
        running_label = tk.Label(app_state.training_window, text="Training in Progress...", fg="green", font=font_style)
        running_label.grid(row=999, column=0, columnspan=2, pady=(10, 0), sticky=tk.W)
        root.update_idletasks() 

        train_data = filter_data_by_files(features, train_files)
        val_data = filter_data_by_files(features, val_files)
        test_data = filter_data_by_files(features, test_files)

    else:
        # Split data without considering files 
        def ask_user_for_small_dataset(parent):
            result = {}

            def on_continue():
                result['answer'] = 'continue'
                win.destroy()

            def on_cancel():
                result['answer'] = 'cancel'
                win.destroy()

            win = tk.Toplevel(parent)
            win.title("Small Dataset detected")
            win.grab_set()
            win.protocol("WM_DELETE_WINDOW", on_cancel)
            win.focus_force()
            
            msg = tk.Label(       
                win,
                text=(
                f"Number of files given is very small! (n = {len(file_indices)})\n"
                "Are you training on one file containing multiple bouts?\n"
                "If not, please add more song files for training.\n"
                "Using data from multiple files is recommended.\n\n"
                "Do you want to continue?"
            ),
            justify="center",
            wraplength=380,
            padx=28, pady=18)
            msg.pack(fill="both", expand=True, padx=12, pady=(16,8))

            button_frame = tk.Frame(win)
            button_frame.pack(pady=(0,18))

            btn1 = tk.Button(button_frame, text="Continue with few files", command=on_continue, width=22)
            btn1.grid(row=0, column=0, padx=10)
            btn2 = tk.Button(button_frame, text="Cancel", command=on_cancel, width=16)
            btn2.grid(row=0, column=1, padx=10)


            win.update_idletasks()
            width = win.winfo_width()
            height = win.winfo_height()
            x = parent.winfo_x() + (parent.winfo_width() // 2) - (width // 2)
            y = parent.winfo_y() + (parent.winfo_height() // 2) - (height // 2)
            win.geometry(f"+{x}+{y}")
            win.minsize(420, 180)

            win.wait_window()
            return result.get('answer', None)
        
        answer = ask_user_for_small_dataset(root) 

        if answer == 'cancel':
            running_label.destroy()
            return
        elif answer == 'continue':
            pass

        train_data, temp_data = train_test_split(features, test_size=0.3, random_state=42)
        val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)
        messagebox.showinfo(
            "Info",
            "Training of segmentation model started. This may take a while, please wait!"
        )
        running_label = tk.Label(app_state.training_window, text="Training in Progress...", fg="green", font=font_style)
        running_label.grid(row=999, column=0, columnspan=2, pady=(10, 0), sticky=tk.W)
        root.update_idletasks() 

    # Remove first column (file index) from the data
    train_data = train_data[:, 1:]
    val_data = val_data[:, 1:]
    test_data = test_data[:, 1:]

    # Extract features and labels
    X_train = train_data[:, :-1].astype('float32')
    y_train = train_data[:, -1].astype('float32')
    X_val = val_data[:, :-1].astype('float32')
    y_val = val_data[:, -1].astype('float32')
    X_test = test_data[:, :-1].astype('float32')
    y_test = test_data[:, -1].astype('float32')

    def downsample_data(data, labels):
        """Downsample data to balance class distribution by undersampling."""
        unique_labels, counts = np.unique(labels, return_counts=True)
        min_count = np.min(counts)

        downsampled_data = []
        downsampled_labels = []

        for label in unique_labels:
            label_indices = np.where(labels == label)[0]
            sampled_indices = np.random.choice(
                label_indices, size=min_count, replace=False
            )
            downsampled_data.append(data[sampled_indices])
            downsampled_labels.append(labels[sampled_indices])

        downsampled_data = np.vstack(downsampled_data)
        downsampled_labels = np.hstack(downsampled_labels)

        return downsampled_data, downsampled_labels

    if downsampling:
        X_train, y_train = downsample_data(X_train, y_train)
        X_val, y_val = downsample_data(X_val, y_val)
        X_test, y_test = downsample_data(X_test, y_test)

    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train)
    y_train_tensor = torch.tensor(y_train).unsqueeze(1)
    X_val_tensor = torch.tensor(X_val)
    y_val_tensor = torch.tensor(y_val).unsqueeze(1)
    X_test_tensor = torch.tensor(X_test)
    y_test_tensor = torch.tensor(y_test).unsqueeze(1)

    # Global normalization: calculate mean and std over all training data
    mean = X_train_tensor.mean()
    std = X_train_tensor.std()

    # Store normalization values in metadata
    metadata['mean'] = mean.item()
    metadata['std'] = std.item()

    # Apply normalization to training, validation, and test data
    X_train_tensor = (X_train_tensor - mean) / std
    X_val_tensor = (X_val_tensor - mean) / std
    X_test_tensor = (X_test_tensor - mean) / std

    # Replace NaN values (if any) with zero
    X_train_tensor[torch.isnan(X_train_tensor)] = 0
    X_val_tensor[torch.isnan(X_val_tensor)] = 0
    X_test_tensor[torch.isnan(X_test_tensor)] = 0

    # Create DataLoaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, drop_last=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, drop_last=True
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConvMLP(input_size=X_train.shape[1]).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Prepare model for Quantization Aware Training if enabled
    if QAT:
        supported_backends = torch.backends.quantized.supported_engines
        available_backends = [backend for backend in supported_backends if backend != 'none']

        if not available_backends:
            raise RuntimeError("No valid quantization backend is available!")

        selected_backend = available_backends[0]
        torch.backends.quantized.engine = selected_backend
        app_state.logger.debug(f"Quantization backend set to: {selected_backend}")

        model.qconfig = torch.quantization.get_default_qat_qconfig(selected_backend)
        model = torch.quantization.prepare_qat(model, inplace=True)
        app_state.logger.debug("Quantization Aware Training activated.")

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    best_model = None

    def calculate_accuracy(outputs, labels):
        """Calculate accuracy for binary classification."""
        probs = torch.sigmoid(outputs)
        preds = probs > 0.5
        accuracy = (preds == labels).float().mean()
        return accuracy

    prefix = f"{training_dataset_name.split('.')[0]}"

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_accuracy = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            acc = calculate_accuracy(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_accuracy += acc.item()

        train_loss /= len(train_loader)
        train_accuracy /= len(train_loader)
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0.0
        val_accuracy = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                acc = calculate_accuracy(outputs, labels)
                val_loss += loss.item()
                val_accuracy += acc.item()

        val_loss /= len(val_loader)
        val_accuracy /= len(val_loader)
        val_losses.append(val_loss)

        # Early stopping based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = copy.deepcopy(model)
            patience_counter = 0
            app_state.logger.debug(f"Best model updated with val loss: {best_val_loss}")
        else:
            patience_counter += 1

        if patience_counter >= early_stopping_patience:
            app_state.logger.info(f'Early stopping at epoch {epoch+1}')
            break

        app_state.logger.info(
            f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, "
            f"Val Loss: {val_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, "
            f"Val Accuracy: {val_accuracy:.4f}"
        )

    # After training: quantize and save the model
    if best_model is not None:
        best_model.eval()
        if QAT:
            best_model = torch.quantization.convert(best_model, inplace=True)
            app_state.logger.debug("Model quantized after QAT.")

        save_path = os.path.join(app_state.config['global_dir'], 'trained_models', f'{prefix}_model.pth')
        # Save the best model
        torch.save(
            {
                'model': best_model,
                'metadata': metadata,
            },
            save_path
        )
        app_state.logger.debug(f"Best model saved with val loss: {best_val_loss:.4f}")

    # Evaluate the best model on the test data
    checkpoint = torch.load(os.path.join(app_state.config['global_dir'], 'trained_models', f'{prefix}_model.pth'))
    model = checkpoint['model']
    metadata = checkpoint['metadata']

    model.to(device)
    test_loss = 0.0
    test_accuracy = 0.0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            acc = calculate_accuracy(outputs, labels)
            test_loss += loss.item()
            test_accuracy += acc.item()

    test_loss /= len(test_loader)
    test_accuracy /= len(test_loader)
    app_state.logger.info(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')

    # remove the label after the operation completes
    running_label.destroy()
    # close the segmentation window
    app_state.training_window.destroy()

    # Include test performance in the final messagebox
    messagebox.showinfo(
        "Info",
        f"Model \"{training_dataset_name}\" trained successfully!\nTest Accuracy: {test_accuracy:.4f}"
    )


def start_classification_training(root, app_state, dataset_name, bird):
    """Start training of classification model using provided dataset and parameters."""

    # check whether a dataset has been selected
    if dataset_name == "Select Training Dataset":
        messagebox.showinfo("Error", "Selected training dataset not valid! Perhaps you forgot to pick a dataset?")
        return
    
    # add label to the GUI
    font_style = tkFont.Font(family="Arial", size=14)
    running_label = tk.Label(app_state.training_window, text="Checking files...", fg="green", font=font_style)
    running_label.grid(row=999, column=0, columnspan=2, pady=(10, 0), sticky=tk.W)
    root.update_idletasks()  # Force GUI to show the label immediately

    downsampling = app_state.train_classification_params['downsampling'].get()
    epochs = int(app_state.train_classification_params['epochs'].get())
    batch_size = int(app_state.train_classification_params['batch_size'].get())
    learning_rate = float(app_state.train_classification_params['learning_rate'].get())
    early_stopping_patience = int(
        app_state.train_classification_params['early_stopping_patience'].get()
    )

    # Define device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    app_state.logger.debug(f"Using device: {device}")

    # Load data
    with open(os.path.join(app_state.config['global_dir'], 'training_data', dataset_name), 'rb') as f:
        data = pickle.load(f)

    df = data['dataframe']
    metadata = data['metadata']

    # Convert 'taf_unflattend_spectrogram' to tensors
    df['taf_unflattend_spectrogram'] = df['taf_unflattend_spectrogram'].apply(np.array).apply(torch.tensor)

    inputs = df['taf_unflattend_spectrogram'].tolist()
    labels = df['label'].tolist()

    # get the smallest number of labels available for one syllable
    all_labels, counts = np.unique(labels, return_counts=True)
    labels_below_threshold = all_labels[counts < 6]
    counts_below_threshold = counts[counts < 6]

    # Map labels from string to integers
    unique_labels = sorted(set(labels))  # Ensure consistent ordering
    label_to_int = {label: i for i, label in enumerate(unique_labels)}
    int_to_label = {i: label for label, i in label_to_int.items()}

    # Convert labels to integers
    labels = [label_to_int[label] for label in labels]
    labels = torch.tensor(labels).long()

    # Save the mapping in metadata
    metadata.update({
        "label_to_int": label_to_int,
        "int_to_label": int_to_label,
    })

    # Get the number of classes automatically
    num_classes = len(unique_labels)

    app_state.logger.debug(f"Available classes: {unique_labels}")
    app_state.logger.debug(f"Number of classes: {num_classes}")

    def preprocess_data(data_list):
        """Convert a list of arrays to tensors with padding and channel dimension."""
        return [F.pad(torch.tensor(array).float().unsqueeze(0), (0, 1, 0, 1)) for array in data_list]

    # Group data by filename
    file_groups = df.groupby('file')

    # Get unique filenames
    filenames = list(file_groups.groups.keys())

    if not filenames:
        running_label.destroy()
        messagebox.showinfo(
            "Error",
            "Given dataset is empty!"
        )
        return

    if any(counts_below_threshold):
        running_label.destroy()
        messagebox.showinfo("Error", f"Number of labels for syllable {labels_below_threshold} is too small (n = {counts_below_threshold})!\nYou need at least 6 labeled syllables per syllable type.")
        return
    
    # Split filenames into train, val, and test sets
    if len(filenames) >= 7:
        train_files, temp_files = train_test_split(filenames, test_size=0.3, random_state=42)
        val_files, test_files = train_test_split(temp_files, test_size=0.5, random_state=42)
        messagebox.showinfo(
            "Info",
            "Training of classification model started. This may take a while, please wait!"
        )
        running_label = tk.Label(app_state.training_window, text="Training in Progress...", fg="green", font=font_style)
        running_label.grid(row=999, column=0, columnspan=2, pady=(10, 0), sticky=tk.W)
        root.update_idletasks()

        # Create new dataframes for each set based on the filenames
        df_train = df[df['file'].isin(train_files)]
        df_val = df[df['file'].isin(val_files)]
        df_test = df[df['file'].isin(test_files)]

        # Extract data and labels from the split dataframes
        train_data = df_train['taf_unflattend_spectrogram'].tolist()
        train_labels = [label_to_int[label] for label in df_train['label'].tolist()]

        val_data = df_val['taf_unflattend_spectrogram'].tolist()
        val_labels = [label_to_int[label] for label in df_val['label'].tolist()]

        test_data = df_test['taf_unflattend_spectrogram'].tolist()
        test_labels = [label_to_int[label] for label in df_test['label'].tolist()]

        # Preprocess the data (add a channel dimension and pad)
        train_data = preprocess_data(train_data)
        val_data = preprocess_data(val_data)
        test_data = preprocess_data(test_data)
        input_shape = train_data[0].shape

    else:
        # Preprocess inputs
        input_data = preprocess_data(inputs)
        # Split data without considering files 
        def ask_user_for_small_dataset(parent):
            result = {}

            def on_continue():
                result['answer'] = 'continue'
                win.destroy()

            def on_cancel():
                result['answer'] = 'cancel'
                win.destroy()

            win = tk.Toplevel(parent)
            win.title("Small Dataset detected")
            win.grab_set()
            win.protocol("WM_DELETE_WINDOW", on_cancel)
            win.focus_force()
            
            msg = tk.Label(       
                win,
                text=(
                f"Number of files given is very small! (n = {len(filenames)})\n"
                "Are you training on one file containing multiple bouts?\n"
                "If not, please add more song files for training.\n"
                "Using data from multiple files is recommended.\n\n"
                "Do you want to continue?"
            ),
            justify="center",
            wraplength=380,
            padx=28, pady=18)
            msg.pack(fill="both", expand=True, padx=12, pady=(16,8))

            button_frame = tk.Frame(win)
            button_frame.pack(pady=(0,18))

            btn1 = tk.Button(button_frame, text="Continue with few files", command=on_continue, width=22)
            btn1.grid(row=0, column=0, padx=10)
            btn2 = tk.Button(button_frame, text="Cancel", command=on_cancel, width=16)
            btn2.grid(row=0, column=1, padx=10)


            win.update_idletasks()
            width = win.winfo_width()
            height = win.winfo_height()
            x = parent.winfo_x() + (parent.winfo_width() // 2) - (width // 2)
            y = parent.winfo_y() + (parent.winfo_height() // 2) - (height // 2)
            win.geometry(f"+{x}+{y}")
            win.minsize(420, 180)

            win.wait_window()
            return result.get('answer', None)
        
        answer = ask_user_for_small_dataset(root) 

        if answer == 'cancel':
            running_label.destroy()
            return
        elif answer == 'continue':
            pass

        # Safe to split
        train_data, temp_data, train_labels, temp_labels = train_test_split(input_data, labels, test_size=0.3,
                                                                            stratify=labels, random_state=42)
        val_data, test_data, val_labels, test_labels = train_test_split(temp_data, temp_labels, test_size=0.5,
                                                                        stratify=temp_labels, random_state=42)
        input_shape = train_data[0].shape
        messagebox.showinfo(
            "Info",
            "Training of classification model started. This may take a while, please wait!"
        )
        running_label = tk.Label(app_state.training_window, text="Training in Progress...", fg="green", font=font_style)
        running_label.grid(row=999, column=0, columnspan=2, pady=(10, 0), sticky=tk.W)
        root.update_idletasks()


    # Shuffle the data
    train_data, train_labels = shuffle(train_data, train_labels, random_state=42)
    val_data, val_labels = shuffle(val_data, val_labels, random_state=42)
    test_data, test_labels = shuffle(test_data, test_labels, random_state=42)

    # Apply Z-Score normalization per sample
    train_data = [
        (array - array.mean()) / array.std() if array.std() != 0 else array
        for array in train_data
    ]
    val_data = [
        (array - array.mean()) / array.std() if array.std() != 0 else array
        for array in val_data
    ]
    test_data = [
        (array - array.mean()) / array.std() if array.std() != 0 else array
        for array in test_data
    ]

    def downsample_data(data, labels):
        """Downsample data to balance class distribution by undersampling."""
        data_df = pd.DataFrame({'data': data, 'labels': labels})
        min_size = data_df['labels'].value_counts().min()
        downsampled_data = pd.DataFrame(columns=data_df.columns)
        for label, group in data_df.groupby('labels'):
            downsampled_data = pd.concat(
                [downsampled_data, group.sample(min_size, random_state=42)]
            )
        return downsampled_data['data'].tolist(), downsampled_data['labels'].tolist()

    if downsampling:
        # Downsample training, validation, and test datasets
        train_data, train_labels = downsample_data(train_data, train_labels)
        val_data, val_labels = downsample_data(val_data, val_labels)
        test_data, test_labels = downsample_data(test_data, test_labels)

    # Convert labels to tensors
    train_labels = torch.tensor(train_labels).long()
    val_labels = torch.tensor(val_labels).long()
    test_labels = torch.tensor(test_labels).long()

    # Create DataLoaders
    train_dataset = TensorDataset(torch.stack(train_data), train_labels)
    val_dataset = TensorDataset(torch.stack(val_data), val_labels)
    test_dataset = TensorDataset(torch.stack(test_data), test_labels)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model, loss function, and optimizer
    model = CNN(input_shape=input_shape, num_classes=num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Early stopping variables
    best_val_loss = float('inf')
    patience_counter = 0
    best_model = None

    # Prefix for saving the model
    prefix = f"{dataset_name.replace('.pkl', '')}"

    def calculate_accuracy(loader, model):
        """Calculate accuracy for classification."""
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        model.train()
        return correct / total

    # Training loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for inputs, labels in train_loader:
            # Data augmentation
            augmented_inputs = []
            for input_tensor in inputs:
                augmented_spec = augment_spectrogram(input_tensor.cpu().numpy())
                augmented_tensor = torch.from_numpy(augmented_spec).float()
                augmented_inputs.append(augmented_tensor)

            augmented_inputs = torch.stack(augmented_inputs).to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(augmented_inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)
        val_loss = 0.0

        # Evaluate on validation set
        model.eval()
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        train_accuracy = calculate_accuracy(train_loader, model)
        val_accuracy = calculate_accuracy(val_loader, model)

        app_state.logger.info(
            f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, "
            f"Val Loss: {val_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, "
            f"Val Accuracy: {val_accuracy:.4f}"
        )

        # Check for early stopping and save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = copy.deepcopy(model)
            patience_counter = 0
            save_path = os.path.join(app_state.config['global_dir'], 'trained_models', f'{prefix}_model.pth')
            torch.save(
                {
                    'model': model,
                    'metadata': metadata,
                },
                save_path
            )
            app_state.logger.debug(f"Best model saved with val loss: {best_val_loss:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                app_state.logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                break

    # Load the best model after training
    checkpoint = torch.load(os.path.join(app_state.config['global_dir'], 'trained_models', f'{prefix}_model.pth'))
    model = checkpoint['model']
    metadata = checkpoint['metadata']
    app_state.logger.debug("Model metadata:")
    app_state.logger.debug(metadata)

    model.to(device)

    # Evaluate on the test set
    test_accuracy = calculate_accuracy(test_loader, model)
    test_loss = 0.0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
    test_loss /= len(test_loader)
    app_state.logger.info(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

    # Compute confusion matrix
    predictions, targets = get_predictions_and_targets(model, test_loader, device)
    cm = confusion_matrix(targets, predictions)
    labels_range = [int_to_label[i] for i in range(num_classes)]

    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(10, 7))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt='.2%',
        cmap='Blues',
        xticklabels=labels_range,
        yticklabels=labels_range
    )
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Normalized Confusion Matrix')
    plt.savefig(os.path.join(app_state.config['global_dir'], 'trained_models', f'{prefix}_confusion_matrix.svg'))
    plt.close()

    running_label.destroy()
    app_state.training_window.destroy()

    # Include test accuracy in the final messagebox
    messagebox.showinfo(
        "Info",
        f"Model \"{dataset_name}\" trained successfully!\nTest Accuracy: {test_accuracy:.4f}"
    )


def calculate_accuracy_and_percent_probabilities(loader, model, device):
    """Calculate accuracy and return class probabilities in percent."""
    model.eval()
    correct = 0
    total = 0
    all_probabilities = []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            probabilities = torch.softmax(outputs, dim=1) * 100  # Convert to percent
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_probabilities.append(probabilities.cpu())

    model.train()
    accuracy = correct / total
    return accuracy, torch.cat(all_probabilities)


def get_predictions_and_targets(model, data_loader, device):
    """Get predictions and true targets from the data loader."""
    model.eval()
    all_predictions = []
    all_targets = []
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    return all_predictions, all_targets


def augment_spectrogram(spec):
    """Apply random augmentations to a spectrogram."""
    import random
    # Decide whether to augment the spectrogram
    if np.random.rand() < 0.2:  # Apply augmentation in 20% of cases
        # List of possible augmentation functions
        augmentations = [
            add_noise_to_spectrogram,
            dynamic_range_compression,
            frequency_mask,
            time_mask
        ]
        # Randomly select one augmentation
        chosen_augmentation = random.choice(augmentations)
        spec = chosen_augmentation(spec)
    return spec


def add_noise_to_spectrogram(spec, noise_level=0.0001):
    """Add Gaussian noise to the spectrogram."""
    noise = noise_level * np.random.randn(*spec.shape)
    return spec + noise


def frequency_mask(spec, F=10, num_masks=1, replace_with_zero=False):
    """Apply frequency masking to the spectrogram."""
    cloned_spec = spec.copy()
    num_freq_channels = spec.shape[0]

    for _ in range(num_masks):
        f = np.random.uniform(low=1, high=min(F, num_freq_channels))
        f = int(f)

        # Ensure f is at least 1 and less than num_freq_channels
        f = max(1, min(f, num_freq_channels - 1))

        max_start = num_freq_channels - f
        if max_start <= 0:
            # Cannot apply mask because it would exceed the spectrogram dimensions
            continue

        f0 = np.random.randint(0, max_start)
        cloned_spec[f0:f0 + f, :] = 0 if replace_with_zero else cloned_spec.mean()

    return cloned_spec


def time_mask(spec, T=10, num_masks=1, replace_with_zero=False):
    """Apply time masking to the spectrogram."""
    cloned_spec = spec.copy()
    num_time_channels = spec.shape[1]

    for _ in range(num_masks):
        t = np.random.uniform(low=1, high=min(T, num_time_channels))
        t = int(t)

        # Ensure t is at least 1 and less than num_time_channels
        t = max(1, min(t, num_time_channels - 1))

        max_start = num_time_channels - t
        if max_start <= 0:
            # Cannot apply mask because it would exceed the spectrogram dimensions
            continue

        t0 = np.random.randint(0, max_start)
        cloned_spec[:, t0:t0 + t] = 0 if replace_with_zero else cloned_spec.mean()

    return cloned_spec


def dynamic_range_compression(spec, compression_factor=0.5):
    """Apply dynamic range compression to the spectrogram."""
    return np.log1p(compression_factor * np.expm1(spec))
