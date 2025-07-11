
<div style="background-color: white; padding: 10px; display: inline-block;">
    <img src="assets/logo_white_bg.png" alt="Möve Logo" width="250">
</div>

# Möve

Möve (Marking Online using Only the Onsets of Vocal Elements) is a novel tool for real-time syllable segmentation and classification of birdsong, designed to enable closed-loop experiments in vocal learning research. Designed to study the learned vocalisations of Bengalese finches, Möve identifies target syllables in a bird's song and provides feedback in real time. Möve provides an out-of-the-box, neural network-based approach to reliably target vocal syllables before their end, enabling a reinforcement protocol where a specific syllable can be targeted with aversive white noise or an alternative feedback stimulus if adjusted.

Möve uses a two-stage architecture: a convolutional-based encoder that segments syllables in the audio signal and a CNN classifier that assigns each detected syllable segment a label, identifying its type based on the initial part of its structure. This design allows Möve to operate at a lower audio chunk duration than other tools, enabling faster and more accurate syllable recognition with minimal latency. Möve includes a GUI for creating training datasets using unsupervised methods and training the networks, as well as a recording script for real-time syllable targeting.

## Installation

### With pip (recommended)

To install Möve, use pip (Python 3.9 or higher required):

```bash
pip install moove
# pre release:
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple moove==0.1.23
```

After installing, a default configuration file (moove_config.ini) will be available at ~/.moove/. This configuration file should be adjusted to fit your experiment setup.

### With poetry

Alternatively, poetry can be used:
```bash
poetry install
```

### PortAudio Installation

Möve uses the `sounddevice` library, which depends on PortAudio. On most systems, PortAudio is already available or bundled. If needed, follow the steps below to install it:

#### Windows
PortAudio is typically preinstalled on Windows, so no additional installation is required. If you encounter issues, install the necessary audio drivers or check the PortAudio website: [www.portaudio.com](http://www.portaudio.com).

#### Linux
Install the PortAudio development library:
```bash
# Debian/Ubuntu
sudo apt update && sudo apt install portaudio19-dev
```

#### macOS
Use Homebrew to install PortAudio:
```bash
brew install portaudio
```

### Enabling ASIO Support for Windows

ASIO provides the lowest latency, which is critical for Möve's real-time targeting capabilities. To enable ASIO support in `sounddevice`, replace the default PortAudio DLL with an ASIO-enabled version.

- Find an ASIO-enabled PortAudio DLL from a trusted source, such as [this one](https://github.com/spatialaudio/portaudio-binaries).

- Locate the PortAudio DLL in your `sounddevice` installation. A common path is:
```
C:\Users\<YourUsername>\AppData\Roaming\Python\<YourPythonVersion>\site-packages\sounddevice\portaudio.dll
```

- Backup the existing `portaudio.dll` and replace it with the downloaded ASIO-enabled DLL, renaming it to `portaudio.dll`.

## Usage

Once installed, Möve offers two main entry points for operation:

- `moovegui`: Opens the GUI for creating labeled datasets and training the segmentation and classification networks.

- `moovetaf`: Starts the recording and targeting application, enabling real-time targeting of specific syllables.

To start, simply type `moovegui` or `moovetaf` in the terminal.

### Requirements

- Python Version: Python 3.9 or higher
- Audio Hardware: A microphone and speaker setup is required for online targeting experiments.

### Workflow Overview

This section outlines the typical workflow for setting up Möve and conducting experiments:

1.  Baseline Recordings
Begin with baseline recordings using MöveTaf to capture the bird’s songs without targeting. In the configuration file (moove_config.ini), set realtime_classification to False for these recordings. Additionally, set db_threshold for bout detection to define when a sequence starts and ends.

2.  Manual Segmentation
In the MöveGUI, use the "ResegmentationWindow" to manually segment recorded songs and adjust the segmentation points as needed.

3.  Train the Segmentation Network
With the segmentation data, open the "TrainingWindow" to train the segmentation network. Once the network is trained, return to the "ResegmentationWindow" to perform an automated re-segmentation on all data, ensuring consistency with segmentation results that would occur in real time.

4.  Label Creation
Use the "ClusterWindow" in the GUI to label syllable segments. The clusters can be automatically labeled and then manually adjusted as needed.

5.  Train the Classification Network
With the labeled syllables, use the "TrainingWindow" again to train the classification network based on the assigned labels.

6.  Real-Time Targeting Setup
Finally, update the configuration file with the names of the trained segmentation and classification networks, set realtime_classification to True, and specify the target syllable for feedback. MöveTaf can now be used for conducting real-time targeting experiments, enabling precise, low-latency feedback during song production.
