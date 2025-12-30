
<div style="background-color: white; padding: 10px; display: inline-block;">
    <img src="assets/logo_white_bg.png" alt="Moove Logo" width="250">
</div>

# Moove

Moove (Marking Online using Only the Onsets of Vocal Elements) is a novel tool for real-time syllable segmentation and classification of birdsong, designed to enable closed-loop experiments in vocal learning research. Designed to study the learned vocalisations of Bengalese finches, Moove identifies target syllables in a bird's song and provides feedback in real time. Moove provides an out-of-the-box, neural network-based approach to reliably target vocal syllables before their end, enabling a reinforcement protocol where a specific syllable can be targeted with aversive white noise or an alternative feedback stimulus if adjusted.

Moove uses a two-stage architecture: a convolutional-based encoder that segments syllables in the audio signal and a CNN classifier that assigns each detected syllable segment a label, identifying its type based on the initial part of its structure. This design allows Moove to operate at a lower audio chunk duration than other tools, enabling faster and more accurate syllable recognition with minimal latency. Moove includes a GUI for creating training datasets using unsupervised methods and training the networks, as well as a recording script for real-time syllable targeting.

## Installation

### With pip (recommended)

To install Moove, use pip (Python 3.9 to including Python 3.12):

```bash
pip install moove
```

For a specific version:
```bash
pip install moove==1.0.0
```

After installing, a default configuration file (moove_config.ini) will be available at ~/.moove/. This configuration file should be adjusted to fit your experiment setup.

### With poetry

Alternatively, poetry can be used:
```bash
poetry install
```

### PortAudio Installation

Moove uses the `sounddevice` library, which depends on PortAudio. On most systems, PortAudio is already available or bundled. If needed, follow the steps below to install it:

#### Windows
PortAudio is typically preinstalled on Windows, so no additional installation is required. If you encounter issues, install the necessary audio drivers or check the PortAudio website: [www.portaudio.com](http://www.portaudio.com).

#### Linux
Install the PortAudio development library:
```bash
# Debian/Ubuntu
python -m venv venv
source venv/bin/activate
sudo apt install python-dev-is-python3 gcc
sudo apt update && sudo apt install portaudio19-dev
sudo apt-get install python3-tk
```
then install the moove package

#### macOS
Use Homebrew to install PortAudio:
```bash
brew install portaudio
```

### Enabling ASIO Support for Windows

ASIO provides the lowest latency, which is critical for Moove's real-time targeting capabilities. To enable ASIO support in `sounddevice`, replace the default PortAudio DLL with an ASIO-enabled version.

- Find an ASIO-enabled PortAudio DLL from a trusted source, such as [this one](https://github.com/spatialaudio/portaudio-binaries).

- Locate the PortAudio DLL in your `sounddevice` installation. A common path is:
```
C:\Users\<YourUsername>\AppData\Roaming\Python\<YourPythonVersion>\site-packages\_sounddevice_data\portaudio_binaries\libportaudio64bit.dll
# oder
C:\Users\<YourUsername>\AppData\Local\Programs\Python\<YourPythonVersion>\Lib\site-packages\_sounddevice_data\portaudio_binaries\libportaudio64bit.dll
```

- Backup the existing `libportaudio64bit.dll` and replace it with the downloaded ASIO-enabled DLL, renaming it to `libportaudio64bit.dll`.

## Configuration

### Default Configuration Location

By default, Moove stores its configuration file (`moove_config.ini`) in `~/.moove/`. This configuration file should be adjusted to fit your experiment setup.

### Custom Configuration Directory

To store the configuration in a different location, use the `MOOVE_CONFIG_DIR` environment variable:

**Windows:**
```cmd
set MOOVE_CONFIG_DIR=D:\moove_config
moovegui
```

**Linux/macOS:**
```bash
export MOOVE_CONFIG_DIR="/path/to/config"
moovegui
```

**Benefits:** Enables different drives, network storage, project isolation, and multi-user setups.

## Usage

Once installed, Moove offers two main entry points for operation:

- `moovegui`: Opens the GUI for creating labeled datasets and training the segmentation and classification networks.

- `moovetaf`: Starts the recording and targeting application, enabling real-time targeting of specific syllables.

To start, simply type `moovegui` or `moovetaf` in the terminal.

### Requirements

- Python Version: Python 3.9 to including Python 3.12
- Audio Hardware: A microphone and speaker setup is required for online targeting experiments.

### Workflow Overview

This section outlines the typical workflow for setting up Moove and conducting experiments:

1.  Baseline Recordings
Begin with baseline recordings using MooveTaf to capture the bird's songs without targeting. In the configuration file (moove_config.ini), set `realtime_classification` to False for these recordings. Additionally, set `dB_threshold` for bout detection to define when a sequence starts and ends.

2.  Manual Segmentation
In the MooveGUI, use the "ResegmentationWindow" to manually segment recorded songs and adjust the segmentation points as needed.

3.  Train the Segmentation Network
With the segmentation data, open the "TrainingWindow" to train the segmentation network. Once the network is trained, return to the "ResegmentationWindow" to perform an automated re-segmentation on all data, ensuring consistency with segmentation results that would occur in real time.

4.  Label Creation
Use the "ClusterWindow" in the GUI to label syllable segments. The clusters can be automatically labeled and then manually adjusted as needed.

5.  Train the Classification Network
With the labeled syllables, use the "TrainingWindow" again to train the classification network based on the assigned labels.

6.  Real-Time Targeting Setup
Finally, update the configuration file with the names of the trained segmentation and classification networks, set `realtime_classification` to True, and specify the target syllable for feedback. MooveTaf can now be used for conducting real-time targeting experiments, enabling precise, low-latency feedback during song production.

A preliminary version of a guide on how to use Moove is available on request. The finished version will be made available soon.

## Contact

For questions, issues, or feedback regarding Moove, please contact:

**Primary contact:**
Lena Veit
lena.veit@uni-tuebingen.de
Nils Riekers  
nils@riekers.it

## License

Moove is licensed under the MIT License. You are free to use, modify, and distribute the software, provided that you retain the copyright notice and give appropriate credit to the original authors.

See [LICENSE](LICENSE) for details.
