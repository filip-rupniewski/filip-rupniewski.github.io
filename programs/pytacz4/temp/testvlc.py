import subprocess
import os

# Get the current script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))

# Assuming the test mp3 file is in the same folder as this script
sound_file = os.path.join(script_dir, "test.mp3")  # Update with your test file name

# Full path to VLC executable (make sure VLC is installed in this location)
vlc_path = r"C:\Program Files\VideoLAN\VLC\vlc.exe"  # Update if necessary

# Run VLC with the specified options
subprocess.run(
    [vlc_path, '--play-and-exit', sound_file],
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL
)
