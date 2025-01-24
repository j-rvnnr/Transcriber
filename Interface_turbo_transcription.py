import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import time
import sounddevice as sd
import threading
import os
import wave
import numpy as np
import whisper
import json
from pathlib import Path

# Global variables
recording = False
paused = False
start_time = None
elapsed_time_on_pause = 0
mic_index = None
stereo_mix_index = None
mic_buffer = []
stereo_mix_buffer = []
stop_event = threading.Event()
samplerate = 16000


# Whisper model
model = whisper.load_model("base", device="cuda")

# Config management
APP_NAME = "Summariser"
# It can't even summarise yet lol



# Define the version number
VERSION = "v0.0.1"

# Findt he config path. I've set this up for all the OS under the sun because I LOVE FUTUREPROOFING
def get_config_path():
    if os.name == "nt":  # Windows
        base_dir = Path(os.getenv("APPDATA")) / APP_NAME

    elif os.name == "posix":
        if "darwin" in os.uname().sysname.lower():  # macOS
            base_dir = Path.home() / "Library" / "Application Support" / APP_NAME

        else:  # Linux
            base_dir = Path.home() / ".config" / APP_NAME

    else:
        raise OSError("Unsupported operating system")

    base_dir.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists
    return base_dir / "config.json"

# Load the Config file function
def load_config():
    config_path = get_config_path()
    if config_path.exists():
        with open(config_path, "r") as f:
            return json.load(f)
    return {}

# Save the confic File function
def save_config(config):
    config_path = get_config_path()
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)
    print(f"Configuration saved to {config_path}")





# Load configuration
config = load_config()
if "default_folder" not in config:
    config["default_folder"] = os.getcwd()
if "mic_index" not in config:
    config["mic_index"] = None
if "stereo_mix_index" not in config:
    config["stereo_mix_index"] = None
save_config(config)

save_folder = config["default_folder"]

# Helper functions

# Audio Callback Function
def audio_callback(indata, frames, time, status, buffer):
    if status:
        print(f"Audio status: {status}")
    buffer.append(indata.copy())

# Audio Recording function
def record_audio(device_index, buffer):
    try:
        with sd.InputStream(samplerate=samplerate, channels=1, device=device_index, dtype="float32") as stream:
            while not stop_event.is_set():
                if not paused:
                    data, _ = stream.read(1024) # Chunksize 1024 works for post transcribing, not so much for real time
                    buffer.append(data)
    except Exception as e:
        print(f"Error recording audio from device: {e}")

# Saving the audio from the buffer to the file
def save_audio(buffer, filename):
    if buffer:
        with wave.open(filename, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(samplerate)
            wf.writeframes((np.concatenate(buffer) * 32767).astype(np.int16).tobytes())
        print(f"Audio saved to {filename}")


# Use whisper to transcribe the audo
def transcribe_audio(file, input_name, output_file):
    try:
        print(f"Transcribing {file} from {input_name}...")
        start_time = time.time()  # Start timing transcription. Useful for observing the ratio between recorded time and transcription time
        result = model.transcribe(file, fp16=False)

        with open(output_file, "w", encoding="utf-8") as f:
            for segment in result["segments"]:
                timestamp = f"[{time.strftime('%H:%M:%S', time.gmtime(segment['start']))} - {time.strftime('%H:%M:%S', time.gmtime(segment['end']))}]"
                text = segment["text"].strip()
                f.write(f"{timestamp} {text}\n")

            # Add transcription duration at the end of the file
            transcription_time = time.time() - start_time
            f.write(f"\nTranscription completed in {transcription_time:.2f} seconds.\n")

        print(f"Transcription for {input_name} saved to {output_file}.")
    except Exception as e:
        print(f"Error during transcription for {input_name}: {e}")


# Combine our transcriptions based on the turbo transcription script
def combine_transcriptions(input1_file, input2_file, combined_file):
    try:
        input1_lines = []
        input2_lines = []

        # Read lines from Input 1
        with open(input1_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip() and not line.startswith("Transcription completed"):
                    timestamp, text = line.split("]", 1)
                    input1_lines.append({
                        # extract timestamp
                        "timestamp": timestamp.strip("["),
                        "source": "Input 1",
                        "line": line.strip()
                    })

        # Read lines from Input 2
        with open(input2_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip() and not line.startswith("Transcription completed"):
                    timestamp, text = line.split("]", 1)
                    input2_lines.append({
                        # extract timestamp
                        "timestamp": timestamp.strip("["),
                        "source": "Input 2",
                        "line": line.strip()
                    })

        # Combine and sot the lines by their timestamp
        combined_lines = input1_lines + input2_lines
        combined_lines.sort(key=lambda x: x["timestamp"])


        # Write all the lines to the combined file
        with open(combined_file, "w", encoding="utf-8") as f:
            for entry in combined_lines:
                f.write(f"{entry['source']}: {entry['line']}\n")

        print(f"Combined transcription saved to {combined_file}.")
    except Exception as e:
        print(f"Error combining transcriptions: {e}")


# Update the timer label
def update_timer(timer_label):
    global elapsed_time_on_pause
    while recording:
        if not paused:
            elapsed_time = time.time() - start_time + elapsed_time_on_pause
            # Format time
            centiseconds = int((elapsed_time % 1) * 100)
            formatted_time = time.strftime('%H:%M:%S', time.gmtime(elapsed_time)) + f".{centiseconds:02d}"
            timer_label.config(text=formatted_time)
        time.sleep(0.01)  # rapid update so the timer looks cool. This is 100% aesthetic

# Recording button function
def start_recording(timer_label, input_1, input_2):
    global recording, paused, start_time, mic_index, stereo_mix_index, mic_buffer, stereo_mix_buffer, elapsed_time_on_pause

    if recording:
        # If already recording, we need this here so that it doesn't throw an error window. Returns nothing
        return

    try:
        mic_index = int(input_1.get().split(' ')[0]) if input_1.get() else None
        stereo_mix_index = int(input_2.get().split(' ')[0]) if input_2.get() else None

        if mic_index is None or stereo_mix_index is None:
            messagebox.showerror("Error", "Please select both input devices.")
            return

        recording = True
        paused = False
        start_time = time.time()
        elapsed_time_on_pause = 0  # Reset the elapsed time on pause
        stop_event.clear()

        # Change Record button colour to red. Extremely cool
        record_button.config(style="Red.TButton")

        # Start the recording threads
        mic_buffer = []
        stereo_mix_buffer = []
        threading.Thread(target=record_audio, args=(mic_index, mic_buffer), daemon=True).start()
        threading.Thread(target=record_audio, args=(stereo_mix_index, stereo_mix_buffer), daemon=True).start()

        # Start the timer thread
        threading.Thread(target=update_timer, args=(timer_label,), daemon=True).start()

        print("Recording started.")
    except Exception as e:
        messagebox.showerror("Error", str(e))

# Pause button function
def pause_recording():
    global paused, start_time, elapsed_time_on_pause
    if not recording:
        messagebox.showerror("Error", "No recording in progress to pause.")
        return

    paused = not paused
    if paused:
        elapsed_time_on_pause += time.time() - start_time
        pause_button.config(text="Resume") # This changes the text so the pause button readds resume while paused
    else:
        start_time = time.time()
        pause_button.config(text="Pause") # and back to paused once we're going again

# stop button function. This handles the saving and transcribing too, as it should only be called at the end of a recording session
def stop_recording():
    global recording
    if not recording:
        messagebox.showerror("Error", "No recording in progress to stop.")
        return

    recording = False
    stop_event.set()
    record_button.config(style="TButton")
    print("Recording stopped.")

    # Save and transcribe the  audio files
    input1_file = os.path.join(save_folder, "input1_audio.wav")
    input2_file = os.path.join(save_folder, "input2_audio.wav")
    input1_transcription_file = os.path.join(save_folder, "input1_transcription.txt")
    input2_transcription_file = os.path.join(save_folder, "input2_transcription.txt")
    combined_file = os.path.join(save_folder, "combined_transcription.txt")

    save_audio(mic_buffer, input1_file)
    save_audio(stereo_mix_buffer, input2_file)
    transcribe_audio(input1_file, "Input 1", input1_transcription_file)
    transcribe_audio(input2_file, "Input 2", input2_transcription_file)
    combine_transcriptions(input1_transcription_file, input2_transcription_file, combined_file)

    messagebox.showinfo("Success", "Recording and transcription completed.")

# Browse folder button, so we can save to a custom location
def browse_folder(folder_label):
    global save_folder
    selected_folder = filedialog.askdirectory()
    if selected_folder:
        save_folder = selected_folder
        folder_label.config(text=save_folder)
        config["default_folder"] = save_folder
        save_config(config)
        print(f"Save folder set to: {save_folder}")





# Main application UI

'''
I don't really know how to use tkinter, so this was a vision quest with the help of a youtube tutorial or ten, and chatgpt

'''

# Main Window Settings
root = tk.Tk()
root.title(f"Summariser {VERSION}")
root.geometry("400x400")

# Tkinter styling
style = ttk.Style()
style.configure("Red.TButton", foreground="red")

# Center frame
center_frame = tk.Frame(root)
center_frame.pack(expand=True)

# Timer
timer_frame = ttk.LabelFrame(center_frame, text="Timer")
timer_frame.grid(row=0, column=0, columnspan=4, pady=10, padx=5)
timer_label = ttk.Label(timer_frame, text="00:00:00.00", font=("Arial", 14))
timer_label.pack(pady=5)

# Buttons
record_button = ttk.Button(center_frame, text="Record", command=lambda: start_recording(timer_label, input_1, input_2))
record_button.grid(row=1, column=0, padx=5, pady=5)

pause_button = ttk.Button(center_frame, text="Pause", command=pause_recording)
pause_button.grid(row=1, column=1, padx=5, pady=5)

stop_button = ttk.Button(center_frame, text="Stop", command=stop_recording)
stop_button.grid(row=1, column=2, padx=5, pady=5)

# Input device dropdowns. We make sure in here the device is capable of recording
input_1_label = ttk.Label(center_frame, text="Input 1:")
input_1_label.grid(row=2, column=0, padx=5, pady=5)

input_1 = ttk.Combobox(
    center_frame,
    width=30,
    values=[
        f"{i} {sd.query_devices(i)['name']}" for i in range(len(sd.query_devices())) if sd.query_devices(i)["max_input_channels"] > 0
    ],
)
input_1.grid(row=2, column=1, padx=5, pady=5)
input_1.set(f"{config['mic_index']} {sd.query_devices(config['mic_index'])['name']}" if config['mic_index'] is not None else "")

input_2_label = ttk.Label(center_frame, text="Input 2:")
input_2_label.grid(row=3, column=0, padx=5, pady=5)

input_2 = ttk.Combobox(
    center_frame,
    width=30,
    values=[
        f"{i} {sd.query_devices(i)['name']}" for i in range(len(sd.query_devices())) if sd.query_devices(i)["max_input_channels"] > 0
    ],
)
input_2.grid(row=3, column=1, padx=5, pady=5)
input_2.set(f"{config['stereo_mix_index']} {sd.query_devices(config['stereo_mix_index'])['name']}" if config['stereo_mix_index'] is not None else "")

# Folder selection
folder_label = ttk.Label(center_frame, text=save_folder, wraplength=300)
folder_label.grid(row=4, column=0, columnspan=2, pady=5)

browse_button = ttk.Button(center_frame, text="Browse", command=lambda: browse_folder(folder_label))
browse_button.grid(row=4, column=2, padx=5, pady=5)

# Settings button
settings_button = ttk.Button(center_frame, text="Settings", command=lambda: print("Settings window doesn't exist yet."))
settings_button.grid(row=5, column=0, columnspan=4, pady=10)

root.mainloop()
