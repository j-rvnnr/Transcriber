import sounddevice as sd
import numpy as np
import wave
import time
import whisper
import threading

# Device indices
mic_index = 2
stereo_mix_index = 1

# Audio settings
samplerate = 16000
channels = 1
buffer_duration = 5  # buffer duration in seconds

# Output file names
mic_audio_file = "mic_audio.wav"
stereo_mix_audio_file = "stereo_mix_audio.wav"
mic_transcription_file = "mic_transcription.txt"
stereo_mix_transcription_file = "stereo_mix_transcription.txt"
combined_transcription_file = "combined_transcription.txt"

# Rolling buffers for recording
mic_buffer = []
stereo_mix_buffer = []

# Stop event for recording
stop_event = threading.Event()

# Whisper model initiation
model = whisper.load_model("base", device="cuda")


def audio_callback(indata, frames, time, status, buffer):
    if status:
        print(f"Audio status: {status}")
    buffer.append(indata.copy())


def record_audio(device_index, buffer, device_name):
    print(f"Recording started on {device_name}.")
    with sd.InputStream(
        samplerate=samplerate,
        channels=channels,
        callback=lambda indata, frames, time, status: audio_callback(indata, frames, time, status, buffer),
        device=device_index,
        dtype="float32",
    ):
        while not stop_event.is_set():
            time.sleep(0.1)  # Keep the script alive


def save_audio(buffer, filename):
    if buffer:
        print(f"Saving audio to {filename}...")
        with wave.open(filename, "wb") as wf:

            wf.setnchannels(channels)
            wf.setsampwidth(2) # This sets it to 16 bit (2 byte = 16 bit)
            wf.setframerate(samplerate)
            wf.writeframes((np.concatenate(buffer) * 32767).astype(np.int16).tobytes()) # scaling to 16 bit

        print(f"Audio saved to {filename}.")
    else:
        print(f"No audio data to save for {filename}.")


def transcribe_audio(file, device_name, output_file):
    try:
        print(f"Transcribing {file} from {device_name}...")
        start_time = time.time()
        result = model.transcribe(file, fp16=False)

        with open(output_file, "w", encoding="utf-8") as f:
            for segment in result["segments"]:
                timestamp = f"[{time.strftime('%H:%M:%S', time.gmtime(segment['start']))} - {time.strftime('%H:%M:%S', time.gmtime(segment['end']))}]"
                text = segment["text"].strip()
                f.write(f"{timestamp} {text}\n")

        duration = time.time() - start_time
        print(f"Transcription for {device_name} completed in {duration:.2f} seconds.")
    except Exception as e:
        print(f"Error during transcription for {device_name}: {e}")


def combine_transcriptions(mic_file, device_file, combined_file):
    print("Combining transcriptions...")
    try:
        mic_lines = []
        device_lines = []

        # read lines from both the files
        with open(mic_file, "r", encoding="utf-8") as f:
            mic_lines = [{"source": "Mic", "line": line.strip()} for line in f if line.strip()]

        with open(device_file, "r", encoding="utf-8") as f:
            device_lines = [{"source": "Device", "line": line.strip()} for line in f if line.strip()]

        # Parse the timestamps and combine based on time
        combined_lines = []
        for entry in mic_lines + device_lines:
            timestamp_start = entry["line"].split("]")[0].split("[")[1].split(" - ")[0]
            combined_lines.append({"timestamp": timestamp_start, "source": entry["source"], "line": entry["line"]})

        # Sort combined lines by timestamps
        combined_lines.sort(key=lambda x: x["timestamp"])

        # Write to the combined file
        with open(combined_file, "w", encoding="utf-8") as combined_file:
            for entry in combined_lines:
                combined_file.write(f"{entry['source']}: {entry['line']}\n")

        print("Combined transcription saved.")
    except Exception as e:
        print(f"Error during transcript combination: {e}")


def main():
    print("Initializing...")
    recording_start_time = time.time()

    # Recording threads
    mic_thread = threading.Thread(target=record_audio, args=(mic_index, mic_buffer, "Mic"), daemon=True)
    stereo_mix_thread = threading.Thread(target=record_audio, args=(stereo_mix_index, stereo_mix_buffer, "Stereo Mix"), daemon=True)

    mic_thread.start()
    stereo_mix_thread.start()

    try:
        input("Recording started. Press Enter to stop...\n")
        stop_event.set()
    except KeyboardInterrupt:
        print("Recording interrupted.")
        stop_event.set()

    mic_thread.join()
    stereo_mix_thread.join()

    # Save audio files
    save_audio(mic_buffer, mic_audio_file)
    save_audio(stereo_mix_buffer, stereo_mix_audio_file)

    # Recording time
    recording_duration = time.time() - recording_start_time
    print(f"Recording completed in {recording_duration:.2f} seconds.")

    # Transcribe saved audio files
    transcription_start_time = time.time()
    transcribe_audio(mic_audio_file, "Mic", mic_transcription_file)
    transcribe_audio(stereo_mix_audio_file, "Stereo Mix", stereo_mix_transcription_file)

    # Combine transcriptions
    combine_transcriptions(mic_transcription_file, stereo_mix_transcription_file, combined_transcription_file)

    # Transcription time
    transcription_duration = time.time() - transcription_start_time
    print(f"Transcription completed in {transcription_duration:.2f} seconds.")
    print("Recording and transcription process completed.")


if __name__ == "__main__":
    main()
