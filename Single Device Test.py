import sounddevice as sd
import numpy as np
import whisper
import queue
import threading
import time
import wave

# Set mic device index
device_index = 2

# Whisper model init
model = whisper.load_model("base", device="cuda")

samplerate = 16000
channels = 1

# Rolling buffer settings
BUFFER_DURATION = 5 # seconds
transcription_interval = 5  # transcription frequency, in seconds

# Queue for audio data
audio_queue = queue.Queue()
stop_event = threading.Event()


def audio_callback(indata, frames, time, status):
    if status:
        print(f"Audio status: {status}")
    audio_queue.put(indata.copy())


def record_audio():
    print("Recording started. Press Enter to stop.")
    with sd.InputStream(
        samplerate=samplerate,
        channels=channels,
        callback=audio_callback,
        dtype="float32",
        device=device_index,
    ):
        while not stop_event.is_set():
            time.sleep(0.1)  # Keep the thread alive


def transcribe_audio():
    rolling_buffer = np.zeros((int(BUFFER_DURATION * samplerate), channels), dtype="float32")
    start_time = time.time()
    transcription_file = "transcription.txt"
    audio_file = "recorded_audio.wav"

    while not stop_event.is_set():
        try:
            # Grab audio data from the queue
            while not audio_queue.empty():
                data = audio_queue.get_nowait()
                rolling_buffer = np.roll(rolling_buffer, -len(data), axis=0)
                rolling_buffer[-len(data):] = data

            # Do the transcription at intervals
            if time.time() - start_time >= transcription_interval:
                start_time = time.time()

                # Normalize the audio buffer for Whisper
                if np.max(np.abs(rolling_buffer)) > 0:
                    audio_data = rolling_buffer.flatten() / np.max(np.abs(rolling_buffer.flatten()))
                else:
                    print("Skipping transcription due to silence.")
                    continue

                # Perform transcription
                result = model.transcribe(audio_data, fp16=False)
                text = result.get("text", "").strip()
                print(f"[{time.strftime('%H:%M:%S')}] {text}")

                # Save the transcription to file
                with open(transcription_file, "a", encoding="utf-8") as f:
                    f.write(f"[{time.strftime('%H:%M:%S')}] {text}\n")

        except Exception as e:
            print(f"Error during transcription: {e}")

    # Save audio to file
    print("Saving audio...")
    with wave.open(audio_file, "wb") as wf:

        wf.setnchannels(channels)
        wf.setsampwidth(2) # This sets it to 16 bit (2 byte = 16 bit)
        wf.setframerate(samplerate)
        wf.writeframes((rolling_buffer.flatten() * 32767).astype(np.int16).tobytes()) # scaling to 16 bit

    print(f"Audio saved to {audio_file}")


def main():
    recording_thread = threading.Thread(target=record_audio, daemon=True)
    transcription_thread = threading.Thread(target=transcribe_audio, daemon=True)

    recording_thread.start()
    transcription_thread.start()

    try:
        input("Recording started. Press Enter to stop...\n")
        stop_event.set()
    except KeyboardInterrupt:
        print("Recording interrupted.")
        stop_event.set()

    recording_thread.join()
    transcription_thread.join()

    print("Recording and transcription completed.")


if __name__ == "__main__":
    main()
