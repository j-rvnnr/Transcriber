import sounddevice as sd

def list_audio_devices():
    """List all available audio devices"""
    print("Available audio devices:")
    devices = sd.query_devices()
    for i, device in enumerate(devices):
        print(f"{i}: {device['name']} (Input Channels: {device['max_input_channels']}, Output Channels: {device['max_output_channels']})")

list_audio_devices()
