
from transformers import pipeline

transcriber = pipeline("automatic-speech-recognition",
                       model="openai/whisper-base")

audio_file = "raw_video_audio.wav"

print(f"Processing '{audio_file}'...\n")

try:
    result = transcriber(audio_file, return_timestamps=True)

    for chunk in result['chunks']:
        start_time = chunk['timestamp'][0]
        end_time = chunk['timestamp'][1]
        text = chunk['text'].strip()

        print(f"[{start_time:.1f}s -> {end_time:.1f}s] : {text}")

except Exception as e:
    print("Completed.")
