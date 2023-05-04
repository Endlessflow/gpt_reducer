# Note: you need to be using OpenAI Python v0.27.0 for the code below to work
import openai
from utils import append_to_file
from pydub import AudioSegment

FILENAME = "Cedille15feb2023.mp3"

audio_file= open(f"audio\{FILENAME}", "rb")

transcript = openai.Audio.transcribe("whisper-1", audio_file, response_format="text")

append_to_file(f"{FILENAME}_transcript.txt", transcript)