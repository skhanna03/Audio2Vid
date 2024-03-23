import librosa
# from transformers import Wav2Vec2ProcessorWithLM, Wav2Vec2ForCTC
from transformers import Wav2Vec2Model, Wav2Vec2Processor

processor = Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-base-960h')
model = Wav2Vec2Model.from_pretrained('facebook/wav2vec2-base-960h')

# # load the processor
# processor = Wav2Vec2ProcessorWithLM.from_pretrained("patrickvonplaten/wav2vec2-base-100h-with-lm")
# model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h")

# load the audio data (use your own wav file here!)
input_audio, sr = librosa.load('/viscam/u/shubh/Moore-AnimateAnyone/configs/inference/audio_files/example.wav', sr=16000)

# tokenize
input_values = processor(input_audio, return_tensors="pt", padding="longest").input_values
print(input_values.size())
# print(input_values.shape)
# retrieve logits
input = input_values[:, 300000]
print(input.size())
logits = model(input_values[:, :400]).last_hidden_state

# decode using n-gram
transcription = processor.batch_decode(logits.detach().numpy()).text

# print the output
print(transcription)