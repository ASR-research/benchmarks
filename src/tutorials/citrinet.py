# NeMo's "core" package
import nemo
# NeMo's ASR collection - this collections contains complete ASR models and
# building blocks (modules) for ASR
import nemo.collections.asr as nemo_asr

from definitions import ROOT_DIR

# This line will download pre-trained QuartzNet15x5 model from NVIDIA's NGC cloud and instantiate it for you
citrinet = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(model_name="stt_en_citrinet_256")

files = [f'{ROOT_DIR}/data/LibriSpeech/sample.flac']
for fname, transcription in zip(files, citrinet.transcribe(paths2audio_files=files)):
    print(f"Audio in {fname} was recognized as: {transcription}")
