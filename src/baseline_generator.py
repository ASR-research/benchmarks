# NeMo's "core" package
import nemo
# NeMo's ASR collection - this collections contains complete ASR models and
# building blocks (modules) for ASR
import nemo.collections.asr as nemo_asr

from functools import wraps
import time

import numpy as np
import pandas as pd

import sys
sys.path.append('../')
from definitions import ROOT_DIR
import os
from pathlib import Path

model_name_to_model = {
    "quartznet": nemo_asr.models.EncDecCTCModel.from_pretrained(model_name="QuartzNet15x5Base-En"),
    "citrinet": nemo_asr.models.EncDecCTCModelBPE.from_pretrained(model_name="stt_en_citrinet_256")
}
models = []
nums_cores = []
WERs = []
sample_ids = []
engines = []
run_ids = []
times = []
text_lens = []
audio_lens = []

def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        times.append(total_time)
        return result
    return timeit_wrapper

@timeit
def inference(model, files):
    return model.transcribe(paths2audio_files=files)

def make_benchmarks():
    for model_name in model_name_to_model.keys():
        model = model_name_to_model[model_name]
        path = Path(ROOT_DIR) / "data" / "LibriSpeech" / "dev-other"
        # for reader_id in path.iterdir():
        #     for chapter_id in reader_id.iterdir():
        reader_id = list(path.glob("116"))[0]
        chapter_id = list(reader_id.glob("288045"))[0]
        texts = list(chapter_id.glob("*.txt"))
        assert len(texts) == 1, "False text assumption!"
        text = texts[0]
        with open(text) as f:
            for line in f.readlines():
                audio_filename, pronounced_text = line.split(' ', 1)
                audios = list(chapter_id.glob(f'{audio_filename}.flac'))
                assert len(audios) == 1, "False audio assumption!"
                audio = audios[0]
                print(pronounced_text, audio)
                for attempt in range(5):
                    inference(model, [audio.as_posix()])
                    models.append(model_name)
                    nums_cores.append("?")
                    WERs.append("?")
                    sample_ids.append(audio.stem)
                    engines.append("NeMo")
                    run_ids.append(f"{audio.stem}-{attempt + 1}")
                    text_lens.append(len(pronounced_text))
                    audio_lens.append("?")


def generate_baseline():
    make_benchmarks()

    benchmarks = {
        "model": models,
        "num_cores": nums_cores,
        "WER": WERs,
        "sample_id": sample_ids,
        "engine": engines,
        "run_id": run_ids,
        "time": times,
        "text_len": text_lens,
        "audio_len": audio_lens,
    }
    columns = benchmarks.keys()
    data = np.array([benchmarks[key] for key in columns]).T

    df = pd.DataFrame(columns=columns, data=data)

    with open(Path(ROOT_DIR) / 'artifacts/result.csv', 'w') as f:
        f.write(df.to_csv())