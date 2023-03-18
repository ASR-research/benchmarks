# NeMo's "core" package
import sys

# NeMo's ASR collection - this collections contains complete ASR models and
# building blocks (modules) for ASR
import nemo.collections.asr as nemo_asr

from functools import wraps
import time
import numpy as np
import pandas as pd
import json
import random
from pathlib import Path

from definitions import ROOT_DIR, SETTINGS

from src.tools.flac_duration import get_flac_duration
from src.tools.read_dataset import read_dataset
from jiwer import wer

model_name_to_model = dict()
models = []
nums_cores = []
WERs = []
sample_ids = []
engines = []
run_ids = []
times = []
text_lens = []
audio_lens = []


def add_row(model_name, num_cores, pronounced_text, recognized_text, audio, engine, attempt):
    models.append(model_name)
    nums_cores.append(num_cores)
    WERs.append(wer(pronounced_text.lower(), recognized_text.lower()))
    sample_ids.append(audio.stem)
    engines.append(engine)
    run_ids.append(attempt + 1)
    text_lens.append(len(pronounced_text))
    audio_lens.append(get_flac_duration(audio.as_posix()))


def setup():
    models_path = Path(ROOT_DIR) / "data" / "models.json"
    with open(models_path.as_posix()) as f:
        for model_info in json.load(f)["models"]:
            for model_class_name, model_names in model_info.items():
                model_class = getattr(sys.modules[nemo_asr.models.__name__], model_class_name)
                for model_name in model_names[:SETTINGS["number_models"]]:
                    model_name_to_model[model_name] = model_class.from_pretrained(model_name=model_name)


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
    recognized_texts = model.transcribe(paths2audio_files=files)
    assert len(recognized_texts) == 1, "False transcribe assumption!"
    return recognized_texts[0]


def make_benchmarks():
    for pronounced_text, audio in read_dataset("dev-other"):
        for model_name in model_name_to_model.keys():
            model = model_name_to_model[model_name]
            for attempt in range(SETTINGS["number_attempts"]):
                recognized_text = inference(model, [audio.as_posix()])
                add_row(model_name, "?", pronounced_text, recognized_text, audio, "NeMo", attempt)


def generate_baseline():
    setup()
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
    columns = list(benchmarks.keys())
    data = np.array([benchmarks[key] for key in columns]).T

    df = pd.DataFrame(columns=columns, data=data)

    with open(Path(ROOT_DIR) / 'artifacts/nemo_results.csv', 'w') as f:
        f.write(df.to_csv())
