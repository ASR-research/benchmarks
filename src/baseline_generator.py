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
import logging
import random
from pathlib import Path

from definitions import ROOT_DIR, SETTINGS

from src.tools.flac_duration import get_flac_duration
from jiwer import wer


logging.basicConfig(filename=f'{ROOT_DIR}/artifacts/logs/nemo_{random.randint(0, int(1e9))}.log', level=logging.DEBUG)

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
    for model_name in model_name_to_model.keys():
        model = model_name_to_model[model_name]
        path = Path(ROOT_DIR) / "data" / "LibriSpeech" / "dev-other"
        reader_ids = list(path.iterdir())
        for i, reader_id in enumerate(reader_ids):
            logging.info(f"reader_id number {i + 1} from {len(reader_ids)}")
            chapter_ids = list(reader_id.iterdir())
            for j, chapter_id in enumerate(chapter_ids):
                logging.info(f"\tchapter_id number {j + 1} from {len(chapter_ids)}")
        # reader_id = list(path.glob("116"))[0]
        # chapter_id = list(reader_id.glob("288045"))[0]
                texts = list(chapter_id.glob("*.txt"))
                assert len(texts) == 1, "False text assumption!"
                text = texts[0]
                with open(text) as f:
                    lines = list(f.readlines())
                    for k, line in enumerate(lines):
                        logging.info(f"\t\taudio number {k + 1} from {len(lines)}")
                        audio_filename, pronounced_text = line.split(' ', 1)
                        audios = list(chapter_id.glob(f'{audio_filename}.flac'))
                        assert len(audios) == 1, "False audio assumption!"
                        audio = audios[0]
                        print(pronounced_text, audio)
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
