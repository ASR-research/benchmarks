# NeMo's "core" package
import nemo
# NeMo's ASR collection - this collections contains complete ASR models and
# building blocks (modules) for ASR
import nemo.collections.asr as nemo_asr

from functools import wraps
import time

import numpy as np
import pandas as pd

benchmarks_list = []
model_name_to_model = {
    "quartznet": nemo_asr.models.EncDecCTCModel.from_pretrained(model_name="QuartzNet15x5Base-En"),
    "citrinet": nemo_asr.models.EncDecCTCModelBPE.from_pretrained(model_name="stt_en_citrinet_256")
}

def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        benchmarks_list.append(total_time)
        return result
    return timeit_wrapper

@timeit
def inference(model, files):
    for fname, transcription in zip(files, model.transcribe(paths2audio_files=files)):
        print(f"Audio in {fname} was recognized as: {transcription}")

def get_models(models_name):
    return [model_name_to_model[model_name] for model_name in models_name]

def get_files():
    return np.array(['sample.flac'])

def get_benchmarks(num_models, num_files):
    global benchmarks_list
    benchmarks_list = np.array(benchmarks_list)
    return np.reshape(benchmarks_list, (num_models, num_files))

def generate_baseline():
    models_name = model_name_to_model.keys()
    models = get_models(models_name)
    files = get_files()

    for model in models:
        inference(model, files)

    benchmarks = get_benchmarks(len(models), len(files))

    df = pd.DataFrame(index=models_name, columns=files, data=benchmarks)

    with open('result.csv', 'w') as f:
        f.write(df.to_csv())
