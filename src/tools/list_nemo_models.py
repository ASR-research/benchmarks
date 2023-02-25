from definitions import ROOT_DIR
import json

import nemo.collections.asr as nemo_asr


def list_models(models_holder):
    model_names = []
    for model in models_holder.list_available_models():
        model_names.append(model.pretrained_model_name)
    return {models_holder.__name__: model_names}


with open(f'{ROOT_DIR}/data/models.json', 'w') as f:
    models = [
        list_models(nemo_asr.models.EncDecCTCModel),
        list_models(nemo_asr.models.EncDecCTCModelBPE),
    ]
    json.dump({"models": models}, f, indent=4)
