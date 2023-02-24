from definitions import ROOT_DIR
import json

import nemo.collections.asr as nemo_asr


def list_models(common_model_name, models_holder):
    with open(f'{ROOT_DIR}/data/models/models-{common_model_name}-like.json', 'w') as f:
      model_names = []
      for model in models_holder.list_available_models():
        model_names.append(model.pretrained_model_name)
      json.dump({"model_names": model_names}, f, indent=4)


list_models('quartznet', nemo_asr.models.EncDecCTCModel)
list_models('citrinet', nemo_asr.models.EncDecCTCModelBPE)