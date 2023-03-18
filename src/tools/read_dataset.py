import logging
import random
from pathlib import Path
from definitions import ROOT_DIR

logging.basicConfig(filename=f'{ROOT_DIR}/artifacts/logs/nemo.log', level=logging.DEBUG, filemode='w')


def read_dataset(dataset_type):
    path = Path(ROOT_DIR) / "data" / "LibriSpeech" / dataset_type # "dev-other"
    # reader_id = list(path.glob("116"))[0]
    reader_ids = list(path.iterdir())
    for i, reader_id in enumerate(reader_ids):
        logging.info(f"reader_id number {i + 1} from {len(reader_ids)}")
        chapter_ids = list(reader_id.iterdir())
        for j, chapter_id in enumerate(chapter_ids):
            logging.info(f"\tchapter_id number {j + 1} from {len(chapter_ids)}")
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
                    path_to_audio = audios[0]
                    yield pronounced_text, path_to_audio
