import numpy as np
import pandas as pd
import plotly.express as px
from pathlib import Path
from definitions import ROOT_DIR

from src.tools.read_dataset import read_dataset
from src.tools.flac_duration import get_flac_duration


def plot_audio_text_correlation():
    def get_duration_by_path_to_audio(path_to_audio):
        return get_flac_duration(path_to_audio.as_posix())
    vectorized_get_duration_by_path_to_audio = np.vectorize(get_duration_by_path_to_audio)
    vectorized_get_len_by_text = np.vectorize(len)
    a = np.array(list(read_dataset("dev-other")))
    a[:, 0] = vectorized_get_len_by_text(a[:, 0])
    a[:, 1] = vectorized_get_duration_by_path_to_audio(a[:, 1])
    df = pd.DataFrame(a)
    text_len = "text_len"
    audio_len = "audio_len"
    df.columns = [text_len, audio_len]
    fig = px.scatter(df, x=audio_len, y=text_len, trendline='ols', opacity=0.5, trendline_color_override="red")
    fig.write_image(Path(ROOT_DIR) / "artifacts" / "dataset_audio_len_to_text_len.png")


if __name__ == "__main__":
    plot_audio_text_correlation()
