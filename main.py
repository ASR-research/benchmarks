import pandas as pd
import numpy as np

libraries = ['TensorFlow', 'PyTorch']

benchmarks = np.zeros(2).reshape(1, 2)


def main():
    df = pd.DataFrame(index=['basic transformer'], columns=libraries, data=benchmarks)

    with open('result.csv', 'w') as f:
        f.write(df.to_csv())


if __name__ == "__main__":
    main()
