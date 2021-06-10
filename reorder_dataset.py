import pandas as pd
import shutil


def main():
    df = pd.read_csv('dataset/train.csv')

    for i in df.index:
        file_name = df.loc[i, 'Image_path']
        label = df.loc[i, 'Condition']

        try:
            shutil.move('dataset/trainImages/' + file_name,
                        'dataset/trainImages/' + str(label) + '/' + file_name)
        except FileNotFoundError:
            pass

    pass


if __name__ == '__main__':
    main()
