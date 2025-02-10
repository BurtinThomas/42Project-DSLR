import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def pair_plot(df):
    sns.pairplot(df, hue='Hogwarts House')
    plt.show()


def main():
    try:
        df = pd.read_csv('../datasets/dataset_train.csv')
        pair_plot(df)
    except Exception as error:
        print(f"{type(error).__name__} : {error}")
    except KeyboardInterrupt:
        print("\nInput interrupted. Exiting...")


if __name__ == "__main__":
    main()
