import pandas as pd
import matplotlib.pyplot as plt


def split_house(df, col):
    houses = df['Hogwarts House'].unique()
    colors = {'Gryffindor': 'red', 'Hufflepuff': 'yellow', 'Ravenclaw': 'blue', 'Slytherin': 'green'}
    for house in houses:
        house_data = df[df['Hogwarts House'] == house].dropna(subset=[col])
        plt.scatter(house_data['Index'], house_data[col], alpha=0.3, label=house, color=colors[house])

        
def scatter(df):
    numeric_columns = [col for col in df.select_dtypes(include=['number']).columns if col != 'Index']
    plt.figure(figsize=(15, 10))
    for i, col in enumerate(numeric_columns, 1):
        plt.subplot(4, 4, i)
        split_house(df, col)
        plt.title(col)
        plt.legend()
        plt.tight_layout()
    plt.show()


def main():
    try:
        df = pd.read_csv('../datasets/dataset_train.csv')
        scatter(df)
    except Exception as error:
        print(f"{type(error).__name__} : {error}")
    except KeyboardInterrupt:
        print("\nInput interrupted. Exiting...")


if __name__ == "__main__":
    main()
