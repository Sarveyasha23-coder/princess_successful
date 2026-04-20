import matplotlib.pyplot as plt
import seaborn as sns

def basic_eda(df):
    print(df.info())
    print(df.describe())

    sns.histplot(df['age'], kde=True)
    plt.title("Age Distribution")
    plt.show()

    sns.boxplot(x=df['numberoffriends'])
    plt.title("Friends Outliers")
    plt.show()