# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import plotly.graph_objs as go
from plotly.subplots import make_subplots


matplotlib.use('TkAgg')

def plot_num_of_medicines_for_drugs():
    # Load the CSV file
    file_path = 'repodb.csv'
    repo_df = pd.read_csv(file_path)

    # Grouping the data by 'ind_name' to count the number of drugs for each disease
    disease_counts = repo_df['ind_name'].value_counts()

    # Plotting the histogram/distribution of the number of drugs suitable for each disease
    plt.figure(figsize=(12, 6))
    plt.hist(disease_counts, bins=1462, edgecolor='black')
    plt.xlabel('Number of Drugs')
    plt.ylabel('Number of Diseases')
    plt.title('Distribution of the Number of Drugs Suitable for Each Disease')
    plt.grid(True)
    plt.show()

def plot_num_of_drugs_for_disease_names():
    # import pandas as pd
    # import matplotlib.pyplot as plt
    #
    # # Replace 'path_to_your_file' with the actual file path
    # repo_df = pd.read_csv('repodb.csv')
    #
    # # Grouping the data by 'ind_name' to count the number of drugs for each disease
    # disease_counts = repo_df['ind_name'].value_counts()
    #
    # # Selecting a subset of diseases for better visualization
    # subset_diseases = disease_counts.head(20)
    #
    # # Plotting the pie chart
    # plt.figure(figsize=(10, 10))
    # plt.pie(subset_diseases.values, labels=subset_diseases.index, autopct='%1.1f%%', startangle=140)
    # plt.title('Distribution of Drugs Among Top 20 Diseases')
    # plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    # plt.show()

    # Replace 'path_to_your_file' with the actual file path
    repo_df = pd.read_csv('repodb.csv')

    # Grouping the data by 'ind_name' to count the number of drugs for each disease
    disease_counts = repo_df['ind_name'].value_counts()

    # Selecting a subset of diseases for better visualization
    subset_diseases = disease_counts.head(1462)

    # Plotting the bar chart
    plt.figure(figsize=(12, 8))
    #sns.barplot(x=subset_diseases.values, y=subset_diseases.index, palette='viridis')
    #sns.barplot(x=subset_diseases.index, y=subset_diseases.values, palette='viridis')
    # The problem is when I change subset_diseases.index to range(let(subset_diseases) because I see each bar vertically
    # Instead of horizontally...
    # What happens when we swap axis...
    sns.barplot(x=range(len(subset_diseases)), y=subset_diseases.values, palette='viridis')

    plt.xlabel('Number of Drugs')
    plt.ylabel('Disease')
    plt.title('Number of Drugs Suitable for Each Disease')
    plt.xticks(ticks=range(-100,1462,100))
    plt.tight_layout()

    plt.show()

def plot_num_of_diseases_for_drug_names():
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Replace 'path_to_your_file' with the actual file path
    repo_df = pd.read_csv('repodb.csv')

    # Grouping the data by 'drug_name' to count the number of diseases each drug is used for
    drug_counts = repo_df['drug_name'].value_counts()

    # Selecting a subset of drugs for better visualization
    subset_drugs = drug_counts.head(2381)

    # Plotting the bar chart
    plt.figure(figsize=(12, 8))
    #sns.barplot(x=subset_drugs.index, y=subset_drugs.values, palette='viridis')
    sns.barplot(x=range(len(subset_drugs)), y=subset_drugs.values, palette='viridis')
    plt.xlabel('Drug')
    plt.ylabel('Number of Diseases')
    plt.title('Number of Diseases Treated by Each Drug')
    #plt.xticks(rotation=90)
    plt.xticks(ticks=range(-100, 2381, 100))
    plt.tight_layout()
    plt.show()


def plot_num_of_medicines_for_top10_drugs():
    # Load the CSV file
    file_path = 'repodb.csv'
    repo_df = pd.read_csv(file_path)

    # Grouping the data by 'ind_name' to count the number of drugs for each disease
    disease_counts = repo_df['ind_name'].value_counts()

    # Further analysis: Creating a bar plot for the top 10 diseases with the most drugs
    # Get the top 10 diseases with the most drugs
    top_10_diseases = disease_counts.head(10)

    # Get the top 10 diseases with the most drugs
    top_10_diseases = disease_counts.head(10)

    # Plotting the bar plot for the top 10 diseases
    plt.figure(figsize=(12, 7))
    top_10_diseases.plot(kind='bar', edgecolor='black')
    plt.xlabel('Disease')
    plt.ylabel('Number of Drugs')
    plt.title('Top 10 Diseases with the Most Drugs')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y')
    plt.show()


def show_distrebution_side_by_side(path):
    repo_df = pd.read_csv(path)
    disease_counts = repo_df['ind_name'].value_counts().sort_values(ascending=False)
    drug_counts = repo_df['drug_name'].value_counts().sort_values(ascending=False)
    dist_fig =  make_subplots(rows=1, cols=2, horizontal_spacing=0.05, vertical_spacing=0.02,
                              subplot_titles=[ 'Distribution of the number of drugs treating a<br> disease across diseases in the database',
                            'Distribution of the number of diseases treated by a<br> drug across drugs in the database'
                        ])
    dist_fig.add_trace(go.Scatter(x=[i+1 for i in range(len(disease_counts))], y=disease_counts, showlegend=False), row=1, col=1)
    dist_fig.add_trace(go.Scatter(x=[i+1 for i in range(len(drug_counts))], y=drug_counts, showlegend=False),row=1, col=2)
    dist_fig.update_xaxes(title_text='Disease', row=1, col=1)
    dist_fig.update_yaxes(title_text='Number of drugs that treat this disease', row=1, col=1)
    dist_fig.update_xaxes(title_text='Drug', row=1, col=2)
    dist_fig.update_yaxes(title_text='Number of diseases treated by this drug', row=1, col=2)
    dist_fig.update_annotations(font=dict(size=20))
    dist_fig.show()

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    #plot_num_of_medicines_for_drugs()
    # plot_num_of_drugs_for_disease_names()
    # plot_num_of_diseases_for_drug_names()
    # plot_num_of_medicines_for_top10_drugs()
    show_distrebution_side_by_side('repodb.csv')
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
