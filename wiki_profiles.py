import os
import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import euclidean_distances
import seaborn as sns
import matplotlib.pyplot as plt


def load_profiles(directory_path):
    drug_profiles = {}
    disease_profiles = {}

    for filename in os.listdir(directory_path):
        if filename.endswith('.json'):
            file_path = os.path.join(directory_path, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                entity_name = data['name']
                content = data.get('content', '')
                category = data.get('category')

                if category == 'drug':
                    drug_profiles[entity_name] = content
                elif category == 'disease':
                    disease_profiles[entity_name] = content

    drugs_df = pd.DataFrame({'profile': drug_profiles.values()}, index=drug_profiles.keys())
    diseases_df = pd.DataFrame({'profile': disease_profiles.values()}, index=disease_profiles.keys())

    return drugs_df, diseases_df


def run_tfidf(df, profile_type):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(df['profile'])
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())
    tfidf_df.index = df.index
    return tfidf_df


def calculate_disease_dist(df):
    tfidf_mat = run_tfidf(df, 'disease')
    tfidf_dist_mat = pd.DataFrame(euclidean_distances(tfidf_mat), index=tfidf_mat.index, columns=tfidf_mat.index)
    return tfidf_dist_mat


def calculate_drug_dist(df):
    tfidf_mat = run_tfidf(df, 'drug')
    tfidf_dist_mat = pd.DataFrame(euclidean_distances(tfidf_mat), index=tfidf_mat.index, columns=tfidf_mat.index)
    return tfidf_dist_mat


def create_diseases_heatmap(df, name):
    disease_dist_matrix = calculate_disease_dist(df)
    dis_heatmap = sns.clustermap(disease_dist_matrix, cmap='viridis', figsize=(10, 10),
                                 xticklabels=False, yticklabels=False)
    dis_heatmap.figure.suptitle("Heatmap of Diseases based on TF-IDF", y=0.9, fontsize=16)
    dis_heatmap.figure.savefig(f"{name}_disease_heatmap_tfidf.png")
    new_order = [df.index[i] for i in dis_heatmap.dendrogram_row.reordered_ind]
    return dis_heatmap, new_order


def create_drugs_heatmap(df, name):
    drug_dist_matrix = calculate_drug_dist(df)
    drug_heatmap = sns.clustermap(drug_dist_matrix, cmap='viridis', figsize=(10, 10),
                                  xticklabels=False, yticklabels=False)
    drug_heatmap.figure.suptitle("Heatmap of Drugs based on TF-IDF", y=0.9, fontsize=16)
    drug_heatmap.figure.savefig(f"{name}_drug_heatmap_tfidf.png")
    new_order = [df.index[i] for i in drug_heatmap.dendrogram_row.reordered_ind]
    return drug_heatmap, new_order



def reorder_df(df, new_order, profile_type):
    reordered_df = df.reindex(new_order)
    return reordered_df

def analyze_wikipedia_profiles():
    drugs_df, diseases_df = load_profiles('C:/Users/edenb/Desktop/DATA_FINAL_PROJECT/wiki_data')
    diseases_heatmap, disease_order = create_diseases_heatmap(diseases_df, 'wikipedia')
    drugs_heatmap, drug_order = create_drugs_heatmap(drugs_df, 'wikipedia')

    reordered_disease_df = reorder_df(diseases_df, disease_order, 'diseases')
    reordered_drug_df = reorder_df(drugs_df, drug_order, 'drugs')

    reordered_disease_names = reordered_disease_df.index.to_frame(index=False)
    reordered_drug_names = reordered_drug_df.index.to_frame(index=False)

    reordered_disease_names.to_csv("C:/Users/edenb/Desktop/DATA_FINAL_PROJECT/reordered_diseases.csv", index=False,
                                   header=["Disease"])
    reordered_drug_names.to_csv("C:/Users/edenb/Desktop/DATA_FINAL_PROJECT/reordered_drugs.csv", index=False,
                                header=["Drug"])
    #print("Analysis complete.")

if __name__ == '__main__':
    analyze_wikipedia_profiles()
