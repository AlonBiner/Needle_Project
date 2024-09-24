import pandas as pd
import numpy as np
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import pairwise_distances
from seaborn import clustermap, heatmap
from collections import defaultdict
import plotly.graph_objects as go
import plotly.express as px
from create_profies import scrape_for_disease_info
import requests
import seaborn as sns
from bokeh.io import output_file, show
from bokeh.models import ColumnDataSource, ImageRGBA, ColorBar, HoverTool
from bokeh.plotting import figure
from bokeh.transform import factor_cmap
from bokeh.palettes import Category10
import re
import rdkit

def diseases_nan_values(diseases_df):
    """
        Analyzes a DataFrame of diseases to count NaN values in specified columns.

        Parameters:
            diseases_df (pd.DataFrame): DataFrame containing disease information.

        Prints:
            Number of NaN values in 'Type', 'Definition', and 'Pubmed_links' columns,
            as well as combinations of these columns.
        """
    type_nans = diseases_df['Type'].isna()
    definition_nans = diseases_df['Definition'].isna()
    links_nans = diseases_df['Pubmed_links'].isna()
    print("number of diseases with nan in Type: " + str(type_nans.sum()))
    print("number of diseases with nan in Definition: " + str(definition_nans.sum()))
    print("number of diseases with nan in Pubmed links: " + str(links_nans.sum()))
    print("number of diseases with nan in Type and Definition: " + str(len(diseases_df[diseases_df.Type.isna() & diseases_df.Pubmed_links.isna()])))
    print("number of diseases with nan in Pubmed links and Definition: " + str(len(diseases_df[diseases_df.Definition.isna() & diseases_df.Pubmed_links.isna()])))
    print("number of diseases with nan in Type and Pubmed links: " + str(len(diseases_df[diseases_df.Definition.isna() & diseases_df.Type.isna()])))
    print("number of diseases with nan in all row: " + str(len(diseases_df[diseases_df.Definition.isna() & diseases_df.Pubmed_links.isna() & diseases_df.Type.isna()])))


def run_tfidf(df):
    """
        Computes the TF-IDF matrix for the 'profile' column in a DataFrame.

        Parameters:
            df (pd.DataFrame): DataFrame containing a 'profile' column with text data.

        Returns:
            pd.DataFrame: A DataFrame representing the TF-IDF matrix.
        """
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(df['profile'])
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())
    tfidf_df.index = df.index
    return tfidf_df

def create_heatmap(df, name, dist_callable):
    """
        Creates a clustered heatmap from a distance matrix derived from the input DataFrame.

        Parameters:
            df (pd.DataFrame): DataFrame containing the data to be clustered.
            name (str): Column name used for labeling in the heatmap.
            dist_callable (function): A callable function that computes distances.

        Returns:
            tuple: The clustered heatmap object and a list of new order of indices.
        """
    dist_mat = dist_callable(df)
    clustered_heatmap = clustermap(dist_mat, figsize=(10, 10), yticklabels=False,xticklabels=False)
    new_order = [df[name][i] for i in clustered_heatmap.dendrogram_row.reordered_ind]
    print(new_order)
    return clustered_heatmap, new_order


def create_diseases_heatmap(name, profile_type):
    """
        Creates a heatmap for diseases by loading disease profiles from a CSV file,
        filtering them, and computing distances.

        Parameters:
            name (str): Column name used for labeling in the heatmap.
            profile_type (str): Type of profile to be indicated in the output filename.

        Returns:
            tuple: The heatmap object and a new order of indices.
        """
    disease_profiles = pd.read_csv("nih_disease_info.csv")
    disease_profiles = disease_profiles[
        disease_profiles.Definition.notna() | disease_profiles.Pubmed_links.notna() | disease_profiles.Type.notna()].reset_index()
    dis_heatmap, new_order = create_heatmap(disease_profiles, name, calculate_disease_dist)
    dis_heatmap.figure.savefig(f"disease_heatmap_{profile_type}_2.png")
    return dis_heatmap, new_order


original_dummy_columns = ['Acquired Abnormality', 'Congenital Abnormality', 'Disease or Syndrome', 'Finding',
                              'Injury or Poisoning', 'Mental or Behavioral Dysfunction', 'Neoplastic Process',
                              'Pathologic Function', 'Sign or Symptom']
def split_and_create_dummies(row):
    """
        Splits a string of categories into dummy variables for a predefined set of columns.

        Parameters:
            row (str): A string of categories separated by semicolons.

        Returns:
            pd.Series: A Series with boolean values indicating presence of categories.
        """
    categories = row.split(';')
    return pd.Series([True] * len(categories) + [False] * (len(original_dummy_columns) - len(categories)), index=original_dummy_columns)

def calculate_disease_dist(df, w_tfidf=0.75, w_type=0.25):
    """
        Calculates a combined distance metric for diseases based on TF-IDF and type dummy variables.

        Parameters:
            df (pd.DataFrame): DataFrame containing disease profiles and types.
            w_tfidf (float): Weight for the TF-IDF distance component.
            w_type (float): Weight for the type distance component.

        Returns:
            pd.DataFrame: A DataFrame representing the combined distance matrix.
        """
    concatenated_df = pd.DataFrame({'index': df['ind_name'],
                                    'profile': df['Definition'].fillna('').str.cat(
                                            df['Pubmed_links'].fillna(''), sep=' ')}).set_index('index')
    tfidf_mat = run_tfidf(concatenated_df)
    tfidf_dist_mat = pd.DataFrame(euclidean_distances(tfidf_mat), index=tfidf_mat.index, columns=tfidf_mat.index)
    type_df = df['Type'].str.get_dummies(sep=';')
    type_dist = pd.DataFrame(euclidean_distances(type_df), index=tfidf_mat.index,
                 columns=tfidf_mat.index)
    return w_tfidf * tfidf_dist_mat + w_type*type_dist



def create_pivoted_dataframe(csv_file):
    """
        Creates a pivoted DataFrame from a CSV file containing drug data.

        Parameters:
            csv_file (str): Path to the CSV file with drug information.

        Returns:
            pd.DataFrame: A pivoted DataFrame indexed by drugbank_id with drug status and phase.
        """
    df = pd.read_csv(csv_file)
    drug_data = defaultdict(lambda: defaultdict(str))
    for _, row in df.iterrows():
        drugbank_id = row['drugbank_id']
        ind_name = row['ind_name']
        status = row['status']
        phase = row['phase']
        if pd.notna(status) and pd.notna(phase):
            combined_status = f"{status} ({phase})"
        elif pd.notna(status):
            combined_status = status
        else:
            combined_status = phase
        drug_data[drugbank_id][ind_name] = combined_status
    pivoted_df = pd.DataFrame(drug_data)

    return pivoted_df



def extract_numeric(s):
    """
        Extracts numeric values from a specific formatted string.

        Parameters:
            s (str): A string containing numeric values in a predefined format.

        Returns:
            tuple: Extracted average and monoisotopic weight as strings, or (0, 0) on failure.
        """
    try:
        avg_match = re.search(r'(\d+\.\d+)', s.split(':')[1])
        mono_match = re.search(r'(\d+\.\d+)', s.split(':')[2])
        return avg_match.group(1) if avg_match else None, \
           mono_match.group(1) if mono_match else None
    except Exception:
        return 0, 0

def calculate_drug_dist(df, w_tfidf=0.75, w_weight=0.25):
    """
        Calculates a combined distance metric for drugs based on TF-IDF and weight attributes.

        Parameters:
            df (pd.DataFrame): DataFrame containing drug profiles.
            w_tfidf (float): Weight for the TF-IDF distance component.
            w_weight (float): Weight for the weight distance component.

        Returns:
            pd.DataFrame: A DataFrame representing the combined distance matrix.
        """
    concatenated_df = pd.DataFrame({'index': df['Generic Name'],
                                    'profile': df['Background'].fillna('').str.cat(
                                        df['Summary'].fillna(''), sep=' ')}).set_index('index')
    tfidf_mat = run_tfidf(concatenated_df)
    tfidf_dist_mat = pd.DataFrame(euclidean_distances(tfidf_mat), index=tfidf_mat.index, columns=tfidf_mat.index)
    weight_dist = pd.DataFrame(euclidean_distances(df[['Average Weight', 'Monoisotopic Weight']]), index=tfidf_mat.index, columns=tfidf_mat.index)
    return w_tfidf * tfidf_dist_mat + w_weight*weight_dist

def create_drugs_heatmap(name, profile_type):
    """
        Creates a heatmap for drugs by loading drug profiles from a CSV file,
        processing weights, and computing distances.

        Parameters:
            name (str): Column name used for labeling in the heatmap.
            profile_type (str): Type of profile to be indicated in the output filename.

        Returns:
            tuple: The heatmap object and a new order of indices.
        """
    drugs_profiles = pd.read_csv("drugbank_info_for_df.csv").drop_duplicates()
    drugs_profiles.replace('Not Available', np.nan, inplace=True)
    drugs_profiles = drugs_profiles[
        drugs_profiles['Summary'].notna() | drugs_profiles['Background'].notna()].reset_index()
    drugs_profiles[['Average Weight', 'Monoisotopic Weight']] = drugs_profiles['Weight'].apply(extract_numeric).apply(pd.Series)
    drugs_profiles['Average Weight'] = drugs_profiles['Average Weight'].astype(float)
    drugs_profiles['Monoisotopic Weight'] = drugs_profiles['Monoisotopic Weight'].astype(float)
    drugs_profiles['Average Weight'] /= 1000
    drugs_profiles['Monoisotopic Weight'] /= 1000
    drugs_heatmap, drugs_order = create_heatmap(drugs_profiles, name, calculate_drug_dist)
    drugs_heatmap.figure.savefig(f"drugs_heatmap_{profile_type}_2.png")
    return drugs_heatmap, drugs_order



def reorder_df(df, disease_order, drug_order):
    """
        Reorders a DataFrame based on specified orders for diseases and drugs.

        Parameters:
            df (pd.DataFrame): The DataFrame to be reordered.
            disease_order (list): The new order of disease indices.
            drug_order (list): The new order of drug columns.

        Returns:
            pd.DataFrame: The reordered DataFrame.
        """
    df_reordered = df.reindex(index=disease_order)
    df_reordered = df_reordered[drug_order]
    return df_reordered


# def get_category(status):
#     categories = ['Approved', 'Terminated', 'Withdrawn', 'Suspended']
#     return next((cat for cat in categories if cat in status), 'Unknown')
#
# def get_phase(status):
#     phases = ['Early Phase 1', 'Phase 1', 'Phase 2', 'Phase 3', 'Phase 1/Phase 2', 'Phase 1/Phase 2', 'Phase 2/Phase 3']
#     return next((phase for phase in phases if phase in status), 'Unknown')
#
#
# def get_color(status):
#     if status == "N/A":
#         return "#FFFFFF"
#     if status == 'Approved':
#         return "#90EE90"
#     category = get_category(status)
#     phase = get_phase(status)
#     base_colors = {
#         'Terminated': '#FFA07A',  # Orange
#         'Withdrawn': '#FF6347',  # Red
#         'Suspended': '#8B4513'  # Brown
#     }
#     shade_factor = int(phase.split()[-1]) / 4
#     # Convert hex color to RGB and calculate shade
#     rgb_color = tuple(int(base_colors[category][i:i + 2], 16) for i in (2, 4, 6))
#     shaded_rgb = tuple(int(channel * shade_factor + 255 * (1 - shade_factor)) for channel in rgb_color)
#     return '#{:02x}{:02x}{:02x}'.format(*shaded_rgb)
#
# def replace_nan(df):
#     # Replace NaN with a custom string
#     df = df.replace({np.nan: 'N/A'})
#     return df
#
#
# def style_dataframe(df):
#     df = replace_nan(df)
#     df_styled = df.style.applymap(get_color)
#     styled_html = df_styled.to_html(index=False, header=False, classes=['minimal-table'])
#     return styled_html
#
#
# def create_color_table(df):
#     df = replace_nan(df)
#     colors = []
#     for _, row in df.iterrows():
#         color_row = []
#         for status in row.values:
#             color = get_color(status)
#             color_row.append(color)
#         colors.append(color_row)
#     return np.array(colors)
#
# def visualize_color_table(df):
#     colors = create_color_table(df)
#     fig = go.Figure(data=[go.Heatmap(
#         z=colors,
#         colorscale=[[0, "#FFFFFF"], [1, "#000000"]],
#         showscale=False,
#         hoverongaps=False,
#         hoverinfo="skip",
#   )])
#
#     fig.update_layout(width=1200, height=600)
#     fig.show()
#
# def create_heatmap_plot(df):
#     # Create color table
#     color_table = create_color_table(df)
#     # Create ColumnDataSource
#     source = ColumnDataSource(data=dict(
#         image=[color_table],
#         x=[0],
#         y=[0],
#         dw=[len(df.columns)],
#         dh=[len(df.index)]
#     ))
#     # Set up figure
#     p = figure(title="Disease-Drug Heatmap", width=800, height=600, toolbar_location=None, tools="hover")
#     # Add heatmap image
#     image = p.image(image='image', x='x', y='y', dw='dw', dh='dh',
#                     source=source)
#     # Customize appearance
#     p.xgrid.grid_line_color = None
#     p.y_range.start = 0
#     p.y_range.end = len(df.index)
#
#     # Set grid line properties for both axes
#     p.xaxis.major_tick_line_color = None
#     p.xaxis.minor_tick_line_color = None
#     p.xaxis.major_label_text_font_size = '0pt'
#     p.xaxis.axis_line_color = None
#
#     p.yaxis.major_tick_line_color = None
#     p.yaxis.minor_tick_line_color = None
#     p.yaxis.major_label_text_font_size = '0pt'
#     p.yaxis.axis_line_color = None
#
#     # Remove top and right borders
#     p.border_fill_alpha = 0
#     p.outline_line_width = 0
#     color_bar = ColorBar(color_mapper=factor_cmap('image', palette=Category10,
#                                                    factors=list(set([status for row in color_table for status in row])),
#                                                    end=1),
#                            location=(0, 0),
#                            title="Phase Progression")
#     p.add_layout(color_bar, 'right')
#     # Add tooltips
#     tooltips = [
#         ("Disease", "@index"),
#         ("Drug", "@columns"),
#         ("Status", "@image")
#     ]
#     p.add_tools(HoverTool(tooltips=tooltips))
#     return p


def analyze_medical_profiles():
    """
        Main function to analyze medical profiles by creating disease and drug heatmaps,
        pivoting data, and saving the result for visualization.

        It orchestrates the overall analysis flow and saves a combined CSV file for final visualization.
        """
    diseases_heatmap, disease_order = create_diseases_heatmap('ind_name', 'medical_profile')
    drugs_heatmap, drugs_order = create_drugs_heatmap('DrugBank Accession Number', 'medical_profile')
    repodb_df = create_pivoted_dataframe("repodb.csv")
    disease_order = disease_order + [i for i in repodb_df.index.values if i not in disease_order]
    drugs_order = drugs_order + [i for i in repodb_df.columns.values if i not in drugs_order]
    # saves the df to csv, later the final visualization is done using R
    reorder_df(repodb_df, disease_order, drugs_order).to_csv("repodb_for_final_visualization_medical_profile_2.csv")


if __name__ == '__main__':
    # create_diseases_heatmap('ind_name', 'medical_profile')
    # create_drugs_heatmap('DrugBank Accession Number', 'medical_profile')
    analyze_medical_profiles()
    pass

