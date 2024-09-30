import numpy as np
import pandas as pd
import sklearn
from recommendation_system import RecommendationSystem
from recommendation_of_disease_treatment_by_drug import *
from analyze_profiles import *
import matplotlib.pyplot as plt
import seaborn as sns
def misclassification_error(y_pred: np.ndarray, y_true: np.ndarray, normalize: bool = True) -> float:
    """
    Calculate misclassification loss

    Parameters
    ----------
    y_true: ndarray of shape (n_samples, )
        True response values
    y_pred: ndarray of shape (n_samples, )
        Predicted response values
    normalize: bool, default = True
        Normalize by number of samples or not

    Returns
    -------
    Misclassification of given predictions
    """
    loss = np.sum(np.abs(y_true)[np.sign(y_true) != np.sign(y_pred)], axis=0)
    if normalize:
        return loss / y_true.shape[0]
    return loss


def adjust_to_rating_of_status(rating):
    if rating >= 11/12:
        return status_to_rating_map['Approved']
    rating_of_statuses = np.array(list(status_to_rating_map.values()))
    abs_distances = np.abs(rating_of_statuses - rating)
    closest_index = abs_distances.argmin()
    return rating_of_statuses[closest_index]

def profiles_form_ratings(data):
    preprocessed_entries_data = preprocess_entries(data)
    rating_matrix = compute_utility_matrix(preprocessed_entries_data)
    return rating_matrix

def calculate_ratings_dist(df):
    df = df.fillna(0)
    df = df.sub(df.mean(axis=1), axis=0)
    dist_mat = pd.DataFrame(pairwise_distances(df, metric='cosine'), index=df.index, columns=df.index)
    return dist_mat




def evaluate_recommendation():
    data = load_data('repodb_for_visualization_medical.csv')  # V
    train_set, test_set = preprocess(data)  # V??
    recommendation_system = RecommendationSystem()  # V
    user_medical_profiles = load_medical_diseases_profiles()
    item_medical_profiles = load_medical_drugs_profiles()
    item_wiki_profiles, user_wiki_profiles  = load_wiki_profiles("wiki_data")
    user_rating_profiles = profiles_form_ratings(data)
    item_rating_profiles = profiles_form_ratings(data).T
    # profiles_and_dist_func_dict = {'user, wiki profiles': (user_wiki_profiles,calculate_wiki_dist)}
    profiles_and_dist_func_dict = {'user, rating matrix': (user_rating_profiles, calculate_ratings_dist),
                                   'user, wiki profiles': (user_wiki_profiles,calculate_wiki_dist),
                                   'user, medical profiles': (user_medical_profiles, calculate_disease_dist)}
                              # 'item, medical profiles': (item_medical_profiles,calculate_drug_dist),
                              #  ,
                              # 'item, wiki profiles': (item_wiki_profiles,calculate_wiki_dist)}
    precisiosns = []
    recalls = []
    misclassification_errors = []
    models_order = []
    for k in [5, 10]:
        for name,profiles_and_dist_func in profiles_and_dist_func_dict.items():
            if name == 'user, medical profiles':
                train_set = train_set[train_set['User'].isin(profiles_and_dist_func[0]['ind_name'])]
                test_set = test_set[test_set['User'].isin(profiles_and_dist_func[0]['ind_name'])]
            elif name == 'user, wiki profiles':
                train_set = train_set[train_set['User'].isin(profiles_and_dist_func[0].index)]
                test_set = test_set[test_set['User'].isin(profiles_and_dist_func[0].index)]
            predictions = recommendation_system.predict(train_set, test_set, profiles_and_dist_func[0], profiles_and_dist_func[1], k)
            predictions['Rating'] = predictions['Rating'].apply(lambda rating: adjust_to_rating_of_status(rating))
            preds_for_recall = np.where(predictions['Rating'] == status_to_rating_map['Approved'], 1, 0)
            test_for_recall = np.where(test_set['Rating'] == status_to_rating_map['Approved'], 1, 0)
            precisiosns.append(sklearn.metrics.precision_score(test_for_recall, preds_for_recall, pos_label=1))
            recalls.append(sklearn.metrics.recall_score(test_for_recall, preds_for_recall, pos_label=1))
            misclassification_errors.append(misclassification_error(test_set['Rating'], predictions['Rating'], normalize=True))
            models_order.append(name+" k="+str(k))
    df_recall = pd.DataFrame({'Model': models_order, 'Metric': ['Recall']*len(recalls), 'Value': recalls})
    df_precision = pd.DataFrame({'Model': models_order, 'Metric': ['Precision']*len(precisiosns), 'Value': precisiosns})
    df_misclass = pd.DataFrame({'Model': models_order, 'Metric': ['Misclassification Error']*len(misclassification_errors),'Value': misclassification_errors})

    # fig, axes = plt.subplots(1, 1, figsize=(12, 8))
    #
    # sns.scatterplot(data=df_combined, x='Model', y='Value', hue='Metric', style='Metric', ax=axes)
    # for metric in df_combined['Metric'].unique():
    #     metric_df = df_combined[df_combined['Metric'] == metric]
    #     axes.plot(metric_df['Model'], metric_df['Value'], label=metric, linewidth=2)
    # handles, labels = axes.get_legend_handles_labels()
    # axes.legend(handles=handles[:3], title='Metric', bbox_to_anchor=(1.05, 1), loc='upper left')
    # axes.tick_params(axis='x', labelrotation=45)
    # axes.set_title('Model Performance Metrics')
    # axes.set_xlabel('Model')
    # axes.set_ylabel('Value')
    # plt.tight_layout()
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    sns.barplot(data=df_recall, x='Metric', y='Value', hue='Model', ax=axes[0])
    axes[0].set_title('Recall')
    axes[0].set_ylabel('Value')
    axes[0].set_xlabel('Model')
    sns.barplot(data=df_precision, x='Metric', y='Value', hue='Model', ax=axes[1])
    axes[1].set_title('Precision')
    axes[1].set_ylabel('Value')
    axes[1].set_xlabel('Model')
    sns.barplot(data=df_misclass, x='Metric', y='Value', hue='Model', ax=axes[2])
    axes[2].set_title('Misclassification Error')
    axes[2].set_ylabel('Value')
    axes[2].set_xlabel('Model')
    for ax in axes[:2]:
        ax.get_legend().set_visible(False)
    plt.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    evaluate_recommendation()




