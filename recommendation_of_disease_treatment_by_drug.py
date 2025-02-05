import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import re
from recommendation_system import RecommendationSystem
from analyze_profiles import calculate_drug_dist, calculate_disease_dist
from analyze_profiles import load_medical_drugs_profiles, load_medical_diseases_profiles

status_to_rating_map = {
    'Approved': 1,
    'Withdrawn (Phase 1)': -1/2, 'Withdrawn (Phase 2)': -2/6, 'Withdrawn (Phase 3)': -1/6,
    'Suspended (Phase 1)': 1/2, 'Suspended (Phase 2)': 4/6, 'Suspended (Phase 3)': 5/6,
    'Terminated (Phase 1)': -1, 'Terminated (Phase 3)': -5/6, 'Terminated (Phase 2)': -4/6,
}

rating_to_status_map = {rating: status for status, rating in status_to_rating_map.items()}


def load_data(file_name):
    data = pd.read_csv(file_name)
    # Set first column to be index
    data = data.set_index(data.columns[0])
    return data

def preprocess(data: DataFrame):
    """
    Preprocess data to feature and labels according to label name

    :param data: Data to preprocess
    :param label_names: labels to split columns of data by
    :return: 4-tuple (train_set_features, train_set_labels, test_set_features, test_set_l) with
        Features and labels for trains set and features and labels for test set.
    """
    preprocessed_entries_data = _preprocess_entries(data)
    utility_matrix = _compute_utility_matrix(preprocessed_entries_data)
    training_set, test_set = split(utility_matrix)
    return training_set, test_set


def _preprocess_entries(data: DataFrame):
    preprocessed_entries = data.copy()
    made_phases_pattern = r"Phase (\d+)/Phase (\d+)"

    for item, item_row in data.iterrows():
        for user in data.columns:
            status = data.loc[item, user]
            # if pd.isna(status): # convert np.nans to Unrated values
            #     # אם הייתי רוצה תהליך הפוך בצורה מושלמת אז כל ערך שהוא nana עבור nan כל ערך שהוא nan לא הייתי נוגע בו...
            #     preprocessed_entries.at[item, user] = 'Unrated'
            # else: המר כל ערך שהוא לא nan באופן הבא:
            if pd.notna(status):
                match = re.search(made_phases_pattern, status)
                if match: # Remove / and later phase from any status containing (Phase x/Phase y)
                    x = int(match.group(1))
                    y = int(match.group(2))
                    status = re.sub(f"/", "", status)
                    status = re.sub(f"Phase {max(x, y)}", "", status)
                status = status.replace("Early ", "")

                if status in status_to_rating_map: #
                    preprocessed_entries.at[item, user] = status
                else: # Unrate any other invalid format key...
                    # כל ערך שהוא לא תקין הייתי הופך ל-nan...
                    #preprocessed_entries.at[item, user] = 'Unrated'
                    preprocessed_entries.at[item, user] = np.nan
    return preprocessed_entries

def _compute_utility_matrix(data: DataFrame):
    """
    Assume there are only valid keys...
    :param data:
    :return:
    """

    return data.applymap(lambda status: status_to_rating_map[status] if pd.notna(status) else status)

def split(utility_matrix: DataFrame , train_proportion=0.8):
    """

    :param data_features:
    :param data_labels:
    :param train_proportion:
    :return: 4-tuple
    """

    utility_matrix_flattened = utility_matrix.stack().reset_index()
    utility_matrix_flattened.columns = ['User', 'Item', 'Rating'] # אני רוצה שuser יהיה מחלה ו-item זה תרופה

    #utility_matrix_flattened = utility_matrix_flattened[utility_matrix_flattened['Rating'] != status_to_rating_map['Unrated']]
    utility_matrix_flattened = utility_matrix_flattened[pd.notna(utility_matrix_flattened['Rating'])]
    train_set = utility_matrix_flattened.sample(frac=train_proportion, random_state=42)
    test_set = utility_matrix_flattened.drop(train_set.index)
    return train_set, test_set

def _compute_drug_to_disease_matrix(utility_matrix: DataFrame):
    return utility_matrix.applymap(lambda rating: rating_to_status_map[rating] if pd.notna(rating) else rating)
    #return data.applymap(lambda status: status_to_rating_map[status])

def postprocess(predictions: DataFrame):
    predictions['Rating'] = predictions['Rating'].apply(lambda rating: _adjust_to_rating_of_status(rating)) #v
    # unified_train_set_and_predictions = pd.concat([train_set, predictions], ignore_index=True)
    # updated_utility_matrix = unified_train_set_and_predictions.pivot(index='User', columns='Item', values='Rating')
    # updated_data = _compute_drug_to_disease_matrix(updated_utility_matrix)
    # return updated_data

def _adjust_to_rating_of_status(rating):
    rating_of_statuses = np.array(list(status_to_rating_map.values()))
    abs_distances = np.abs(rating_of_statuses - rating)
    closest_index = abs_distances.argmin() # הדירוג שממזער את המטריצה
    return rating_of_statuses[closest_index]


def limit_to_items_with_profile(train_set, test_set, item_encoding, item_profiles):
    limited_train_set = train_set[train_set['Item'].isin(item_profiles['DrugBank Accession Number'])]
    limited_test_set = test_set[test_set['Item'].isin(item_profiles['DrugBank Accession Number'])]
    return limited_train_set, limited_test_set

if __name__ == "__main__":
    data = load_data('repodb_for_final_visualization_medical_profile.csv')
    train_set, test_set = preprocess(data)
    recommendation_system = RecommendationSystem()
    recommendation_system.fit() #  אין אימון כרגע... לעדכן.

    # Predict user-profiles
    user_profiles = load_medical_diseases_profiles()
    distance_func_for_users = calculate_disease_dist
    #
    train_set = train_set[train_set['User'].isin(user_profiles['ind_name'])] # TODO: גם כאן, אני לא אמור לדעת מה השם של העמודה של המחלות של פרופיל משתמש מסוים. צריך שאלומה תשנה את זה.
    test_set = train_set[train_set['User'].isin(user_profiles['ind_name'])]
    disease_name_to_disease_name_map = {disease: disease for disease in user_profiles['ind_name']}  # TODO: זה משהו שאלומה צריכה להגדיר אצלה. אני לא מור לדעת מה זה genric name.
    # עבור שגיאה עבור מיפוי אז שגיאה מאז מיפוי... למה שגיאה במיפוי? אין key בשם ind name??
    predictions = recommendation_system.predict(train_set, test_set, user_profiles, distance_func_for_users,
                                                disease_name_to_disease_name_map, k=5)
    postprocess(predictions)


    # item_profiles = load_medical_drugs_profiles()
    # distance_func_for_items = calculate_drug_dist
    # drug_name_to_dbank_number_map = item_profiles.set_index('Generic Name')['DrugBank Accession Number'].to_dict()  # TODO: זה משהו שאלומה צריכה להגדיר אצלה. אני לא מור לדעת מה זה genric name.
    # # TODO: אפשר שתקודד אצלה בתור drug name
    # # TODO: לא אמור לדעת מה זה DrugBankAccessor Number אפשר שתקודד אצלה בתור drug band number...
    # train_set, test_set = limit_to_items_with_profile(train_set, test_set, 'DrugBank Accession Number', item_profiles)
    # predictions_item_item = recommendation_system.predict_item_item(train_set, test_set, item_profiles, distance_func_for_items, drug_name_to_dbank_number_map, k=5)
    # updated_data = postprocess(predictions_item_item)
    # updated_data.to_csv('updated_data.csv')

    # איך נציג את התוצאות?
    # רוצים לעשות השוואה לפי פרופיל...
    # לפי פרופיל ספרציפי של אלומה:
    # 1. לחשב precision ו-recall על מה ש-approved
    # להסתכל על מה ש-false positive שמתעלם משאר הדירוגים בווצאפ
    # אפשר גם לחשב misclassification error על כל התוצאות...
    # לא מדורג בחוץ
    # שתי המדדים לכל אחד מהפרופילים... ואפשר אולי
    # עדן שלך, לפי מחלות ותרופות...
    # 4 - מודלים
    # אם רוצים k=5 או k=10 יכפיל ל-8...

