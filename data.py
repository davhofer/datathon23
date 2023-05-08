import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, KNNImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn import linear_model
from datetime import datetime, timedelta

"""
This file contains functions that are used to load and clean the data.
"""


def impute_missing_values(df, strategy, add_indicator_flag=True, missing_values=np.nan):
    """  
    This function imputes missing values in a dataframe using either MICE or KNN imputation.

    Parameters:
    df (pandas dataframe): dataframe to be imputed
    strategy (str): imputation strategy, either 'mice' or 'knn'
    add_indicator_flag (bool): whether to add a column indicating whether a value was imputed
    missing_values (int, float, str, np.nan): value to be treated as missing
    """
    orig_df = df.copy()
    df = df[['distance', 'uphill', 'downhill', 'duration',
        'temperature', 'avg_hr', 'max_hr', 'power']].copy()
    df = df.select_dtypes(include=np.number)

    indicator_cols = {}
    if add_indicator_flag:
        for col in df.columns:
            imputed_flag = df[col].isna().astype(int)
            if imputed_flag.sum() > 0:
                # df[col + '_imputed_flag'] = imputed_flag
                indicator_cols[col] = imputed_flag


                
    
    if strategy == 'mice':
        mice_imputer = IterativeImputer(estimator=linear_model.BayesianRidge(), n_nearest_features=None, imputation_order='ascending', missing_values=missing_values)
        df_imputed = pd.DataFrame(mice_imputer.fit_transform(df), columns=df.columns)
    elif strategy == 'knn':
        knn_scaler = MinMaxScaler(feature_range=(0, 1))
        df = pd.DataFrame(knn_scaler.fit_transform(df), columns = df.columns)
        knn_imputer = KNNImputer(n_neighbors=5, weights='uniform', metric='nan_euclidean', missing_values=missing_values)
        df_imputed = pd.DataFrame(knn_imputer.fit_transform(df), columns=df.columns)
    else:
        print("'strategy' should be either 'mice' or 'knn'")
        return None
    
    for col in df_imputed.columns:
        orig_df[col] = df_imputed[col]
        if col in indicator_cols.keys():
            orig_df[col + '_imputed_flag'] = indicator_cols[col]

    return orig_df


def remove_heartrate(df):
    """
    This function removes heartrate values that are not physiologically/mathematically possible.
    """

    df = df.copy()
    
    def hr_crit(row):
        if row['avg_hr'] > 220:
            return np.nan
        if row['avg_hr'] < 40:
            return np.nan 
        if not np.isnan(row['max_hr']):
            if row['avg_hr'] > row['max_hr']:
                 return np.nan
        
        return row['avg_hr']

    df['avg_hr'] = df.apply(hr_crit, axis=1)

    return df

def haversine_distance(lat1, lon1, lat2, lon2):
    import numpy as np
    R = 6371  # radius of the Earth in kilometers

    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)

    a = np.sin(delta_phi / 2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    d = R * c
    return d*1000


def additional_features(df_run_logs):
    import pandas as pd
    import numpy as np
    df_run_logs["time"] = pd.to_datetime(df_run_logs["time"])
    df_run_logs.sort_values(by=['training_id', 'time'], inplace=True)

    hr_filter = np.logical_and(df_run_logs['hr'] >= 40, df_run_logs['hr'] <= 220)
    df_run_logs['hr'] = hr_filter.astype(int) * df_run_logs['hr']
    df_run_logs['hr'] = df_run_logs['hr'].replace(0, np.nan)
    
    df_run_logs['slat'] = df_run_logs['lat'].shift(1)
    df_run_logs['slng'] = df_run_logs['lng'].shift(1)
    df_run_logs['stime'] = df_run_logs['time'].shift(1)
    df_run_logs['straining_id'] = df_run_logs['training_id'].shift(1)
    mask = df_run_logs['straining_id'] != df_run_logs['training_id']
    df_run_logs.loc[mask, ['slat', 'slng', 'stime']] = np.nan

    # calculate the distance between lat, lng and slat, slng
    lat = df_run_logs['lat']
    lng = df_run_logs['lng']
    slat = df_run_logs['slat']
    slng = df_run_logs['slng']
    time = df_run_logs['time']
    stime = df_run_logs['stime']

    # Use vectorized operations to compute 'distance', 'duration', and 'speed'
    df_run_logs = df_run_logs.assign(
        distance=haversine_distance(lat, lng, slat, slng),
        duration=(time - stime).dt.total_seconds(),
    )
    df_run_logs['speed'] = df_run_logs['distance'] / df_run_logs['duration']

    # rolling distance of the last 30 seconds
    df_run_logs['cumsum_duration'] = df_run_logs.groupby('training_id')['duration'].cumsum()
    # chunk into 30 second intervals
    df_run_logs['interval'] = df_run_logs['cumsum_duration'] // 30
    # groupby interval and training_id
    speed = df_run_logs.groupby(['interval', 'training_id'])[["distance", "duration", "speed"]].mean()
    # reset index
    speed = speed.reset_index()
    speed = speed.groupby('training_id').aggregate({'speed': ['min', 'max', 'std', 'mean', 'median']})
    # rename columns with min,max,std,mean, append _30s to each column name
    speed.columns = [f'{i}_{j}_30s' for i,j in speed.columns]
    speed['skewness_30s'] = df_run_logs.groupby('training_id')['speed'].skew()

    hr_agg = ['min', 'max', 'std', 'mean']
    time_agg = ['first','last']
    lat_agg = ['mean','max','std','min']
    lng_agg = ['mean','max','std','min']
    ele_agg = ['mean','max','std','min']

    aggregates = {'hr': hr_agg, 'time': time_agg, 'lat': lat_agg, 'lng': lng_agg,'ele': ele_agg}
    print(aggregates)

    add_features = df_run_logs.groupby('training_id').aggregate(aggregates)
    add_features.columns = ['_'.join(col) for col in add_features.columns]

    add_features = add_features.merge(speed, on='training_id', how='left')
    return add_features


def remove_outliers(df_trainings):
    categories = ['distance', 'duration', 'uphill', 'downhill']

    remove_ids = set()

    for c in categories:
        arr = df_trainings[c].values

        lower = np.percentile(arr, 1)
        upper = np.percentile(arr, 99)
        df_remove = df_trainings[(df_trainings[c] < lower) & (df_trainings[c] > upper)]
        remove_ids.update(df_remove['training_id'].values)

    df_trainings = df_trainings[~df_trainings['training_id'].isin(remove_ids)]
    return df_trainings


def compute_more_features(features):
    # compute diff of avg hr for each run to the overall max hr for the user
    maxes = features[['user_id', 'hr_max']].groupby('user_id').max()
    features['hr_diff_to_overall_max'] = features.apply(lambda row: maxes['hr_max'][row['user_id']] - row['hr_mean'], axis=1)

    return features


def data_preprocessing(df_trainings, df_additional_features, remove_type=True):

    df_trainings = df_trainings.copy()
    df_additional_features = df_additional_features.copy()

    df_trainings = remove_heartrate(df_trainings)
    df_trainings = remove_outliers(df_trainings)

    df_trainings['start_date'] = pd.to_datetime(df_trainings['start_date'])

    df_trainings['month'] = df_trainings['start_date'].dt.month
    df_trainings['weekday'] = df_trainings['start_date'].dt.weekday
    df_trainings['hour'] = df_trainings['start_date'].dt.hour

    
    df_trainings = impute_missing_values(df_trainings, 'mice', add_indicator_flag=True)






    for c in df_additional_features.columns:
        # check if column is not numeric
        if not np.issubdtype(df_additional_features[c].dtype, np.number):
            if c == 'training_id':
                continue
            # drop column
            df_additional_features.drop(c, axis=1, inplace=True)

    df_trainings = df_trainings.merge(df_additional_features, on='training_id')

    df_trainings = compute_more_features(df_trainings)


    # TODO: recent runs


    def recent_runs_feature(df):
        interesting_cols = ['weekday','distance', 'duration', 'speed_std_30s', 'speed_mean_30s', 'hr_diff_to_overall_max', 'hr_std', 'hr_mean', 'ele_std']

        print("count recent run types")
        for t in df['type'].unique():
            print(t)
            if pd.isnull(t):
                continue 

            def do2(row):
                user_id = row['user_id']
                timestamp = row['start_date']
                tmp = df[df['user_id'] == user_id]
                tmp = tmp[tmp['start_date'] < timestamp]
                tmp = tmp[tmp['start_date'] >= timestamp - pd.Timedelta(days=7)].copy()

                # approach 2:
                n = len(tmp[tmp['type'] == t])
                return n
            df['recent_' + str(t) + '_count'] = df.apply(do2, axis=1)

        print("aggregate recent runs")
        for c in interesting_cols:
            print(c)
            def do(row):
                user_id = row['user_id']
                timestamp = row['start_date']
                tmp = df[df['user_id'] == user_id]
                tmp = tmp[tmp['start_date'] < timestamp]
                tmp = tmp[tmp['start_date'] >= timestamp - pd.Timedelta(days=7)].copy()

                # approach 1:
                tmp = tmp[interesting_cols]
                return tmp.mean()[c]
            df['recent_runs_' + c] = df.apply(do, axis=1)

        return df
    
    df_trainings = recent_runs_feature(df_trainings)


    ############################
    # use custom metric as feature
    print("custom metric!")
    df_trainings['day'] = pd.to_datetime(df_trainings['start_date']).dt.dayofyear
    df_trainings['days_discount'] = 365 - df_trainings['day']
    df_trainings = df_trainings.sort_values(by=['user_id', 'days_discount'], ascending=False)

    def calculate_hr_split_score(df_merged, training_id):
        run = df_merged[df_merged['training_id'] == training_id]
        user_id = run['user_id'].values[0]
        # get the max heart rate of that person
        max_hr = df_merged[df_merged['user_id'] == user_id]['hr_max'].max()
        max_hr = max_hr
        mean_hr = run['hr_mean'].values[0]
        hr_dist = max_hr - mean_hr
        split_avg = 1/run['speed_mean_30s'].values[0] * 50 / 3
        return hr_dist/split_avg/5
    

    df_trainings['hr_split_score'] = 0
    df_trainings['duration_score'] = df_trainings['duration'] / 60  / 60
    for training_id in df_trainings['training_id'].unique():
        hr_split = calculate_hr_split_score(df_trainings, training_id)
        df_trainings.loc[df_trainings['training_id'] == training_id, 'hr_split'] = hr_split


    factor = 0.995
    # fill na with mean of user
    df_trainings['hr_split'] = df_trainings.groupby('user_id')['hr_split'].fillna(df_trainings['hr_split'].mean())
    df_trainings['duration_score'] = df_trainings.groupby('user_id')['duration_score'].fillna(df_trainings['duration_score'].mean())
    # for every person change the first fitness to 0
    df_trainings['hr_split'] = df_trainings.groupby('user_id')['hr_split'].shift(1)
    df_trainings['duration_score'] = df_trainings.groupby('user_id')['duration_score'].shift(1)
    # for every person fill the NaN with 25% percentile
    df_trainings['hr_split'] = df_trainings.groupby('user_id')['hr_split'].apply(lambda x: x.fillna(x.quantile(0.5)*67))
    df_trainings['duration_score'] = df_trainings.groupby('user_id')['duration_score'].apply(lambda x: x.fillna(x.quantile(0.5)*67))

    df_trainings['hr_split_discounted'] = df_trainings['hr_split'] * factor**df_trainings['days_discount']
    df_trainings['hr_split_discounted_cumsum'] = df_trainings.groupby('user_id')['hr_split_discounted'].cumsum()
    
    df_trainings['duration_score_discounted'] = df_trainings['duration_score'] * factor**df_trainings['days_discount']
    df_trainings['duration_score_discounted_cumsum'] = df_trainings.groupby('user_id')['duration_score_discounted'].cumsum()
    

    df_trainings['fitness_hr_split'] = df_trainings['hr_split_discounted_cumsum'] / factor**df_trainings['days_discount']
    df_trainings['fitness_duration'] = df_trainings['duration_score_discounted_cumsum'] / factor**df_trainings['days_discount']
    df_trainings['difference'] = df_trainings['fitness_hr_split'] - df_trainings['fitness_duration']
    ############################

    cols = ['day', 'days_discount', 'hr_split_score', 'duration_score', 'hr_split_score', 'hr_split', 'hr_split_discounted', 'hr_split_discounted_cumsum', 'duration_score_discounted', 'duration_score_discounted_cumsum', 'fitness_hr_split', 'fitness_duration', 'difference']
    for c in cols:
        if c in df_trainings.columns:
            df_trainings.drop(c, axis=1, inplace=True)


    # drop column start_date from df_trainings
    df_trainings.drop('start_date', axis=1, inplace=True)

    
    # df_trainings.drop('training_id', axis=1, inplace=True)

    if remove_type:
        df_trainings.drop('type', axis=1, inplace=True)

    return df_trainings



def split_by_user(df_features):
    split = []
    for uid in df_features['user_id'].unique():
        df = df_features[df_features['user_id'] == uid].copy()
        df.drop('user_id', axis=1, inplace=True)
        split.append((uid, df))

    return split


def train_test_split(df_features, test_size=0.02):
    n = len(df_features)

    uids = list(df_features['user_id'].unique())

    # df_features['user_id'] = df_features['user_id'].apply(lambda x: uids.index(x))
    df_features = df_features[~df_features['type'].isna()]
    
    n_test = int(n * test_size)
    n_test_balanced = n_test // 2

    n_classes = len(df_features['type'].unique())
    n_per_class = n_test_balanced // n_classes

    print(f"sampling {n_per_class} samples per class")

    # prepare dataframe for combining samples
    df_test_balanced = pd.DataFrame(columns=df_features.columns)
    for label in df_features['type'].unique():
        class_samples = len(df_features[df_features['type'] == label])
        print(label, class_samples)
        # sample n_per_class for each label
        df_sample = df_features[df_features['type'] == label].sample(min(n_per_class, class_samples//3))
        df_features = df_features[~df_features.index.isin(df_sample.index)]
        # aggregate df_sample in df_test_balanced
        df_test_balanced = pd.concat([df_test_balanced, df_sample])

    n_test_balanced = len(df_test_balanced)
    n_test_sampled = n_test_balanced

    df_test_sampled = df_features.sample(n_test_sampled)
    df_features = df_features[~df_features.index.isin(df_test_sampled.index)]
    df_train = df_features

    return df_train, df_test_sampled, df_test_balanced, uids



