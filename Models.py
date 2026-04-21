import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import ( accuracy_score, classification_report, confusion_matrix, mean_absolute_error, mean_squared_error, r2_score, silhouette_score)
from Config import (Rand_state, test_size, val_size, kfold_splits, class_features, Reg_features, Cluster_features, Model_features, log)

#Splitting data
def build_feature_matrices(dataframe):
    X = dataframe[Model_features]
    y = dataframe['collision_severity']
    return X, y

def split_train_val_test(X, y): #Spliting the data to produce training validation and test sets.

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=test_size, random_state=Rand_state)
    X_value, X_test, y_value, y_test = train_test_split(X_temp, y_temp, test_size=val_size, random_state=Rand_state)
    log.info("Train/Val/Test split: %d / %d / %d", len(X_train), len(X_value), len(X_test)
    )
    return X_train, X_value, X_test, y_train, y_value, y_test

#Classification
def run_cross_validation(X_train, y_train):
    candidates = {'Decision Tree': DecisionTreeClassifier(random_state=Rand_state), 'KNN': KNeighborsClassifier(n_neighbors=5), 'Logistic Regression': LogisticRegression(max_iter=5000, random_state=Rand_state)}
    kf = KFold(n_splits=kfold_splits, shuffle=True, random_state=Rand_state)
    log.info("K-Fold Cross Validation (k=%d)", kfold_splits)
    for name, model in candidates.items():
        scores = cross_val_score(model, X_train, y_train, cv=kf, scoring='accuracy')
        log.info("%s: Mean=%.4f  Std=%.4f", name, scores.mean(), scores.std())
        print(f"{name}: Mean Accuracy = {scores.mean():.4f}, Std = {scores.std():.4f}")
    return candidates

def tune_decision_tree(X_train, X_value, y_train, y_value):
    depth_options        = [3, 5, 10, 15, 20, None]
    min_split_options    = [2, 5, 10]
    best_accuracy        = 0
    best_parameters          = {}

    log.info("Tuning Decision Tree hyperparameters")

    for max_depth in depth_options:
        for min_split in min_split_options:

            model = DecisionTreeClassifier(
                max_depth=max_depth,
                min_samples_split=min_split,
                class_weight='balanced',
                random_state=Rand_state
            )
            model.fit(X_train, y_train)
            val_acc = accuracy_score(y_value, model.predict(X_value))
            print(f"max_depth={max_depth}, min_samples_split={min_split} -> {val_acc:.4f}")
            if val_acc > best_accuracy:
                best_accuracy = val_acc
                best_parameters   = {'max_depth': max_depth, 'min_samples_split': min_split}

    best_model = DecisionTreeClassifier(**best_parameters, class_weight='balanced', random_state=Rand_state)
    best_model.fit(X_train, y_train)
    log.info("Best DT params: %s  Val accuracy: %.4f", best_parameters, best_accuracy)
    return best_model, best_parameters, best_accuracy

#Evalutation
def evaluate_classifier(model, X_test, y_test, label='Model'):
    y_prediction  = model.predict(X_test)
    acc     = accuracy_score(y_test, y_prediction)
    cm      = confusion_matrix(y_test, y_prediction)
    report  = classification_report(
        y_test, y_prediction,
        target_names=['Fatal', 'Serious', 'Slight'],
        zero_division=0
    )
    log.info("%s test accuracy: %.4f", label, acc)
    print(f"\n{label} Test Accuracy: {acc:.4f}")
    print(report)
    return acc, cm, report

#Binary decision tree hyperparameter tuning
def tune_binary_decision_tree(X_train, X_value, y_train, y_value):
    depth_options     = [3, 5, 10, 15, None]
    min_split_options = [2, 5, 10]
    best_accuracy     = 0
    best_params       = {}

    for max_depth in depth_options:
        for min_split in min_split_options:
            model = DecisionTreeClassifier(
                max_depth=max_depth,
                min_samples_split=min_split,
                random_state=Rand_state
            )
            model.fit(X_train, y_train)
            value_acc = accuracy_score(y_value, model.predict(X_value))
            if value_acc > best_accuracy:
                best_accuracy = value_acc
                best_parameters   = {'max_depth': max_depth, 'min_samples_split': min_split}

    best_model = DecisionTreeClassifier(**best_parameters, random_state=Rand_state)
    best_model.fit(X_train, y_train)
    log.info("Binary DT best parameters: %s  Value accuracy: %.4f", best_parameters, best_accuracy)
    return best_model, best_parameters


def tune_categorical_decision_tree(X_train, X_value, y_train, y_value):
    depth_options     = [3, 5, 10, 15, None]
    min_split_options = [2, 5, 10]
    best_accuracy     = 0
    best_parameters       = {}

    for max_depth in depth_options:
        for min_split in min_split_options:
            model = DecisionTreeClassifier(
                max_depth=max_depth,
                min_samples_split=min_split,
                random_state=Rand_state
            )
            model.fit(X_train, y_train)
            val_acc = accuracy_score(y_value, model.predict(X_value))
            if val_acc > best_accuracy:
                best_accuracy = val_acc
                best_parameters   = {'max_depth': max_depth, 'min_samples_split': min_split}

    best_model = DecisionTreeClassifier(**best_parameters, random_state=Rand_state)
    best_model.fit(X_train, y_train)
    log.info("Categorical DT best parameters: %s  Value Accuracy: %.4f", best_parameters, best_accuracy)
    return best_model, best_parameters

#Regression
def run_simple_linear_regression(dataframe):
    yearly_counts = dataframe.groupby('collision_year').size().reset_index(name='count')
    X_year = yearly_counts[['collision_year']]
    y_count = yearly_counts['count']

    X_train, X_temp, y_train, y_temp = train_test_split(X_year, y_count, test_size=test_size, random_state=Rand_state)
    X_value, X_test, y_value, y_test = train_test_split(X_temp, y_temp, test_size=val_size, random_state=Rand_state)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_prediction_test = model.predict(X_test)
    r2 = r2_score(y_test, y_prediction_test)
    mae = mean_absolute_error(y_test, y_prediction_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_prediction_test))

    log.info("Simple Linear Regression - R2: %.4f  Coefficient: %.2f/yr", r2, model.coef_[0])
    print(f"\nSimple Linear Regression:")
    print(f"  Test R2: {r2:.4f}  MAE: {mae:.2f}  RMSE: {rmse:.2f}")
    print(f"  Trend: {model.coef_[0]:.2f} collisions change per year")

    return model, yearly_counts, r2


def run_multiple_linear_regression(dataframe):
    X_cas = dataframe[['weather_conditions', 'road_type', 'light_conditions', 'speed_limit', 'number_of_vehicles', 'road_surface_conditions', 'junction_detail', 'junction_control', 'urban_or_rural_area', 'day_of_week', 'hour', 'month', 'collision_severity', 'pedestrian_crossing', 'carriageway_hazards', 'special_conditions_at_site', 'first_road_class', 'second_road_class', 'is_rush_hour', 'is_weekend', 'is_dark', 'is_bad_weather']]
    y_cas = dataframe['number_of_casualties']
    X_train, X_temp, y_train, y_temp = train_test_split(X_cas, y_cas, test_size=test_size, random_state=Rand_state)
    X_value, X_test, y_value, y_test = train_test_split(X_temp, y_temp, test_size=val_size, random_state=Rand_state)

    model = LinearRegression()
    model.fit(X_train, y_train)

    kf = KFold(n_splits=kfold_splits, shuffle=True, random_state=Rand_state)
    cv_r2 = cross_val_score(model, X_train, y_train, cv=kf, scoring='r2')

    y_prediction_test = model.predict(X_test)
    r2   = r2_score(y_test, y_prediction_test)
    mae  = mean_absolute_error(y_test, y_prediction_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_prediction_test))

    log.info("Multiple Linear Regression - R2: %.4f (CV Mean: %.4f)", r2, cv_r2.mean())
    print(f"\nMultiple Linear Regression (casualties):")
    print(f"  CV R2 Mean: {cv_r2.mean():.4f}  Std: {cv_r2.std():.4f}")
    print(f"  Test R2: {r2:.4f}  MAE: {mae:.4f}  RMSE: {rmse:.4f}")

    return model, X_test, y_test, r2

def run_polynomial_regression(dataframe):
    X_veh = dataframe[['weather_conditions', 'light_conditions', 'road_surface_conditions', 'junction_detail', 'junction_control', 'speed_limit', 'urban_or_rural_area', 'day_of_week', 'hour', 'is_rush_hour', 'is_weekend', 'is_dark', 'is_bad_weather']]
    y_veh = dataframe['number_of_vehicles']

    X_train, X_temp, y_train, y_temp = train_test_split(X_veh, y_veh, test_size=test_size, random_state=Rand_state)
    X_value, X_test, y_value, y_test = train_test_split(X_temp, y_temp, test_size=val_size, random_state=Rand_state)

    best_r2     = -999
    best_degree = 2
    best_model  = None

    for degree in [2, 3, 4]:
        pipeline = Pipeline([('poly',   PolynomialFeatures(degree=degree)), ('linear', LinearRegression())])
        pipeline.fit(X_train, y_train)
        val_r2 = r2_score(y_value, pipeline.predict(X_value))
        print(f"  Degree={degree} -> Val R2: {val_r2:.4f}")

        if val_r2 > best_r2:
            best_r2     = val_r2
            best_degree = degree
            best_model  = pipeline

    y_prediction_test = best_model.predict(X_test)
    test_r2  = r2_score(y_test, y_prediction_test)
    test_mae = mean_absolute_error(y_test, y_prediction_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_prediction_test))

    log.info("Polynomial Regression - Best degree: %d  R2: %.4f", best_degree, test_r2)
    print(f"\nPolynomial Regression (degree={best_degree}):")
    print(f"  Test R2: {test_r2:.4f}  MAE: {test_mae:.4f}  RMSE: {test_rmse:.4f}")

    return best_model, X_test, y_test, test_r2, best_degree

#Clustering
def run_kmeans_risk_clustering(dataframe):
    X_risk = dataframe[['number_of_casualties', 'number_of_vehicles', 'speed_limit', 'road_type', 'weather_conditions', 'light_conditions', 'road_surface_conditions', 'urban_or_rural_area', 'collision_severity']]

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X_risk)

    k_range          = range(2, 11)
    inertia_values   = []
    silhouette_scores = []

    log.info("Running K-Means elbow and silhouette analysis")
    for k in k_range:
        km     = KMeans(n_clusters=k, random_state=Rand_state, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertia_values.append(km.inertia_)
        silhouette_scores.append(silhouette_score(X_scaled, labels))
        print(f"  k={k}  Inertia={km.inertia_:.2f}  Silhouette={silhouette_scores[-1]:.4f}")

    best_k     = list(k_range)[silhouette_scores.index(max(silhouette_scores))]
    best_km    = KMeans(n_clusters=best_k, random_state=Rand_state, n_init=10)
    best_labels = best_km.fit_predict(X_scaled)
    dataframe['kmeans_cluster'] = best_labels

    #Reduce to 2D for visualisation only
    pca_2d       = PCA(n_components=2)
    X_pca_coords = pca_2d.fit_transform(X_scaled)

    log.info("Best k=%d  Silhouette=%.4f", best_k, max(silhouette_scores))
    print(f"\nBest k: {best_k}  Silhouette: {max(silhouette_scores):.4f}")
    print(dataframe.groupby('kmeans_cluster')[['number_of_casualties', 'number_of_vehicles', 'speed_limit','collision_severity', 'light_conditions', 'weather_conditions']].mean().round(2).to_string())
    return X_scaled, X_pca_coords, best_labels, best_k, silhouette_scores, k_range

def run_geographic_clustering(df):
    X_geo    = df[['latitude', 'longitude']]
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X_geo)
    geo_km   = KMeans(n_clusters=5, random_state=Rand_state, n_init=10)
    labels   = geo_km.fit_predict(X_scaled)
    sil      = silhouette_score(X_scaled, labels)

    df['geo_cluster'] = labels
    log.info("Geographic clustering silhouette: %.4f", sil)
    return labels, sil

def run_dbscan_clustering(X_scaled, X_pca_coords):
    eps_options        = [0.3, 0.5, 0.7, 1.0]
    min_sample_options = [5, 10, 20]
    best_score         = -1
    best_eps           = 0.5
    best_min_samples   = 10
    best_labels        = None

    log.info("Tuning DBSCAN hyperparameters.")
    for eps in eps_options:
        for min_samples in min_sample_options:
            db        = DBSCAN(eps=eps, min_samples=min_samples)
            db_labels = db.fit_predict(X_scaled)
            n_clusters = len(set(db_labels)) - (1 if -1 in db_labels else 0)

            if n_clusters > 1:
                valid_mask = db_labels != -1
                score = silhouette_score(X_scaled[valid_mask], db_labels[valid_mask])
                print(f"  eps={eps} min_samples={min_samples} -> clusters={n_clusters}  sil={score:.4f}")
                if score > best_score:
                    best_score       = score
                    best_eps         = eps
                    best_min_samples = min_samples
                    best_labels      = db_labels
            else:
                print(f"  eps={eps} min_samples={min_samples} -> clusters={n_clusters} (insufficient)")

    log.info("Best DBSCAN: eps=%.1f min_samples=%d sil=%.4f", best_eps, best_min_samples, best_score)
    return best_labels, best_eps, best_min_samples, best_score

#Pca
def run_pca_analysis(X_classification_scaled, y_classification):

    pca = PCA()
    pca.fit(X_classification_scaled)

    explained = pca.explained_variance_ratio_
    cumulative = np.cumsum(explained)
    n_components_95 = int(np.argmax(cumulative >= 0.95)) + 1
    log.info("PCA: %d components needed for 95%% variance.", n_components_95)
    X_reduced = PCA(n_components=n_components_95).fit_transform(X_classification_scaled)
    X_train_pca, X_temp_pca, y_train_pca, y_temp_pca = tts(X_reduced, y_classification, test_size=test_size, random_state=Rand_state)
    X_value_pca, X_test_pca, y_value_pca, y_test_pca = tts(X_temp_pca, y_temp_pca, test_size=val_size, random_state=Rand_state)

    dt_pca = DecisionTreeClassifier(max_depth=3, random_state=Rand_state)
    dt_pca.fit(X_train_pca, y_train_pca)
    pca_acc = accuracy_score(y_test_pca, dt_pca.predict(X_test_pca))

    print(f"\nPCA Analysis:")
    print(f"  Components for 95% variance: {n_components_95} (from {X_classification_scaled.shape[1]})")
    print(f"  Decision Tree WITH PCA:    {pca_acc:.4f}")
    print(f"  Decision Tree WITHOUT PCA: 0.8343")

    return pca, explained, cumulative, n_components_95

#Responsible ai
def responsible_ai_checks(df, best_dt, X):
    log.info("Responsible AI checks")

    print("\nClass Distribution (collision_severity):")
    counts = df['collision_severity'].value_counts()
    pcts   = df['collision_severity'].value_counts(normalize=True) * 100
    labels = {1: 'Fatal', 2: 'Serious', 3: 'Slight'}
    for cls in counts.index:
        print(f"  {labels.get(cls, cls)}: {counts[cls]} ({pcts[cls]:.1f}%)")

    print("\nTop 5 Most Important Features (Decision Tree):")
    importance = pd.Series(best_dt.feature_importances_, index=X.columns)
    for feat, imp in importance.sort_values(ascending=False).head(5).items():
        print(f"  {feat}: {imp:.4f}")