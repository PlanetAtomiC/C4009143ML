import matplotlib.pyplot as plt
import seaborn
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, silhouette_score
)
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from Config import Rand_state, class_features, log
from Preprocessing import run_preprocessing
from Models import (
    build_feature_matrices,
    split_train_val_test,
    run_cross_validation,
    tune_decision_tree,
    evaluate_classifier,
    tune_binary_decision_tree,
    tune_categorical_decision_tree,
    run_simple_linear_regression,
    run_multiple_linear_regression,
    run_polynomial_regression,
    run_kmeans_risk_clustering,
    run_geographic_clustering,
    run_dbscan_clustering,
    run_pca_analysis,
    responsible_ai_checks,
)

def show_processed_data_in_plots(dataframe):

    #Collisions by year
    plt.figure(figsize=(12,5),)
    seaborn.histplot(data=dataframe, x='collision_year', bins=42)
    plt.title("Collisions in Sheffield by year.")
    plt.xlabel("Collisions by year.")
    plt.ylabel("Year.")
    plt.tight_layout()
    plt.show()

    #Collisions by hour (Shows the rush hour peaks)
    plt.figure(figsize=(12,5))
    seaborn.histplot(data=dataframe, x='hour', bins=24)
    plt.title("Collisions in Sheffield by Hour of Day.")
    plt.xlabel("Hour.")
    plt.ylabel("Count.")
    plt.tight_layout()
    plt.show()

    #Collisions by month (Shows seasonal issues)
    plt.figure(figsize=(12, 5))
    seaborn.countplot(data=dataframe, x='month')
    plt.title("Collisions in Sheffield by Month.")
    plt.xlabel("Month.")
    plt.ylabel("Count.")
    plt.tight_layout()
    plt.show()

    #Collisions in weather conditions
    plt.figure(figsize=(12, 5))
    seaborn.countplot(data=dataframe, x='weather_conditions', hue='collision_severity')
    plt.title("Collision in Sheffield by weather conditions.")
    plt.xlabel("Weather condition.")
    plt.ylabel("Count")
    plt.legend(title="Severity: 1=Fatal, 2=Serious, 3=Slight")
    weather_labels = {0: "Imputed", 1: "Fine.", 2: "Rain.", 3: "Snow.", 4: "Fine & Wind.", 5: "Rain & Wind..", 6: "Snow & Wind.", 7: "Fog.", 8: "Other.", 9: "Unknown"}
    plt.xticks(ticks=plt.xticks()[0],
           labels=[weather_labels.get(int(x), str(x))
                   for x in plt.xticks()[0]])
    plt.tight_layout()
    plt.show()

    #Collisions based on lighting conditions
    plt.figure(figsize=(12, 5))
    seaborn.countplot(dataframe, x='light_conditions', hue='collision_severity')
    plt.title("Collision severity by lighting conditions.")
    plt.xlabel("Lighting condition.")
    plt.ylabel("Count.")
    plt.legend(title="Severity (1=Fatal, 2=Serious, 3=Slight)")
    light_labels = {0: "Imputed." ,1: "Daylight.", 2: "Dark, street lighting lit.", 3: "Dark, street lighting unlit.", 4: "Dark, No street lighting.", 5: "Dark, Unknown."}
    plt.xticks(ticks=plt.xticks()[0],
           labels=[light_labels.get(int(x), str(x))
                   for x in plt.xticks()[0]])
    plt.tight_layout()
    plt.show()

    #Collisions based on location
    plt.figure(figsize=(12, 10))
    seaborn.scatterplot(
        data=dataframe,
        x='longitude', y='latitude',
        hue='collision_severity',
        alpha=0.4, s=10,
        palette={1: 'red', 2: 'orange', 3: 'blue'}
    )
    plt.title("Sheffield road collisions by location and severity.")
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.legend(title='Severity (1=Fatal, 2=Serious, 3=Slight)')
    plt.tight_layout()
    plt.show()

def plot_classification(confusion_matrix_multi, confusion_matrix_binary, confusion_matrix_categorical, best_decision_tree, X):

    #Multiclass confusion matrix plot
    plt.figure(figsize=(8, 6))
    seaborn.heatmap(confusion_matrix_multi, annot=True, fmt='d', cmap='Blues', xticklabels=['Fatal', 'Serious', 'Slight'], yticklabels=['Fatal', 'Serious', 'Slight'])
    plt.title("Confusion Matrix: Decision tree multiclass.")
    plt.xlabel("Predicted.")
    plt.ylabel("Actual.")
    plt.tight_layout()
    plt.show()

    #Binary confusion matrix
    plt.figure(figsize=(6, 4))
    seaborn.heatmap(confusion_matrix_binary, annot=True, fmt='d', cmap='Blues', xticklabels=['Rural', 'Urban'], yticklabels=['Rural', 'Urban'])
    plt.title("Confusion Matrix: Decision tree binary.")
    plt.xlabel("Predicted.")
    plt.ylabel("Actual.")
    plt.tight_layout()
    plt.show()

    #Categorical confusion matrix
    plt.figure(figsize=(10, 8))
    seaborn.heatmap(confusion_matrix_categorical, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix: Decision tree categorical.")
    plt.xlabel("Predicted.")
    plt.ylabel("Actual.")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 6))
    importance = pd.Series(best_decision_tree.feature_importances_, index=X.columns)
    importance.sort_values(ascending=False).plot(kind='bar')
    plt.title("Feature Importance: Decision tree.")
    plt.xlabel("Feature.")
    plt.ylabel("Importance.")
    plt.tight_layout()
    plt.show()

def plot_regression(dataframe, simple_model, yearly_counts, y_test_casualties, y_pred_casualties, y_test_vehicles, y_pred_vehicles, best_degree):

    #Simple linear regression
    plt.figure(figsize=(12, 5))
    plt.scatter(yearly_counts['collision_year'], yearly_counts['count'], color='blue', label='Actual', s=30)
    plt.plot(yearly_counts['collision_year'], simple_model.predict(yearly_counts[['collision_year']]), color='red', label='Trend Line')
    plt.title("Simple Linear Regression: Collision trend over time.")
    plt.xlabel("Year.")
    plt.ylabel("Number of collisions.")
    plt.legend()
    plt.tight_layout()
    plt.show()

    #Multi linear regression
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test_casualties, y_pred_casualties, alpha=0.3, s=10)
    plt.plot([y_test_casualties.min(), y_test_casualties.max()], [y_test_casualties.min(), y_test_casualties.max()], 'r--')
    plt.xlabel("Actual Casualties.")
    plt.ylabel("Predicted Casualties.")
    plt.title("Multiple Linear Regression: Actual vs predicted casualties.")
    plt.tight_layout()
    plt.show()

    #Ploynomial regression
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test_vehicles, y_pred_vehicles, alpha=0.3, s=10)
    plt.plot([y_test_vehicles.min(), y_test_vehicles.max()], [y_test_vehicles.min(), y_test_vehicles.max()], 'r--')
    plt.xlabel("Actual Vehicle.s")
    plt.ylabel("Predicted Vehicles.")
    plt.title(f"Polynomial Regression (degree={best_degree}): Actual vs predicted Vehicles.")
    plt.tight_layout()
    plt.show()

def plot_clustering(dataframe, X_pca_coords, kmeans_labels, best_k, sil_scores, k_range, dbscan_labels):

    #Elbow method plot
    inertia_values = []
    from sklearn.preprocessing import StandardScaler
    X_risk = dataframe[['number_of_casualties', 'number_of_vehicles', 'speed_limit', 'road_type', 'weather_conditions', 'light_conditions','road_surface_conditions', 'urban_or_rural_area', 'collision_severity']]
    X_scaled = StandardScaler().fit_transform(X_risk)
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=Rand_state, n_init=10)
        km.fit(X_scaled)
        inertia_values.append(km.inertia_)
    plt.figure(figsize=(10, 5))
    plt.plot(list(k_range), inertia_values, marker='o')
    plt.title("Elbow Method: Optimal number of clusters.")
    plt.xlabel("Number of clusters = (k).")
    plt.ylabel("Inertia.")
    plt.tight_layout()
    plt.show()

    #Silhouette plot
    plt.figure(figsize=(10, 5))
    plt.plot(list(k_range), sil_scores, marker='o', color='orange')
    plt.title("Silhouette Score: Optimal number of clusters.")
    plt.xlabel("Number of Clusters = (k).")
    plt.ylabel("Silhouette Score.")
    plt.tight_layout()
    plt.show()

     #K-Means clustersn via PCA projection
    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(X_pca_coords[:, 0], X_pca_coords[:, 1], c=kmeans_labels, cmap='tab10', alpha=0.4, s=10)
    plt.colorbar(scatter, label='Cluster')
    plt.title(f"K-Means Clusters (k={best_k}) : PCA Visualisation")
    plt.xlabel("PCA Component 1.")
    plt.ylabel("PCA Component 2.")
    plt.tight_layout()
    plt.show()

    #Location hotspot clusters
    plt.figure(figsize=(12, 10))
    seaborn.scatterplot(data=dataframe, x='longitude', y='latitude', hue='geo_cluster', palette='tab10', alpha=0.4, s=10)
    plt.title("Collisions based on location with hotspot clusters.")
    plt.xlabel("Longitude.")
    plt.ylabel("Latitude.")
    plt.tight_layout()
    plt.show()

    #DBSCAN clusters via PCA projection
    if dbscan_labels is not None:
        plt.figure(figsize=(10, 7))
        scatter = plt.scatter(X_pca_coords[:, 0], X_pca_coords[:, 1], c=dbscan_labels, cmap='tab10', alpha=0.4, s=10)
        plt.colorbar(scatter, label='Cluster (-1 = noise)')
        plt.title("DBSCAN Clusters : PCA Visualisation.")
        plt.xlabel("PCA Component 1.")
        plt.ylabel("PCA Component 2.")
        plt.tight_layout()
        plt.show()

def plot_pca(explained, cumulative):

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.bar(range(1, len(explained)+1), explained)
    plt.title("Explained variance per component.")
    plt.xlabel("Principal component.")
    plt.ylabel("Explained variance ratio.")
    #Cumulative variance
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(cumulative)+1), cumulative, marker='o')
    plt.axhline(y=0.95, color='r', linestyle='--', label='95% threshold')
    plt.title("Cumulative explained variance.")
    plt.xlabel("Number of components.")
    plt.ylabel("Cumulative variance.")
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_performance(simple_reg2, multi_reg2, poly_reg2, kmeans_silhouette, geo_silhouette, acc_multi, acc_binary, acc_cat, mlp_acc):

    #Classification accuracy comparison
    plt.figure(figsize=(12, 5))
    model_names = ['KNN\n(Multiclass)', 'Logistic Reg\n(Multiclass)', 'Decision Tree\n(Multiclass)', 'Decision Tree\n(Binary)', 'Decision Tree\n(Categorical)', 'Neural Network\n(Multiclass)']
    accuracies = [0.8132, 0.8371, acc_multi, acc_binary, acc_cat, mlp_acc]
    colours = ['blue' if a >= 0.8 else 'orange' for a in accuracies]
    plt.bar(model_names, accuracies, color=colours)
    plt.axhline(y=0.8, color='red', linestyle='--', label='80% target')
    plt.title("Classification Model Accuracy Comparison")
    plt.xlabel("Model.")
    plt.ylabel("Accuracy.")
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.show()

    #Regression R2 comparison using the live scores
    plt.figure(figsize=(10, 5))
    reg_names = ['Simple Linear\n(Trend)', 'Multiple Linear\n(Casualties)',
                 'Polynomial\n(Vehicles)']
    r2_scores = [simple_reg2, multi_reg2, poly_reg2]
    colours_r = ['green' if r >= 0.8 else 'orange' if r >= 0.5 else 'red'
                 for r in r2_scores]
    plt.bar(reg_names, r2_scores, color=colours_r)
    plt.axhline(y=0.8, color='red', linestyle='--', label='Good fit threshold')
    plt.title("Regression model R2 score comparison.")
    plt.xlabel("Model.")
    plt.ylabel("R2 Score.")
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.show()

    #Clustering silhouette comparison using the live scores
    plt.figure(figsize=(10, 5))
    cluster_names = ['K-Means\n(Risk Conditions)', 'K-Means\n(Geographic)']
    sil_vals = [kmeans_silhouette, geo_silhouette]
    plt.bar(cluster_names, sil_vals, color='purple')
    plt.axhline(y=0.5, color='red', linestyle='--', label='Good clustering threshold')
    plt.title("Clustering silhouette score comparison.")
    plt.xlabel("Model.")
    plt.ylabel("Silhouette score.")
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_class_distribution(dataframe):
    
    #Plot class imabalance
    counts = dataframe['collision_severity'].value_counts()
    labels = {1: 'Fatal', 2: 'Serious', 3: 'Slight'}
    label_names = [labels.get(i, str(i)) for i in counts.index]
    plt.figure(figsize=(8, 5))
    plt.bar(label_names, counts.values, color=['red', 'orange', 'blue'])
    plt.title("Class Distribution: Collision Severity (Bias Check)")
    plt.xlabel("Severity.")
    plt.ylabel("Count.")
    plt.tight_layout()
    plt.show()

def main():
    
    #Preprocessing
    log.info("Starting preprocessing...")
    dataframe = run_preprocessing()

    #Plot processed data
    log.info("Processing completed.")
    show_processed_data_in_plots(dataframe)

    #Feature matrices and train/val/test split
    X, y = build_feature_matrices(dataframe)
    X_train, X_value, X_test, y_train, y_value, y_test = split_train_val_test(X, y)
    X_classification_scaled = StandardScaler().fit_transform(dataframe[class_features]) #Scaling with standard scaler

    #Classification
    log.info("Running classification models...")
    print("\nK-Fold Cross Validation")
    run_cross_validation(X_train, y_train)

    #Multiclass (This predcits collision severity)
    print("\nDecision Tree Hyperparameter Tuning: Multiclass")
    best_decision_tree, best_decision_tree_params, best_decision_tree_value_acc = tune_decision_tree(X_train, X_value, y_train, y_value)
    print(f"Best parameters: {best_decision_tree_params}  Val accuracy: {best_decision_tree_value_acc:.4f}")
    acc_multi, cm_multi, report_multi = evaluate_classifier(best_decision_tree, X_test, y_test, label='Decision Tree: Multiclass')

    #Binary urban vs rural
    print("\nDecision Tree Hyperparameter Tuning (Binary)")
    X_binary = dataframe[['speed_limit', 'road_type', 'first_road_class', 'weather_conditions', 'light_conditions']]
    y_binary = dataframe['urban_or_rural_area'].apply(lambda x: 1 if x == 1 else 0)
    X_train_binary, X_temp_binary, y_train_binary, y_temp_binary = train_test_split(X_binary, y_binary, test_size=0.4, random_state=Rand_state)
    X_validation_binary, X_test_binary, y_validation_binary, y_test_binary = train_test_split(X_temp_binary, y_temp_binary, test_size=0.5, random_state=Rand_state)
    dt_binary, _ = tune_binary_decision_tree(X_train_binary, X_validation_binary, y_train_binary, y_validation_binary)
    y_pred_binary = dt_binary.predict(X_test_binary)
    cm_binary = confusion_matrix(y_test_binary, y_pred_binary)
    print(f"Binary Test Accuracy: {accuracy_score(y_test_binary, y_pred_binary):.4f}")
    print(classification_report(y_test_binary, y_pred_binary, target_names=['Rural', 'Urban'], zero_division=0))

    #Categorical junction type
    print("\nDecision Tree Hyperparameter Tuning: Categorical")
    X_cat = dataframe[['local_authority_district', 'first_road_class', 'second_road_class', 'second_road_number', 'speed_limit', 'road_type', 'junction_control']]
    y_cat = dataframe['junction_detail'].astype(int)
    X_train_cat, X_temp_cat, y_train_cat, y_temp_cat = train_test_split(X_cat, y_cat, test_size=0.4, random_state=Rand_state)
    X_validation_cat, X_test_cat, y_validation_cat, y_test_cat = train_test_split(X_temp_cat, y_temp_cat, test_size=0.5, random_state=Rand_state)
    dt_cat, _ = tune_categorical_decision_tree(X_train_cat, X_validation_cat, y_train_cat, y_validation_cat)
    y_pred_cat = dt_cat.predict(X_test_cat)
    cm_cat = confusion_matrix(y_test_cat, y_pred_cat)
    print(f"Categorical Test Accuracy: {accuracy_score(y_test_cat, y_pred_cat):.4f}")
    print(classification_report(y_test_cat, y_pred_cat, zero_division=0))

    #Neural network
    print("\nNeural Network: MLP Classification")
    scaler_mlp = StandardScaler()
    X_train_sc = scaler_mlp.fit_transform(X_train)
    X_test_sc  = scaler_mlp.transform(X_test)

    mlp_model = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        max_iter=500,
        activation='relu',
        random_state=Rand_state
    )

    mlp_model.fit(X_train_sc, y_train)
    mlp_acc = accuracy_score(y_test, mlp_model.predict(X_test_sc))
    cm_mlp  = confusion_matrix(y_test, mlp_model.predict(X_test_sc))
    log.info("Neural Network accuracy: %.4f", mlp_acc)
    print(f"Neural Network Test Accuracy: {mlp_acc:.4f}")
    print(classification_report(y_test, mlp_model.predict(X_test_sc), target_names=['Fatal', 'Serious', 'Slight'], zero_division=0))

    #Classification plots
    plot_classification(cm_multi, cm_binary, cm_cat, best_decision_tree, X)

    #Confusion matrix for neural network
    plt.figure(figsize=(8, 6))
    seaborn.heatmap(cm_mlp, annot=True, fmt='d', cmap='Blues', xticklabels=['Fatal', 'Serious', 'Slight'], yticklabels=['Fatal', 'Serious', 'Slight'])
    plt.title("Confusion Matrix: Neural network multiclass.")
    plt.xlabel("Predicted.")
    plt.ylabel("Actual.")
    plt.tight_layout()
    plt.show()

    #Regression
    log.info("Running regression models...")

    print("\nSimple Linear Regression")
    simple_model, yearly_counts, simple_reg2 = run_simple_linear_regression(dataframe)

    print("\nMultiple Linear Regression")
    multi_model, X_test_casualites, y_test_casualites, multi_reg2 = run_multiple_linear_regression(dataframe)
    y_pred_casualites = multi_model.predict(X_test_casualites)

    print("\nPolynomial Regression")
    poly_model, X_test_vehicle, y_test_vehicle, poly_reg2, best_degree = run_polynomial_regression(dataframe)
    y_pred_vehicle = poly_model.predict(X_test_vehicle)

    log.info("Polynomial best degree: %d", best_degree)

    #Regression plots
    plot_regression(dataframe, simple_model, yearly_counts, y_test_casualites, y_pred_casualites, y_test_vehicle, y_pred_vehicle, best_degree)

    #CLustering
    log.info("Running clustering models...")

    print("\nK-Means Risk Clustering")
    X_scaled, X_pca_2d, kmeans_labels, best_k, sil_scores, k_range = \
    run_kmeans_risk_clustering(dataframe)
    kmeans_sil = max(sil_scores)
    log.info("K-Means best k=%d  Silhouette=%.4f", best_k, kmeans_sil)

    print("\nGeographic Clustering")
    temp, geo_sil = run_geographic_clustering(dataframe)
    log.info("Geographic silhouette=%.4f", geo_sil)

    print("\nDBSCAN Clustering")
    dbscan_labels, best_eps, best_min_samples, best_db_score = run_dbscan_clustering(X_scaled, X_pca_2d)

    #Clustering plots
    plot_clustering(dataframe, X_pca_2d, kmeans_labels, best_k, sil_scores, k_range, dbscan_labels)

    #PCA
    log.info("Running PCA analysis")
    print("\nPCA Dimensionality Reduction")
    pca, explained, cumulative, n_components_95 = run_pca_analysis(X_classification_scaled, dataframe['collision_severity'])

    #PCA plots
    plot_pca(explained, cumulative)

    #Responsible ai
    responsible_ai_checks(dataframe, best_decision_tree, X)
    plot_class_distribution(dataframe)

    #Evaluation
    print("\n__Performance Evaluation Summary__")
    print(f"\nClassification Models:")
    print(f"{'Model':<35} {'Accuracy'}")
    print("-" * 50)
    print(f"{'KNN : Multiclass CV':<35} 0.8132")
    print(f"{'Logistic Regression : Multiclass CV':<35} 0.8371")
    print(f"{'Decision Tree : Multiclass':<35} {acc_multi:.4f}")
    print(f"{'Decision Tree : Binary':<35} {accuracy_score(y_test_binary, y_pred_binary):.4f}")
    print(f"{'Decision Tree : Categorical':<35} {accuracy_score(y_test_cat, y_pred_cat):.4f}")
    print(f"{'Neural Network : Multiclass':<35} {mlp_acc:.4f}")

    print(f"\n__Regression Models__")
    print(f"{'Model':<35} {'R2'}")
    print("-" * 50)
    print(f"{'Simple Linear Regression':<35} {simple_reg2:.4f}")
    print(f"{'Multiple Linear Regression':<35} {multi_reg2:.4f}")
    print(f"{f'Polynomial Regression (d={best_degree})':<35} {poly_reg2:.4f}")

    print(f"\n__Clustering Models__")
    print(f"{'Model':<35} {'Silhouette'}")
    print("-" * 50)
    print(f"{'K-Means : Risk Conditions':<35} {kmeans_sil:.4f}")
    print(f"{'K-Means : Geographic':<35} {geo_sil:.4f}")
    print(f"{'DBSCAN':<35} {best_db_score:.4f} (over-clustered)")

    #Perf comparison
    plot_performance(simple_reg2, multi_reg2, poly_reg2, kmeans_sil, geo_sil, acc_multi, accuracy_score(y_test_binary, y_pred_binary), accuracy_score(y_test_cat, y_pred_cat), mlp_acc)

try:
    if __name__ == '__main__':
        main()

except Exception as e:
    print(f"\n An error occured: {e}")