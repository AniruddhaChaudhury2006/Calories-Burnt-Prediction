# Calorie Prediction Model

## Introduction
This project aims to build a machine learning model to predict the number of calories burned during exercise based on various personal and exercise-related features. The model is built using the XGBoost Regressor algorithm.

## Dataset
The project utilizes two datasets:
- `calories.csv`: Contains the `User_ID` and `Calories` burned.
- `exercise.csv`: Contains `User_ID`, `Gender`, `Age`, `Height`, `Weight`, `Duration` of exercise, `Heart_Rate`, and `Body_Temp`.

These two datasets are merged based on `User_ID` to form a comprehensive dataset for model training.

## Data Preprocessing
1.  **Loading Data**: Both `calories.csv` and `exercise.csv` were loaded into pandas DataFrames.
2.  **Merging Data**: The datasets were concatenated using `pd.concat` on `User_ID` and `Calories` columns.
3.  **Handling Missing Values**: Checked for null values, and the dataset was found to be clean with no missing values.
4.  **Categorical Data Encoding**: The 'Gender' column, which is a categorical feature ('male', 'female'), was converted into numerical representation (0 for 'male', 1 for 'female') using `replace`.
5.  **Feature Selection**: The `User_ID` column was dropped as it's not relevant for prediction, and `Calories` was set as the target variable (`Y`). Other columns formed the features (`X`).
6.  **Data Splitting**: The data was split into training and testing sets using `train_test_split` with an 80/20 ratio and `random_state=2` for reproducibility.

## Exploratory Data Analysis (EDA)
-   **Data Information**: `calories_data.info()` provided a summary of column types and non-null counts.
-   **Descriptive Statistics**: `calories_data.describe()` showed statistical summaries of numerical features.
-   **Gender Distribution**: `sns.countplot(calories_data['Gender'])` was used to visualize the distribution of genders.
-   **Numerical Feature Distributions**: `sns.distplot` was used to visualize the distributions of 'Age', 'Height', and 'Weight'.
-   **Correlation Heatmap**: A heatmap was generated using `sns.heatmap` to visualize the correlation between numerical features and the target variable.

## Model Training
An XGBoost Regressor model was initialized and trained on the `X_train` and `Y_train` datasets.

## Model Evaluation
After training, the model's performance was evaluated on the test set:
-   **Predictions**: `model.predict(X_test)` was used to get predictions on the test data.
-   **Mean Absolute Error (MAE)**: The `metrics.mean_absolute_error` was calculated to quantify the average magnitude of errors in the predictions. A low MAE indicates a good model performance.

## Prediction
The trained model can be used to predict calories burned for new input data. An example prediction was made using a sample input, demonstrating how to reshape the input data for the model.
