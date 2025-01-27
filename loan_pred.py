import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, ConfusionMatrixDisplay
)
from imblearn.over_sampling import ADASYN
import pickle


class LoanPredictionPipeline:
    def __init__(self, train_path, test_path, econ_path):
        """
        Initialize the pipeline with file paths.
        """
        self.train_path = train_path
        self.test_path = test_path
        self.econ_path = econ_path
        self.train = None
        self.test = None
        self.economic_data = None
        self.train_processed = None
        self.test_processed = None
        self.features = None
        self.scaler = None
        self.model = None

    def load_data(self):
        """
        Load datasets into memory, handling both DataFrame and file inputs.
        """
        # Handle cases where input might already be a DataFrame
        if isinstance(self.train_path, pd.DataFrame):
            self.train = self.train_path
        else:
            # For Streamlit uploaded files, we need to read the data directly
            self.train = pd.read_csv(self.train_path)
            
        if isinstance(self.test_path, pd.DataFrame):
            self.test = self.test_path
        else:
            self.test = pd.read_csv(self.test_path)
            
        if isinstance(self.econ_path, pd.DataFrame):
            self.economic_data = self.econ_path
        else:
            self.economic_data = pd.read_csv(self.econ_path)

    def preprocess_economic_data(self, selected_countries):
        """
        Preprocess the economic indicators data.
        """
        econ_df = self.economic_data[self.economic_data['Country'].isin(selected_countries)]
        econ_df = econ_df[econ_df['Indicator'].isin(['Deposit interest rate (%)', 
                                                    'Inflation, consumer prices (annual %)', 
                                                    'Official exchange rate (LCU per US$, period average)', 
                                                    'Unemployment rate'])]
        econ_df = pd.melt(
            econ_df,
            id_vars=['Country', 'Indicator'],
            value_vars=[col for col in econ_df.columns if col.startswith('YR')],
            var_name='Year', value_name='Value'
        )
        econ_df['Year'] = econ_df['Year'].str.replace('YR', '').astype(int)
        # econ_df = econ_df.pivot_table(
        #     index=['Country', 'Year'], columns='Indicator', values='Value'
        # ).reset_index()   
        rename_mapping = {
            'Deposit interest rate (%)': 'DIR',
            'Inflation, consumer prices (annual %)': 'inflation',
            'Official exchange rate (LCU per US$, period average)': 'exchange_rate',
            'Unemployment rate': 'unemployment_rate'
        }
        econ_df['Indicator'] = econ_df['Indicator'].replace(rename_mapping)

        pivoted_df = econ_df.pivot_table(
            index=['Country', 'Year'],  # Rows will be Country and Year
            columns='Indicator',        # Columns will be the unique Indicators
            values='Value'            # The values will correspond to the 'Value' column
            # aggfunc='first'             # In case there are multiple values for the same Country and Year, use the first one
        )

        # Reset the index to make the DataFrame easier to work with
        pivoted_df.reset_index(inplace=True)
        pivoted_df = pivoted_df.rename(columns={'Country': 'country_id'})

        self.economic_data = pivoted_df

    def merge_data(self):
        """
        Merge the economic data with train and test datasets.
        """
        for dataset in [self.train, self.test]:
            dataset['disbursement_date'] = pd.to_datetime(dataset['disbursement_date'])
            dataset['Year'] = dataset['disbursement_date'].dt.year
        print(self.economic_data.columns.to_list())
        self.economic_data.rename(columns={'Country':'country_id'}, inplace=True)
        self.train = self.train.merge(self.economic_data, on=['Year', 'country_id'], how='left')
        self.test = self.test.merge(self.economic_data, on=['Year', 'country_id'], how='left')
        # Forward fill economic data
        econ_cols = ['DIR', 'exchange_rate', 'inflation', 'unemployment_rate']
        for dataset in [self.train, self.test]:
            dataset.sort_values(by='Year', inplace=True)
            dataset[econ_cols] = dataset[econ_cols].fillna(method='ffill')

    def preprocess_features(self):
        """
        Handle missing values and feature encoding.
        """
        econ_cols = ['DIR', 'exchange_rate', 'inflation', 'unemployment_rate']
        for dataset in [self.train, self.test]:
            dataset.sort_values(by='Year', inplace=True)
            dataset[econ_cols] = dataset[econ_cols].fillna(method='ffill') 

        self.train.drop(['DIR','inflation','unemployment_rate'],axis = 1, inplace = True)
        self.test.drop(['DIR','inflation','unemployment_rate'],axis = 1, inplace = True) 

        for col in ['Total_Amount', 'Total_Amount_to_Repay', 'Amount_Funded_By_Lender', 'Lender_portion_to_be_repaid']:
            self.train[f'{col}'] = self.train[col] / self.train['exchange_rate']

        for col in ['Total_Amount', 'Total_Amount_to_Repay', 'Amount_Funded_By_Lender', 'Lender_portion_to_be_repaid']:
            self.test[f'{col}'] = self.test[col] / self.test['exchange_rate']

        # Log transformation
        for col in ['Total_Amount', 'Total_Amount_to_Repay', 'Amount_Funded_By_Lender', 'Lender_portion_to_be_repaid']:
            self.train[f'{col}'] = np.log1p(self.train[col])

        # Do the same for test data
        for col in ['Total_Amount', 'Total_Amount_to_Repay', 'Amount_Funded_By_Lender', 'Lender_portion_to_be_repaid']:
            self.test[f'{col}'] = np.log1p(self.test[col])
        
        # Frequency encoding for loan type
        frequency_map = self.train['loan_type'].value_counts(normalize=True)
        for dataset in [self.train, self.test]:
            dataset['loan_type'] = dataset['loan_type'].map(frequency_map)

        # Label encode categorical variables
        cat_cols = self.train.select_dtypes(include='object').columns
        # Combine train and test data to ensure all labels are captured
        combined_data = pd.concat([self.train, self.test], axis=0)

        # Fit LabelEncoder on the combined dataset
        le = LabelEncoder()
        for col in [col for col in cat_cols if col not in ['loan_type', 'ID']]:
            le.fit(combined_data[col])
            self.train[col] = le.transform(self.train[col])
            self.test[col] = le.transform(self.test[col])
        
        return self.train, self.test

    def scale_features(self, X_train, X_test, features):
        """
        Scale the numerical features using StandardScaler and save the scaler.
        """
        self.scaler = StandardScaler()
        X_train[features] = self.scaler.fit_transform(X_train[features])
        X_test[features] = self.scaler.transform(X_test[features])
        self.save_scaler('scaler.pkl')
        return X_train[features], X_test[features]

    def oversample_data(self, X, y) :
        """
        Handle class imbalance using ADASYN.
        """
        adasyn = ADASYN(sampling_strategy=1.0, n_neighbors=5)
        X_res, y_res = adasyn.fit_resample(X, y)
        return X_res, y_res

    def train_model(self, model, X_train, y_train):
        """
        Train a machine learning model and save it as a pickle file.
        """
        self.model = model
        model.fit(X_train, y_train)
        self.save_model(model, 'model.pkl')
        return model

    def evaluate_model(self, model, X_valid, y_valid):
        """
        Evaluate the model using various metrics and a confusion matrix.
        """
        y_pred = model.predict(X_valid)
        y_pred_proba = model.predict_proba(X_valid)[:, 1] if hasattr(model, "predict_proba") else None

        metrics = {
            "Accuracy": accuracy_score(y_valid, y_pred),
            "Precision": precision_score(y_valid, y_pred),
            "Recall": recall_score(y_valid, y_pred),
            "F1 Score": f1_score(y_valid, y_pred),
            "ROC AUC": roc_auc_score(y_valid, y_pred_proba) if y_pred_proba is not None else None
        }

        print("\n".join([f"{key}: {value:.4f}" for key, value in metrics.items()]))
        # ConfusionMatrixDisplay.from_predictions(y_valid, y_pred, cmap=plt.cm.Blues)
        # plt.show()
        return metrics

    def save_scaler(self, file_name):
        """
        Save the trained scaler as a pickle file.
        """
        with open(file_name, 'wb') as f:
            pickle.dump(self.scaler, f)

    def save_model(self, model, file_name):
        """
        Save the trained model as a pickle file.
        """
        with open(file_name, 'wb') as f:
            pickle.dump(model, f)