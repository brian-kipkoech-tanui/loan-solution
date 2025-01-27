import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns
from loan_pred import LoanPredictionPipeline

class StreamlitLoanPredictionApp:
    def __init__(self):
        st.set_page_config(page_title="Loan Prediction Pipeline", layout="wide")
        # Define default XGBoost configuration
        self.default_xgboost_params = {
            'use_label_encoder': False,
            'eval_metric': 'logloss',
            'n_jobs': 5,
            'subsample': 0.9,
            'reg_lambda': 20,
            'reg_alpha': 0,
            'n_estimators': 500,
            'max_depth': 8,
            'learning_rate': 0.3,
            'colsample_bytree': 1.0,
            'random_state': 42
        }
        
        self.models = {
            "XGBoost": xgb.XGBClassifier,
            "Random Forest": RandomForestClassifier,
            "Gradient Boosting": GradientBoostingClassifier,
            "Logistic Regression": LogisticRegression,
            "Support Vector Machine": SVC
        }
        self.pipeline = None
        self.train_data = None
        self.test_data = None
        self.economic_data = None
        self.prediction_data = None

    def upload_data(self):
        st.sidebar.header("Data Upload")
        
        try:
            # Train Data Upload
            train_file = st.sidebar.file_uploader("Upload Train Data", type=['csv'])
            if train_file is not None:
                self.train_data = pd.read_csv(train_file)
                st.sidebar.success("Train Data Uploaded Successfully")
            
            # Test Data Upload
            test_file = st.sidebar.file_uploader("Upload Test Data", type=['csv'])
            if test_file is not None:
                self.test_data = pd.read_csv(test_file)
                st.sidebar.success("Test Data Uploaded Successfully")
            
            # Economic Indicators Upload
            econ_file = st.sidebar.file_uploader("Upload Economic Indicators", type=['csv'])
            if econ_file is not None:
                self.economic_data = pd.read_csv(econ_file)
                st.sidebar.success("Economic Indicators Uploaded Successfully")
        except Exception as e:
            st.error(f"Error uploading files: {str(e)}")
            return False
        
        return True

    def configure_model(self):
        st.sidebar.header("Model Configuration")
        
        # Model Selection with XGBoost as default
        selected_model = st.sidebar.selectbox(
            "Select Model", 
            list(self.models.keys()),
            index=0  # Set XGBoost as default
        )
        
        # Hyperparameter Configuration
        model_class = self.models[selected_model]
        hyperparams = {}
        
        if selected_model == "XGBoost":
            # Use default XGBoost parameters but allow modification
            hyperparams = self.default_xgboost_params.copy()
            
            # Add sliders for key parameters
            hyperparams['n_estimators'] = st.sidebar.slider("Number of Estimators", 100, 1000, 500)
            hyperparams['max_depth'] = st.sidebar.slider("Max Depth", 3, 15, 8)
            hyperparams['learning_rate'] = st.sidebar.slider("Learning Rate", 0.01, 0.5, 0.3)
            hyperparams['subsample'] = st.sidebar.slider("Subsample", 0.5, 1.0, 0.9)
            hyperparams['reg_lambda'] = st.sidebar.slider("L2 Regularization", 1, 50, 20)
            
        elif selected_model == "Random Forest":
            hyperparams['n_estimators'] = st.sidebar.slider("Number of Estimators", 10, 200, 100)
            hyperparams['max_depth'] = st.sidebar.slider("Max Depth", 1, 20, 10)
            hyperparams['random_state'] = 42
        
        elif selected_model == "Gradient Boosting":
            hyperparams['n_estimators'] = st.sidebar.slider("Number of Estimators", 10, 200, 100)
            hyperparams['learning_rate'] = st.sidebar.slider("Learning Rate", 0.01, 1.0, 0.1)
            hyperparams['max_depth'] = st.sidebar.slider("Max Depth", 1, 20, 5)
            hyperparams['random_state'] = 42
        
        elif selected_model == "Logistic Regression":
            hyperparams['penalty'] = st.sidebar.selectbox("Penalty", ['l2'])
            hyperparams['C'] = st.sidebar.slider("Regularization Strength", 0.1, 10.0, 1.0)
            hyperparams['random_state'] = 42
        
        elif selected_model == "Support Vector Machine":
            hyperparams['kernel'] = st.sidebar.selectbox("Kernel", ['linear', 'rbf'])
            hyperparams['C'] = st.sidebar.slider("Regularization Parameter", 0.1, 10.0, 1.0)
            hyperparams['random_state'] = 42

        # Initialize the selected model with hyperparameters
        model = model_class(**hyperparams)
        
        return selected_model, model

    def interpret_metrics(self, metrics):
        """Provide interpretation of model metrics"""
        interpretations = {
            "Accuracy": {
                "excellent": 0.90,
                "good": 0.80,
                "fair": 0.70,
                "poor": 0.60,
                "interpretation": lambda x: (
                    "Excellent" if x >= 0.90 else
                    "Good" if x >= 0.80 else
                    "Fair" if x >= 0.70 else
                    "Poor" if x >= 0.60 else
                    "Very Poor"
                )
            },
            "Precision": {
                "interpretation": lambda x: f"Out of all loans predicted as defaulting, {x*100:.1f}% actually defaulted"
            },
            "Recall": {
                "interpretation": lambda x: f"The model correctly identified {x*100:.1f}% of all actual loan defaults"
            },
            "F1 Score": {
                "excellent": 0.80,
                "good": 0.70,
                "fair": 0.60,
                "poor": 0.50,
                "interpretation": lambda x: (
                    "Excellent balance" if x >= 0.80 else
                    "Good balance" if x >= 0.70 else
                    "Fair balance" if x >= 0.60 else
                    "Poor balance" if x >= 0.50 else
                    "Very poor balance"
                )
            },
            "ROC AUC": {
                "excellent": 0.90,
                "good": 0.80,
                "fair": 0.70,
                "poor": 0.60,
                "interpretation": lambda x: (
                    "Excellent discrimination" if x >= 0.90 else
                    "Good discrimination" if x >= 0.80 else
                    "Fair discrimination" if x >= 0.70 else
                    "Poor discrimination" if x >= 0.60 else
                    "Very poor discrimination"
                )
            }
        }

        interpretations_text = {}
        for metric, value in metrics.items():
            if value is not None:
                interpretations_text[metric] = interpretations[metric]["interpretation"](value)
        
        return interpretations_text

    def plot_confusion_matrix(self, y_true, y_pred):
        """Plot confusion matrix using matplotlib"""
        fig, ax = plt.subplots(figsize=(8, 6))
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(ax=ax, cmap='Blues')
        plt.title('Confusion Matrix')
        return fig

    def upload_prediction_data(self):
        """Upload data for prediction"""
        st.sidebar.header("Prediction Data Upload")
        pred_file = st.sidebar.file_uploader("Upload Data for Prediction", type=['csv'])
        if pred_file is not None:
            try:
                self.prediction_data = pd.read_csv(pred_file)
                st.sidebar.success("Prediction Data Uploaded Successfully")
                return True
            except Exception as e:
                st.error(f"Error uploading prediction data: {str(e)}")
                return False
        return False

    def make_predictions(self, model, data, features):
        """Make predictions on new data"""
        try:
            # Select only the features used by the model
            X_test = data[features]
            predictions = model.predict(X_test)
            probabilities = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
            return predictions, probabilities
        except Exception as e:
            st.error(f"Error making predictions: {str(e)}")
            return None, None

    def run_pipeline(self):
        if not all([self.train_data is not None, self.test_data is not None, self.economic_data is not None]):
            st.error("Please upload all required data files")
            return None, None, None, None

        try:
            # Initialize Pipeline with DataFrames
            self.pipeline = LoanPredictionPipeline(
                train_path=self.train_data, 
                test_path=self.test_data, 
                econ_path=self.economic_data
            )
            
            # Load and Preprocess Data
            self.pipeline.load_data()
            
            # Get unique countries from the data
            selected_countries = list(set(self.train_data['country_id'].unique()) | 
                                   set(self.test_data['country_id'].unique()))
            
            self.pipeline.preprocess_economic_data(selected_countries)
            self.pipeline.merge_data()
            train, test = self.pipeline.preprocess_features()

            # Feature Preparation
            date_cols = ['disbursement_date', 'due_date']
            features_for_modelling = [
                col for col in train.columns 
                if col not in date_cols + ['ID', 'target', 'customer_id', 'country_id', 'tbl_loan_id', 'lender_id']
            ]

            # Split Data
            X_train = train[features_for_modelling]
            y_train = train['target']
            X_valid = test[features_for_modelling]
            y_valid = test['target'] if 'target' in test.columns else None

            # Scale Features
            X_train_scaled, X_test_scaled = self.pipeline.scale_features(
                X_train=X_train, 
                X_test=X_valid, 
                features=features_for_modelling
            )

            return X_train_scaled, X_test_scaled, y_train, y_valid

        except Exception as e:
            st.error(f"Error in pipeline execution: {str(e)}")
            return None, None, None, None

    def main(self):
        st.title("Loan Prediction Pipeline")
        
        # Create tabs for training and prediction
        tab1, tab2 = st.tabs(["Model Training", "Make Predictions"])
        
        with tab1:
            # Upload Data Section
            upload_success = self.upload_data()
            
            # Store model configuration in session state
            if 'selected_model' not in st.session_state:
                st.session_state.selected_model = None
            
            # Configure model and store in session state
            current_model_name, current_model = self.configure_model()
            st.session_state.selected_model = current_model
            
            if st.sidebar.button("Run Pipeline"):
                if not upload_success:
                    st.error("Please ensure all files are uploaded correctly")
                    return
                    
                try:
                    with st.spinner("Running pipeline..."):
                        # Use the model from session state
                        model = st.session_state.selected_model
                        
                        # Run Pipeline
                        X_train, X_test, y_train, y_valid = self.run_pipeline()
                        
                        if X_train is not None:
                            try:
                                # Train Model
                                self.pipeline.train_model(model, X_train, y_train)
                                
                                # Evaluate Model
                                if y_valid is not None:
                                    try:
                                        metrics = self.pipeline.evaluate_model(model, X_test, y_valid)
                                        
                                        if metrics is None:
                                            st.error("Model evaluation failed to return metrics")
                                            return
                                        
                                        # Display Results
                                        st.header(f"Model Performance: {current_model_name}")
                                        
                                        # Create two columns for metrics and confusion matrix
                                        col1, col2 = st.columns([1, 1])
                                        
                                        with col1:
                                            # Display metrics with interpretations
                                            if isinstance(metrics, dict) and metrics:
                                                interpretations = self.interpret_metrics(metrics)
                                                for metric, value in metrics.items():
                                                    if value is not None:
                                                        st.metric(metric, f"{value:.4f}")
                                                        st.info(interpretations[metric])
                                            else:
                                                st.warning("No metrics were calculated during evaluation")
                                        
                                        with col2:
                                            # Plot confusion matrix
                                            predictions = model.predict(X_test)
                                            fig = self.plot_confusion_matrix(y_valid, predictions)
                                            st.pyplot(fig)
                                            
                                    except Exception as eval_error:
                                        st.error(f"Error during model evaluation: {str(eval_error)}")
                                else:
                                    st.info("No target values in test data. Skipping evaluation.")
                            except Exception as train_error:
                                st.error(f"Error during model training: {str(train_error)}")
                        else:
                            st.error("Failed to prepare training data")
                except Exception as e:
                    st.error(f"Pipeline execution failed: {str(e)}")
        
        with tab2:
            st.header("Make Predictions on New Data")
        
            # Upload prediction data
            pred_upload_success = self.upload_prediction_data()
            
            if pred_upload_success and st.session_state.get('selected_model') is not None:
                if st.button("Generate Predictions"):
                    try:
                        # First, ensure we have a trained model
                        if not st.session_state.get('selected_model'):
                            with st.spinner("Training default XGBoost model..."):
                                # Create default XGBoost model
                                default_model = xgb.XGBClassifier(**self.default_xgboost_params)
                                st.session_state.selected_model = default_model

                        # Initialize or get the main pipeline
                        if self.pipeline is None:
                            # Initialize the main pipeline
                            self.pipeline = LoanPredictionPipeline(
                                train_path=self.train_data,
                                test_path=self.test_data,
                                econ_path=self.economic_data
                            )
                            self.pipeline.load_data()
                            selected_countries_train = list(set(self.train_data['country_id'].unique()))
                            self.pipeline.preprocess_economic_data(selected_countries_train)
                            self.pipeline.merge_data()
                            train, _ = self.pipeline.preprocess_features()

                            # Prepare and train the model
                            date_cols = ['disbursement_date', 'due_date']
                            features_for_modelling = [
                                col for col in train.columns 
                                if col not in date_cols + ['ID', 'target', 'customer_id', 'country_id', 'tbl_loan_id', 'lender_id']
                            ]
                            
                            X_train = train[features_for_modelling]
                            y_train = train['target']
                            
                            # Scale features
                            X_train_scaled, _ = self.pipeline.scale_features(
                                X_train=X_train,
                                X_test=X_train,
                                features=features_for_modelling
                            )
                            
                            # Train the model
                            st.info("Training model on the data...")
                            self.pipeline.train_model(st.session_state.selected_model, X_train_scaled, y_train)
                            st.success("Model trained successfully!")

                        # Now proceed with prediction pipeline
                        pred_pipeline = LoanPredictionPipeline(
                            train_path=self.train_data,
                            test_path=self.prediction_data,
                            econ_path=self.economic_data
                        )
                        
                        # Load and preprocess prediction data
                        pred_pipeline.load_data()
                        
                        # Get countries from prediction data and merge with training countries
                        selected_countries = list(set(
                            list(self.prediction_data['country_id'].unique()) +
                            list(self.train_data['country_id'].unique())
                        ))
                        
                        # Preprocess and merge economic data
                        pred_pipeline.preprocess_economic_data(selected_countries)
                        pred_pipeline.merge_data()
                        
                        # Preprocess features
                        _, processed_pred_data = pred_pipeline.preprocess_features()
                        
                        # Get the features used in training
                        date_cols = ['disbursement_date', 'due_date']
                        features_for_modelling = [
                            col for col in processed_pred_data.columns 
                            if col not in date_cols + ['ID', 'target', 'customer_id', 'country_id', 'tbl_loan_id', 'lender_id']
                        ]
                        
                        # Scale features using the prediction pipeline's scaler
                        _, X_pred = pred_pipeline.scale_features(
                            X_train=processed_pred_data,
                            X_test=processed_pred_data,
                            features=features_for_modelling
                        )
                        
                        # Use the trained model from pipeline for predictions
                        if hasattr(self.pipeline, 'model') and self.pipeline.model is not None:
                            model = self.pipeline.model
                        else:
                            model = st.session_state.selected_model
                            if not hasattr(model, 'predict'):
                                st.error("Model not properly trained. Please train the model first.")
                                return
                        
                        # Make predictions
                        predictions, probabilities = self.make_predictions(
                            model,
                            processed_pred_data,
                            features_for_modelling
                        )
                        
                        if predictions is not None:
                            # Create results DataFrame with original IDs
                            results_df = pd.DataFrame({
                                'ID': self.prediction_data['ID'],
                                'Prediction': predictions,
                                'Default_Probability': probabilities if probabilities is not None else [-1] * len(predictions)
                            })
                            
                            # Display predictions with custom formatting
                            st.subheader("Prediction Results")
                            styled_df = results_df.style.format({
                                'Default_Probability': '{:.2%}'
                            }).background_gradient(
                                subset=['Default_Probability'],
                                cmap='RdYlGn_r'
                            )
                            
                            st.dataframe(styled_df)
                            
                            # Show summary statistics
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Total Predictions", len(predictions))
                                st.metric("Predicted Defaults", sum(predictions == 1))
                            with col2:
                                st.metric("Predicted Non-Defaults", sum(predictions == 0))
                                if probabilities is not None:
                                    st.metric("Average Default Probability", f"{np.mean(probabilities):.2%}")
                            
                            # Add download button for predictions
                            csv = results_df.to_csv(index=False)
                            st.download_button(
                                "Download Predictions",
                                csv,
                                "loan_predictions.csv",
                                "text/csv",
                                key='download-csv'
                            )
                            
                    except Exception as e:
                        st.error(f"Error generating predictions: {str(e)}")
            else:
                # Check and display appropriate messages based on what's missing
                missing_requirements = []
                
                if not pred_upload_success:
                    missing_requirements.append("prediction data")
                if st.session_state.get('selected_model') is None:
                    missing_requirements.append("trained model")
                if self.train_data is None:
                    missing_requirements.append("training data")
                if self.economic_data is None:
                    missing_requirements.append("economic indicators data")
                
                if missing_requirements:
                    requirements_text = ", ".join(missing_requirements)
                    st.warning(f"Please provide the following to make predictions: {requirements_text}")
                    
                    # Provide specific guidance
                    st.info("""
                    To make predictions:
                    1. First train a model in the 'Model Training' tab
                    2. Upload your prediction data here
                    3. Ensure you have all required data files (training data and economic indicators)
                    4. Click 'Generate Predictions'
                    """)       
if __name__ == "__main__":
    app = StreamlitLoanPredictionApp()
    app.main()