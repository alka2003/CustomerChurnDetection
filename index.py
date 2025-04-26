import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import SelectKBest, f_classif
import warnings
warnings.filterwarnings('ignore')

# Load datasets
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# Since this is transaction data, we need to aggregate at customer level for churn prediction
# Let's assume churn is when a customer stops transacting (is_fraud might not be relevant here)
# We'll need to create a churn label based on activity patterns

# Feature Engineering - Create customer-level features from transaction data
def create_customer_features(df):
    # Convert transaction time to datetime
    df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
    
    # Group by customer (cc_num) and create features
    customer_features = df.groupby('cc_num').agg({
        'trans_date_trans_time': ['max', 'min', 'count'],
        'amt': ['mean', 'sum', 'std'],
        'category': lambda x: x.nunique(),
        'merchant': lambda x: x.nunique(),
        'city': lambda x: x.nunique(),
        'state': lambda x: x.nunique(),
        'is_fraud': 'sum'
    })
    
    # Flatten multi-index columns
    customer_features.columns = ['_'.join(col).strip() for col in customer_features.columns.values]
    
    # Rename columns for clarity
    customer_features = customer_features.rename(columns={
        'trans_date_trans_time_max': 'last_transaction',
        'trans_date_trans_time_min': 'first_transaction',
        'trans_date_trans_time_count': 'transaction_count',
        'amt_mean': 'avg_transaction_amount',
        'amt_sum': 'total_spent',
        'amt_std': 'std_transaction_amount',
        'category_<lambda>': 'unique_categories',
        'merchant_<lambda>': 'unique_merchants',
        'city_<lambda>': 'unique_cities',
        'state_<lambda>': 'unique_states',
        'is_fraud_sum': 'fraud_count'
    })
    
    # Calculate customer tenure in days
    customer_features['tenure_days'] = (customer_features['last_transaction'] - 
                                       customer_features['first_transaction']).dt.days
    
    # Add demographic info (assuming first record per customer has their details)
    demo_info = df.drop_duplicates('cc_num').set_index('cc_num')[['first', 'last', 'gender', 'city', 'state', 'zip', 'lat', 'long', 'city_pop', 'job', 'dob']]
    customer_features = customer_features.join(demo_info)
    
    # Calculate age from dob
    customer_features['dob'] = pd.to_datetime(customer_features['dob'])
    customer_features['age'] = (pd.to_datetime('today') - customer_features['dob']).dt.days / 365.25
    
    return customer_features

# Create customer-level datasets
train_customers = create_customer_features(train_df)
test_customers = create_customer_features(test_df)

# Define churn - assuming customers who haven't transacted in the last 30 days are churned
# For training data, we need a reference date (typically max date in data minus 30 days)
reference_date_train = train_customers['last_transaction'].max() - pd.Timedelta(days=30)
train_customers['churn'] = (train_customers['last_transaction'] < reference_date_train).astype(int)

# For test data, we might not have the label (depending on problem setup)
# For this example, we'll proceed with training the model on train_customers

# EDA - Check churn distribution
print("Churn Distribution in Training Data:")
print(train_customers['churn'].value_counts(normalize=True))

# Feature selection and preprocessing
# Drop columns that are identifiers or not useful
drop_cols = ['first', 'last', 'last_transaction', 'first_transaction', 'dob', 'city', 'state', 'zip']
X = train_customers.drop(columns=['churn'] + drop_cols, errors='ignore')
y = train_customers['churn']

# Identify numerical and categorical columns
num_cols = X.select_dtypes(include=['int64', 'float64']).columns
cat_cols = X.select_dtypes(include=['object']).columns

# Preprocessing pipeline
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, num_cols),
        ('cat', categorical_transformer, cat_cols)])

# Split data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Handle class imbalance with SMOTE
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(preprocessor.fit_transform(X_train), y_train)

# Model training
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42)
}

results = {}
for name, model in models.items():
    model.fit(X_train_res, y_train_res)
    y_pred = model.predict(preprocessor.transform(X_val))
    y_prob = model.predict_proba(preprocessor.transform(X_val))[:,1]
    
    results[name] = {
        'accuracy': accuracy_score(y_val, y_pred),
        'roc_auc': roc_auc_score(y_val, y_prob),
        'report': classification_report(y_val, y_pred),
        'confusion_matrix': confusion_matrix(y_val, y_pred)
    }

# Display results
for model_name, metrics in results.items():
    print(f"\n{model_name} Performance:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"ROC AUC: {metrics['roc_auc']:.4f}")
    print("Classification Report:")
    print(metrics['report'])
    
    plt.figure()
    sns.heatmap(metrics['confusion_matrix'], annot=True, fmt='d')
    plt.title(f'{model_name} Confusion Matrix')
    plt.show()

# Feature importance analysis (using best model)
best_model = GradientBoostingClassifier(random_state=42)
best_model.fit(X_train_res, y_train_res)

# Get feature names after preprocessing
preprocessor.fit(X)
feature_names = (list(num_cols) + 
                list(preprocessor.named_transformers_['cat']
                     .named_steps['onehot']
                     .get_feature_names_out(cat_cols)))

# Plot feature importance
importances = best_model.feature_importances_
indices = np.argsort(importances)[-20:]  # top 20 features
plt.figure(figsize=(10, 8))
plt.title('Top 20 Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()

# Final model training on full data
X_full = preprocessor.fit_transform(X)
y_full = y

# Apply SMOTE to full data
X_full_res, y_full_res = smote.fit_resample(X_full, y_full)

final_model = GradientBoostingClassifier(random_state=42)
final_model.fit(X_full_res, y_full_res)

# Prepare test data (if we want to predict on test set)
X_test = test_customers.drop(columns=drop_cols, errors='ignore')
X_test_processed = preprocessor.transform(X_test)

# Predict churn probabilities on test set
test_customers['churn_probability'] = final_model.predict_proba(X_test_processed)[:,1]
test_customers['predicted_churn'] = (test_customers['churn_probability'] > 0.5).astype(int)

# Save predictions
test_customers[['cc_num', 'churn_probability', 'predicted_churn']].to_csv('churn_predictions.csv', index=False)

# Insights
print("\nKey Insights:")
print("1. Top factors contributing to churn:")
top_features = pd.DataFrame({
    'feature': feature_names,
    'importance': final_model.feature_importances_
}).sort_values('importance', ascending=False).head(10)
print(top_features)

print("\n2. Churn rate by demographic factors:")
# Example: Churn by gender
if 'gender' in train_customers.columns:
    print(train_customers.groupby('gender')['churn'].mean())

# Example: Churn by age group
if 'age' in train_customers.columns:
    train_customers['age_group'] = pd.cut(train_customers['age'], bins=[0, 30, 40, 50, 60, 100])
    print(train_customers.groupby('age_group')['churn'].mean())