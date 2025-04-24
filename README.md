# 🎧 Podcast Listening Time Prediction

[![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/competitions/playground-series-s5e4)
[![Python 3.9+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

A data science project to predict [podcast listening duration](https://www.kaggle.com/competitions/playground-series-s5e4/overview).

## **Created by: Antonio Kevin**
🌐 [**Kaggle**](https://www.kaggle.com/akkevin) | 💼 [**LinkedIn**](https://www.linkedin.com/in/antonio-kevin/) | 🧑‍💻 [**GitHub**](https://github.com/akkevinn)

## **Table of Contents**
1. [Problem Understanding](#problem-understanding)
2. [Approach](#approach)
3. [Results](#results)
4. [Data Preprocessing](#data-preprocessing)
5. [Feature Engineering](#feature-engineering)
6. [Model Training](#model-training)
7. [Prediction & Submission](#prediction-and-submission)


## Problem Understanding

### **Business Objective**:
The objective of this model is to **predict podcast listening durations** to help content creators improve their podcasts and increase engagement. This prediction will assist with:

- **Optimizing episode length**: Helping creators decide on the ideal episode duration.
- **Improving publishing schedules**: Suggesting the best times and days for publishing.
- **Guest selection**: Determining which types of guests or topics result in higher engagement.

### **Evaluation Metric**:
The model's performance will be evaluated using **Root Mean Squared Error (RMSE)**, which measures the difference between the predicted and actual listening durations in **minutes**. The goal is to minimize this error to ensure more accurate predictions.

## Approach
1. **Data Preprocessing**: Clean and prepare the data for modeling.
2. **Feature Engineering**: Create meaningful features for model training.
3. **Model Training**: Train and fine-tune a suitable model for the prediction task.
4. **Evaluation**: Assess model performance using RMSE.
5. **Prediction on Test Data and Submission**:
    - Use the trained model to make predictions on the test dataset.
    - Format the predictions for submission according to the competition requirements.
    - Save the results to a CSV file for submission.

## Results
RMSE = 12.59510

## Data Preprocessing

### 1. Data Integration
Combined training and test datasets to ensure consistent preprocessing:
```python
train_data = pd.read_csv('train.csv')
train_data['dataset'] = 'train'

test_data = pd.read_csv('test.csv')
test_data['dataset'] = 'test'

data = pd.concat([train_data, test_data]).reset_index(drop=True)
```

### 2. Missing Value Handling
| Column | Strategy | Rationale |
|--------|----------|-----------|
| `Number_of_Ads` | Median imputation | Preserved ad distribution |
| `Guest_Popularity_percentage` | Zero imputation | Missing = no guest |
| `Episode_Length_minutes` | Podcast-specific median | Same podcast tends to have specific podcast lengh |

### 3. Outlier Treatment
Capped numerical features at reasonable maxima:
```python
data['Number_of_Ads'] = data['Number_of_Ads'].clip(upper=3)  # Max 3 ads
data['Episode_Length_minutes'] = data['Episode_Length_minutes'].clip(upper=120)  # Max 2hrs
data[['Host_Popularity_percentage', 'Guest_Popularity_percentage']] = \
    data[['Host_Popularity_percentage', 'Guest_Popularity_percentage']].clip(upper=100)
```

## Feature Engineering

### 1. Temporal Features
#### Cyclical Date Encoding:
```python
day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday',
             'Friday', 'Saturday', 'Sunday']
data['day_num'] = data['Publication_Day'].map(lambda x: day_order.index(x))
data['day_sin'] = np.sin(2 * np.pi * data['day_num'] / 7)
data['day_cos'] = np.cos(2 * np.pi * data['day_num'] / 7)
```

#### Weekend Indicator:
```python
data['is_weekend'] = data['Publication_Day'].isin(['Saturday', 'Sunday']).astype(int)
```

### 2. Categorical Encoding
| Feature | Encoding Type | Description |
|---------|---------------|-------------|
| `Genre` | One-Hot | Separate columns for each Genre with binary values |
| `Publication_Time` | One-Hot | Separate columns for each Publication Time with binary values |
| `Episode_Sentiment` | Ordinal | Mapped {Positive: 2, Neutral: 1, Negative: 0} |

### 3. Derived Features
#### Episode Metadata:
```python
data['episode_num'] = data['Episode_Title'].str.extract('(\d+)').astype(float)  # Extract episode number
data['length_bin'] = pd.cut(data['Episode_Length_minutes'], bins=[0,30,60,90,120])  # Duration categories
```

#### Host-Guest Dynamics:
```python
data['host_guest_ratio'] = Host_Popularity / (Guest_Popularity + 1e-6)
data['host_guest_diff'] = Host_Popularity - Guest_Popularity
```

#### Sentiment-Weighted Metrics:
```python
data['host_pop_sentiment'] = Host_Popularity * sentiment_encoded
data['guest_pop_sentiment'] = Guest_Popularity * sentiment_encoded
```

## Model Training

### 1. Feature Set
```python
features = [
    'day_sin', 'day_cos', 'sentiment_encoded',
    'episode_num', 'Episode_Length_minutes',
    'Host_Popularity_percentage', 'Guest_Popularity_percentage',
    'Number_of_Ads',
    'length_bin',
    'is_weekend',
    'host_guest_ratio',
    'host_guest_diff',
    'host_pop_sentiment',
    'guest_pop_sentiment',
    'Podcast_Name'
] + [col for col in data.columns if col.startswith('time_')] \
    + [col for col in data.columns if col.startswith('genre_')]
```

### 2. XGBoost Configuration
```python
model = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=1000,
    learning_rate=0.02,
    max_depth=15,
    subsample=0.8,
    early_stopping_rounds=50,
    eval_metric='rmse'
)
```

### 3. Validation Strategy
#### 5-Fold Cross Validation:
```python
for (train_idx, val_idx) in gkf.split(X, y):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    # Initialize target encoder
    encoder = TargetEncoder(cols=['Podcast_Name'], smoothing=10)

    # Fit and transform on training fold
    X_train_encoded = encoder.fit_transform(X_train, y_train)

    # Transform validation fold (using training fold's encoding)
    X_val_encoded = encoder.transform(X_val)

    # Drop original Podcast Name column
    X_train_encoded = X_train_encoded.drop(columns=['Podcast_Name'])
    X_val_encoded = X_val_encoded.drop(columns=['Podcast_Name'])

    # Train model
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=1000,
        learning_rate=0.02,
        max_depth=15,
        subsample=0.8,
        early_stopping_rounds=50,
        eval_metric='rmse'
    )
    model.fit(X_train_encoded, y_train, eval_set=[(X_val_encoded, y_val)], verbose=False)

    # Validate
    val_preds = model.predict(X_val_encoded)
    rmse = np.sqrt(mean_squared_error(y_val, val_preds))
    rmse_scores.append(rmse)
    print(f"Fold RMSE: {rmse:.2f} minutes")

print(f"\nAverage RMSE: {np.mean(rmse_scores):.2f} ± {np.std(rmse_scores):.2f}")

```

#### Early Stopping: monitors validation RMSE to prevent overfitting.

### 4. Final Training
```python
# Final model training (RMSE: 12.59510)
encoder = TargetEncoder(cols=['Podcast_Name'], smoothing=10)
X_encoded = encoder.fit_transform(X, y)
X_encoded = X_encoded.drop(columns=['Podcast_Name'])
X_test_encoded = encoder.transform(X_test).drop(columns=['Podcast_Name'])

final_model = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=1000,
    learning_rate=0.02,
    max_depth=15,
    subsample=0.8,
    eval_metric='rmse'
)
final_model.fit(X_encoded, y)
```

## Prediction and Submission
```python
test_preds = final_model.predict(X_test_encoded)
submission_data = pd.DataFrame({
    'id': test_data['id'],
    'Listening_Time_minutes': test_preds
})
submission_data.to_csv('final_submission_v13.csv', index=False)
```