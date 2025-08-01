# Aviation Damage Prediction System

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![FastAPI](https://img.shields.io/badge/Framework-FastAPI-green)
![Accuracy](https://img.shields.io/badge/Accuracy-86%25-brightgreen)

Machine learning system for predicting aircraft damage severity using gradient boosting classification.

## Model Training Process

1. **Data Preprocessing**:
   - Imputed missing values using median/mean for numerical and mode for categorical data
   - Encoded categorical features using Label Encoding

2. **Feature Selection**:
   - Generated correlation heatmap to identify feature importance
   - Selected top 5 most correlated features with target (`aircraft.damage`)

3. **Model Development**:
   - Trained Gradient Boosting Classifier
   - Addressed class imbalance using SMOTE oversampling
   - Achieved **86% prediction accuracy**

4. **Deployment Prep**:
   - Serialized model to `.pkl` file
   - Created FastAPI wrapper for predictions

## API Deployment

### Render Hosting

```bash
1. New Web Service → Connect GitHub repo
2. Runtime: Python 3.8+
3. Build Command: pip install -r requirements.txt
4. Start Command: uvicorn main:app --host 0.0.0.0 --port $PORT
```

### API Documentation

Interactive endpoints available at:
- Swagger UI: `/docs`
- ReDoc: `/redoc`

**Prediction Endpoint**:
```python
POST /predict
{
  "Investigation_Type_Incident": 1,
  "Schedule_SCHD": 1,
  "Total_Uninjured": 3.0,
  "Injury_Severity_freq": 250,
  "Engine_Type_Turbo_Fan": 1
}
```

## Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Launch dev server
uvicorn main:app --reload
```

Check it out: https://aviationdamage.onrender.com/docs#/default/predict_predict_post

## Key Files
- Jupyter notebook with EDA and model training
- Serialized trained model
- FastAPI application

