from fastapi import FastAPI 
from pydantic import BaseModel
import joblib
import numpy as np 

model = joblib.load('CLV_model.joblib')
feature_name = joblib.load('modelfeatures.joblib')

app = FastAPI(title='Customer Lifetime Value Prediction API')

class CLVinput(BaseModel):
    Customer_Age: int
    Annual_Income: float
    Tenure_Months: int
    Monthly_Spend: float
    Visits_Per_Month: int
    Avg_Basket_Value: float
    Support_Tickets: int
    
@app.get("/")
def health_check():
    return {"status": "API is running"}

@app.post('/predict-CLV')
def predict_CLV(data:CLVinput):
    x = np.array([[getattr(data,f) for f in feature_name]])
    prediction = model.predict(x)[0]
    return{'predicted_CLV': prediction}