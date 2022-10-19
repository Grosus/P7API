# 1. Library imports
import uvicorn
import numpy as np
from fastapi import FastAPI
from Model import ClientModel, ClientData
from Model import preprocessing as preprocessing

# 2. Create app and model objects
app = FastAPI()
model = ClientModel()

# 3. Expose the prediction functionality, make a prediction from the passed
#    JSON data and return the predicted flower species with the confidence
@app.post('/predict')
def predict(data: ClientData):
    data=data.dict()
    with open('column.npy', 'rb') as f:
        cols=np.load(f,allow_pickle=True)
    df=preprocessing(data)
    df=df[cols]
    prediction, probability = model.predict_target(df)
    return {
        'prediction': prediction,
        'probability': probability
    }

@app.post('/prepro')
def prepro(data: ClientData):
    data=data.dict()
    with open('column.npy', 'rb') as f:
        cols=np.load(f,allow_pickle=True)
    df=preprocessing(data)
    df=df[cols]  
    return {
        df.to_dict()
    }


# 4. Run the API with uvicorn
#    Will run on http://127.0.0.1:8000
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
