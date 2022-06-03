import pandas as pd
import numpy as np
import onnxruntime
import onnxmltools
import onnx
import json, shutil, os

### Sample Input Data
#input_data = {
#  "acc_now_delinq": "4",
#  "acc_open_past_24mths": "329.08",
#  "annual_inc": "1",
#  "avg_cur_bal": "1",
#  "funded_amnt": "1"
#}

model_path = onnx.load("model.onnx").SerializeToString()

so = onnxruntime.SessionOptions()
so.add_session_config_entry('model.onnx', 'ONNX')

session = onnxruntime.InferenceSession(model_path)
output = session.get_outputs()[0] 
inputs = session.get_inputs()
    
def run(input_data):
    
    # Reformatting Input
    df = pd.DataFrame(input_data, index=[0])
    df.columns = ['acc_now_delinq', 'acc_open_past_24mths', 'annual_inc', 'avg_cur_bal', 'funded_amnt']
    df['acc_now_delinq'] = df['acc_now_delinq'].astype(float)
    df['acc_open_past_24mths'] = df['acc_open_past_24mths'].astype(float)
    df['annual_inc'] = df['annual_inc'].astype(float)
    df['avg_cur_bal'] = df['avg_cur_bal'].astype(float)
    df['funded_amnt'] = df['funded_amnt'].astype(float)
    
    # ONNX Scoring
    try:
        input_data= {i.name: v for i, v in zip(inputs, df.values.reshape(len(inputs),1,1).astype(np.float32))}
        output = session.run(None, input_data)
        pred = pd.DataFrame(output)[0][0]

        #print('[INFO] Results was ' + json.dumps(pred))
        return {"result": pred}

    except Exception as e:
        result_dict = {"error": str(e)}
    
    return result_dict