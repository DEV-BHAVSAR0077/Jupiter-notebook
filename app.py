# app.py (Streamlit)
import streamlit as st, pandas as pd, numpy as np
from io import StringIO
import xml.etree.ElementTree as ET
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.datasets import fetch_california_housing
from tensorflow.keras import models, layers
from sklearn.model_selection import train_test_split

st.title("Minimal California Predictor")

# simple parser
def parse_upload(f):
    name = f.name.lower()
    txt = f.read()
    try:
        if name.endswith('.csv'): return pd.read_csv(StringIO(txt.decode()))
        if name.endswith('.json'): return pd.read_json(StringIO(txt.decode()))
        if name.endswith('.xml'):
            root = ET.fromstring(txt.decode()); rows=[]
            for r in list(root): rows.append({c.tag: c.text for c in r})
            return pd.DataFrame(rows)
    except Exception as e:
        st.error("Parse error: "+str(e)); return None

# define features (same sets small)
M1=['MedInc','HouseAge','AveRooms','AveBedrms','Population']
M2=['MedInc','AveRooms','AveOccup','Latitude','Longitude']
M3=['MedInc','HouseAge','AveOccup','Latitude','Population']
all_models = {'LR Model1':M1,'LR Model2':M2,'LR Model3':M3}

# quick train of LR models on app start
data=fetch_california_housing(as_frame=True)
X=data.frame.drop(columns=['MedHouseVal']); y=data.frame['MedHouseVal']
Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=0.2,random_state=42)
trained = {}
for name, feats in all_models.items():
    s=StandardScaler(); Xt=s.fit_transform(Xtr[feats]); lr=LinearRegression().fit(Xt,ytr)
    trained[name] = {'model':lr,'scaler':s,'feats':feats}

uploaded = st.file_uploader("Upload CSV / JSON / XML")
if uploaded:
    df = parse_upload(uploaded)
    if df is not None:
        st.write("Preview:", df.head())
        model_choice = st.selectbox("Choose model", list(trained.keys()))
        info = trained[model_choice]; feats = info['feats']
        # try map columns case-insensitively
        cols_map = {c.lower().strip():c for c in df.columns}
        if not all(f.lower() in cols_map for f in feats):
            st.error("Missing required features for chosen model: " + ", ".join(feats))
        else:
            df_in = df[[cols_map[f.lower()] for f in feats]].astype(float)
            Xp = info['scaler'].transform(df_in)
            preds = info['model'].predict(Xp)
            out = df_in.copy(); out['Predicted_MedHouseVal']=preds
            st.dataframe(out)
            st.download_button("Download CSV", out.to_csv(index=False).encode(), "preds.csv")
