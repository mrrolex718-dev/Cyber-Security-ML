from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import os
import uvicorn

app = FastAPI(title="Phishing Detection API")

# ===============================
# Load Model
# ===============================
model_filename = "xgboost_model.joblib"

try:
    model = joblib.load(model_filename)
    print(f"Model loaded successfully from {model_filename}")
except Exception as e:
    print(f"Model loading failed: {e}")
    model = None

# ===============================
# Expected Column Order
# MUST match training dataset
# ===============================
COLUMN_ORDER = [
    'NumDots', 'SubdomainLevel', 'PathLevel', 'UrlLength', 'NumDash',
    'NumDashInHostname', 'AtSymbol', 'TildeSymbol', 'NumUnderscore',
    'NumPercent', 'NumQueryComponents', 'NumAmpersand', 'NumHash',
    'NumNumericChars', 'NoHttps', 'RandomString', 'IpAddress',
    'DomainInSubdomains', 'DomainInPaths', 'HttpsInHostname',
    'HostnameLength', 'PathLength', 'QueryLength', 'DoubleSlashInPath',
    'NumSensitiveWords', 'EmbeddedBrandName', 'PctExtHyperlinks',
    'PctExtResourceUrls', 'ExtFavicon', 'InsecureForms',
    'RelativeFormAction', 'ExtFormAction', 'AbnormalFormAction',
    'PctNullSelfRedirectHyperlinks', 'FrequentDomainNameMismatch',
    'FakeLinkInStatusBar', 'RightClickDisabled', 'PopUpWindow',
    'SubmitInfoToEmail', 'IframeOrFrame', 'MissingTitle',
    'ImagesOnlyInForm', 'SubdomainLevelRT', 'UrlLengthRT',
    'PctExtResourceUrlsRT', 'AbnormalExtFormActionR',
    'ExtMetaScriptLinkRT', 'PctExtNullSelfRedirectHyperlinksRT'
]

# ===============================
# Request Schema
# ===============================
class PredictionInput(BaseModel):
    NumDots: int
    SubdomainLevel: int
    PathLevel: int
    UrlLength: int
    NumDash: int
    NumDashInHostname: int
    AtSymbol: int
    TildeSymbol: int
    NumUnderscore: int
    NumPercent: int
    NumQueryComponents: int
    NumAmpersand: int
    NumHash: int
    NumNumericChars: int
    NoHttps: int
    RandomString: int
    IpAddress: int
    DomainInSubdomains: int
    DomainInPaths: int
    HttpsInHostname: int
    HostnameLength: int
    PathLength: int
    QueryLength: int
    DoubleSlashInPath: int
    NumSensitiveWords: int
    EmbeddedBrandName: int
    PctExtHyperlinks: float
    PctExtResourceUrls: float
    ExtFavicon: int
    InsecureForms: int
    RelativeFormAction: int
    ExtFormAction: int
    AbnormalFormAction: int
    PctNullSelfRedirectHyperlinks: float
    FrequentDomainNameMismatch: int
    FakeLinkInStatusBar: int
    RightClickDisabled: int
    PopUpWindow: int
    SubmitInfoToEmail: int
    IframeOrFrame: int
    MissingTitle: int
    ImagesOnlyInForm: int
    SubdomainLevelRT: int
    UrlLengthRT: int
    PctExtResourceUrlsRT: int
    AbnormalExtFormActionR: int
    ExtMetaScriptLinkRT: int
    PctExtNullSelfRedirectHyperlinksRT: int

# ===============================
# Health Check Endpoint
# ===============================
@app.get("/")
def health_check():
    return {"status": "API Running Successfully 🚀"}

# ===============================
# Prediction Endpoint
# ===============================
@app.post("/predict")
async def predict(data: PredictionInput):

    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        feature_values = data.model_dump()

        input_df = pd.DataFrame([feature_values])
        input_df = input_df[COLUMN_ORDER]

        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        if probability > 0.8:
            risk_level = "High Risk"
        elif probability > 0.5:
            risk_level = "Medium Risk"
        else:
            risk_level = "Low Risk"

        return {
            "prediction": int(prediction),
            "phishing_probability": round(float(probability), 4),
            "risk_level": risk_level
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ===============================
# Run Server (Render Compatible)
# ===============================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("app:app", host="0.0.0.0", port=port)
