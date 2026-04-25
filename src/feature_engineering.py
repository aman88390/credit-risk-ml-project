def feature_engineering(df):
    # Map categorical variables to more meaningful categories
    df["Saving accounts"] = df["Saving accounts"].replace({
        "little": "low",
        "moderate": "medium",
        "quite rich": "high",
        "rich": "high",
        "Unknown": "unknown"
    })
    
    df["Checking account"] = df["Checking account"].replace({
        "little": "low",
        "moderate": "medium",
        "quite rich": "high",
        "rich": "high",
        "Unknown": "unknown"
    })
    
    df["Credit_per_month"] = df["Credit amount"] / df["Duration"]
    
    return df