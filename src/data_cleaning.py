import pandas as pd

def data_clean_and_handling_missing_values(df):
    #Removing Uncessary Columns
    df= df.drop("Unnamed: 0", axis=1)
    
    #Handling Missing Values
    df["Saving accounts"] = df["Saving accounts"].fillna("Unknown")
    df["Checking account"] = df["Checking account"].fillna("Unknown")
    
    return df

    
    

    