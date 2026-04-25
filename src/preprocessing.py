from sklearn.preprocessing import OneHotEncoder
import pandas as pd

def encode_data(X_train, X_test):
    # Select categorical columns
    cat_cols = X_train.select_dtypes(include=["object", "category"]).columns

    # Initialize encoder (FIX HERE)
    encoder = OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False)

    # Fit on train, transform both
    X_train_enc = encoder.fit_transform(X_train[cat_cols])
    X_test_enc = encoder.transform(X_test[cat_cols])

    # Convert to DataFrame
    X_train_enc_df = pd.DataFrame(
        X_train_enc,
        columns=encoder.get_feature_names_out(cat_cols),
        index=X_train.index
    )

    X_test_enc_df = pd.DataFrame(
        X_test_enc,
        columns=encoder.get_feature_names_out(cat_cols),
        index=X_test.index
    )

    # Combine
    X_train_final = pd.concat(
        [X_train.drop(columns=cat_cols), X_train_enc_df],
        axis=1
    )

    X_test_final = pd.concat(
        [X_test.drop(columns=cat_cols), X_test_enc_df],
        axis=1
    )

    return X_train_final, X_test_final