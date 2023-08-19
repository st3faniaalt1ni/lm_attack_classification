import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from scipy.sparse import hstack, csr_matrix
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, roc_curve, auc
from sklearn.model_selection import train_test_split

arithmetic = ["Task", "ProcessID", "ParentProcessId", "SourcePort", "DestinationPort",
              "path_count_in_norm", "image_count_in_norm",
              "name_count_in_norm", "extension_count_in_norm"]
ambg = ["Unnamed: 0", "ThreadID", "Version", "EventRecordID", "ProcessGuid",
        "EventID", "ProcessId"]
categorical = ["IntegrityLevel", "Protocol", "Initiated", "SourceIsIpv6", "SourceHostname", "DestinationIsIpv6",
               "EventType", "QueryStatus", "time_category", "SourceIpType", "destIpType", "C:\\Users", "T1053", "T1031",
               "T1050", "T1122", "T1101", "Context", "DeviceConnectedOrUpdated",
               "InvDB-Ver", "InvDB-CompileTimeClaim", "InvDB-Pub", "InvDB-Path", "InvDB-DriverVer", "T1042", "Caution",
               "RDP", "DLL", "EXE", "T1023", "T1060", "RunKey", "Downloads", "T1158", "T1037",
               "T1484", "Tamper-Winlogon", "T1183", "IFEO", "InvDB", "T1176", "Usermode", "ProcessHostingdotNETCode",
               "Suspicious", "ImageBeginWithBackslash", "T1099", "T1089", "Tamper-Defender",
               "T1137", "Alert", "Sysinternals Tool Used", "ModifyRemoteDesktopState", "Proxy", "Metasploit", "T1088",
               "rel_to_windows_sys", "rel_to_user_app_data", "rel_to_program_files",
               "UserTaxonomy", "isSidSystem"]

dataset_data_schema = {
    "Task": "float64",
    "ProcessID": "float64",
    "ParentProcessId": "float64",
    "SourcePort": "float64",
    "DestinationPort": "float64",
    "path_count_in_norm": "float64",
    "image_count_in_norm": "float64",
    "name_count_in_norm": "float64",
    "extension_count_in_norm": "float64",
    "ThreadID": "float64",
    "Version": "float64",
    "EventRecordID": "float64",
    "ProcessGuid": "float64",
    "ProcessId.1": "float64",
    "EventID": "str",
    "IntegrityLevel": "str",
    "Protocol": "str",
    "Initiated": "str",
    "SourceIsIpv6": "str",
    "SourceHostname": "str",
    "DestinationIsIpv6": "str",
    "EventType": "str",
    "QueryStatus": "str",
    "time_category": "str",
    "SourceIpType": "str",
    "destIpType": "str",
    "C:\\Users": "str",
    "T1053": "str",
    "T1031": "str",
    "T1050": "str",
    "T1122": "str",
    "T1101": "str",
    "Context": "str",
    "DeviceConnectedOrUpdated": "str",
    "InvDB-Ver": "str",
    "InvDB-CompileTimeClaim": "str",
    "InvDB-Pub": "str",
    "InvDB-Path": "str",
    "InvDB-DriverVer": "str",
    "T1042": "str",
    "Caution": "str",
    "RDP": "str",
    "DLL": "str",
    "EXE": "str",
    "T1023": "str",
    "T1060": "str",
    "RunKey": "str",
    "Downloads": "str",
    "T1158": "str",
    "T1037": "str",
    "T1484": "str",
    "Tamper-Winlogon": "str",
    "T1183": "str",
    "IFEO": "str",
    "InvDB": "str",
    "T1176": "str",
    "Usermode": "str",
    "ProcessHostingdotNETCode": "str",
    "Suspicious": "str",
    "ImageBeginWithBackslash": "str",
    "T1099": "str",
    "T1089": "str",
    "Tamper-Defender": "str",
    "T1137": "str",
    "Alert": "str",
    "Sysinternals Tool Used": "str",
    "ModifyRemoteDesktopState": "str",
    "Proxy": "str",
    "Metasploit": "str",
    "T1088": "str",
    "rel_to_windows_sys": "str",
    "rel_to_user_app_data": "str",
    "rel_to_program_files": "str",
    "UserTaxonomy": "str",
    "isSidSystem": "str",
    "Label": "str"
}


def import_data(path_to_data):
    df = pd.read_csv(path_to_data, low_memory=True, dtype=dataset_data_schema)
    return df


if __name__ == '__main__':
    data_dir = Path(".")

    df = import_data(data_dir.joinpath("features_df_3105_76.csv"))
    df = df.dropna(subset=["Label"])
    y = df["Label"]
    y = np.array(y).astype(np.float64)
    df = df.drop("Label", axis=1, inplace=False)

    df_clean = df.drop(ambg, axis=1)

    df_categorical = df_clean[categorical]
    df_categorical.fillna("-", inplace=True)

    df_arithmetic = df_clean[arithmetic]
    df_categorical.fillna(0, inplace=True)

    print(df_arithmetic.info())

    # Categorical data treatment
    print(f"df categorical {df_categorical.shape}")
    enc = OneHotEncoder(handle_unknown='ignore')
    df_categorical_ohe = enc.fit_transform(df_categorical)
    print(f"df categorical (one hot encoded) {df_categorical_ohe.shape}")
    type(df_categorical_ohe)

    # Arithmetic data treatment
    scaler = MinMaxScaler()
    arithmetic_data_transformed = scaler.fit_transform(df_arithmetic)
    print(f"arithmetic data shape {arithmetic_data_transformed.shape}")

    # Combine DFs to create matrix X
    # Convert arithmetic_data_transformed to a sparse matrix
    arithmetic_data_sparse = csr_matrix(arithmetic_data_transformed)

    # Horizontally stack the two sparse matrices
    X = hstack((df_categorical_ohe, arithmetic_data_sparse))
    X = X.toarray()
    X = np.nan_to_num(X, 0)
    print(f"X shape : {X.shape}")

    # Run ML algorithms
    target_names = ["normal", "EoRS", "EoHT"]
    np.savez("final_dataset.npz", X=X, y=y)

