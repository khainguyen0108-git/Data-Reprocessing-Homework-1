import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

data=pd.read_csv("greengrocer_sales.csv")
data=data.drop_duplicates()

# convert all anomalies into NaN
string_columns=["item_name","customer_region"]
numeric_columns=["quantity", "price_per_unit"]
data[numeric_columns]=data[numeric_columns].apply(pd.to_numeric,errors="coerce")
data[numeric_columns]=np.where(data[numeric_columns]<0,np.nan,data[numeric_columns])

date_columns=["order_date"]
data[date_columns]=data[date_columns].apply(pd.to_datetime,errors="coerce")

# convert the customer region's values columns into title case values
data["customer_region"] = data["customer_region"].str.title()

#Imputing
imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
data[numeric_columns] = imputer.fit_transform(data[numeric_columns])

#Creating total_spent columns(dependent variable)
data["total_spent"] = data["quantity"] * data["price_per_unit"]

#Independent and dependent columns
X = data[["customer_region", "item_name", "quantity", "price_per_unit"]]
y = data["total_spent"]

#Encoding
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(sparse_output=False), string_columns)], remainder='passthrough')
X = np.array(ct.fit_transform(X))

# Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)

# Feature Scaling
sc = StandardScaler()
X_train[:, -2:] = sc.fit_transform(X_train[:, -2:])
X_test[:, -2:] = sc.transform(X_test[:, -2:])
