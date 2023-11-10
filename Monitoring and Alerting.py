# Databricks notebook source
# MAGIC %md
# MAGIC #Monitoring and Alerting

# COMMAND ----------

# MAGIC %md
# MAGIC ##Installing dependencies and making necessary imports

# COMMAND ----------

!pip install mlflow

# COMMAND ----------

with open('tokenfile', 'w') as f:
    f.write(dbutils.secrets.get(scope="tokens", key="dbtoken"))
!databricks configure --host https://adb-6724577987585661.1.azuredatabricks.net/ --token-file tokenfile
!rm -rf tokenfile

# COMMAND ----------

import mlflow
import smtplib
from email.mime.text import MIMEText
import pandas as pd
from sklearn.metrics import accuracy_score
import joblib

# COMMAND ----------

# MAGIC %md
# MAGIC ##Email Alert Function

# COMMAND ----------

def send_email_alert(subject, message):
    # Email configuration
    sender_email = "kbhavesh@sigmoidanalytics.com"
    recipient_email = "kbhavesh@sigmoidanalytics.com"
    smtp_server = "smtp.gmail.com"
    smtp_port = 587
    smtp_username = "kbhavesh@sigmoidanalytics.com"
    smtp_password = dbutils.secrets.get(scope="tokens", key="emailtoken")

    # Create the email message
    email_message = MIMEText(message)
    email_message["Subject"] = subject
    email_message["From"] = sender_email
    email_message["To"] = recipient_email

    # Send the email
    with smtplib.SMTP(smtp_server, smtp_port) as server:
        server.starttls()
        server.login(smtp_username, smtp_password)
        server.sendmail(sender_email, [recipient_email], email_message.as_string())

# COMMAND ----------

# MAGIC %md
# MAGIC ##Monitor Model Function

# COMMAND ----------

mlflow.set_experiment("/Users/bhaveshkak26122000@gmail.com/my_experiment")

# COMMAND ----------

def monitor_model():
    # Load the latest registered model from MLflow
    registered_model_name = "titanic_model"
    client = mlflow.tracking.MlflowClient()
    model_version = client.get_latest_versions(registered_model_name, stages=["Production"])[0]
    model_uri = f"runs:/{model_version.run_id}/model"

    # Load the model from the URI
    model = mlflow.pyfunc.load_model(model_uri)

    # Load the test data
    test_data = pd.read_csv("/dbfs/mnt/datamount/data/test.csv")
    test_labels = pd.read_csv("/dbfs/mnt/datamount/data/gender_submission.csv")

    #Preprocessing test data to remove NaN values
    test_df = pd.concat([test_data, test_labels["Survived"]], axis=1)
    test_df = test_df.drop(columns=['Cabin'])
    test_df = test_df.dropna()
    test_df = test_df.drop_duplicates()
    test_df['FareZScore'] = abs(test_df['Fare'] - test_df['Fare'].mean()) / test_df['Fare'].std()
    df_no_outliers = test_df[test_df['FareZScore'] <= (threshold := 0.1)].drop(columns=['FareZScore'])

    # Extract features and target from the test data
    features = ["Pclass", "Age", "SibSp", "Parch", "Fare"]
    target = "Survived"
    X_test = df_no_outliers[features]
    y_test = df_no_outliers[target]

    # Make predictions using the model
    y_pred = model.predict(X_test.fillna(0))

    # Calculate model accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(accuracy)

    # Log the model accuracy in MLflow
    with mlflow.start_run():
        mlflow.log_metric("model_accuracy", accuracy)

    # Check for conditions that trigger email alerts
    if accuracy < 0.9:
        send_email_alert("Model Accuracy Alert", "Model accuracy is below the threshold!")

# COMMAND ----------

# MAGIC %md
# MAGIC ##Call Monitor Model Function

# COMMAND ----------

monitor_model()

# COMMAND ----------


