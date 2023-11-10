# Databricks notebook source
!pip install mlflow
with open('tokenfile', 'w') as f:
    f.write(dbutils.secrets.get(scope="tokens", key="dbtoken"))
!databricks configure --host https://adb-6724577987585661.1.azuredatabricks.net/ --token-file tokenfile
!rm -rf tokenfile

# COMMAND ----------

import mlflow

# COMMAND ----------

registered_model_name = "titanic_model"
client = mlflow.tracking.MlflowClient()
model_version = client.get_latest_versions(registered_model_name)[0]

# COMMAND ----------

model_version.run_id

# COMMAND ----------

metrics = client.get_run(model_version.run_id).data.metrics

# COMMAND ----------

if metrics.get("accuracy") > 0.4 and model_version.current_stage != (prod := "Production"):
    client.transition_model_version_stage(
        name=registered_model_name,
        version=model_version.version,
        stage=prod,
    )

# COMMAND ----------

model_version.current_stage

# COMMAND ----------

model_version.version

# COMMAND ----------


