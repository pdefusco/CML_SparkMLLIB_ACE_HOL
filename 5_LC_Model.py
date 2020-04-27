from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import VectorAssembler, StandardScaler, Imputer
from pyspark.ml import PipelineModel
from pyspark.ml.linalg import DenseVector
import numpy as np
from pyspark.mllib.stat import Statistics

from pyspark.sql import SparkSession

spark = SparkSession\
    .builder\
    .appName("PythonSQL")\
    .config("spark.hadoop.fs.s3a.s3guard.ddb.region","us-east-1")\
    .config("spark.yarn.access.hadoopFileSystems","s3a://demo-aws-2/")\
    .getOrCreate()
    
modelPipeline = PipelineModel.load("s3a://demo-aws-2/datalake/pdefusco/pipeline")

from pyspark.sql.types import *

feature_schema = StructType([StructField("acc_now_delinq", FloatType(), True),
StructField("acc_open_past_24mths", FloatType(), True),
StructField("annual_inc", FloatType(), True),
StructField("avg_cur_bal", FloatType(), True),
StructField("funded_amnt", FloatType(), True)])

#data = {"feature":"0,8,65000,10086,12000"}
#{"result": 0}

def predict(data):
  
  request_data = data["feature"].split(",")
  
  df = spark.createDataFrame([
    (
    float(request_data[0]), 
    float(request_data[1]),
    float(request_data[2]),
    float(request_data[3]),
    float(request_data[4])
    )
  ], schema=feature_schema)
  
  pred = modelPipeline.transform(df).collect()[0].prediction
  
  return {"result": pred}
    
    
    