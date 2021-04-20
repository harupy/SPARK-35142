from pyspark.ml.classification import LogisticRegression, OneVsRest
from pyspark.ml.linalg import Vectors
from pyspark.sql import SparkSession
from sklearn.datasets import load_irisspark = SparkSession.builder.getOrCreate()

X, y = load_iris(return_X_y=True)
df = spark.createDataFrame(
 [(Vectors.dense(features), int(label)) for features, label in zip(X, y)], ["features", "label"]
)
train, test = df.randomSplit([0.8, 0.2])

lor = LogisticRegression(maxIter=5)
ovr = OneVsRest(classifier=lor)ovrModel = ovr.fit(train)
pred = ovrModel.transform(test)

pred.printSchema()
# This prints out:
# root
#  |-- features: vector (nullable = true)
#  |-- label: long (nullable = true)
#  |-- rawPrediction: string (nullable = true)  # <- should not be string
#  |-- prediction: double (nullable = true)
