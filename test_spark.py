# Save as test_spark.py and run
from pyspark.sql import SparkSession
import pyspark

print(f"PySpark version: {pyspark.__version__}")

# Create Spark session
spark = SparkSession.builder \
    .appName("Project2-Test") \
    .master("local[*]") \
    .getOrCreate()

# Suppress verbose logging
spark.sparkContext.setLogLevel("ERROR")

# Quick test
data = [(1, "World", 100),
        (2, "Sports", 200),
        (3, "Business", 150)]

df = spark.createDataFrame(
    data, ["id", "category", "count"])

print("\n✅ PySpark working correctly")
print(f"Spark version: {spark.version}")
df.show()

spark.stop()