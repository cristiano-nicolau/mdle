## Cristiano Nicolau - 108536

from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import KMeans
from pyspark.ml.linalg import Vectors, DenseVector
from pyspark.sql.types import IntegerType, FloatType
from pyspark.sql import functions as F
import numpy as np
import math
from collections import defaultdict
import json

class BFRClustering:
    def __init__(self, spark, k=8, chunk_size=15000, threshold=None, data=None):
        self.data = data
        self.num_features = len(data.columns) - 1
        self.spark = spark
        self.k = k
        self.chunk_size = chunk_size
        self.threshold = threshold if threshold is not None else 2 * np.sqrt(num_features)
        # Clusters
        self.DS = {}
        self.CS = {}
        self.RS = []
        self.track_to_DS = {}
        self.track_to_CS = {}
        self.track_to_RS = []

    def mahalanobis(self, x, stats):
        ''' Calculate the Mahalanobis distance.'''
        N, SUM, SUMSQ = stats["N"], stats["SUM"], stats["SUMSQ"]
        centroid = SUM / N
        var = (SUMSQ / N) - (centroid ** 2)
        var[var == 0] = 1e-9
        return np.sqrt(np.sum(((x - centroid) ** 2) / var))

    def initialize_clusters(self):
        ''' Initialize clusters using KMeans on the first chunk of data.'''
        first_chunk = self.data.filter((F.col("track_id") >= 0) & (F.col("track_id") < self.chunk_size))
        model = KMeans(k=self.k, seed=42, featuresCol="features").fit(first_chunk)
        preds = model.transform(first_chunk).select("track_id", "features", "prediction")

        for row in preds.collect():
            cid = row["prediction"]
            x = np.array(row["features"])
            tid = row["track_id"]
            if cid not in self.DS:
                self.DS[cid] = {"N": 0, "SUM": np.zeros_like(x), "SUMSQ": np.zeros_like(x)}
            self.DS[cid]["N"] += 1
            self.DS[cid]["SUM"] += x
            self.DS[cid]["SUMSQ"] += x ** 2
            self.track_to_DS[tid] = cid

    def process_chunks(self):
        ''' Process the data in chunks and update clusters.'''
        current = self.chunk_size
        while True:
            chunk = self.data.filter((F.col("track_id") >= current) & (F.col("track_id") < current + self.chunk_size))
            if chunk.count() == 0:
                break
            current += self.chunk_size

            for row in chunk.collect():
                tid = row["track_id"]
                x = np.array(row["features"])

                # DS
                min_dist, best_cid = float("inf"), None
                for cid, stats in self.DS.items():
                    dist = self.mahalanobis(x, stats)
                    if dist < min_dist:
                        min_dist = dist
                        best_cid = cid

                if min_dist < self.threshold:
                    self.DS[best_cid]["N"] += 1
                    self.DS[best_cid]["SUM"] += x
                    self.DS[best_cid]["SUMSQ"] += x ** 2
                    self.track_to_DS[tid] = best_cid
                    continue

                # CS
                min_dist, best_cid = float("inf"), None
                for cid, stats in self.CS.items():
                    dist = self.mahalanobis(x, stats)
                    if dist < min_dist:
                        min_dist = dist
                        best_cid = cid

                if min_dist < self.threshold:
                    self.CS[best_cid]["N"] += 1
                    self.CS[best_cid]["SUM"] += x
                    self.CS[best_cid]["SUMSQ"] += x ** 2
                    self.track_to_CS[tid] = best_cid
                    continue

                self.RS.append((tid, x))
                self.track_to_RS.append(tid)

            if len(self.RS) >= 100:
                self.cluster_RS()

    def cluster_RS(self):
        ''' Cluster the RS points using KMeans and update CS clusters.'''
        rows = [(tid, Vectors.dense(x)) for tid, x in self.RS]
        rs_df = self.spark.createDataFrame(rows, ["track_id", "features"])
        k_rs = min(10, len(self.RS) // 10)
        model = KMeans(k=k_rs, seed=42, featuresCol="features").fit(rs_df)
        preds = model.transform(rs_df).collect()

        new_clusters = defaultdict(list)
        new_ids = defaultdict(list)
        for row in preds:
            new_clusters[row["prediction"]].append(np.array(row["features"]))
            new_ids[row["prediction"]].append(row["track_id"])

        self.RS.clear()
        for cid, points in new_clusters.items():
            if len(points) > 1:
                points = np.array(points)
                self.CS[len(self.CS)] = {
                    "N": len(points),
                    "SUM": np.sum(points, axis=0),
                    "SUMSQ": np.sum(points ** 2, axis=0)
                }
                for tid in new_ids[cid]:
                    self.track_to_CS[tid] = len(self.CS) - 1
            else:
                self.RS.append((new_ids[cid][0], points[0]))
                self.track_to_RS.append(new_ids[cid][0])

    def merge_CS_to_DS(self):
        ''' Merge CS clusters into DS clusters based on the threshold distance.'''
        for csid in list(self.CS.keys()):
            stats = self.CS[csid]
            centroid = stats["SUM"] / stats["N"]
            for dsid, ds_stats in self.DS.items():
                dist = self.mahalanobis(centroid, ds_stats)
                if dist < self.threshold:
                    self.DS[dsid]["N"] += stats["N"]
                    self.DS[dsid]["SUM"] += stats["SUM"]
                    self.DS[dsid]["SUMSQ"] += stats["SUMSQ"]
                    for tid, cid in list(self.track_to_CS.items()):
                        if cid == csid:
                            self.track_to_DS[tid] = dsid
                            del self.track_to_CS[tid]
                    del self.CS[csid]
                    break

    def save_result(self, output_path):
        final_output = {"DS": defaultdict(list), "CS": defaultdict(list), "RS": {}}
        for tid, cid in self.track_to_DS.items():
            final_output["DS"][str(cid)].append(tid)
        for tid, cid in self.track_to_CS.items():
            final_output["CS"][str(cid)].append(tid)
        for idx, tid in enumerate(self.track_to_RS):
            final_output["RS"][str(idx)] = tid

        with open(output_path, "w") as f:
            json.dump(final_output, f, indent=4)

# Spark session
spark = SparkSession.builder \
    .appName("BFR") \
    .config("spark.driver.memory", "8g") \
    .config("spark.executor.memory", "8g") \
    .master("local[*]") \
    .getOrCreate()

# Data loading and preprocessing
file_path = "_input/fma_metadata/features.csv"
data_rdd = spark.read.csv(file_path, header=True)
data_rdd = data_rdd.subtract(data_rdd.limit(4))

num_features = len(data_rdd.columns) - 1
columns = data_rdd.columns
data_df = data_rdd.select(
    data_rdd[columns[0]].cast(IntegerType()).alias("track_id"),
    *[data_rdd[c].cast(FloatType()).alias(c) for c in columns[1:]]
)

def compute_stats(rows):
    sums = np.zeros(num_features)
    sumsqs = np.zeros(num_features)
    count = 0
    for row in rows:
        x = np.array(row[1:]).astype(float)
        sums += x
        sumsqs += x**2
        count += 1
    return [(sums, sumsqs, count)]

def normalize_row(row):
    x = np.array(row[1:]).astype(float)
    x_norm = (x - means_b.value) / (stds_b.value + 1e-9) 
    return (row[0], Vectors.dense(x_norm))

partial = data_df.rdd.mapPartitions(compute_stats)
sums, sumsqs, count = partial.reduce(lambda a, b: (a[0]+b[0], a[1]+b[1], a[2]+b[2]))

means = sums / count
stds = np.sqrt((sumsqs / count) - (means ** 2))
means_b = spark.sparkContext.broadcast(means)
stds_b = spark.sparkContext.broadcast(stds)

data_df = data_df.rdd \
    .map(normalize_row) \
    .toDF(["track_id", "features"])

data = data_df.repartition(8).cache()

# Based on the analysis in the first exercise, the best k is 8 with the highest average density
bfr = BFRClustering(spark, k=8, chunk_size=15000, threshold=2 * np.sqrt(num_features), data=data)
bfr.initialize_clusters()
bfr.process_chunks()
bfr.merge_CS_to_DS()
bfr.save_result("_output/bfr_result.json")

# Stop the Spark session
spark.stop()



