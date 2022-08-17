#nullable disable

namespace AmericanExpress
{
    using System;
    using System.Collections.Generic;
    using System.Diagnostics;
    using System.Globalization;
    using System.Text.Json.Serialization;
    using System.Drawing.Imaging;
    using System.Linq;
    using System.Data;
    using Windows.Foundation.Collections;
    using System.Security.Cryptography.Xml;
    using Microsoft.ML;
    using Microsoft.ML.Data;
    using Microsoft.ML.Trainers.LightGbm;
    using Microsoft.ML.Calibrators;
    using Newtonsoft.Json.Linq;
    using System.Numerics;

    partial class AmericanExpress
    {
        private const String fileDirectory = @"C:\Users\tordm\Documents\Kaggle Data\American Express\";
        //private const String fileDirectory = @"C:\Users\tomal12\American Express\";
        private const Int32 numberCategories = 13;
        private const Int32 numberBins = 512;
        private static readonly Int32[] categoricalIndexes = new Int32[] { 53, 54, 60, 62, 105, 108, 112, 145, 155, 157, 158, 161, 167 };
        private static readonly NumberFormatInfo numberFormatInfo = new() { NumberDecimalSeparator = "." };
        private static readonly Dictionary<String, Category>[] categoriesLookup = new Dictionary<String, Category>[numberCategories];
        private static readonly SortedList<Int32, TransformerNaN> transformerNaN = new();
        private static readonly Dictionary<Int32, Bin[]> binsLookup = new();
        private static Int32 featuresLength;
        private static List<Double>[] transformers;

        static void Main()
        {
            DataSet trainDataSet = new();
            trainDataSet.Read();

            DataRows testDataRows = new();
            testDataRows.Read();

            WriteFeature(138, trainDataSet, testDataRows);

            //for (Double topRate = .1d; topRate < .9d; topRate += .1d)
            //{
            //    for (Double otherRate = .1d; otherRate <= 1 - topRate; otherRate += .1d)
            //    {
            //        Double metric2 = trainDataSet.Lgbm(topRate, otherRate, false, featureIndexes);
            //        Debug.WriteLine(topRate.ToString() + ";"
            //            + otherRate.ToString() + ";"
            //            + metric2.ToString());
            //    }
            //}

            //var lgbmResults = trainDataSet.Lgbm(.5d, .5d, true, featureIndexes);

            //trainDataSet.StepDiscriminate();
            //trainDataSet.StepDiscriminateReverse();

            List<Int32> featureIndexes = Enumerable.Range(0, featuresLength).Except(new List<Int32>() { 52, 9, 133, 0, 107, 181, 160, 49, 172, 59 }).ToList();
            featureIndexes = new() { 13,
15,
22,
18,
24,
154,
21,
16,
52,
28,
172,
135,
17,
148,
19,
58,
14,
62,
165,
127,
38,
2,
159,
74,
92,
138,
61,
136,
29,
35,
98,
36,
132,
27,
20,
95,
169,
107,
101,
151,
113,
168,
150,
46,
122,
59,
163,
64,
56,
103,
81,
116,
90,
97,
69,
79,
34,
118,
130,
119,
73,
112,
126,
43,
77,
115,
106,
131,
44
};

            (Double metric, _, DA da) = trainDataSet.Discriminate(featureIndexes);

            DataSet.Submit(da, featureIndexes);
        }

        private static void WriteFeature(Int32 featureIndex, DataSet trainDataSet, DataRows testDataRows)
        {
            using StreamWriter debugOutput = new(new FileStream(fileDirectory + "debug.csv", FileMode.Create));

            foreach (Boolean target in trainDataSet.Keys)
            {
                DataRows dataRows = trainDataSet[target];

                foreach (DataRow dataRow in dataRows.Values)
                {
                    debugOutput.WriteLine((target ? "1" : "0") + ";"
                        + dataRow.features[featureIndex].ToString());
                }
            }

            foreach (DataRow dataRow in testDataRows.Values)
            {
                debugOutput.WriteLine("2;"
                    + dataRow.features[featureIndex].ToString());
            }

            debugOutput.Close();
        }
        private static void CvM(DataSet trainDataSet, DataRows testDataRows)
        {

            for (Int32 featureIndex = 0; featureIndex < featuresLength; featureIndex++)
            {
                List<Double> trainFeatures = trainDataSet[false].GetFeatureValues(featureIndex);
                List<Double> testFeatures = testDataRows.GetFeatureValues(featureIndex);

                List<(Double x, Boolean isFirstSample)> items = new();

                foreach (Double x in trainFeatures)
                {
                    items.Add((x, true));
                }

                foreach (Double x in testFeatures)
                {
                    items.Add((x, false));
                }

                items.Sort((firstItem, secondItem) =>
                {
                    if (firstItem.x.Equals(secondItem.x))
                    {
                        return firstItem.isFirstSample.CompareTo(secondItem.isFirstSample);
                    }
                    else
                    {
                        return firstItem.x.CompareTo(secondItem.x);
                    }
                });

                Double rankX = 0;
                Double rankY = 0;
                Double sumY2 = 0;
                Double sumX2 = 0;
                Double sumXY = 0;

                foreach ((Double x, Boolean isSignal) in items)
                {
                    if (isSignal)
                    {
                        rankX++;
                    }
                    else
                    {
                        rankY++;
                    }

                    sumY2 += rankY * rankY;
                    sumXY += rankX * rankY;
                    sumX2 += rankX * rankX;
                }

                items.Clear();

                Double cvm = (rankY / rankX * sumX2 + rankX / rankY * sumY2 - 2d * sumXY) / Math.Pow(rankX + rankY, 2);
                Debug.WriteLine(featureIndex.ToString() + " ; " + cvm.ToString());
            }

        }

        private sealed class DataSet : Dictionary<Boolean, DataRows> // true / false
        {
            private Int32 workersLock;
            private (Int32, Double)[] workersResults;

            internal DataSet()
            {
                Add(true, new());
                Add(false, new());
            }

            private List<(Boolean, Double)> GetFeatureValues(Int32 featureIndex)
            {
                List<(Boolean, Double)> featureValues = new();
                featureValues.AddRange(this[true].GetFeatureValues(featureIndex).Select(x => (true, x)).ToList());
                featureValues.AddRange(this[false].GetFeatureValues(featureIndex).Select(x => (false, x)).ToList());
                return featureValues;
            }
            private void FillBins(Int32 featureIndex)
            {
                Debug.WriteLine(featureIndex.ToString());
                List<(Boolean, Double)> featureValues = GetFeatureValues(featureIndex);

                featureValues.Sort((firstItem, secondItem) =>
                {
                    return firstItem.Item2.CompareTo(secondItem.Item2);
                });

                Bin[] bins = new Bin[numberBins];

                for (Int32 binNumber = 0; binNumber < numberBins; binNumber++)
                {
                    bins[binNumber] = new Bin(featureValues[(Int32)(featureValues.Count / (Double)numberBins * (binNumber + 1) - 1)].Item2);
                }

                Int32 binIndex = 0;

                foreach ((Boolean target, Double featureValue) in featureValues)
                {
                    if (featureValue <= bins[binIndex].upperValue)
                    {
                        bins[binIndex].Increment(target);
                    }
                    else
                    {
                        binIndex++;

                        if (binIndex.Equals(numberBins))
                        {
                            throw new Exception("Arse");
                        }
                    }
                }

                featureValues.Clear();

                Double previousAverage = Double.NaN;

                for (Int32 binNumber = 0; binNumber < numberBins; binNumber++)
                {
                    if (Double.IsNaN(bins[binNumber].Average))
                    {
                        if (!Double.IsNaN(previousAverage))
                        {
                            bins[binNumber].Average = previousAverage;
                        }
                    }

                    previousAverage = bins[binNumber].Average;
                }

                previousAverage = Double.NaN;

                for (Int32 binNumber = numberBins-1; binNumber >= 0; binNumber--)
                {
                    if (Double.IsNaN(bins[binNumber].Average))
                    {
                        if (!Double.IsNaN(previousAverage))
                        {
                            bins[binNumber].Average = previousAverage;
                        }
                    }

                    previousAverage = bins[binNumber].Average;
                }

                do { Thread.Sleep(1); } while (Interlocked.CompareExchange(ref workersLock, 1, 0) == 1);
                binsLookup.Add(featureIndex, bins);
                Interlocked.Exchange(ref workersLock, 0);
            }
            private void DiscriminateWorker(Object parameter)
            {
                Object[] objects = (Object[])parameter;
                Int32 workerNumber = (Int32)objects[0];
                List<Int32> featureIndexes = new();
                featureIndexes.AddRange((List<Int32>)objects[1]);

                Double metricMaximum = Double.MinValue;
                Int32 featureIndexMaximum = -1;

                for (Int32 featureIndex = workerNumber; featureIndex < featuresLength; featureIndex += Environment.ProcessorCount)
                {
                    if (!featureIndexes.Contains(featureIndex))
                    {
                        featureIndexes.Add(featureIndex);
                        (Double metric, Double metricError, _) = Discriminate(featureIndexes);

                        if (metric - 3 * metricError > metricMaximum)
                        {
                            metricMaximum = metric - 3 * metricError;
                            featureIndexMaximum = featureIndex;
                            //Debug.WriteLine("    " + workerNumber.ToString() + " ; " + featureIndex.ToString() + " ; " + metric.ToString() + " ; " + metricError.ToString());
                        }

                        featureIndexes.Remove(featureIndex);
                    }
                }

                do { Thread.Sleep(1); } while (Interlocked.CompareExchange(ref workersLock, 1, 0) == 1);
                workersResults[workerNumber] = (featureIndexMaximum, metricMaximum);
                Interlocked.Exchange(ref workersLock, 0);
            }
            private void DiscriminateReverseWorker(Object parameter)
            {
                Object[] objects = (Object[])parameter;
                Int32 workerNumber = (Int32)objects[0];
                List<Int32> featureIndexes = new();
                featureIndexes.AddRange((List<Int32>)objects[1]);

                Double metricMaximum = Double.MinValue;
                Int32 featureIndexMaximum = -1;

                for (Int32 featureIndex = workerNumber; featureIndex < featuresLength; featureIndex += Environment.ProcessorCount)
                {
                    if (featureIndexes.Contains(featureIndex))
                    {
                        featureIndexes.Remove(featureIndex);
                        (Double metric, Double metricError, _) = Discriminate(featureIndexes);

                        if (metric - 3 * metricError > metricMaximum)
                        {
                            metricMaximum = metric - 3 * metricError;
                            featureIndexMaximum = featureIndex;
                            //Debug.WriteLine("    " + workerNumber.ToString() + " ; " + featureIndex.ToString() + " ; " + metric.ToString() + " ; " + metricError.ToString());
                        }

                        featureIndexes.Add(featureIndex);
                    }
                }

                do { Thread.Sleep(1); } while (Interlocked.CompareExchange(ref workersLock, 1, 0) == 1);
                workersResults[workerNumber] = (featureIndexMaximum, metricMaximum);
                Interlocked.Exchange(ref workersLock, 0);
            }
            private void FillTransformers(Int32 featureIndex)
            {
                List<Double> featureValues = this[true].GetFeatureValues(featureIndex);
                featureValues.Sort();
                transformers[featureIndex] = featureValues;
            }

            internal (Double, Double, DA) Discriminate(List<Int32> featureIndexes)
            {
                DA da = new(featureIndexes.Count, 1);

                foreach (Boolean target in Keys)
                {
                    DataRows dataRows = this[target];

                    foreach (String customerId in dataRows.Keys)
                    {
                        DataRow dataRow = dataRows[customerId];
                        List<Double> features = new();

                        foreach (Int32 featureIndex in featureIndexes)
                        {
                            features.Add(dataRow.features[featureIndex]);
                        }

                        da.Add(features.ToArray(), target);
                    }
                }

                da.Solve();
                Double auc = .5d;
                Double metricError = .5d;
                Double recallAt4 = 0;

                if (da.SolutionExists)
                {
                    List<(Double outcome, Boolean target)> daItems = new();

                    foreach (Boolean target in Keys)
                    {
                        foreach (String customerId in this[target].Keys)
                        {
                            daItems.Add((this[target][customerId].Outcome(da, featureIndexes), target));
                        }
                    }

                    daItems.Sort((firstItem, secondItem) =>
                    {
                        if (firstItem.outcome.Equals(secondItem.outcome))
                        {
                            return secondItem.target.CompareTo(firstItem.target);
                        }
                        else
                        {
                            return firstItem.outcome.CompareTo(secondItem.outcome);
                        }
                    });

                    Double countBackground = 0;
                    Double countSignal = 0;
                    Double aucPessimistic = 0;
                    Double thresholdAt4 = .96d * (20d * daItems.Count - 19d * daItems.Count(x => x.target));
                    Boolean thresholdAt4Found = false;
                    Double countSignalAt4 = 0;

                    foreach ((Double outcome, Boolean target) in daItems)
                    {
                        if (target)
                        {
                            countSignal++;
                            aucPessimistic += countBackground;
                        }
                        else
                        {
                            countBackground += 20d;
                        }

                        if (!thresholdAt4Found)
                        {
                            if (countSignal + countBackground >= thresholdAt4)
                            {
                                thresholdAt4Found = true;
                                countSignalAt4 = countSignal;
                            }
                        }
                    }

                    aucPessimistic /= countSignal * countBackground;
                    Double recallAt4Pessimistic = 1d - countSignalAt4 / countSignal;
                    { }
                    daItems.Sort((firstItem, secondItem) =>
                    {
                        if (firstItem.outcome.Equals(secondItem.outcome))
                        {
                            return firstItem.target.CompareTo(secondItem.target);
                        }
                        else
                        {
                            return firstItem.outcome.CompareTo(secondItem.outcome);
                        }
                    });

                    countBackground = 0;
                    countSignal = 0;
                    Double aucOptimistic = 0;
                    thresholdAt4 = .96d * (20d * daItems.Count - 19d * daItems.Count(x => x.target));
                    thresholdAt4Found = false;
                    countSignalAt4 = 0;

                    foreach ((Double outcome, Boolean target) in daItems)
                    {
                        if (target)
                        {
                            countSignal++;
                            aucOptimistic += countBackground;
                        }
                        else
                        {
                            countBackground += 20;
                        }

                        if (!thresholdAt4Found)
                        {
                            if (countSignal + countBackground >= thresholdAt4)
                            {
                                thresholdAt4Found = true;
                                countSignalAt4 = countSignal;
                            }
                        }
                    }

                    aucOptimistic /= countSignal * countBackground;
                    Double recallAt4Optimistic = 1d - countSignalAt4 / countSignal;

                    Double aucErrorSampleSize = .25d * (1d / countSignal + 1d / countBackground);
                    metricError = Math.Sqrt(
                          Math.Pow(aucErrorSampleSize, 2)
                        + Math.Pow(aucOptimistic - aucPessimistic, 2)
                        + Math.Pow(recallAt4Optimistic - recallAt4Pessimistic, 2)
                        );
                    auc = (aucPessimistic + aucOptimistic) / 2d;
                    recallAt4 = (recallAt4Pessimistic + recallAt4Optimistic) / 2d;
                    daItems.Clear();
                }

                return (auc - .5d + .5d * recallAt4, metricError, da);
            }
            internal void Read()
            {
                Dictionary<String, Boolean> lookup = new();

                using (StreamReader dataInput = new(new FileStream(fileDirectory + "train_labels.csv", FileMode.Open)))
                {
                    String[] header = dataInput.ReadLine().Split(',');

                    while (dataInput.EndOfStream is false)
                    {
                        String[] dataStrings = dataInput.ReadLine().Split(',');
                        lookup.Add(dataStrings[0], dataStrings[1].Equals("1"));
                    }

                    dataInput.Close();
                }


                using (StreamReader dataInput = new(new FileStream(fileDirectory + "train_data.csv", FileMode.Open)))
                {
                    String[] header = dataInput.ReadLine().Split(',');

                    Int32 n = 0;
                    Int32 seconds = 0;
                    String previousCustomerId = String.Empty;
                    DataRow dataRow = null;
                    Stopwatch stopwatch = Stopwatch.StartNew();

                    while (dataInput.EndOfStream is false)
                    {
                        String[] dataStrings = dataInput.ReadLine().Split(',');
                        String customerId = dataStrings[0];

                        if (!customerId.Equals(previousCustomerId))
                        {
                            if (lookup.ContainsKey(previousCustomerId))
                            {
                                this[lookup[previousCustomerId]][previousCustomerId] = dataRow;
                            }

                            dataRow = new();
                        }

                        dataRow.Update(dataStrings);
                        previousCustomerId = customerId;

7                        n++;

                        if (!stopwatch.Elapsed.Seconds.Equals(seconds))
                        {
                            Debug.WriteLine((n / 5_531_451d * 100d).ToString("##.0"));
                            seconds = stopwatch.Elapsed.Seconds;
                        }
                    }

                    dataInput.Close();

                    if (lookup.ContainsKey(previousCustomerId))
                    {
                        this[lookup[previousCustomerId]][previousCustomerId] = dataRow;
                    }
                }



                featuresLength = this.First().Value.First().Value.features.Length;


                Debug.WriteLine("AmalgamateCategories");

                for (Int32 i = 0; i < numberCategories; i++)
                {
                    categoriesLookup[i] = new();
                }

                this[true].AmalgamateCategories(true);
                this[false].AmalgamateCategories(false);


                Debug.WriteLine("TransformCategorical");
                Parallel.ForEach(Values, new ParallelOptions { MaxDegreeOfParallelism = Environment.ProcessorCount }, x => x.TransformCategorical());


                Debug.WriteLine("AmalgamateNaN");

                for (Int32 featureIndex = 0; featureIndex < featuresLength; featureIndex++)
                {
                    transformerNaN.Add(featureIndex, new());
                }

                this[true].AmalgamateNaN();
                this[false].AmalgamateNaN();


                Debug.WriteLine("ReplaceNaN");
                Parallel.ForEach(Values, new ParallelOptions { MaxDegreeOfParallelism = Environment.ProcessorCount }, x => x.ReplaceNaN());


                Debug.WriteLine("FillBins");
                workersLock = 0;
                Parallel.For(0, featuresLength, new ParallelOptions { MaxDegreeOfParallelism = Environment.ProcessorCount }, x => FillBins(x));

                Debug.WriteLine("SetFeatureValues");
                Parallel.ForEach(Values, new ParallelOptions { MaxDegreeOfParallelism = Environment.ProcessorCount }, x => x.SetFeatureValues());


                Debug.WriteLine("FillTransformers");
                transformers = new List<Double>[featuresLength];
                Parallel.For(0, featuresLength, new ParallelOptions { MaxDegreeOfParallelism = Environment.ProcessorCount }, x => FillTransformers(x));

                Debug.WriteLine("Probability integral transform");
                Parallel.ForEach(Values, new ParallelOptions { MaxDegreeOfParallelism = Environment.ProcessorCount }, x => x.ProbabilityIntegralTransform());
            }
            internal void StepDiscriminate()
            {
                List<Int32> featureIndexes = new();// { 16, 18, 20, 51, 22, 25, 27, 24, 17, 57, 55, 31, 65 };

                while (true)
                {
                    Thread[] workers = new Thread[Environment.ProcessorCount];
                    workersResults = new (Int32, Double)[Environment.ProcessorCount];
                    workersLock = 0;

                    for (Int32 workerNumber = 0; workerNumber < Environment.ProcessorCount; workerNumber++)
                    {
                        workers[workerNumber] = new Thread(DiscriminateWorker);
                        workers[workerNumber].Start(new Object[] { workerNumber, featureIndexes });
                    }

                    foreach (Thread worker in workers)
                    {
                        worker.Join();
                    }

                    Int32 featureIndexMaximum = -1;
                    Double metricMaximum = Double.MinValue;

                    foreach ((Int32 featureIndexMaximum, Double metricMaximum) workerResults in workersResults)
                    {
                        if (workerResults.metricMaximum > metricMaximum)
                        {
                            metricMaximum = workerResults.metricMaximum;
                            featureIndexMaximum = workerResults.featureIndexMaximum;
                        }
                    }

                    featureIndexes.Add(featureIndexMaximum);
                    Debug.WriteLine(featureIndexMaximum.ToString() + " ; " + metricMaximum.ToString());
                }
            }
            internal void StepDiscriminateReverse()
            {
                List<Int32> featureIndexes = Enumerable.Range(0, featuresLength).Except(new Int32[] { 52, 9, 133, 0, 107, 181, 160, 49, 172, 59 }).ToList();

                while (true)
                {
                    Thread[] workers = new Thread[Environment.ProcessorCount];
                    workersResults = new (Int32, Double)[Environment.ProcessorCount];
                    workersLock = 0;

                    for (Int32 workerNumber = 0; workerNumber < Environment.ProcessorCount; workerNumber++)
                    {
                        workers[workerNumber] = new Thread(DiscriminateReverseWorker);
                        workers[workerNumber].Start(new Object[] { workerNumber, featureIndexes });
                    }

                    foreach (Thread worker in workers)
                    {
                        worker.Join();
                    }

                    Int32 featureIndexMaximum = -1;
                    Double metricMaximum = Double.MinValue;

                    foreach ((Int32 featureIndexMaximum, Double metricMaximum) workerResults in workersResults)
                    {
                        if (workerResults.metricMaximum > metricMaximum)
                        {
                            metricMaximum = workerResults.metricMaximum;
                            featureIndexMaximum = workerResults.featureIndexMaximum;
                        }
                    }

                    featureIndexes.Remove(featureIndexMaximum);
                    Debug.WriteLine(featureIndexMaximum.ToString() + " ; " + metricMaximum.ToString());
                }
            }
            internal Double Lgbm(Double topRate, Double otherRate, Boolean submit, List<Int32> featureIndexes)
            {
                LightGbmBinaryTrainer.Options options = new()
                {
                    Booster = new GossBooster.Options
                    {
                        TopRate = topRate,
                        OtherRate = otherRate
                    },
                    EvaluationMetric = LightGbmBinaryTrainer.Options.EvaluateMetricType.Logloss
                };

                MLContext mlContext = new();
                LightGbmBinaryTrainer lightGbmBinaryTrainer = mlContext.BinaryClassification.Trainers.LightGbm(options);



                List<MLData> preData = new();

                foreach (Boolean target in Keys)
                {
                    foreach (DataRow dataRow in this[target].Values)
                    {
                        preData.Add(new MLData(target, dataRow.features, featureIndexes));
                    }
                }

                IDataView data = mlContext.Data.LoadFromEnumerable(preData);

                DataOperationsCatalog.TrainTestData splitData = mlContext.Data.TrainTestSplit(data, .5d);

                if (submit)
                {
                    var binaryPredictionTransformer = lightGbmBinaryTrainer.Fit(data);

                    using StreamWriter submission = new(new FileStream(fileDirectory + "submission.csv", FileMode.Create)) { AutoFlush = true };
                    submission.WriteLine("customer_ID,prediction");

                    using (StreamReader dataInput = new(new FileStream(fileDirectory + "test_data.csv", FileMode.Open)))
                    {
                        // test data is sorted and nice
                        String[] header = dataInput.ReadLine().Split(',');
                        String previousCustomerId = String.Empty;
                        DataRow submissionDataRow = null;
                        Int32 n = 0;
                        Int32 seconds = 0;
                        Stopwatch stopwatch = Stopwatch.StartNew();

                        while (dataInput.EndOfStream is false)
                        {
                            String dataString = dataInput.ReadLine();
                            String[] dataStrings = dataString.Split(',');
                            String customerId = dataStrings[0];

                            if (!customerId.Equals(previousCustomerId))
                            {
                                if (submissionDataRow is not null)
                                {
                                    submissionDataRow.TransformCategorical();
                                    submissionDataRow.ReplaceNaN();
                                    submissionDataRow.SetFeatureValues();
                                    submissionDataRow.ProbabilityIntegralTransform();

                                    preData = new List<MLData>() { new MLData(true, submissionDataRow.features, featureIndexes) };
                                    IDataView submissionDataBajs = mlContext.Data.LoadFromEnumerable(preData);
                                    submissionDataBajs = binaryPredictionTransformer.Transform(submissionDataBajs);
                                    List<Prediction> predictionsBajs = mlContext.Data.CreateEnumerable<Prediction>(submissionDataBajs, reuseRowObject: false, ignoreMissingColumns: true).ToList();

                                    submission.WriteLine(previousCustomerId + "," + predictionsBajs[0].Score.ToString(numberFormatInfo));
                                }

                                submissionDataRow = new();
                            }

                            submissionDataRow.Update(dataStrings);
                            previousCustomerId = customerId;

                            n++;

                            if (!stopwatch.Elapsed.Seconds.Equals(seconds))
                            {
                                Debug.WriteLine((n / 11_363_762d * 100d).ToString("##.0") + "   " + n.ToString("### ### ###"));
                                seconds = stopwatch.Elapsed.Seconds;
                            }
                        }

                        dataInput.Close();

                        submissionDataRow.TransformCategorical();
                        submissionDataRow.ReplaceNaN();
                        submissionDataRow.SetFeatureValues();
                        submissionDataRow.ProbabilityIntegralTransform();

                        preData = new List<MLData>() { new MLData(true, submissionDataRow.features, featureIndexes) };
                        IDataView submissionData = mlContext.Data.LoadFromEnumerable(preData);
                        submissionData = binaryPredictionTransformer.Transform(submissionData);
                        List<Prediction> predictions = mlContext.Data.CreateEnumerable<Prediction>(submissionData, reuseRowObject: false, ignoreMissingColumns: true).ToList();

                        submission.WriteLine(previousCustomerId + "," + predictions[0].Score.ToString(numberFormatInfo));
                    }

                    submission.Close();

                    return Double.NaN;
                }
                else
                {
                    var binaryPredictionTransformer = lightGbmBinaryTrainer.Fit(splitData.TrainSet);

                    IDataView testData = binaryPredictionTransformer.Transform(splitData.TestSet);
                    List<Prediction> predictions = mlContext.Data.CreateEnumerable<Prediction>(testData, reuseRowObject: false, ignoreMissingColumns: true).ToList();

                    predictions.Sort((firstItem, secondItem) =>
                    {
                        if (firstItem.Score.Equals(secondItem.Score))
                        {
                            return secondItem.Label.CompareTo(firstItem.Label);
                        }
                        else
                        {
                            return firstItem.Score.CompareTo(secondItem.Score);
                        }
                    });

                    Double countBackground = 0;
                    Double countSignal = 0;
                    Double aucPessimistic = 0;
                    Double thresholdAt4 = .96d * (20d * predictions.Count - 19d * predictions.Count(x => x.Label));
                    Boolean thresholdAt4Found = false;
                    Double countSignalAt4 = 0;

                    foreach (Prediction prediction in predictions)
                    {
                        if (prediction.Label)
                        {
                            countSignal++;
                            aucPessimistic += countBackground;
                        }
                        else
                        {
                            countBackground += 20d;
                        }

                        if (!thresholdAt4Found)
                        {
                            if (countSignal + countBackground >= thresholdAt4)
                            {
                                thresholdAt4Found = true;
                                countSignalAt4 = countSignal;
                            }
                        }
                    }

                    aucPessimistic /= countSignal * countBackground;
                    Double recallAt4Pessimistic = 1d - countSignalAt4 / countSignal;

                    predictions.Sort((firstItem, secondItem) =>
                    {
                        if (firstItem.Score.Equals(secondItem.Score))
                        {
                            return firstItem.Label.CompareTo(secondItem.Label);
                        }
                        else
                        {
                            return firstItem.Score.CompareTo(secondItem.Score);
                        }
                    });

                    countBackground = 0;
                    countSignal = 0;
                    Double aucOptimistic = 0;
                    thresholdAt4 = .96d * (20d * predictions.Count - 19d * predictions.Count(x => x.Label));
                    thresholdAt4Found = false;
                    countSignalAt4 = 0;

                    foreach (Prediction prediction in predictions)
                    {
                        if (prediction.Label)
                        {
                            countSignal++;
                            aucOptimistic += countBackground;
                        }
                        else
                        {
                            countBackground += 20;
                        }

                        if (!thresholdAt4Found)
                        {
                            if (countSignal + countBackground >= thresholdAt4)
                            {
                                thresholdAt4Found = true;
                                countSignalAt4 = countSignal;
                            }
                        }
                    }

                    aucOptimistic /= countSignal * countBackground;
                    Double recallAt4Optimistic = 1d - countSignalAt4 / countSignal;

                    Double aucErrorSampleSize = .25d * (1d / countSignal + 1d / countBackground);
                    Double metricError = Math.Sqrt(
                            Math.Pow(aucErrorSampleSize, 2)
                          + Math.Pow(aucOptimistic - aucPessimistic, 2)
                          + Math.Pow(recallAt4Optimistic - recallAt4Pessimistic, 2)
                          );
                    Double auc = (aucPessimistic + aucOptimistic) / 2d;
                    Double recallAt4 = (recallAt4Pessimistic + recallAt4Optimistic) / 2d;
                    return auc - .5d + .5d * recallAt4;
                }

            }
            internal static void Submit(DA da, List<Int32> featureIndexes)
            {
                using StreamWriter submission = new(new FileStream(fileDirectory + "submission.csv", FileMode.Create)) { AutoFlush = true };
                submission.WriteLine("customer_ID,prediction");

                using (StreamReader dataInput = new(new FileStream(fileDirectory + "test_data.csv", FileMode.Open)))
                {
                    // test data is sorted and nice
                    String[] header = dataInput.ReadLine().Split(',');
                    String previousCustomerId = String.Empty;
                    DataRow submissionDataRow = null;
                    Int32 n = 0;
                    Int32 seconds = 0;
                    Stopwatch stopwatch = Stopwatch.StartNew();

                    while (dataInput.EndOfStream is false)
                    {
                        String dataString = dataInput.ReadLine();
                        String[] dataStrings = dataString.Split(',');
                        String customerId = dataStrings[0];

                        if (!customerId.Equals(previousCustomerId))
                        {
                            if (submissionDataRow is not null)
                            {
                                submissionDataRow.TransformCategorical();
                                submissionDataRow.ReplaceNaN();
                                submissionDataRow.SetFeatureValues();
                                submissionDataRow.ProbabilityIntegralTransform();
                                submission.WriteLine(previousCustomerId + "," + submissionDataRow.Outcome(da, featureIndexes).ToString(numberFormatInfo));
                            }

                            submissionDataRow = new();
                        }

                        submissionDataRow.Update(dataStrings);
                        previousCustomerId = customerId;

                        n++;

                        if (!stopwatch.Elapsed.Seconds.Equals(seconds))
                        {
                            Debug.WriteLine((n / 11_363_762d * 100d).ToString("##.0") + "   " + n.ToString("### ### ###"));
                            seconds = stopwatch.Elapsed.Seconds;
                        }
                    }

                    dataInput.Close();

                    submissionDataRow.TransformCategorical();
                    submissionDataRow.ReplaceNaN();
                    submissionDataRow.SetFeatureValues();
                    submissionDataRow.ProbabilityIntegralTransform();
                    submission.WriteLine(previousCustomerId + "," + submissionDataRow.Outcome(da, featureIndexes).ToString(numberFormatInfo));
                }

                submission.Close();
            }
            internal new void Clear()
            {
                Parallel.ForEach(Values, new ParallelOptions { MaxDegreeOfParallelism = Environment.ProcessorCount }, x => x.Clear());
                base.Clear();
            }
            internal void CvM()
            {
                for (Int32 featureIndex = 0; featureIndex < featuresLength; featureIndex++)
                {
                    List<Double> trainFeatures = this[true].GetFeatureValues(featureIndex);
                    List<Double> testFeatures = this[false].GetFeatureValues(featureIndex);

                    List<(Double x, Boolean isFirstSample)> items = new();

                    foreach (Double x in trainFeatures)
                    {
                        items.Add((x, true));
                    }

                    foreach (Double x in testFeatures)
                    {
                        items.Add((x, false));
                    }

                    items.Sort((firstItem, secondItem) =>
                    {
                        if (firstItem.x.Equals(secondItem.x))
                        {
                            return firstItem.isFirstSample.CompareTo(secondItem.isFirstSample);
                        }
                        else
                        {
                            return firstItem.x.CompareTo(secondItem.x);
                        }
                    });

                    Double rankX = 0;
                    Double rankY = 0;
                    Double sumY2 = 0;
                    Double sumX2 = 0;
                    Double sumXY = 0;

                    foreach ((Double x, Boolean isSignal) in items)
                    {
                        if (isSignal)
                        {
                            rankX++;
                        }
                        else
                        {
                            rankY++;
                        }

                        sumY2 += rankY * rankY;
                        sumXY += rankX * rankY;
                        sumX2 += rankX * rankX;
                    }

                    items.Clear();

                    Double cvm = (rankY / rankX * sumX2 + rankX / rankY * sumY2 - 2d * sumXY) / Math.Pow(rankX + rankY, 2);
                    Debug.WriteLine(featureIndex.ToString() + " ; " + cvm.ToString());
                }
            }
        }

        private sealed class Prediction
        {
            public Boolean Label { get; set; }
            public Single Score { get; set; }
        }

        private sealed class MLData
        {
            public Boolean Label { get; set; }
            [VectorType(188 - 40)]
            public Single[] Features { get; set; }

            internal MLData(Boolean target, Double[] features, List<Int32> featureIndexes)
            {
                Label = target;
                List<Single> preFeatures = new();

                foreach (Int32 featuresIndex in featureIndexes)
                {
                    preFeatures.Add((Single)features[featuresIndex]);
                }

                Features = preFeatures.ToArray();
            }
        }

        private sealed class DataRows : Dictionary<String, DataRow> // customerId
        {
            internal void AmalgamateCategories(Boolean target)
            {
                foreach (String customerId in Keys)
                {
                    DataRow dataRow = this[customerId];

                    for (Int32 j = 0; j < numberCategories; j++)
                    {
                        String category = dataRow.categories[j];

                        if (category is not null)
                        {
                            if (!categoriesLookup[j].ContainsKey(category))
                            {
                                categoriesLookup[j].Add(category, new());
                            }

                            categoriesLookup[j][category].Increment(target);
                        }
                    }
                }
            }
            internal void AmalgamateNaN()
            {
                foreach (String customerId in Keys)
                {
                    DataRow dataRow = this[customerId];

                    for (Int32 j = 0; j < featuresLength; j++)
                    {
                        if (!Double.IsNaN(dataRow.features[j]))
                        {
                            transformerNaN[j].Add(dataRow.features[j]);
                        }
                    }
                }
            }
            internal void TransformCategorical()
            {
                foreach (DataRow dataRow in Values)
                {
                    dataRow.TransformCategorical();
                }
            }
            internal void ReplaceNaN()
            {
                foreach (DataRow dataRow in Values)
                {
                    dataRow.ReplaceNaN();
                }
            }
            internal List<Double> GetFeatureValues(Int32 featureIndex)
            {
                List<Double> featureValues = new();

                foreach (DataRow dataRow in Values)
                {
                    if (!Double.IsNaN(dataRow.features[featureIndex]))
                    {
                        featureValues.Add(dataRow.features[featureIndex]);
                    }
                }

                return featureValues;
            }
            internal void SetFeatureValues()
            {
                Parallel.ForEach(Values, new ParallelOptions { MaxDegreeOfParallelism = Environment.ProcessorCount }, x => x.SetFeatureValues());
            }
            internal void ProbabilityIntegralTransform()
            {
                Parallel.ForEach(Values, new ParallelOptions { MaxDegreeOfParallelism = Environment.ProcessorCount }, x => x.ProbabilityIntegralTransform());
            }
            internal void Read()
            {
                using (StreamReader dataInput = new(new FileStream(fileDirectory + "test_data.csv", FileMode.Open)))
                {
                    String[] header = dataInput.ReadLine().Split(',');

                    Int32 n = 0;
                    Int32 seconds = 0;
                    String previousCustomerId = String.Empty;
                    DataRow dataRow = null;
                    Stopwatch stopwatch = Stopwatch.StartNew();

                    while (dataInput.EndOfStream is false)
                    {
                        String[] dataStrings = dataInput.ReadLine().Split(',');
                        String customerId = dataStrings[0];

                        if (!customerId.Equals(previousCustomerId))
                        {
                            if (dataRow is not null)
                            {
                                this[previousCustomerId] = dataRow;
                            }

                            dataRow = new();
                        }

                        dataRow.Update(dataStrings);
                        previousCustomerId = customerId;

                        n++;

                        if (!stopwatch.Elapsed.Seconds.Equals(seconds))
                        {
                            Debug.WriteLine((n / 11_363_762d * 100d).ToString("##.0"));
                            seconds = stopwatch.Elapsed.Seconds;
                        }
                    }

                    dataInput.Close();

                    this[previousCustomerId] = dataRow;
                }

                Debug.WriteLine("TransformCategorical");
                Parallel.ForEach(Values, new ParallelOptions { MaxDegreeOfParallelism = Environment.ProcessorCount }, x => x.TransformCategorical());

                Debug.WriteLine("ReplaceNaN");
                Parallel.ForEach(Values, new ParallelOptions { MaxDegreeOfParallelism = Environment.ProcessorCount }, x => x.ReplaceNaN());

                Debug.WriteLine("SetFeatureValues");
                Parallel.ForEach(Values, new ParallelOptions { MaxDegreeOfParallelism = Environment.ProcessorCount }, x => x.SetFeatureValues());

                Debug.WriteLine("Probability integral transform");
                Parallel.ForEach(Values, new ParallelOptions { MaxDegreeOfParallelism = Environment.ProcessorCount }, x => x.ProbabilityIntegralTransform());
            }
        }

        private sealed class DataRow
        {
            internal String[] categories;
            internal Double[] features;

            internal DataRow()
            {
                categories = new String[numberCategories];
                features = Enumerable.Repeat(Double.NaN, 188).ToArray();
            }

            internal void Update(String[] dataStrings)
            {
                for (Int32 j = 0; j < numberCategories; j++)
                {
                    if (!dataStrings[categoricalIndexes[j]].Equals(String.Empty))
                    {
                        categories[j] = dataStrings[categoricalIndexes[j]];
                    }
                }

                Int32 i = 0;

                for (Int32 j = 0; j < 188; j++)
                {
                    if (!categoricalIndexes.Contains(2 + j))
                    {
                        if (!dataStrings[2 + j].Equals(String.Empty))
                        {
                            Double x = Double.Parse(dataStrings[2 + j], numberFormatInfo);
                            features[i + numberCategories] = x;
                        }

                        i++;
                    }
                }
            }
            internal void TransformCategorical()
            {
                for (Int32 j = 0; j < numberCategories; j++)
                {
                    if (categories[j] is null)
                    {
                        features[j] = Double.NaN;
                    }
                    else
                    {
                        features[j] = categoriesLookup[j][categories[j]].Average;
                    }

                    categories[j] = String.Empty;
                }

                categories = null;
            }
            internal void ReplaceNaN()
            {
                foreach (Int32 j in transformerNaN.Keys)
                {
                    if (Double.IsNaN(features[j]))
                    {
                        features[j] = transformerNaN[j].Transform;
                    }
                }

                foreach (Double feature in features)
                {
                    if (Double.IsNaN(feature))
                    {
                        throw new Exception("feature is NaN");
                    }
                }
            }
            internal void SetFeatureValues()
            {
                for (Int32 featureIndex = 0; featureIndex < featuresLength; featureIndex++)
                {
                    for (Int32 binNumber = 0; binNumber < numberBins; binNumber++)
                    {
                        if (features[featureIndex] <= binsLookup[featureIndex][binNumber].upperValue)
                        {
                            features[featureIndex] = binsLookup[featureIndex][binNumber].Average;
                            break;
                        }
                    }
                }
            }
            internal Double Outcome(DA da, List<Int32> featureIndexes)
            {
                List<Double> featuresDa = new();

                foreach (Int32 featureIndex in featureIndexes)
                {
                    featuresDa.Add(features[featureIndex]);
                }

                return da.Outcome(featuresDa.ToArray());
            }
            internal void ProbabilityIntegralTransform()
            {
                for (Int32 featureIndex = 0; featureIndex < featuresLength; featureIndex++)
                {
                    Double feature = features[featureIndex];
                    Int32 searchIndex = transformers[featureIndex].BinarySearch(feature);

                    if (searchIndex < 0)
                    {
                        searchIndex = ~searchIndex;

                        if (searchIndex.Equals(transformers[featureIndex].Count))
                        {
                            searchIndex--;
                        }
                    }

                    features[featureIndex] = (Double)searchIndex / transformers[featureIndex].Count;
                }
            }
        }

        private sealed class Category
        {
            private Double sum;
            private Double count;

            internal Double Average => sum / count;

            internal Category()
            {
                sum = 0;
                count = 0;
            }

            internal void Increment(Boolean target)
            {
                if (target)
                {
                    sum++;
                }

                count++;
            }
        }

        private sealed class DA
        {
            private readonly Int32 matrixSize;
            internal readonly Byte degree;
            private readonly Int32 numberVariables;
            private readonly Double[] xTmp;
            private readonly Double[] signalMeans;
            private readonly Double[] backgroundMeans;
            private readonly Double[,] signalDispersionMatrix;
            private readonly Double[,] backgroundDispersionMatrix;
            private Double signalWeight;
            private Double backgroundWeight;

            internal Double[] A { get; private set; }
            internal Double SignalOutputMean { get; private set; }
            internal Double BackgroundOutputMean { get; private set; }
            internal Boolean SolutionExists { get; private set; }
            internal Int32 DegreesFreedom => matrixSize;

            internal DA(in Int32 numberVariables, in Byte degree)
            {
                if (degree > 6)
                {
                    throw new NotImplementedException();
                }

                this.numberVariables = numberVariables;
                this.degree = degree;

                do
                {
                    matrixSize = numberVariables;
                    if (degree > 1)
                    {
                        matrixSize += numberVariables * (numberVariables + 1) / 2;
                    }

                    if (degree > 2)
                    {
                        matrixSize += numberVariables * (numberVariables + 1) * (numberVariables + 2) / 6;
                    }

                    if (degree > 3)
                    {
                        matrixSize += numberVariables * (numberVariables + 1) * (numberVariables + 2) * (numberVariables + 3) / 24;
                    }

                    if (degree > 4)
                    {
                        matrixSize += numberVariables * (numberVariables + 1) * (numberVariables + 2) * (numberVariables + 3) * (numberVariables + 4) / 120;
                    }

                    if (degree > 5)
                    {
                        matrixSize += numberVariables * (numberVariables + 1) * (numberVariables + 2) * (numberVariables + 3) * (numberVariables + 4) * (numberVariables + 5) / 720;
                    }

                    if ((matrixSize > 4096) | (matrixSize < 0))
                    {
                        throw new NotImplementedException();
                    }
                } while ((matrixSize > 4096) || (matrixSize < 0));

                xTmp = new Double[matrixSize];
                signalMeans = new Double[matrixSize];
                backgroundMeans = new Double[matrixSize];
                signalDispersionMatrix = new Double[matrixSize, matrixSize];
                backgroundDispersionMatrix = new Double[matrixSize, matrixSize];
                SolutionExists = false;
                A = new Double[matrixSize];
            }
            internal DA(List<Double> daCoefficients, Byte degree) : this(daCoefficients.Count, degree)
            {
                A = daCoefficients.ToArray();
            }

            private void Expand(in Double[] xVar)
            {
                Int32 idx = 0;

                if (degree.Equals(1))
                {
                    for (Int32 i1 = 0; i1 < numberVariables; i1++)
                    {
                        xTmp[idx++] = xVar[i1];
                    }
                }
                else if (degree.Equals(2))
                {
                    for (Int32 i1 = 0; i1 < numberVariables; i1++)
                    {
                        xTmp[idx++] = xVar[i1];
                        for (Int32 i2 = i1; i2 < numberVariables; i2++)
                        {
                            xTmp[idx++] = xVar[i1] * xVar[i2];
                        }
                    }
                }
                else if (degree.Equals(3))
                {
                    for (Int32 i1 = 0; i1 < numberVariables; i1++)
                    {
                        xTmp[idx++] = xVar[i1];
                        for (Int32 i2 = i1; i2 < numberVariables; i2++)
                        {
                            xTmp[idx++] = xVar[i1] * xVar[i2];
                            for (Int32 i3 = i2; i3 < numberVariables; i3++)
                            {
                                xTmp[idx++] = xVar[i1] * xVar[i2] * xVar[i3];
                            }
                        }
                    }
                }
                else if (degree.Equals(4))
                {
                    for (Int32 i1 = 0; i1 < numberVariables; i1++)
                    {
                        xTmp[idx++] = xVar[i1];
                        for (Int32 i2 = i1; i2 < numberVariables; i2++)
                        {
                            xTmp[idx++] = xVar[i1] * xVar[i2];
                            for (Int32 i3 = i2; i3 < numberVariables; i3++)
                            {
                                xTmp[idx++] = xVar[i1] * xVar[i2] * xVar[i3];
                                for (Int32 i4 = i3; i4 < numberVariables; i4++)
                                {
                                    xTmp[idx++] = xVar[i1] * xVar[i2] * xVar[i3] * xVar[i4];
                                }
                            }
                        }
                    }
                }
                else if (degree.Equals(5))
                {
                    for (Int32 i1 = 0; i1 < numberVariables; i1++)
                    {
                        xTmp[idx++] = xVar[i1];
                        for (Int32 i2 = i1; i2 < numberVariables; i2++)
                        {
                            xTmp[idx++] = xVar[i1] * xVar[i2];
                            for (Int32 i3 = i2; i3 < numberVariables; i3++)
                            {
                                xTmp[idx++] = xVar[i1] * xVar[i2] * xVar[i3];
                                for (Int32 i4 = i3; i4 < numberVariables; i4++)
                                {
                                    xTmp[idx++] = xVar[i1] * xVar[i2] * xVar[i3] * xVar[i4];
                                    for (Int32 i5 = i4; i5 < numberVariables; i5++)
                                    {
                                        xTmp[idx++] = xVar[i1] * xVar[i2] * xVar[i3] * xVar[i4] * xVar[i5];
                                    }
                                }
                            }
                        }
                    }
                }
                else if (degree.Equals(6))
                {
                    for (Int32 i1 = 0; i1 < numberVariables; i1++)
                    {
                        xTmp[idx++] = xVar[i1];
                        for (Int32 i2 = i1; i2 < numberVariables; i2++)
                        {
                            xTmp[idx++] = xVar[i1] * xVar[i2];
                            for (Int32 i3 = i2; i3 < numberVariables; i3++)
                            {
                                xTmp[idx++] = xVar[i1] * xVar[i2] * xVar[i3];
                                for (Int32 i4 = i3; i4 < numberVariables; i4++)
                                {
                                    xTmp[idx++] = xVar[i1] * xVar[i2] * xVar[i3] * xVar[i4];
                                    for (Int32 i5 = i4; i5 < numberVariables; i5++)
                                    {
                                        xTmp[idx++] = xVar[i1] * xVar[i2] * xVar[i3] * xVar[i4] * xVar[i5];
                                        for (Int32 i6 = i5; i6 < numberVariables; i6++)
                                        {
                                            xTmp[idx++] = xVar[i1] * xVar[i2] * xVar[i3] * xVar[i4] * xVar[i5] * xVar[i6];
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                else
                {
                    throw new Exception("Degree = " + degree.ToString());
                }
            }

            internal void Add(in Double[] xVar, in Boolean target)
            {
                Expand(xVar);

                if (target)
                {
                    signalWeight++;

                    for (Int32 i = 0; i < matrixSize; i++)
                    {
                        signalMeans[i] += xTmp[i];

                        for (Int32 j = i; j < matrixSize; j++)
                        {
                            signalDispersionMatrix[i, j] += xTmp[i] * xTmp[j];
                        }
                    }
                }
                else
                {
                    backgroundWeight++;
                    for (Int32 i = 0; i < matrixSize; i++)
                    {
                        backgroundMeans[i] += xTmp[i];
                        for (Int32 j = i; j < matrixSize; j++)
                        {
                            backgroundDispersionMatrix[i, j] += xTmp[i] * xTmp[j];
                        }
                    }
                }
            }
            internal void Solve()
            {
                if ((signalWeight <= 0)
                    || (backgroundWeight <= 0)
                    || matrixSize.Equals(0))
                {
                    SolutionExists = false;
                    return;
                }

                Double[,] mat = new Double[matrixSize, matrixSize];
                Double[] y = new Double[matrixSize];
                Boolean[] rowDone = new Boolean[matrixSize];
                Int32[] rowIndexes = new Int32[matrixSize];

                for (Int32 i = 0; i < matrixSize; i++)
                {
                    rowDone[i] = false;
                    rowIndexes[i] = -1;

                    for (Int32 j = i; j < matrixSize; j++)
                    {
                        signalDispersionMatrix[i, j] = (signalDispersionMatrix[i, j] - signalMeans[i] * signalMeans[j] / signalWeight) / signalWeight;
                        backgroundDispersionMatrix[i, j] = (backgroundDispersionMatrix[i, j] - backgroundMeans[i] * backgroundMeans[j] / backgroundWeight) / backgroundWeight;
                        mat[i, j] = signalDispersionMatrix[i, j] + backgroundDispersionMatrix[i, j];
                    }

                    signalMeans[i] /= signalWeight;
                    backgroundMeans[i] /= backgroundWeight;
                    y[i] = signalMeans[i] - backgroundMeans[i];
                }

                for (Int32 j = 0; j < matrixSize; j++)
                {
                    for (Int32 k = j; k < matrixSize; k++)
                    {
                        mat[k, j] = mat[j, k];
                    }
                }

                // **** SOLVER ********
                Int32 currentColumn;
                Double maxValue;
                Int32 rowIndexMax;
                Int32 rowIndex;
                Double factor;
                Int32 columnIndex;
                Double[] tmpRow = new Double[matrixSize];
                Double tmpY;
                SolutionExists = true;

                for (currentColumn = 0; currentColumn < matrixSize; currentColumn++)
                {
                    maxValue = -1;
                    rowIndexMax = -1;

                    for (rowIndex = 0; rowIndex < matrixSize; rowIndex++)
                    {
                        if (!rowDone[rowIndex])
                        {
                            if (Math.Abs(mat[currentColumn, rowIndex]) > maxValue)
                            {
                                rowIndexMax = rowIndex;
                                maxValue = Math.Abs(mat[currentColumn, rowIndexMax]);
                            }
                        }
                    }

                    if (rowIndexMax >= 0)
                    {
                        if (maxValue > Double.Epsilon)
                        {
                            factor = 1 / mat[currentColumn, rowIndexMax];

                            for (columnIndex = currentColumn; columnIndex < matrixSize; columnIndex++)
                            {
                                mat[columnIndex, rowIndexMax] *= factor;
                                tmpRow[columnIndex] = mat[columnIndex, rowIndexMax];
                            }

                            y[rowIndexMax] *= factor;
                            tmpY = y[rowIndexMax];
                            rowDone[rowIndexMax] = true;

                            for (rowIndex = 0; rowIndex < matrixSize; rowIndex++)
                            {
                                if (!rowDone[rowIndex])
                                {
                                    factor = mat[currentColumn, rowIndex];

                                    for (columnIndex = currentColumn; columnIndex < matrixSize; columnIndex++)
                                    {
                                        mat[columnIndex, rowIndex] -= tmpRow[columnIndex] * factor;
                                    }

                                    y[rowIndex] -= tmpY * factor;
                                }
                            }

                            rowIndexes[currentColumn] = rowIndexMax;
                        }
                        else
                        {
                            SolutionExists = false;
                            break;
                        }
                    }
                    else
                    {
                        SolutionExists = false;
                        break;
                    }
                }

                if (SolutionExists)
                {
                    Double tmpA;
                    SignalOutputMean = 0;
                    BackgroundOutputMean = 0;

                    for (currentColumn = matrixSize - 1; currentColumn >= 0; currentColumn--)
                    {
                        rowIndex = rowIndexes[currentColumn];
                        tmpA = y[rowIndex];

                        for (columnIndex = matrixSize - 1; columnIndex > currentColumn; columnIndex--)
                        {
                            tmpA -= A[columnIndex] * mat[columnIndex, rowIndex];
                        }

                        A[currentColumn] = tmpA;
                        SignalOutputMean += tmpA * signalMeans[currentColumn];
                        BackgroundOutputMean += tmpA * backgroundMeans[currentColumn];
                    }

                    if (SignalOutputMean < BackgroundOutputMean)
                    {
                        for (currentColumn = 0; currentColumn < matrixSize; currentColumn++)
                        {
                            A[currentColumn] = -A[currentColumn];
                        }
                    }
                }
            }
            internal Double Outcome(in Double[] xVar)
            {
                Expand(xVar);
                Double x = 0;
                for (Int32 i = 0; i < matrixSize; i++)
                {
                    x += A[i] * xTmp[i];
                }

                return x;
            }
        }

        private class Amalgamator
        {
            private Double sum;
            private Int32 count;

            internal Double Mean => sum / count;

            internal Amalgamator()
            {
                sum = 0;
                count = 0;
            }

            internal void Add(Double outcome)
            {
                sum += outcome;
                count++;
            }
        }

        private sealed class TransformerNaN
        {
            private Double sum;
            private Double count;

            internal Double Transform => sum / count;

            internal TransformerNaN()
            {
                sum = 0;
                count = 0;
            }

            internal void Add(Double value)
            {
                sum += value;
                count++;
            }
        }

        private sealed class Bin
        {
            internal readonly Double upperValue;
            internal Int32 count;
            internal Double sum;

            internal Double Average
            {
                get => sum / count;
                set { sum = value; count = 1; }
            }

            internal Bin(Double upperValue)
            {
                this.upperValue = upperValue;
                count = 0;
                sum = 0;
            }

            internal void Increment(Boolean target)
            {
                if (target)
                {
                    sum++;
                }

                count++;
            }
        }

        private sealed class Accumulator : Amalgamator
        {
            internal readonly Boolean target;

            internal Accumulator(Boolean target)
            {
                this.target = target;
            }
        }
    }
}