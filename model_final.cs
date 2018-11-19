using System;
using Emgu.CV;
using Emgu.CV.ML;
using System.IO;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace MLT_Proj
{
    public class Program
    {
        string path = @"File2-Cleaned.csv";
        Matrix<float> TrainData;
        Matrix<int> TrainLabel;

        RTrees rtrees;  

        private void LoadTrainData()
        {
            List<float[]> TrainList = new List<float[]>();
            List<string> trainLabel = new List<string>();
            Dictionary<string,int> lookup = new Dictionary<string, int>();
            List<int> trainAnswer = new List<int>();

            StreamReader reader = new StreamReader(path);
            string line = "";
            while ((line = reader.ReadLine()) != null)
            {
                int lastIndex = line.LastIndexOf(',');
                string currentData = line.Substring(0, lastIndex);
                double[] data = currentData.Split(',').Select(x => double.Parse(x)).ToArray();

                float[] floatArray = Array.ConvertAll(data, x => Convert.ToSingle(x));
                float[] floatNoNegInfinityArray = floatArray.Select(x => (float.IsNegativeInfinity(x) ? float.MinValue : x)).ToArray();
                float[] floatNoInfinityArray = floatNoNegInfinityArray.Select(x => (float.IsPositiveInfinity(x) ? float.MaxValue : x)).ToArray();

                
                string currentLabel = line.Substring(lastIndex + 1);
                TrainList.Add(floatNoInfinityArray);
                
                trainLabel.Add(currentLabel);
            }

            TrainData = new Matrix<float>(To2D<float>(TrainList.ToArray()));

            // getting server names in integer
            List<string> distinct = trainLabel.Distinct().ToList();
           
            int counter = 1;
            
            foreach (string value in distinct)
            {
                //Console.WriteLine("After: {0}", value);
                lookup.Add(value, counter);
                counter++;
            }
           // System.Console.WriteLine("Counter : {0}", counter - 1);
            System.Console.WriteLine("List size earlier : {0}", trainLabel.Count);
            /*
            foreach (var element in lookup)
            {
                Console.WriteLine(element.Value);
            }
            */

            foreach(string value in trainLabel)
            {
                foreach (var kvp in lookup)
                    if (kvp.Key == value)
                        trainAnswer.Add(kvp.Value);
            }

            int number_of_values_counter = 1;
            foreach (var value in trainAnswer)
            {
                // Console.WriteLine(value);
                number_of_values_counter++;
            }
            TrainLabel = new Matrix<int>(trainAnswer.ToArray());
            System.Console.WriteLine("List size earlier : {0}", trainAnswer.Count);
        }

        private T[,] To2D<T>(T[][] source)
        {
            try
            {
                int FirstDim = source.Length;
                int SecondDim = source.GroupBy(row => row.Length).Single().Key; // throws InvalidOperationException if source is not rectangular

                var result = new T[FirstDim, SecondDim];
                for (int i = 0; i < FirstDim; ++i)
                    for (int j = 0; j < SecondDim; ++j)
                        result[i, j] = source[i][j];

                return result;
            }
            catch (InvalidOperationException)
            {
                throw new InvalidOperationException("The given jagged array is not rectangular.");
            }
        }

        private void train()
        {
            try
            {
                rtrees = new RTrees();
                rtrees.MaxDepth = 2;
                rtrees.MinSampleCount = 10;
                rtrees.MaxCategories = 43;
                
                rtrees.Train(TrainData, Emgu.CV.ML.MlEnum.DataLayoutType.RowSample,TrainLabel);
                rtrees.Save("randomforest.xml");
            }
            catch (System.AccessViolationException)
            {
                //recognier = new EigenFaceRecognizer(80, double.PositiveInfinity);
                System.Console.WriteLine("Exception");
            }

        }

         private void test()
        {
            
            try
            {
                int counter = 0;
                for (int i = 0; i < TrainData.Rows; i++)
                {
                    Matrix<float> row = TrainData.GetRow(i);
                    float predict = rtrees.Predict(row);
                    if(predict==TrainLabel[i,0])
                    {
                        counter++;
                    }
                }

                System.Console.WriteLine("Accuracy = {0} ", (counter / (float)(TrainData.Rows)));
                System.Console.ReadKey();
            }
            catch (Exception ex)
            {
                System.Console.WriteLine(ex.Message);
            }
        }

        static void Main(string[] args)
        {
            
                Program p = new Program();
                p.LoadTrainData();
                p.train();
                p.test();
        }

    }   
}
