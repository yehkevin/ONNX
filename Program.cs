using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Drawing;
using System.Text;
using System.Threading.Tasks;
using Accord.Imaging.Converters;
using Accord.IO;
using Microsoft.ML.OnnxRuntime;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.Data.Matlab;
//using System.Numerics.Tensors;
using Microsoft.ML.OnnxRuntime.Tensors;
using Accord.Imaging.Formats;

namespace ConsoleApp2
{
    internal class Program
    {
        static void Main(string[] args)
        {

            /*
            string fileName = @"1.mat";
            var reader = new MatReader(fileName);
            float[,] X = reader.Read<float[,]>("x");

            MatrixToImage conv = new MatrixToImage(min: 0.0, max: 1.0);
            // Declare an image and store the pixels on it
            Bitmap image; conv.Convert(X, out image);
            // Show the image on screen
            image.Save("input.png");
            */

            Bitmap image = ImageDecoder.DecodeFromFile("input.png");
            ImageToMatrix conv = new ImageToMatrix(min: 0, max: 1);
            float[,] X; conv.Convert(image, out X);




            //float max = X.Cast<float>().Max(); */
            System.Random random = new System.Random();
            int a = 512;
            int b = 512;
            /*float[,] X = new float [a, b];

            for (int i = 0; i < a; ++i)
                for (int j = 0; j < b; ++j)
                    X[i, j] = (float)random.NextDouble();
            */
            
            newfunc(ref X);
            Console.WriteLine("Done");
            Console.ReadLine();

        }

        static void newfunc(ref float[,] X)
        {
            int gpuDeviceId = 0;
            var session = new InferenceSession("ONNXModels/model.onnx");
            Tensor<float> input = new DenseTensor<float>(new[] { 1, 1, 256, 256 });
            for (int i = 0; i < X.GetLength(0); i++)
                for (int j = 0; j < X.GetLength(1); j++)
                    
                    input[0,0,i,j] = (X[i, j]-1.5f)/1.5f;
            Console.WriteLine("Output for {0}", input[0,0,100, 100]);
            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor<float>("actual_input",  input)
            };
            var watch = new System.Diagnostics.Stopwatch();
            watch.Start();
            var results = session.Run(inputs);
            watch.Stop();
            Console.WriteLine($"Execution Time: {watch.ElapsedMilliseconds} ms");
            
            foreach (var r in results)
            {
                //Console.WriteLine("Output for {0}", r);
                var count = 0;
                /*
                for (int i = 0; i < X.GetLength(0); i++)
                    for (int j = 0; j < X.GetLength(1); j++)
                    {
                        X[i, j] = r.AsTensor<float>().ToArray()[count] * 1.5f + 1.5f;
                        count++;
                    }
                */
               
                for (int i = 0; i < X.GetLength(0); i++)
                    for (int j = 0; j < X.GetLength(1); j++)
                    {
                        X[i, j] = (r.AsTensor<float>().ToArray()[count]*1.5f+1.5f) ;
                        count++;
                    }

                float max = X.Cast<float>().Max();
                Console.WriteLine(max);

                //file saving

                MatrixToImage conv = new MatrixToImage(min: 0.0, max: 1.0);
                
                // Declare an image and store the pixels on it
                Bitmap image; conv.Convert(X, out image);

                // Show the image on screen
                image.Save("output.png");
            }
        }
    }
}

