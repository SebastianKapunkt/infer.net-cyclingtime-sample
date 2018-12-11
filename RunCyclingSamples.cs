using System;
using Microsoft.ML.Probabilistic.Models;
using Microsoft.ML.Probabilistic.Distributions;
using static cyclingtime.CyclistBase;

namespace cyclingtime
{
    class Program
    {
        static void Main(string[] args)
        {
            RunCyclingTime1();
        }

        public static void RunCyclingTime1()
        {
            //[1] The model
            Variable<double> averageTime = Variable.GaussianFromMeanAndPrecision(15, 0.01);
            Variable<double> trafficNoise = Variable.GammaFromShapeAndScale(2.0, 0.5);
            Variable<double> travelTimeMonday =
            Variable.GaussianFromMeanAndPrecision(averageTime, trafficNoise);
            Variable<double> travelTimeTuesday =
            Variable.GaussianFromMeanAndPrecision(averageTime, trafficNoise);
            Variable<double> travelTimeWednesday =
            Variable.GaussianFromMeanAndPrecision(averageTime, trafficNoise);

            //[2] Train the model
            travelTimeMonday.ObservedValue = 13;
            travelTimeTuesday.ObservedValue = 17;
            travelTimeWednesday.ObservedValue = 16;
            InferenceEngine engine = new InferenceEngine();
            Gaussian averageTimePosterior = engine.Infer<Gaussian>(averageTime);
            Gamma trafficNoisePosterior = engine.Infer<Gamma>(trafficNoise);
            Console.WriteLine("averageTimePosterior: " + averageTimePosterior);
            Console.WriteLine("trafficNoisePosterior: " + trafficNoisePosterior);

            //[3] Make predictions
            //Add a prediction variable and retrain the model
            Variable<double> tomorrowsTime =
            Variable.GaussianFromMeanAndPrecision(averageTime, trafficNoise);
            Gaussian tomorrowsTimeDist = engine.Infer<Gaussian>(tomorrowsTime);
            double tomorrowsMean = tomorrowsTimeDist.GetMean();
            double tomorrowsStdDev = Math.Sqrt(tomorrowsTimeDist.GetVariance());
            // Write out the results.
            Console.WriteLine("Tomorrows predicted time: {0:f2} plus or minus {1:f2}",
            tomorrowsMean, tomorrowsStdDev);
            // Ask other questions of the model
            double probTripTakesLessThan18Minutes = engine.Infer<Bernoulli>(tomorrowsTime <
           18.0).GetProbTrue();
            Console.WriteLine("Probability that the trip takes less than 18 min: {0:f2}",
            probTripTakesLessThan18Minutes);
        }
    }
}
