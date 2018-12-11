using System;
using Microsoft.ML.Probabilistic.Models;
using Microsoft.ML.Probabilistic.Distributions;


namespace cyclingtime
{
    public class CyclistTraining : CyclistBase
    {
        protected VariableArray<double> TravelTimes;
        protected Variable<int> NumTrips;

        public override void CreateModel()
        {
            base.CreateModel();
            NumTrips = Variable.New<int>();
            Range tripRange = new Range(NumTrips);
            TravelTimes = Variable.Array<double>(tripRange);
            using (Variable.ForEach(tripRange))
            {
                TravelTimes[tripRange] =
                Variable.GaussianFromMeanAndPrecision(AverageTime, TrafficNoise);
            }
        }
        public ModelData InferModelData(double[] trainingData)
        {
            ModelData posteriors;
            NumTrips.ObservedValue = trainingData.Length;
            TravelTimes.ObservedValue = trainingData;
            posteriors.AverageTimeDist = InferenceEngine.Infer<Gaussian>(AverageTime);
            posteriors.TrafficNoiseDist = InferenceEngine.Infer<Gamma>(TrafficNoise);
            return posteriors;
        }
    }
}