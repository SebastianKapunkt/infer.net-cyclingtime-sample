using System;
using Microsoft.ML.Probabilistic.Models;
using Microsoft.ML.Probabilistic.Distributions;

namespace cyclingtime
{
    public class CyclistPrediction : CyclistBase
    {
        private Gaussian tomorrowsTimeDist;
        public Variable<double> TomorrowsTime;

        public override void CreateModel()
        {
            base.CreateModel();
            TomorrowsTime =
            Variable.GaussianFromMeanAndPrecision(AverageTime, TrafficNoise);
        }

        public Gaussian InferTomorrowsTime()
        {
            tomorrowsTimeDist = InferenceEngine.Infer<Gaussian>(TomorrowsTime);
            return tomorrowsTimeDist;
        }

        public Bernoulli InferProbabilityTimeLessThan(double time)
        {
            return InferenceEngine.Infer<Bernoulli>(TomorrowsTime < time);
        }
    }
}