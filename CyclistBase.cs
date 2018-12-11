using System;
using Microsoft.ML.Probabilistic.Models;
using Microsoft.ML.Probabilistic.Distributions;

namespace cyclingtime
{

    public class CyclistBase
    {
        public InferenceEngine InferenceEngine;
        protected Variable<double> AverageTime;
        protected Variable<double> TrafficNoise;
        protected Variable<Gaussian> AverageTimePrior;
        protected Variable<Gamma> TrafficNoisePrior;

        public virtual void CreateModel()
        {
            AverageTimePrior = Variable.New<Gaussian>();
            TrafficNoisePrior = Variable.New<Gamma>();
            AverageTime = Variable.Random<double, Gaussian>(AverageTimePrior);
            TrafficNoise = Variable.Random<double, Gamma>(TrafficNoisePrior);
            if (InferenceEngine == null)
            {
                InferenceEngine = new InferenceEngine();
            }
        }

        public virtual void SetModelData(ModelData priors)
        {
            AverageTimePrior.ObservedValue = priors.AverageTimeDist;
            TrafficNoisePrior.ObservedValue = priors.TrafficNoiseDist;
        }

        public struct ModelData
        {
            public Gaussian AverageTimeDist;
            public Gamma TrafficNoiseDist;
            public ModelData(Gaussian mean, Gamma precision)
            {
                AverageTimeDist = mean;
                TrafficNoiseDist = precision;
            }
        }
    }
}