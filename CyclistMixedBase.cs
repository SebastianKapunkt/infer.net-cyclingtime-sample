using System;
using Microsoft.ML.Probabilistic.Models;
using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Math;
using Microsoft.ML.Probabilistic.Algorithms;

namespace cyclingtime
{

    public class CyclistMixedBase
    {
        protected InferenceEngine InferenceEngine;
        protected int NumComponents;
        protected VariableArray<Gaussian> AverageTimePriors;
        protected VariableArray<double> AverageTime;
        protected VariableArray<Gamma> TrafficNoisePriors;
        protected VariableArray<double> TrafficNoise;
        protected Variable<Dirichlet> MixingPrior;
        protected Variable<Vector> MixingCoefficients;

        public virtual void CreateModel()
        {
            InferenceEngine = new InferenceEngine(new VariationalMessagePassing());
            NumComponents = 2;
            Range ComponentRange = new Range(NumComponents);
            AverageTimePriors = Variable.Array<Gaussian>(ComponentRange);
            TrafficNoisePriors = Variable.Array<Gamma>(ComponentRange);
            AverageTime = Variable.Array<double>(ComponentRange);
            TrafficNoise = Variable.Array<double>(ComponentRange);
            using (Variable.ForEach(ComponentRange))
            {
                AverageTime[ComponentRange] =
                Variable.Random<double, Gaussian>(AverageTimePriors[ComponentRange]);
                TrafficNoise[ComponentRange] =
                Variable.Random<double, Gamma>(TrafficNoisePriors[ComponentRange]);
            }
            //Mixing coefficients
            MixingPrior = Variable.New<Dirichlet>();
            MixingCoefficients = Variable<Vector>.Random(MixingPrior);
            MixingCoefficients.SetValueRange(ComponentRange);
        }

        public virtual void SetModelData(ModelDataMixed modelData)
        {
            AverageTimePriors.ObservedValue = modelData.AverageTimeDist;
            TrafficNoisePriors.ObservedValue = modelData.TrafficNoiseDist;
            MixingPrior.ObservedValue = modelData.MixingDist;
        }
    }

    public struct ModelDataMixed
    {
        public Gaussian[] AverageTimeDist;
        public Gamma[] TrafficNoiseDist;
        public Dirichlet MixingDist;
    }
}