using System;
using Microsoft.ML.Probabilistic.Models;
using Microsoft.ML.Probabilistic.Distributions;


namespace cyclingtime
{
    public class CyclistMixedWithEvidence : CyclistMixedTraining
    {
        protected Variable<bool> Evidence;
        public override void CreateModel()
        {
            Evidence = Variable.Bernoulli(0.5);
            using (Variable.If(Evidence))
            {
                base.CreateModel();
            }
        }
        public double InferEvidence(double[] trainingData)
        {
            double logEvidence;
            ModelDataMixed posteriors = base.InferModelData(trainingData);
            logEvidence = InferenceEngine.Infer<Bernoulli>(Evidence).LogOdds;
            return logEvidence;
        }
    }
}