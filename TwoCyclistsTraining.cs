using System;
using Microsoft.ML.Probabilistic.Models;
using Microsoft.ML.Probabilistic.Distributions;
using static cyclingtime.CyclistBase;

namespace cyclingtime
{
    public class TwoCyclistsTraining
    {
        private CyclistTraining cyclist1, cyclist2;

        public void CreateModel()
        {
            cyclist1 = new CyclistTraining();
            cyclist1.CreateModel();
            cyclist2 = new CyclistTraining();
            cyclist2.CreateModel();
        }

        public void SetModelData(ModelData modelData)
        {
            cyclist1.SetModelData(modelData);
            cyclist2.SetModelData(modelData);
        }

        public ModelData[] InferModelData(
            double[] trainingData1,
            double[] trainingData2
        )
        {
            ModelData[] posteriors = new ModelData[2];
            posteriors[0] = cyclist1.InferModelData(trainingData1);
            posteriors[1] = cyclist2.InferModelData(trainingData2);
            return posteriors;
        }
    }
}