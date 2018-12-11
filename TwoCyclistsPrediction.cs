using System;
using Microsoft.ML.Probabilistic.Models;
using Microsoft.ML.Probabilistic.Distributions;
using static cyclingtime.CyclistBase;

namespace cyclingtime
{
    public class TwoCyclistsPrediction
    {
        private CyclistPrediction cyclist1, cyclist2;
        private Variable<double> TimeDifference;
        private Variable<bool> Cyclist1IsFaster;
        private InferenceEngine CommonEngine;

        public void CreateModel()
        {
            CommonEngine = new InferenceEngine();
            cyclist1 = new CyclistPrediction() { InferenceEngine = CommonEngine };
            cyclist1.CreateModel();
            cyclist2 = new CyclistPrediction() { InferenceEngine = CommonEngine };
            cyclist2.CreateModel();
            TimeDifference = cyclist1.TomorrowsTime - cyclist2.TomorrowsTime;
            Cyclist1IsFaster = cyclist1.TomorrowsTime < cyclist2.TomorrowsTime;
        }

        public void SetModelData(ModelData[] modelData)
        {
            cyclist1.SetModelData(modelData[0]);
            cyclist2.SetModelData(modelData[1]);
        }

        public Gaussian[] InferTomorrowsTime()
        {
            Gaussian[] tomorrowsTime = new Gaussian[2];
            tomorrowsTime[0] = cyclist1.InferTomorrowsTime();
            tomorrowsTime[1] = cyclist2.InferTomorrowsTime();
            return tomorrowsTime;
        }

        public Gaussian InferTimeDifference()
        {
            return CommonEngine.Infer<Gaussian>(TimeDifference);
        }

        public Bernoulli InferCyclist1IsFaster()
        {
            return CommonEngine.Infer<Bernoulli>(Cyclist1IsFaster);
        }
    }
}