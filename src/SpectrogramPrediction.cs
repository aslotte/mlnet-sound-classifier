using System;
using Microsoft.ML.Data;

namespace SoundClassifier 
{
    public class ImagePrediction
    {
        [ColumnName("Score")]
        public float[] Score;

        [ColumnName("PredictedLabel")]
        public UInt32 PredictedLabel;
    }
}   