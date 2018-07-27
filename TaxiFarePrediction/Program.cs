using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Models;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;
using System;
using System.IO;
using System.Threading.Tasks;

namespace TaxiFarePrediction
{
    class Program
    {
        static readonly string _datapath = Path.Combine(Environment.CurrentDirectory, "Data", "taxi-fare-train.csv"); //Tahmin için kullanılacak veri kümesinin yolu.
        static readonly string _testdatapath = Path.Combine(Environment.CurrentDirectory, "Data", "taxi-fare-test.csv"); //Test için kullanılan veri kümesinin yolu.
        static readonly string _modelpath = Path.Combine(Environment.CurrentDirectory, "Data", "Model.zip"); //Eğitimli modelin depolandığı dosyanın yolu.

        static async Task Main(string[] args)
        {
            Console.WriteLine(Environment.CurrentDirectory);
            PredictionModel<TaxiTrip, TaxiTripFarePrediction> model = await Train();
            TaxiTripFarePrediction prediction = model.Predict(TestTrips.Trip1);
            Console.WriteLine("Predicted fare: {0}, actual fare: 29.5", prediction.FareAmount);
            Evaluate(model);                                         
        }

        public static async Task<PredictionModel<TaxiTrip, TaxiTripFarePrediction>> Train()
        {
            var pipeline = new LearningPipeline{ //LearningPipline metodu tüm dataları ve gerekli algoritmaları Train metoduna ekler.
                new TextLoader(_datapath).CreateFrom<TaxiTrip>(useHeader: true, separator: ','), //useHeader ile ilk satırı gözardı ettik. separator ile dosyadaki sütunları virgül ile ayırdık.
                new ColumnCopier(("FareAmount", "Label")),
                new CategoricalOneHotVectorizer("VendorId",
                                             "RateCode",
                                             "PaymentType"), //Kategorik değerleri numerik değerlere dönüştürdük.
                new ColumnConcatenator("Features",
                                    "VendorId",
                                    "RateCode",
                                    "PassengerCount",
                                    "TripDistance",
                                    "PaymentType"), //concatenator ile yeni değerlere sahip sütunları birleştirdik.
                new FastTreeRegressor() //Tahmin için kullanacağımız algoritma.
            };
            PredictionModel<TaxiTrip, TaxiTripFarePrediction> model = pipeline.Train<TaxiTrip, TaxiTripFarePrediction>();
            await model.WriteAsync(_modelpath); //Eğitilen modeli model.zip klasörüne kaydediyoruz.
            return model;
        }
        private static void Evaluate(PredictionModel<TaxiTrip, TaxiTripFarePrediction> model)
        {
            var testData = new TextLoader(_testdatapath).CreateFrom<TaxiTrip>(useHeader: true, separator: ',');
            var evaluator = new RegressionEvaluator();
            RegressionMetrics metrics = evaluator.Evaluate(model, testData);
            Console.WriteLine($"Rms = {metrics.Rms}");
            Console.WriteLine($"RSquared = {metrics.RSquared}");
            Console.ReadLine();
        }
    }
}
