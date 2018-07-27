using System;
using System.Collections.Generic;
using System.Text;

namespace TaxiFarePrediction
{
    static class TestTrips
    {
        internal static readonly TaxiTrip Trip1 = new TaxiTrip
        {
            VendorId = "CMT",
            RateCode = "1",
            PassengerCount = 1,
            TripDistance = 10.33f,
            PaymentType = "CSH",
            FareAmount =  0 
        };
    }
}
