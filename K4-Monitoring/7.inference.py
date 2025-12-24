"""
Load Test / Inference Script for Real Model Traffic
Author: Anwar-Rohmadi
Uses realistic house price features matching the trained model
"""
import requests
import time
import argparse
from concurrent.futures import ThreadPoolExecutor
import json

ENDPOINT = "http://localhost:8000/predict"

# Realistic house feature ranges based on Ames Housing dataset
# These are typical encoded/scaled values after preprocessing
REALISTIC_FEATURES = {
    # Numeric features with realistic ranges
    "MSSubClass": (20, 190),
    "MSZoning": (0, 7),  # Encoded categorical
    "LotFrontage": (21, 313),
    "LotArea": (1300, 215245),
    "Street": (0, 1),
    "Alley": (0, 2),
    "LotShape": (0, 3),
    "LandContour": (0, 3),
    "Utilities": (0, 3),
    "LotConfig": (0, 4),
    "LandSlope": (0, 2),
    "Neighborhood": (0, 24),  # Encoded
    "Condition1": (0, 8),
    "Condition2": (0, 7),
    "BldgType": (0, 4),
    "HouseStyle": (0, 7),
    "OverallQual": (1, 10),  # Key feature
    "OverallCond": (1, 10),
    "YearBuilt": (1872, 2010),
    "YearRemodAdd": (1950, 2010),
    "RoofStyle": (0, 5),
    "RoofMatl": (0, 7),
    "Exterior1st": (0, 14),
    "Exterior2nd": (0, 15),
    "MasVnrType": (0, 4),
    "MasVnrArea": (0, 1600),
    "ExterQual": (0, 4),
    "ExterCond": (0, 4),
    "Foundation": (0, 5),
    "BsmtQual": (0, 5),
    "BsmtCond": (0, 4),
    "BsmtExposure": (0, 4),
    "BsmtFinType1": (0, 6),
    "BsmtFinSF1": (0, 5644),
    "BsmtFinType2": (0, 6),
    "BsmtFinSF2": (0, 1474),
    "BsmtUnfSF": (0, 2336),
    "TotalBsmtSF": (0, 6110),
    "Heating": (0, 5),
    "HeatingQC": (0, 4),
    "CentralAir": (0, 1),
    "Electrical": (0, 4),
    "1stFlrSF": (334, 4692),
    "2ndFlrSF": (0, 2065),
    "LowQualFinSF": (0, 572),
    "GrLivArea": (334, 5642),  # Key feature
    "BsmtFullBath": (0, 3),
    "BsmtHalfBath": (0, 2),
    "FullBath": (0, 3),
    "HalfBath": (0, 2),
    "BedroomAbvGr": (0, 8),
    "KitchenAbvGr": (0, 3),
    "KitchenQual": (0, 4),
    "TotRmsAbvGrd": (2, 14),
    "Functional": (0, 6),
    "Fireplaces": (0, 3),
    "FireplaceQu": (0, 5),
    "GarageType": (0, 6),
    "GarageYrBlt": (1900, 2010),
    "GarageFinish": (0, 3),
    "GarageCars": (0, 4),  # Key feature
    "GarageArea": (0, 1418),
    "GarageQual": (0, 5),
    "GarageCond": (0, 5),
    "PavedDrive": (0, 2),
    "WoodDeckSF": (0, 857),
    "OpenPorchSF": (0, 547),
    "EnclosedPorch": (0, 552),
    "3SsnPorch": (0, 508),
    "ScreenPorch": (0, 480),
    "PoolArea": (0, 738),
    "PoolQC": (0, 4),
    "Fence": (0, 4),
    "MiscFeature": (0, 4),
    "MiscVal": (0, 15500),
    "MoSold": (1, 12),
    "YrSold": (2006, 2010),
    "SaleType": (0, 8),
    "SaleCondition": (0, 5),
    "TotalSF": (334, 11752),  # Engineered feature
}

def generate_realistic_features():
    """Generate realistic house features for inference"""
    import random
    features = []
    for name, (min_val, max_val) in REALISTIC_FEATURES.items():
        if isinstance(min_val, int) and isinstance(max_val, int):
            features.append(random.randint(min_val, max_val))
        else:
            features.append(random.uniform(min_val, max_val))
    return features

def send_request():
    """Send a single prediction request with realistic features"""
    features = generate_realistic_features()
    try:
        response = requests.post(ENDPOINT, json={"features": features}, timeout=10)
        return response.status_code, response.json() if response.ok else None
    except Exception as e:
        return 500, str(e)

def load_test(n_requests=100, n_workers=5, delay=0.1):
    """Run load test with concurrent requests"""
    print(f"Starting load test: {n_requests} requests, {n_workers} workers")
    print(f"Using {len(REALISTIC_FEATURES)} realistic house price features")
    
    success = 0
    errors = 0
    latencies = []
    predictions = []
    
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        for i in range(n_requests):
            start = time.time()
            future = executor.submit(send_request)
            status, result = future.result()
            latency = time.time() - start
            
            if status == 200:
                success += 1
                latencies.append(latency)
                if result and 'prediction' in result:
                    predictions.append(result['prediction'])
                    print(f"[{i+1}/{n_requests}] OK - ${result['prediction']:,.0f} - {latency:.3f}s")
                else:
                    print(f"[{i+1}/{n_requests}] OK - {latency:.3f}s")
            else:
                errors += 1
                print(f"[{i+1}/{n_requests}] ERROR - {result}")
            
            time.sleep(delay)
    
    print("\n" + "="*60)
    print(f"LOAD TEST COMPLETE (REAL MODEL)")
    print("="*60)
    print(f"Total: {n_requests}, Success: {success}, Errors: {errors}")
    if latencies:
        print(f"Avg latency: {sum(latencies)/len(latencies):.3f}s")
    if predictions:
        print(f"Prediction range: ${min(predictions):,.0f} - ${max(predictions):,.0f}")
        print(f"Avg prediction: ${sum(predictions)/len(predictions):,.0f}")
    print("="*60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load test for House Price Prediction API")
    parser.add_argument("--requests", type=int, default=100, help="Number of requests")
    parser.add_argument("--workers", type=int, default=5, help="Concurrent workers")
    parser.add_argument("--delay", type=float, default=0.1, help="Delay between requests")
    args = parser.parse_args()
    
    load_test(args.requests, args.workers, args.delay)
