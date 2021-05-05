#!/bin/bash
python generate_fats_features.py
python train_rf_models.py -mode all
python train_rf_models.py -mode sne