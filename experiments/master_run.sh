#!/bin/bash
#python generate_fats_features.py
python train_rf_models.py -mode sne
python train_rf_models.py -mode all
python export_2dprojections.py -mode sne
python export_2dprojections.py -mode all