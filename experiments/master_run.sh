#!/bin/bash
SECONDS=0
#python generate_fats_features.py
#python export_2dprojections.py -mode all
#python export_2dprojections.py -mode spm
#python export_2dprojections.py -mode sne
#python train_rf_models.py -mode all
python train_rf_models.py -mode spm
python train_rf_models.py -mode sne

echo echo "Time Elapsed : ${$((SECONDS/60))} minutes"