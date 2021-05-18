#!/bin/bash
SECONDS=0

#python generate_fats_features.py
for mode in 'spm' 'sne' 'all'
do
python export_2dprojections.py -mode $mode
#python train_rf_models.py -mode $mode
done

echo echo "Time Elapsed : $(($SECONDS/60)) minutes"