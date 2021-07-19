#!/bin/bash
SECONDS=0
clear

methods=(
	linear-fstw
	bspline-fstw
	# spm-mle-fstw
	# spm-mle-estw
	spm-mcmc-fstw
	spm-mcmc-estw
	)
modes=(
	all
	# spm
	# sne
	)

for method in "${methods[@]}"; do
	for kf in {0..4}; do
	# for kf in 1 2 3 4; do
		#python generate_fats_features.py
		echo
	done
done

for method in "${methods[@]}"; do
	# for mid in 7000; do
	for mid in {1000..1004}; do
		for kf in {0..4}; do
			for mode in "${modes[@]}"; do
				# script="python export_2dprojections.py --method $method --mid $mid --kf $kf --mode $mode"
				script="python train_rf_models.py --method $method --mid $mid --kf $kf --mode $mode"
				echo "$script"; eval "$script"
			done
		done
	done
done

mins=$((SECONDS/60))
echo echo "Time Elapsed : ${mins} minutes"