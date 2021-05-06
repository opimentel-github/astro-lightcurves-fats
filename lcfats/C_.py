import sys
import numpy as np

###################################################################################################################################################

MAX_DAY = 100.
EPS = 1e-5
REC_LOSS_EPS = .01 # ***

### JOBLIB
import os
JOBLIB_BACKEND = 'loky' # loky multiprocessing threading
N_JOBS = -1 # The number of jobs to use for the computation. If -1 all CPUs are used. If 1 is given, no parallel computing code is used at all, which is useful for debugging. For n_jobs below -1, (n_cpus + 1 + n_jobs) are used. Thus for n_jobs = -2, all CPUs but one are used.
CHUNK_SIZE = os.cpu_count() if N_JOBS<0 else N_JOBS

SNE_SELECTED_FEATURES = [
	'SF_ML_amplitude',
	'IAR_phi',
	'LinearTrend',
	'GP_DRW_sigma', # slow
	'GP_DRW_tau', # slow
	
	'SPM_t0',
	'SPM_A',
	'SPM_chi',
	'SPM_gamma',
	'SPM_tau_rise',
	'SPM_tau_fall',
	'SPM_beta',

	'MHPS_PN_flag',
	'MHPS_high',
	'MHPS_low',
	'MHPS_non_zero',
	'MHPS_ratio',
	]

ALERCE_SPM_FEATURES = [
	'SPM_t0',
	'SPM_A',
	'SPM_chi',
	'SPM_gamma',
	'SPM_tau_rise',
	'SPM_tau_fall',
	'SPM_beta',
	]

MPHS_FEATURES = [
	'MHPS_PN_flag',
	'MHPS_high',
	'MHPS_low',
	'MHPS_non_zero',
	'MHPS_ratio',
	]

HARMONICS_FEATURES = [
	'Harmonics_mag_2_1',
	'Harmonics_mag_3_1',
	'Harmonics_mag_4_1',
	'Harmonics_mag_5_1',
	'Harmonics_mag_6_1',
	'Harmonics_mag_7_1',
	'Harmonics_mse_1',
	'Harmonics_phase_2_1',
	'Harmonics_phase_3_1',
	'Harmonics_phase_4_1',
	'Harmonics_phase_5_1',
	'Harmonics_phase_6_1',
	'Harmonics_phase_7_1',
	]

FATS_FEATURES = [
	'Amplitude',
	'AndersonDarling',
	'Autocor_length',
	'Beyond1Std',
	'Con',
	'Eta_e',
	'ExcessVar',
	'GP_DRW_sigma', # slow
	'GP_DRW_tau', # slow
	'Gskew',
	'Harmonics', # slow
	'IAR_phi',
	'LinearTrend',
	'MaxSlope',
	'Mean',
	'Meanvariance',
	'MedianAbsDev',
	'MedianBRP',
	'PairSlopeTrend',
	'PercentAmplitude',
	'Pvar',
	'Q31',
	'Rcs',
	'SF_ML_amplitude',
	'SF_ML_gamma',
	'Skew',
	'SmallKurtosis',
	'Std',
	'StetsonK',
	'CAR_sigma',
	'CAR_mean',
	'CAR_tau',
	'FluxPercentileRatioMid20',
	'FluxPercentileRatioMid35',
	'FluxPercentileRatioMid50',
	'FluxPercentileRatioMid65',
	'FluxPercentileRatioMid80',
	'PercentDifferenceFluxPercentile',
	'PeriodLS_v2', # slow
	'Period_fit_v2', # slow
	]

NOT_IMPLEMENTED = [
	'median_diffmaglim_before_fid',
	'Period_band',
	'mean_mag',
	'delta_period',
	'n_pos',
	'min_mag',
	'first_mag',
	'positive_fraction',
	'Psi_CS',
	'Psi_eta',
	'dmag_non_det_fid',
	'MeanvariancePairSlopeTrend',
	'delta_mag_fid',
	'delta_mjd_fid',
	'dmag_first_det_fid',
	'iqr',
	'last_diffmaglim_before_fid',
	'last_mjd_before_fid',
	'max_diffmaglim_after_fid',
	'max_diffmaglim_before_fid',
	'n_det',
	'n_neg',
	'n_non_det_after_fid',
	'n_non_det_before_fid',
	'VariabilityIndex',
	]