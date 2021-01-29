import sys
import numpy as np

###################################################################################################################################################

EPS = 1e-10
N_JOBS = 6 # The number of jobs to use for the computation. If -1 all CPUs are used. If 1 is given, no parallel computing code is used at all, which is useful for debugging. For n_jobs below -1, (n_cpus + 1 + n_jobs) are used. Thus for n_jobs = -2, all CPUs but one are used.
CHUNK_SIZE = N_JOBS*1
NAN_VALUE = -999

OLD_FEATURES = [
    #'MeanvariancePairSlopeTrend',
    'CAR_sigma',
    'CAR_mean',
    'CAR_tau',
    'FluxPercentileRatioMid20',
    'FluxPercentileRatioMid35',
    'FluxPercentileRatioMid50',
    'FluxPercentileRatioMid65',
    'FluxPercentileRatioMid80',
    'PercentDifferenceFluxPercentile',
    #'PeriodLS_v2', # slow?
    #'Period_fit_v2', # slow?
    #'VariabilityIndex',
]

ALERCE_SPM_FEATURES = [
    'SF_ML_amplitude',
    'IAR_phi',
    'LinearTrend',
    #'GP_DRW_sigma', # slow?
    #'GP_DRW_tau', # slow?
]

ALERCE_FEATURES = [
    'Amplitude',
    'AndersonDarling',
    'Autocor_length',
    'Beyond1Std',
    'Con',
    'Eta_e',
    'ExcessVar',
    #'GP_DRW_sigma', # slow?
    #'GP_DRW_tau', # slow?
    'Gskew',
    #'Harmonics', # slow?
    #(11) - Harmonics_mag_2_1
    #(12) - Harmonics_mag_3_1
    #(13) - Harmonics_mag_4_1
    #(14) - Harmonics_mag_5_1
    #(15) - Harmonics_mag_6_1
    #(16) - Harmonics_mag_7_1
    #(17) - Harmonics_mse_1
    #(18) - Harmonics_phase_2_1
    #(19) - Harmonics_phase_3_1
    #(20) - Harmonics_phase_4_1
    #(21) - Harmonics_phase_5_1
    #(22) - Harmonics_phase_6_1
    #(23) - Harmonics_phase_7_1
    'IAR_phi',
    'LinearTrend',
    #'MHPS_PN_flag',
    #'MHPS_high',
    #'MHPS_low',
    #'MHPS_non_zero',
    #'MHPS_ratio',
    'MaxSlope',
    'Mean',
    'Meanvariance',
    'MedianAbsDev',
    'MedianBRP',
    'PairSlopeTrend',
    'PercentAmplitude',
    #'Period_band',
    #'Psi_CS',
    #'Psi_eta',
    'Pvar',
    'Q31',
    'Rcs',
    'SF_ML_amplitude',
    'SF_ML_gamma',
    #'SPM_A',
    #'SPM_beta',
    #'SPM_chi',
    #'SPM_gamma',
    #'SPM_t0',
    #'SPM_tau_fall',
    #'SPM_tau_rise',
    'Skew',
    'SmallKurtosis',
    'Std',
    'StetsonK',
    #'delta_mag_fid',
    #'delta_mjd_fid',
    #'delta_period',
    #'dmag_first_det_fid',
    #'dmag_non_det_fid',
    #'first_mag',
    #'iqr',
    #'last_diffmaglim_before_fid',
    #'last_mjd_before_fid',
    #'max_diffmaglim_after_fid',
    #'max_diffmaglim_before_fid',
    #'mean_mag',
    #'median_diffmaglim_after_fid',
    #'median_diffmaglim_before_fid',
    #'min_mag',
    #'n_det',
    #'n_neg',
    #'n_non_det_after_fid',
    #'n_non_det_before_fid',
    #'n_pos',
    #'positive_fraction',
]