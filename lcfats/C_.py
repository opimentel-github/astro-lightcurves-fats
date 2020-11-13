import sys
import numpy as np

###################################################################################################################################################

N_JOBS = 4
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
    'PeriodLS_v2',
    'Period_fit_v2',
    #'VariabilityIndex',
]

ALERCE_FEATURES = [
    'Amplitude',
    'AndersonDarling',
    'Autocor_length',
    'Beyond1Std',
    'Con',
    'Eta_e',
    'ExcessVar',
    'GP_DRW_sigma',
    'GP_DRW_tau',
    'Gskew',
    'Harmonics',
    #(11) - Harmonics_mag_2_1: 0.06727560624790807
    #(12) - Harmonics_mag_3_1: 0.1119856901696408
    #(13) - Harmonics_mag_4_1: 0.12274118261108367
    #(14) - Harmonics_mag_5_1: 0.05710346340791948
    #(15) - Harmonics_mag_6_1: 0.06308390416802627
    #(16) - Harmonics_mag_7_1: 0.005024974773290941
    #(17) - Harmonics_mse_1: 0.04065159637204645
    #(18) - Harmonics_phase_2_1: 2.0009756351458234
    #(19) - Harmonics_phase_3_1: 6.04389265879566
    #(20) - Harmonics_phase_4_1: 6.168103655501289
    #(21) - Harmonics_phase_5_1: 1.8012355360321504
    #(22) - Harmonics_phase_6_1: 2.4778906023433684
    #(23) - Harmonics_phase_7_1: 0.04875669736539834
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