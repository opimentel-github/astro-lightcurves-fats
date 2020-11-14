from typing import List

#from ..core.base import FeatureExtractorSingleBand
import pandas as pd
import numpy as np
import logging
import mhps


class MHPSExtractor():
    def __init__(self, t1=100, t2=10, dt=3.0, mag0=19.0, epsilon=1.0):
        self.t1 = t1
        self.t2 = t2
        self.dt = dt
        self.mag0 = mag0
        self.epsilon = epsilon

    def get_features_keys(self) -> List[str]:
        return [
            'MHPS_ratio',
            'MHPS_low',
            'MHPS_high',
            'MHPS_non_zero',
            'MHPS_PN_flag']

    def get_required_keys(self) -> List[str]:
        return ["magpsf_ml", "sigmapsf_ml", "mjd"]

    def compute_feature_in_one_band_(self, mag, magerr, time, **kwargs):
        mhps_results = []
        ratio, low, high, non_zero, pn_flag = mhps.statistics(mag,
                                                              magerr,
                                                              time,
                                                              self.t1,
                                                              self.t2)
        values = np.array([[ratio, low, high, non_zero, pn_flag]])
        mhps_df = pd.DataFrame(values, columns=self.get_features_keys(), index=[''])
        return mhps_df
        mhps_results.append(mhps_df)
        mhps_results = pd.concat(mhps_results, axis=0)
        #mhps_results.index.name = 'oid'
        return mhps_results

    def compute_feature_in_one_band(self, detections, band=None, **kwargs):
        oids = detections.index.unique()
        mhps_results = []

        detections = detections.sort_values('mjd')
        columns = self.get_features_keys_with_band(band)
        for oid in oids:
            oid_detections = detections.loc[[oid]]
            if band not in oid_detections.fid.values:
                logging.info(
                    f'extractor=MHPS object={oid} required_cols={self.get_required_keys()} band={band}')
                nan_df = self.nan_df(oid)
                nan_df.columns = columns
                mhps_results.append(nan_df)
                continue

            oid_band_detections = oid_detections[oid_detections.fid == band]

            mag = oid_band_detections.magpsf_ml
            magerr = oid_band_detections.sigmapsf_ml
            time = oid_band_detections.mjd
            ratio, low, high, non_zero, pn_flag = mhps.statistics(mag.values,
                                                                  magerr.values,
                                                                  time.values,
                                                                  self.t1,
                                                                  self.t2)
            values = np.array([[ratio, low, high, non_zero, pn_flag]])
            mhps_df = pd.DataFrame(values, columns=columns, index=[oid])
            mhps_results.append(mhps_df)
        mhps_results = pd.concat(mhps_results, axis=0)
        mhps_results.index.name = 'oid'
        return mhps_results