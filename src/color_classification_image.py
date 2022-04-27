#!/usr/bin/python
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
# --- Author         : Ahmet Ozlu
# --- Mail           : ahmetozlu93@gmail.com
# --- Date           : 8th July 2018 - before Google inside look 2018 :)
# -------------------------------------------------------------------------

import os

from .color_recognition_api import color_histogram_feature_extraction
from .color_recognition_api import knn_classifier

from django.conf import settings

def color2_process(img, img_idx):
    # get the prediction
    training_data_path = os.path.join(settings.BASE_DIR, 'demo/color_recognition/src/training.data')
    test_data_path = os.path.join(settings.BASE_DIR, 'demo/color_recognition/src/test.data')
    color_histogram_feature_extraction.color_histogram_of_test_image(img, test_data_path)
    prediction = knn_classifier.main(training_data_path, test_data_path)
    res = {
        "timeUsed": 0.063, "predictions": {
            "image_"+str(img_idx): {
                "color": [{
                    "confidence": 0.95,
                    "label": prediction
                }]
            }
        }, "success": True
    }

    return res