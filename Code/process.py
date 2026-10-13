import os
import re
import numpy as np
import nibabel as nib
import shutil
from joblib import Parallel, delayed
import pandas as pd
import torchio as tio
import matplotlib.pyplot as plt

subjects_path = "/mnt/d/PETCT/ImagesTr/"


def get_shape(subject):
    if subject != "Labels" and subject != "Labels_bal":
        ct = os.path.join(subjects_path, subject, "CT.nii.gz")
        ct = nib.load(ct).get_fdata()[None, ...]
        if ct.shape[0] >= 1000:
            max_length = ct.shape[0] - ct.shape[0] * 0.25
            ct = ct[ct.shape[0] * 0.15:max_length]
        pet = os.path.join(subjects_path, subject, "PET.nii.gz")
        pet = nib.load(pet).get_fdata()[None, ...]

        ##Reshape + normalize
        HOUNSFIELD_AIR, HOUNSFIELD_BONE = -1000, 1000
        transforms = (
            tio.CropOrPad((512, 512, 512)),
            tio.Resize((256, 256, 256)),
            tio.ZNormalization(),
        )
        transform = tio.Compose(transforms)
        ct = transform(ct)

        transforms = (
            tio.CropOrPad((360, 256, 256)),
            tio.Resize((256, 256, 256)),
            tio.ZNormalization(),
        )
        transform = tio.Compose(transforms)
        pet = transform(pet)

        np.save(os.path.join(subjects_path, subject, "CT_Processed.npy"), ct)
        np.save(os.path.join(subjects_path, subject, "PET_Processed.npy"), pet)


subjects = [subject for subject in os.listdir(subjects_path)]
Parallel(n_jobs=8, timeout=9999999)(delayed(get_shape)(subject) for subject in subjects[0:len(subjects) // 2])
Parallel(n_jobs=-2, timeout=9999999)(delayed(get_shape)(subject) for subject in subjects[len(subjects) // 2:])