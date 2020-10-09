# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 11:57:42 2020

@author: joony
"""

import pandas as pd
import numpy as np
import mne
import os
import xarray as xr


os.getcwd()

#os.chdir("/Users/jk8sd/Box Sync/Practice")

dirPath = "C:/Users/joony/Documents/myGit/CAPSTONE/SRM/data/"
folderName = "S001/"
fileName = "S001R04.edf"

# turn into xarray object
def convertToXarray(file) :
    #code

# function for getting an individual file 
def getDataForFile(dirP, folderN, fileN) :
    filePath = dirP + folderN + fileN
    print("Getting file for " + folderN + ", file : " + fileN)
    raw = mne.io.read_raw_edf(filePath)
    ev = mne.events_from_annotations(raw)
    da = xr.DataArray(
    ev[0],
    [
        ("index", range(len(ev[0]))),
        ("description", ["data", "what_is_this", "type_B_LandB_RandB"]),
    ],
    )
    return da


# function for getting all the files in an individual
def getDataForSubject(dirP, folderN) :
    # code
    print("######################################")
    print("Starting subject " + folderN)

def getData(dirP) :
    #code


shape = raw.get_data().shape
print(shape)

samp_freq = raw.info['sfreq']
nyq = samp_freq / 2
print(nyq)

shape[1] / (samp_freq)

ev = mne.events_from_annotations(raw)
print(ev[1])




