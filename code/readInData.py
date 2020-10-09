# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 11:57:42 2020

@author: SRM TEAM
"""

import pandas as pd
import numpy as np
import mne
import os
import xarray as xr


os.getcwd()

#os.chdir("/Users/jk8sd/Box Sync/Practice")

url = "https://github.com/VIXSoh/SRM/raw/master/data/
#folderName = "S001/"
#fileName = "S001R04.edf"

# turn into xarray object
def convertToXarray(file) :
    #code

# function for getting an individual file 
def getDataForFile(dirP, folderN, fileN) :
    filePath = dirP + folderN + fileN
    print("Getting file for " + folderN + ", file : " + fileN)
    edf = wget.download(filePath)
    raw = mne.io.read_raw_edf(edf)
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
    for i in range(1,14) :
        if i < 10 :
            i = "0" + str(i)
        fileN =  folderN + "R" + i + ".edf"
        iterDaAr = getDataForFile(dirP, folderN, fileN)

def getData(dirP) :
    #code






