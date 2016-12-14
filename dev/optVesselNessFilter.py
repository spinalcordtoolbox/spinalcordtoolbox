#!/usr/bin/env python
#########################################################################################
#
# Script to optimize the vesselness filter on a dataset with T1w and T2w images.
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2015 Polytechnique Montreal <www.neuro.polymtl.ca>
# Author: Benjamin De Leener
# Modified: 2015-03-10
#
# About the license: see the file LICENSE.TXT
#########################################################################################
from msct_parser import Parser
import sys
import os
import sct_utils as sct
import time
import sct_register_pipeline
import numpy as np

input_vessel_params = '0.15,1.0,5.0,1.5,4.5,5'

#=======================================================================================================================
# Start program
#=======================================================================================================================
if __name__ == "__main__":
    begin = time.time()

    # contrast: t1, t2 or both
    input_t = "t1"

    # fixed parameters
    beta = 1.0;
    gamma = 5.0;
    numberOfSigmaSteps = 10;

    # parameters to optimize
    minimalPath_alpha = np.arange(0.1,0.21,0.1); #arange(0.01,0.31,0.01);
    minimalPath_sigmaMinimum = np.arange(0.5,1.1,0.5); #arange(0.5,2.51,0.1);
    minimalPath_sigmaMaximum = np.arange(4.0,5.1,1.0); #arange(3.0,7.01,0.1);

    results = np.empty([len(minimalPath_alpha), len(minimalPath_sigmaMinimum), len(minimalPath_sigmaMaximum)])

    for x,alpha in enumerate(minimalPath_alpha):
        for y,sigmaMinimum in enumerate(minimalPath_sigmaMinimum):
            for z,sigmaMaximum in enumerate(minimalPath_sigmaMaximum):
                # vesselness parameters
                input_vessel_params = str(alpha)+","+str(beta)+","+str(gamma)+","+str(sigmaMinimum)+","+str(sigmaMaximum)+","+str(numberOfSigmaSteps)

                # copy of the folder
                folder_name = "data_"+str(alpha)+"_"+str(beta)+"_"+str(gamma)+"_"+str(sigmaMinimum)+"_"+str(sigmaMaximum)+"_"+str(numberOfSigmaSteps)
                sct.runProcess("mkdir "+folder_name)
                sct.runProcess("cp -R original_data/ "+folder_name)

                pipeline_test = sct_register_pipeline.Pipeline(folder_name, input_t, vessel=True, vessel_params=input_vessel_params)
                pipeline_test.compute()

                results[x,y,z] = pipeline_test.vesselnessFilterMetric

    max_value = results.argmax()
    x_min,y_min,z_min = np.unravel_index(max_value, results.shape)
    print "Maximum= "+max_value
    print "Optimal parameters:"
    print "  "+str(minimalPath_alpha(x_min))
    print "  "+str(beta)
    print "  "+str(gamma)
    print "  "+str(minimalPath_sigmaMinimum(y_min))
    print "  "+str(minimalPath_sigmaMaximum(z_min))
    print "  "+str(numberOfSigmaSteps)

    elapsed_time = round(time.time() - begin, 2)