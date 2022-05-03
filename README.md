# NMR-Echo-Train-Inversion-to-created-a-typical-NMR-log
This repository contains the python code for a python appication for NMR Echo Train Inversion to created a typical NMR log T2 distributions and NMR log outputs from the Time-Domain echo train data. 

Back in 1994 when we were preparing our own software for NMR Inversion from the Time-Domain Echo Train to the T2 distribution for the MIRL C tool, Dan Georgi initiated some very intuitive software for the process as shown below. The objective was to determine the NMR T2 bin porosities from the echo train data and create the typical MRIL outputs of the effective NMR porosity (MPHI), NMR Capillary Bound Water (MBVI) and Free Fluid (MFFI). The example below shows this process over a short section of NMR log.

We have created the python code working with SciPy curve_fit to perform this NMR inversion. This application might be a bit unorthodox for NMR Echo Train inversion, but it does serve as a python's SciPy's example. The sample code and sample data can be found in this repository:

![Geolog_Image](NMR_log.gif)

Please consider simple example as work in progress as we are investigating other methods too. 
