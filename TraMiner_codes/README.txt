This folder includes the code used in the paper:
 
TraMiner: Vision-based Analysis of Locomotion Traces for Cognitive Assessment in Smart-homes, 
by Samaneh Zolfaghari, Elham Khodabandehloo and Daniele Riboni, published in Cognitive Computation Special Issue on Advances in Deep Learning for Clinical and Healthcare Applications

The full dataset is available here: http://casas.wsu.edu/datasets/assessmentdata.zip
In this dataset the sensors can be categorized by:

   Mxx:       motion sensor
   Ixx:       item sensor for selected items in the kitchen
   Dxx:       door sensor
   AD1-A:     burner sensor
   AD1-B:     hot water sensor
   AD1-C:     cold water sensor
   Txx:       temperature sensors
   P001:      whole-apartment electricity usage

1) First of all, we made 'dataxy.csv' which is a subset of CASAS row data by simple python script. 
   In this file we extract row motion sensors (Mxx) and door sensors (Dxx) data with states of “ON” and “Open” from all available data.

2) Then we made 'FileID.csv' file which reports the IDs of the individuals used in our experimentation, as explained in the paper.

3) In order to prepare input file for trajectory classification in TRAJECTORY FEATURE EXTRACTION part of paper, you should run these scripts:

- 'featureExtraction_GVFE.py' script is related to preparing "GVFE images", as explained in the paper.

- 'featureExtraction_TS.py'script is related to preparing "TRAJ images" and "SPEED images", as explained in the paper.

4) In the next step you should run 'trajectoryClassification.py' script whcih performs the proposed two inputs MLP DNN Classification on TRAJ and SPEED trajectory images, 
   are made by previous scripts. The DCNN function and one input MLP DNN function related to using just TRAJ or SPEED for comparison is also implemented in this file.


The dashboard can be accessed at: https://sites.unica.it/domusafe/traminer/

* All use of the code must cite the publication: 
- Zolfaghari, S., Khodabandehloo, E., & Riboni, D. (2021). TraMiner: Vision-based analysis of locomotion traces for cognitive assessment in smart-homes. Cognitive Computation, 1-22.
