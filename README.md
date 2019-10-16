# CMIP6 Ocean Atlas

## Quick Start
1. Download GLODAPv2.2019
```wget https://www.nodc.noaa.gov/archive/arc0133/0186803/2.2/data/0-data/GLODAPv2.2019_Merged_Master_File.csv```
2. Run the filter_glodap_sections.ipynb
3. Run qc_lines.ipynb

Check ```/interp_examples/``` for example notebooks

## Summary
This repository contains the data processing notebooks for a web based ocean atlas (similar to the WOCE Atlas, shown above) of CMIP6 models sampled as WOCE/GO-SHIP cruises. 

Ultimately these will be included in a website where one will be able select an ocean transect (A16, P16,S4, etc) and then compare the CMIP6 model mean, single model ensembles, and individual realizations to observations. The high quality bottle samples will come from the GLODAPv2 bottle data product. When the hackathon is over, we will have a functioning website and begin a draft of the publication that will accompany the dataset of remapped files. Much of the data processing workflow has been mapped out, therefore I expect much of our collaboration to be focused on developing the web based atlas. Questions to consider will be:

 - How should we compare observations to single model ensembles?
 -  What skill metrics should we use?
 -  What variables should we include? 

With this atlas we can make the CMIP6 data more accessible to observationalists, and modelers will spend much less time creating comparisons of simulated tracer fields and observed tracer fields from WOCE/GO-SHIP cruises.

# Included Sections 
!['sections'](https://github.com/sridge/CMIP6_OceanAtlas/blob/master/qc_images/sections_qc.png "Sections")

