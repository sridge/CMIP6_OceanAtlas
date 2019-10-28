# CMIP6 Ocean Atlas

## Quick Start
1. login to [ocean.pangeo.io](https://ocean.pangeo.io) with ORCID
2. Clone this repository:\
```git clone https://github.com/sridge/CMIP6_OceanAtlas.git```
3. ```cd CMIP6_OceanAtlas/```
4. Download GLODAPv2.2019 to the ```qc/``` folder:\
```cd qc/ && wget https://www.nodc.noaa.gov/archive/arc0133/0186803/2.2/data/0-data/GLODAPv2.2019_Merged_Master_File.csv```
5. Run ```qc/filter_glodap_sections.ipynb```
6. Run ```qc/qc_lines.ipynb```
7. [Download master files from figshare](https://figshare.com/articles/CMIP6_Ocean_Atlas/10052342) (8.12 GB, compressed) or generate your own (```notebooks/model_master_files.ipynb```)

Now you can run the example notebook: ```notebooks/example_notebook.pynb```

## Included Sections 
!['sections'](https://github.com/sridge/CMIP6_OceanAtlas/blob/master/notebooks/sections_qc.png "Sections")

## Summary
This repository contains the data processing notebooks for a web based ocean atlas of CMIP6 models sampled as WOCE/GO-SHIP cruises. 

Ultimately these will be included in a website where one will be able select an ocean transect (A16, P16, S04, etc) and then compare the CMIP6 model mean, single model ensembles, and individual realizations to observations.

With this atlas we can make the CMIP6 data more accessible to observationalists, and modelers will spend much less time creating comparisons of simulated tracer fields and observed tracer fields from WOCE/GO-SHIP cruises.



