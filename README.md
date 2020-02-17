# SPADE
Discovery of spatial marker by deep learning with spatial transcriptome data


## Requirements
python >=3.7
tensorflow >= 1.14.0
keras >= 2.3

R >=3.6.1
Seurat >=3.1.2. (For Spatial Transcriptome Data)
limma >=3.42.

##  How to Run
> Command Line 
```bash
python spade_spatial_marker_by_deeplearning.py --position [Tissue Position List File] --image [High Res Image File] --scale [Scale for High Res Image] --meta [metadata csv file] --outdir [Output directory]
```
* Files Info

Tissue Position List file : Tissue coordinate file includes barcodes, row and col coordinates.
High Res Image File : PNG, JPEG or TIFF for high resolution tissue images
Scale : Scale for high resolution image. For Visium, find from a scalefactos_json file. 
meta : meta data for barcodes and clusters. 


* Example

Check breastca_spatial_SPADE.R file for an example.
