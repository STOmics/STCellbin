# StereoCell v2.0

## Introduction
StereoCell v2.0 expands the application on the basic of the previous version. StereoCell v2.0 registers the spatial gene expression map and the cell membrane/wall staining image with the nuclei staining image as an intermediary, thus the cell membrane/wall outlines can be directly segmented from the cell membrane/wall staining image without correcting the nuclei boundaries.

<div align="center">
  <img src="docs/main_figure.png" width=567>
    <h6>
      Enhanced application on generating single-cell gene expression profile for high-resolution spatial transcriptomics.
    </h6>
</div>
<br>

## Installation
StereoCell v2.0 is developed by Python scripts. Please make sure Conda is installed before installation.

Download the [project resource code](https://codeload.github.com/STOmics/StereoCell_v2.0/zip/refs/heads/main) and install requirements.txt in a python==3.8 environment.

```text
# python3.8 in conda env
conda create --name=StereoCellv2 python=3.8
conda activate StereoCellv2
cd StereoCell_v2.0-main
pip install -r requirements.txt (需要给出requirements.txt，对于一些特殊依赖，例如pyvips需要额外说明)  # install
```

## Tutorials

### Test dataset
The demo datasets have been deposited into Spatial Transcript Omics DataBase (STOmics DB) of China National GeneBank DataBase (CNGBdb) with accession number [STT0000048](https://db.cngb.org/stomics/project/STT0000048).

### Command Line
StereoCell v2.0 in one-stop is performed by command (参考这个例子写):

```text
python stereocellv2.py
--tiles_path /data/SS200000135TL_D1
--gene_exp_data /data/SS200000135TL_D1.gem.gz
--output_path /data/result
--chip_no SS200000135TL_D1
```

* ```--tiles_path```  The path of all tiles.
* ```--gene_exp_data``` The compressed file of spatial gene expression data.
* ```--output_path``` The output path.
* ```--chip_no``` Chip number of the Stereo-seq data.

## License and Citation
StereoCell v2.0 is released under the MIT license.

Please cite StereoCell v2.0 in your publications if it helps your research:

```text
B. Zhang et al. StereoCell v2.0 expands the application to generate single-cell gene expression profile for high-resolution spatial transcriptomics. Preprint in bioRxiv. 2023. (引用根据实际情况更新)
```

## Reference
```text
M. Li et al. StereoCell enables highly accurate single-cell segmentation for spatial transcriptomics. Preprint in bioRxiv. 2023.
```
