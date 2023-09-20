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
git clone https://github.com/STOmics/StereoCell_v2.0.git
conda create --name=StereoCellv2 python=3.8
conda activate StereoCellv2
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 -c pytorch
cd StereoCell_v2.0
pip install -r requirements.txt # install
```

* The ```pyvips``` package needs to be installed separately. The following is referenced from [pyvips](https://libvips.github.io/pyvips/README.html#non-conda-install)

**On Windows**, first you need to use pip to install like,
```text
$ pip install --user pyvips==2.2.1
```
then you need to download the compiled library from [vips-dev-8.12](https://github.com/libvips/libvips/releases),
To set PATH from within Python, you need something like this at the start:

```python
import os
vipshome = 'c:\\vips-dev-8.7\\bin'
os.environ['PATH'] = vipshome + ';' + os.environ['PATH']
```

**On Linux**,
```text
$ conda install --channel conda-forge pyvips==2.2.1
```


## Tutorials

### Test dataset
The demo datasets have been deposited into Spatial Transcript Omics DataBase (STOmics DB) of China National GeneBank DataBase (CNGBdb) with accession number [STT0000048](https://db.cngb.org/stomics/project/STT0000048).

### Command Line
StereoCell v2.0 in one-stop is performed by command:

```text
python StereoCell_v2.0/stereocell_v2.py
-i data/C01344C4/C01344C4,data/C01344C4/C01344C4_Actin_IF
-m /data/C01344C4.gem.gz
-o /data/result
-c C01344C4
```

* ```-i```  The path of all tiles.
* ```-m``` The compressed file of spatial gene expression data.
* ```-o``` The output path.
* ```-c``` Chip number of the Stereo-seq data.

## License and Citation
StereoCell v2.0 is released under the MIT license.

Please cite StereoCell v2.0 in your publications if it helps your research:

```text
B. Zhang et al. StereoCell v2.0 expands the application to generate single-cell gene expression profile for high-resolution spatial transcriptomics. Preprint in bioRxiv. 2023.
```

## Reference
```text
M. Li et al. StereoCell enables highly accurate single-cell segmentation for spatial transcriptomics. Preprint in bioRxiv. 2023.
```
> https://github.com/matejak/imreg_dft <br>
> https://github.com/rezazad68/BCDU-Net <br>
> https://github.com/libvips/pyvips <br>
