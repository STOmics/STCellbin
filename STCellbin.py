import os
import sys
from glob import glob
import argparse
from distutils.util import strtobool

from src.cellbin.utils import clog

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

QC_CONFIG = os.path.join(ROOT, 'scripts', 'config.json')
QC_ENTRY = os.path.join(ROOT, 'scripts', 'qc.py')
PIPELINE_CONFIG = os.path.join(ROOT, 'scripts', 'pipeline_config.json')
PIPELINE_ENTRY = os.path.join(ROOT, 'scripts', 'pipeline.py')
WEIGHT_DIR = os.path.join(ROOT, 'src', 'weights')

PROG_VERSION = "1.0.0"


def rename_files(paths, new_name_dapi, new_name_else):
    for path in paths:
        dir_path, img_name_ext = os.path.split(path)
        img_name, ext = os.path.splitext(img_name_ext)
        if 'DAPI' in img_name:
            new_name = new_name_dapi
        else:
            new_name = new_name_else
        new_path = os.path.join(dir_path, new_name)
        os.rename(path, new_path)


def filter_files_rename(paths, new_name="cell_mask.tif"):
    filter_mask_path = ""
    non_filter_mask_path = ""
    for path in paths:
        if "filter" in path:
            filter_mask_path = path
        else:
            non_filter_mask_path = path

    if filter_mask_path != "":
        mask_to_use = filter_mask_path
        if non_filter_mask_path != "":
            clog.info(f"Removing {non_filter_mask_path}")
            os.remove(non_filter_mask_path)
    else:
        mask_to_use = non_filter_mask_path
    if mask_to_use != "":
        clog.info(f"Using mask {mask_to_use}")
        cell_path_dir, cell_mask_name = os.path.split(mask_to_use)
        new_path = mask_to_use.replace(cell_mask_name, new_name)
        os.rename(mask_to_use, new_path)


def gem_to_txt(gem_path):
    dir_path, file_name = os.path.split(gem_path)
    new_path = gem_path.replace(file_name, "profile.txt")
    import gzip
    with gzip.open(gem_path, 'rb') as f, open(new_path, 'wb') as f_out:
        f_out.write(f.read())


class Runner(object):
    def __init__(self):
        pass

    def run(self, src_imgs, gene_exp, chip_no, output_dir, iffilter):
        qc_out = os.path.join(output_dir, "qc")
        os.makedirs(qc_out, exist_ok=True)
        os.system(f"python {QC_ENTRY} -i {src_imgs} -c {chip_no} -o {qc_out} -j {QC_CONFIG} -s DAPI --zoo {WEIGHT_DIR}")
        files = list(filter(os.path.isfile, glob(qc_out + "/*.ipr")))
        files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        the_most_recent_ipr = files[0]
        pipe_out = os.path.join(output_dir, "pipeline")
        os.system(f"python {PIPELINE_ENTRY} -i {src_imgs} -m {gene_exp} -r {the_most_recent_ipr} -o {pipe_out} "
                  f"-cf {PIPELINE_CONFIG} -z {WEIGHT_DIR} -f {iffilter}")
        self.rename_fs(os.path.join(pipe_out, "registration"))

    def rename_fs(self, result_dir):
        stitch_paths = glob(os.path.join(result_dir, "fov_stitched_**.tif"))
        rename_files(stitch_paths, new_name_dapi="stitched_nuclei.tif", new_name_else="stitched_membrane_wall.tif")
        regist_paths = glob(os.path.join(result_dir, "**regist_**.tif"))
        rename_files(
            regist_paths,
            new_name_dapi="registered_gene_nuclei.tif",
            new_name_else="registered_nuclei_membrane_wall.tif"
        )
        # remove
        ipr_paths = glob(os.path.join(result_dir, "**.ipr"))
        clog.info(f"Removing {ipr_paths[0]}")
        os.remove(ipr_paths[0])
        tissue_cut_paths = glob(os.path.join(result_dir, "**_tissue_cut**.tif"))
        clog.info(f"Removing {tissue_cut_paths[0]}")
        os.remove(tissue_cut_paths[0])

        cell_mask_paths = glob(os.path.join(result_dir, "**mask**.tif"))
        filter_files_rename(cell_mask_paths)

        # gem
        gem_paths = glob(os.path.join(result_dir, "**.gem.gz"))
        filter_files_rename(gem_paths, new_name='profile.gem.gz')
        keep_gem = os.path.join(result_dir, 'profile.gem.gz')
        gem_to_txt(keep_gem)
        clog.info(f"Removing {keep_gem}")
        os.remove(keep_gem)


def main(args, para):
    runner = Runner()
    clog.info(args)
    runner.run(
        src_imgs=args.input,
        gene_exp=args.matrix,
        chip_no=args.chip_name,
        output_dir=args.output,
        iffilter=args.filter
    )
    # runner.rename_fs(result_dir=result_dir)


def arg_parser():
    usage = '''
    python STCellbin.py
    -i "D:\Data\mif\test_mif_data\C01344C4\C01344C4","D:\Data\mif\test_mif_data\C01344C4\C01344C4_Actin_IF" 
    -m "D:\Data\mif\test_mif_data\C01344C4\C01344C4.gem.gz" -o D:\Data\mif\pipeline_out_0920 -c C01344C4
    '''

    clog.info(PROG_VERSION)
    parser = argparse.ArgumentParser(usage=usage)
    # Must have
    parser.add_argument("--version", action="version", version=PROG_VERSION)
    parser.add_argument("-i", "--input", action="store", dest="input", type=str, required=True,
                        help="Tar file / Input image dir.")
    parser.add_argument("-m", "--matrix", action="store", dest="matrix", type=str, required=True,
                        help="Input gene matrix.")
    parser.add_argument("-o", "--output", action="store", dest="output", type=str, required=True,
                        help="Result output dir.")
    parser.add_argument("-f", "--filter", action="store", dest="filter", type=lambda x: bool(strtobool(x)),
                        default=True, help="filter or not")
    parser.add_argument("-c", "--chip", action="store", dest="chip_name", type=str, required=True, help="Chip name")

    parser.set_defaults(func=main)
    (para, args) = parser.parse_known_args()
    para.func(para, args)


if __name__ == '__main__':
    arg_parser()
