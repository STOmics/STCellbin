import os
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

QC_CONFIG = os.path.join(ROOT, 'scripts', 'config.json')
QC_ENTRY = os.path.join(ROOT, 'scripts', 'qc.py')
PIPELINE_CONFIG = os.path.join(ROOT, 'scripts', 'pipeline_config.json')
PIPELINE_ENTRY = os.path.join(ROOT, 'scripts', 'pipeline.py')
print("asd")


class Runner(object):
    def __init__(self):
        pass

    def run(self, src_imgs, gene_exp, chip_no, output_dir):
        os.system(f"python {QC_ENTRY} -i {src_imgs} ")
