import os
import requests
from cellbin.utils import clog


weights = {
        'clarity_eval_mobilev3small05064_DAPI_20230202_pytorch.onnx': "https://pan.genomics.cn/ucdisk/api/2.0/share/link/download?shareEventId=share_202331611295523_28e42b24e57844768733f13eab17c1e4&nodeId=8a80804a867c36b90186e43c9b034785&code=",
        'points_detect_yolov5obb_SSDNA_20220513_pytorch.onnx': "https://pan.genomics.cn/ucdisk/api/2.0/share/link/download?shareEventId=share_202331691548476_d9f05ef0d23a4a069ffcbff5a49ac860&nodeId=8a80804a867c36b90186e43cb7e4478a&code=",
        'tissueseg_yolo_SH_20230131_th.onnx': "https://pan.genomics.cn/ucdisk/api/2.0/share/link/download?shareEventId=share_20233117301767_10ff2d6b54c94351a4654eb8c9bd3ddb&nodeId=8a808043867c394401869c7d01282e29",
        'tissueseg_bcdu_SDI_220822_tf.onnx': "https://pan.genomics.cn/ucdisk/api/2.0/share/link/download?shareEventId=share_20233117301767_10ff2d6b54c94351a4654eb8c9bd3ddb&nodeId=8a808043867c394401869c7d1f062e38",
        'tissueseg_bcdu_H_221101_tf.onnx': "https://pan.genomics.cn/ucdisk/api/2.0/share/link/download?shareEventId=share_20233117301767_10ff2d6b54c94351a4654eb8c9bd3ddb&nodeId=8a808043867c394401869c7debdd2e57",
        'tissueseg_bcdu_rna_220909_tf.onnx': "https://pan.genomics.cn/ucdisk/api/2.0/share/link/download?shareEventId=share_20233117301767_10ff2d6b54c94351a4654eb8c9bd3ddb&nodeId=8a808043867c394401869c7d87842e4b",
        'cellseg_bcdu_SHDI_221008_tf.onnx': "https://pan.genomics.cn/ucdisk/api/2.0/share/link/download?shareEventId=share_20233117301767_10ff2d6b54c94351a4654eb8c9bd3ddb&nodeId=8a808043867c394401869c7d10192e32",
    }

# pwd = os.path.dirname(__file__)

# Just for test
# checkpoints = {
#     'ClarityEvaler': {
#         'Local': 'ST_TP_Mobile_small_050.onnx',
#         'Remote': ''
#     },
#     'PointsDetector': {
#         'Local': '',
#         'Remote': ''
#     },
#     'PointsFilter': {
#         'Local': '',
#         'Remote': ''
#     },
#     'TissueYolo': {
#         'Local': '',
#         'Remote': ''
#     },
#     'TissueBCDU': {
#         'Local': '',
#         'Remote': ''
#     },
#     'CellBCDU': {
#         'Local': '',
#         'Remote': ''
#     }
# }


def auto_download_weights(save_dir, names):
    for k, url in weights.items():
        if k not in names:
            continue
        weight = os.path.join(save_dir, k)
        if not os.path.exists(weight):
            try:
                clog.info('Download {} from remote {}'.format(k, url))
                r = requests.get(url)
                with open(os.path.join(save_dir, k), 'wb') as fd:
                    fd.write(r.content)
            except Exception as e:
                clog.error('FAILED! (Download {} from remote {})'.format(k, url))
                print(e)
                return 1
        else:
            clog.info('{} already exists'.format(k))
    return 0


if __name__ == '__main__':
    save_dir = r"D:\Data\qc\new_qc_test_data\clarity\bad\test_imgs\test_download"
    names = weights.keys()
    auto_download_weights(
        save_dir=save_dir,
        names=names,
    )
