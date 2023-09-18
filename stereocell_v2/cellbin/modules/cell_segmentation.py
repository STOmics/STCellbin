from cellbin.modules import CellBinElement
from cellbin.dnn.cseg.detector import Segmentation
from cellbin.dnn.cseg.cell_trace import get_trace as get_t
from cellbin.utils import clog


class CellSegmentation(CellBinElement):
    def __init__(self, model_path, gpu="-1", num_threads=0):
        """
        Args:
            model_path(str): network model file path
            gpu(str): gpu index
            num_threads(int): default is 0,When you use the CPU,
            you can use it to control the maximum number of threads
        """
        super(CellSegmentation, self).__init__()

        self._MODE = "onnx"
        self._NET = "bcdu"
        self._WIN_SIZE = (256, 256)
        self._INPUT_SIZE = (256, 256, 1)
        self._OVERLAP = 16

        self._gpu = gpu
        self._model_path = model_path
        self._num_threads = num_threads

        self._cell_seg = Segmentation(
            net=self._NET,
            mode=self._MODE,
            gpu=self._gpu,
            num_threads=self._num_threads,
            win_size=self._WIN_SIZE,
            intput_size=self._INPUT_SIZE,
            overlap=self._OVERLAP
        )
        clog.info("start reading model weight")
        self._cell_seg.f_init_model(model_path=self._model_path)
        clog.info("end reading model weight")

    def run(self, img):
        """
        run cell predict
        Args:
            img(ndarray): img array

        Returns(ndarray):cell mask

        """
        clog.info("start cell seg")
        mask = self._cell_seg.f_predict(img)
        clog.info("end cell seg")
        return mask

    @staticmethod
    def get_trace(mask):
        return get_t(mask)
