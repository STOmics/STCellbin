import numpy as np
from stio.ipr.meta.hardware import ImageInfo
from stio.ipr.meta.research import Research
from stio.microscopy import SlideType
from stio.ipr.ipr_factory import IPRFactory


def slide2ipr0d0d1(msp):
    slide = SlideType(msp.device.manufacturer)
    ipr = IPRFactory().create_ipr_by_name('0.0.1')

    if slide == SlideType.Unknown:
        ipr.image_info.stitched_image = True
        ipr.research.set_fovs_tag(msp.get_fovs_tag())
        ipr.stitch.scope_stitch.global_width = msp.mosaic_width
        ipr.stitch.scope_stitch.global_height = msp.mosaic_height
    else:
        ipr.image_info.app_file_ver = msp.device.app_file_ver
        ipr.image_info.bit_depth = msp.bit_depth()
        ipr.image_info.channel_count = msp.scan.fov_channel
        ipr.image_info.device_sn = msp.device.device_sn
        ipr.image_info.exposure_time = msp.scan.exposure_time
        ipr.image_info.fov_height = msp.scan.fov_height
        ipr.image_info.fov_width = msp.scan.fov_width
        ipr.image_info.manufacturer = msp.device.manufacturer
        ipr.image_info.overlap = msp.scan.overlap
        ipr.image_info.pixel_size_x = msp.scan.scale_x
        ipr.image_info.pixel_size_y = msp.scan.scale_y
        ipr.image_info.qc_result_file = ''  # optional
        ipr.image_info.scan_channel = ''
        ipr.image_info.scan_cols = msp.scan.fov_cols
        ipr.image_info.scan_objective = msp.scan.objective
        ipr.image_info.scan_rows = msp.scan.fov_rows
        ipr.image_info.scan_time = msp.scan.scan_time
        ipr.image_info.stereo_resep_version = ''
        ipr.image_info.rgb_scale = np.array(msp.device.rgb_scale)

        ipr.stitch.scope_stitch.global_loc = msp.fov_location
        ipr.stitch.scope_stitch.global_width = msp.scan.mosaic_width
        ipr.stitch.scope_stitch.global_height = msp.scan.mosaic_height

        ipr.research.set_fovs_tag(msp.get_fovs_tag())
        # if slide == SlideType.DaXingCheng: pass
        # elif slide == SlideType.ChangGuang: pass
        # elif slide == SlideType.Leica: pass
        # elif slide == SlideType.Zeiss: pass
        # elif slide == SlideType.Olympus: pass
        # elif slide == SlideType.Motic: pass
        # else: pass

    return ipr



# def main():
#     slide_file = r'D:\data\guojing\221107\SS200001153BR_D5'
#     s2i = Slide2IPR()
#     s2i.convert(slide_file=slide_file, ipr='')
#
#
# if __name__ == '__main__': main()
