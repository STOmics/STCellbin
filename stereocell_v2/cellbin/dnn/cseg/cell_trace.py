import cv2


def get_trace(mask):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    h, w = mask.shape[: 2]
    output = []
    for i in range(num_labels):
        box_w, box_h, area = stats[i][2:]
        if box_h == h and box_w == w:
            continue
        output.append([box_h, box_w, area])  # 更改，删除list of list，方便外面numpy array保存到ipr by dzh
    return output


if __name__ == '__main__':
    import tifffile

    mask = tifffile.imread('/media/Data/dzh/neq_qc_test_data/SS200000464BL_C4/test_out/targz_fov_out/registration/SS200000464BL_C4_mask.tif')
    output = get_trace(mask)
    print("Asd")
