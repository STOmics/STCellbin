import os


def search_files(file_path, exts):
    files_ = list()
    for root, dirs, files in os.walk(file_path):
        if len(files) == 0:
            continue
        for f in files:
            fn, ext = os.path.splitext(f)
            if ext in exts: files_.append(os.path.join(root, f))

    return files_


def filename2index(file_name, style, row_len=None):
    file_name = os.path.basename(file_name)
    if style.lower() == 'motic':
        tags = os.path.splitext(file_name)[0].split('_')
        xy = list()
        for tag in tags:
            if (len(tag) == 4) and tag.isdigit(): xy.append(tag)
        x_str = xy[0]
        y_str = xy[1]
        return [int(y_str), int(x_str)]
    elif style.lower() == 'zeiss':
        name, info, tail = os.path.splitext(file_name)[0].split('_')
        x_start, x_y, y_len = info.split('-')
        x_start = int(x_start.split('x')[1])
        x_len, y_start = x_y.split('y')
        x_len = int(x_len)
        y_start = int(y_start)
        y_len = int(y_len.split('m')[0])
        overlap_x = int(x_len * 0.9)
        overlap_y = int(y_len * 0.9)
        i = round(x_start / overlap_x)
        j = round(y_start / overlap_y)
        return [i, j]
    elif style.lower() == 'leica':
        prefix, num = os.path.splitext(file_name)[0].split('--Stage')
        x = int(int(num) / row_len)
        y = int(int(num) % row_len)
        return [x, y]
    else:
        return None
