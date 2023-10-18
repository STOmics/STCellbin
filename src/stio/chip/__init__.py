from os.path import join, dirname, abspath

# 区域号范围
s13_shard_set = ("ABCDEFGHJKLMN", "123456789ABCD")
s6_shard_set = ("ABCDEF", "123456")


class STOmicsChip(object):
    chip_dir = current = dirname(abspath(__file__))

    def __init__(self):
        self.pool = self._load_items()

    def _load_items(self, ):
        import ast
        from Crypto.Cipher import AES

        key_path = join(self.chip_dir, 'key.bin')
        data_path = join(self.chip_dir, 'encrypted.bin')
        with open(key_path, "rb") as f:
            key = f.read()

        file_in = open(data_path, "rb")
        nonce, tag, ciphertext = [file_in.read(x) for x in (16, 16, -1)]
        cipher = AES.new(key, AES.MODE_EAX, nonce)
        data = cipher.decrypt_and_verify(ciphertext, tag)
        data = ast.literal_eval(data.decode("UTF-8"))
        return data

    def get_chip_grids(self, chip_no):
        return self.pool[chip_no]['grids']

    def get_chip_pitch(self, chip_no):
        return self.pool[chip_no]['pitch']

    def get_chip_info(self, chip_no):
        return self.pool[chip_no]

    def has(self, ):
        return list(self.pool.keys())

    def check_long_no(self, chip_no):
        if not chip_no:
            return False
        for i in [chip_no[:3], chip_no[:4]]:
            if i in self.has():
                return True
        return False

    # 短码校验
    def check_short_no(self, chip_no):
        if not chip_no:
            return False
        if len(chip_no) == 8:
            if chip_no[1:6].isdigit():
                if chip_no[0] in 'ABCDY' and chip_no[6:] == '00':
                    return True
                elif chip_no[0] in 'ABCD':
                    return chip_no[6] in s6_shard_set[0] and chip_no[7] in s6_shard_set[1]
                elif chip_no[0] == "Y":
                    return chip_no[6] in s13_shard_set[0] and chip_no[7] in s13_shard_set[1]
                else:
                    return False
        else:
            if chip_no[1:6].isdigit():
                if chip_no[0] in 'ABCD':
                    return chip_no[6] in s6_shard_set[0] and chip_no[7] in s6_shard_set[1]
                elif chip_no[0] == "Y":
                    return chip_no[6] in s13_shard_set[0] and chip_no[7] in s13_shard_set[1]
                else:
                    return False
        return False

    def is_chip_number_legal(self, chip_no):
        result = self.check_long_no(chip_no) or self.check_short_no(chip_no)
        return result

    def get_valid_chip_no(self, chip_name):
        chipNo = "SS2"
        for chip in self.has():
            if chip in chip_name:
                chipNo = chip
        return chipNo


if __name__ == '__main__':
    st_chip = STOmicsChip()
    # test1 = [
    #     "SS200000302TL_B5",
    #     "SS200000302TL_B5",
    #     "DP8400000302TL_B5",
    #     "FP200000302TL_B5",
    #     "Y00035MD",
    #     "s2"
    # ]
    # for t in test1:
    #     re = st_chip.is_chip_number_legal(t)
    #     print(re)
    st_chip.get_valid_chip_no("DP8400")
