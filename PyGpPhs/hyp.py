class Hyp:

    def __init__(self):
        self.sn_ = [1, True]
        self.sd_ = [1, True]
        self.l_ = [[1, 1], [True, True]]
        self.JR_vec_ = [[0, 1, -1], [False, False, True]]

    def get_SN(self):
        return self.sn_

    def get_SD(self):
        return self.sd_

    def get_L(self):
        return self.l_

    def get_JRvec(self):
        return self.JR_vec_

    def set_SN(self, new_SN):
        self.sn_ = new_SN

    def set_SD(self, newSD):
        self.sd_ = newSD

    def set_L(self, newL):
        self.l_ = newL

    def set_JRvec(self, newJRvec):
        self.JR_vec_ = newJRvec
