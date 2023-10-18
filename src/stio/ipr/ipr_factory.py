from stio.utils import clog
from stio.ipr import IPRVersion
from stio.ipr.ipr_v0d1d0 import IPRV0d1d0
from stio.ipr.ipr_v0d0d1 import IPRv0d0d1


class IPRFactory(object):
    def __init__(self):
        self.version = IPRVersion.V0D0D1

    @staticmethod
    def create_ipr_by_name(version: str):
        iv = IPRVersion(version)
        clog.info('IPR version is {}'.format(version))
        if iv == IPRVersion.V0D0D1:
            return IPRv0d0d1()
        elif iv == IPRVersion.V0D1D0:
            return IPRV0d1d0()
        else:
            return None
