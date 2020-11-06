import logging

logger = logging.getLogger(__name__)


class Coupler_compatability_mixin():
    def __init__(self, existing=None):
        if existing:
            msg = (
                'Deprecation Warning! '
                'Do not use keyword argument "existing" on Coupler. '
                'Instead use keyword argument "base"')
            logger.warning(msg)
            self.base = existing

    def _couple(self, modifier):
        msg = (
            'Deprecation Warning! '
            'Do not use method "_couple" on Coupler. '
            'Instead use method "couple".')
        logger.warning(msg)
        return self.couple(modifier)


class Parameter_Compatability_mixin:
    @property
    def _coupler(self):
        return self.coupler

    @_coupler.setter
    def _coupler(self, val):
        self._coupler = val
