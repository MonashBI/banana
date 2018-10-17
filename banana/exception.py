class bananaError(Exception):
    pass


class bananaUsageError(bananaError):
    pass


class bananaMissingHeaderValue(bananaError):
    pass


class bananaRuntimeError(bananaError):
    pass
