class BananaError(Exception):
    pass


class BananaUsageError(BananaError):
    pass


class BananaMissingHeaderValue(BananaError):
    pass


class BananaRuntimeError(BananaError):
    pass


class BananaTestSetupError(BananaError):
    pass
