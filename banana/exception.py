class NiAnalysisError(Exception):
    pass


class NiAnalysisUsageError(NiAnalysisError):
    pass


class NiAnalysisMissingHeaderValue(NiAnalysisError):
    pass


class NiAnalysisRuntimeError(NiAnalysisError):
    pass
