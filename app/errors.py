"""领域异常类型。

所有"非法输入"与"无法求解"的情况都抛出 :class:`SelectionError` 的子类，
由 HTTP 层统一转成带说明的 422 响应，保证它们在进入求根过程之前被挡下。
"""


class SelectionError(ValueError):
    """选型计算中所有可预期错误的基类。"""


class ValidationError(SelectionError):
    """输入本身非法（管径为零、管长为负、曲线为空等）。"""


class ExtrapolationError(SelectionError):
    """采样型风机曲线在声明 ``reject`` 模式时被请求到采样范围之外。"""


class SolverDomainError(SelectionError):
    """无法构造合法的求根括号（例如曲线范围覆盖不到可行区间）。"""


class NoIntersectionError(SelectionError):
    """风机曲线与管阻曲线在可行流量范围内不存在交点。"""
