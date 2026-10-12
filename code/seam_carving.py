import numpy as np
from scipy import ndimage


# =========================================================
# 1. Energy Functions
# =========================================================
def energy_basic(image, **kwargs):
    """
    基础梯度能量函数。

    Parameters
    ----------
    image : ndarray
        输入二维图像。

    Returns
    -------
    ndarray
        归一化梯度能量图。
    """
    image = image.astype(np.float32)

    grad_x = ndimage.sobel(image, axis=1)
    grad_y = ndimage.sobel(image, axis=0)

    energy = np.sqrt(grad_x ** 2 + grad_y ** 2)

    max_energy = energy.max()

    if max_energy <= 1e-8:
        return np.zeros_like(energy, dtype=np.float32)

    return (energy / (max_energy + 1e-8)).astype(np.float32)


def energy_medical(
    image,
    lung_mask=None,
    lambda_protect=None,
    **kwargs
):
    """
    医学感知能量函数：
    Gradient Energy + Lung Protection Energy

    E_medical = E_gradient + lambda * lung_mask

    Parameters
    ----------
    image : ndarray
        当前 seam carving 图像。

    lung_mask : ndarray or None
        与当前图像尺寸一致的肺区 mask。

    lambda_protect : float or None
        肺区保护权重。
        若为 None，则读取 config.LAMBDA_PROTECT。

    Returns
    -------
    ndarray
        Medical-aware energy map。
    """
    from config import LAMBDA_PROTECT

    if lambda_protect is None:
        lambda_protect = LAMBDA_PROTECT

    E_grad = energy_basic(image)

    if lung_mask is not None:
        if lung_mask.shape != image.shape:
            raise ValueError(
                f"lung_mask shape {lung_mask.shape} "
                f"!= image shape {image.shape}"
            )

        E_lung = (
            lung_mask.astype(np.float32)
            * float(lambda_protect)
        )

        return (
            E_grad + E_lung
        ).astype(np.float32)

    return E_grad


# =========================================================
# 2. Find Minimum-Energy Vertical Seam
# =========================================================
def find_seam(energy):
    """
    使用动态规划寻找最低能量垂直 seam。

    性能优化：
    - 前向 DP 在每一行内使用 NumPy 向量化；
    - 回溯仍保持原始 left -> up -> right 的 tie-breaking；
    - 返回格式与旧版完全一致：list[int]。

    Parameters
    ----------
    energy : ndarray
        H x W 能量图。

    Returns
    -------
    list[int]
        长度为 H。
        seam[i] 表示第 i 行的 seam 列坐标。
    """
    if energy.ndim != 2:
        raise ValueError(
            f"energy must be 2-D, got shape {energy.shape}"
        )

    rows, cols = energy.shape

    if cols <= 0:
        raise ValueError("energy image has zero width")

    dp = energy.astype(np.float64).copy()

    # 动态规划累计能量：
    # 保留 row 方向依赖，只对每一行的所有列做向量化。
    for i in range(1, rows):
        prev = dp[i - 1]

        left = np.empty_like(prev)
        left[0] = np.inf
        left[1:] = prev[:-1]

        right = np.empty_like(prev)
        right[-1] = np.inf
        right[:-1] = prev[1:]

        dp[i] += np.minimum(
            np.minimum(left, prev),
            right
        )

    # 从最后一行最小值开始回溯。
    # 这里刻意保留旧版 np.argmin([left, up, right]) 的
    # left -> up -> right tie-breaking 顺序。
    seam = []

    j = int(
        np.argmin(dp[-1])
    )

    seam.append(j)

    for i in range(
        rows - 2,
        -1,
        -1
    ):

        left = (
            dp[i, j - 1]
            if j > 0
            else np.inf
        )

        up = dp[i, j]

        right = (
            dp[i, j + 1]
            if j < cols - 1
            else np.inf
        )

        best = int(
            np.argmin(
                [left, up, right]
            )
        )

        j = int(
            np.clip(
                j + (best - 1),
                0,
                cols - 1
            )
        )

        seam.append(j)

    seam.reverse()

    return seam

# =========================================================
# 3. Remove One Vertical Seam
# =========================================================
def remove_seam(image, seam):
    """
    从二维数组中删除一条垂直 seam。

    可用于：
    - CT image
    - lung mask
    - effective anatomy mask
    - coordinate map

    性能优化：
    使用一次二维布尔索引代替逐行 np.delete()。

    Parameters
    ----------
    image : ndarray
        H x W 二维数组。

    seam : list[int] or ndarray
        长度为 H。

    Returns
    -------
    ndarray
        H x (W-1) 数组。
    """
    if image.ndim != 2:
        raise ValueError(
            f"image must be 2-D, got shape {image.shape}"
        )

    rows, cols = image.shape

    if cols <= 1:
        raise ValueError(
            "Cannot remove seam: image width <= 1"
        )

    seam_arr = np.asarray(
        seam,
        dtype=np.int64
    )

    if seam_arr.ndim != 1 or len(seam_arr) != rows:
        raise ValueError(
            f"seam length {len(seam_arr)} "
            f"!= image rows {rows}"
        )

    if (
        np.any(seam_arr < 0)
        or np.any(seam_arr >= cols)
    ):
        bad_rows = np.where(
            (seam_arr < 0)
            | (seam_arr >= cols)
        )[0]

        first_bad = int(bad_rows[0])

        raise ValueError(
            f"Invalid seam position "
            f"at row {first_bad}: "
            f"{int(seam_arr[first_bad])}, "
            f"width={cols}"
        )

    keep_mask = np.ones(
        (rows, cols),
        dtype=bool
    )

    keep_mask[
        np.arange(rows),
        seam_arr
    ] = False

    return image[
        keep_mask
    ].reshape(
        rows,
        cols - 1
    )

# =========================================================
# 4. Original Seam Carving
# =========================================================
def run_seam_carving(
    image,
    energy_func,
    num_seams,
    lung_mask=None,
    verbose=True
):
    """
    原始 Seam Carving 主流程。

    此函数保留原实验逻辑，
    方便与以前 100/150/200 seams 结果进行一致性验证。

    返回：
    1. seam carving 后图像
    2. 所有 seam 在原始图像上的坐标

    Parameters
    ----------
    image : ndarray
        原始二维图像。

    energy_func : callable
        energy_basic 或 energy_medical。

    num_seams : int
        要删除的垂直 seam 数量。

    lung_mask : ndarray or None
        Medical SC 使用。
        Basic SC 可为 None。

    verbose : bool
        是否打印进度。

    Returns
    -------
    current : ndarray

    all_seams : list
        每条 seam 都映射回原始图像坐标。
    """
    if image.ndim != 2:
        raise ValueError(
            f"image must be 2-D, got {image.shape}"
        )

    if num_seams < 0:
        raise ValueError(
            "num_seams must be >= 0"
        )

    rows, cols = image.shape

    if num_seams >= cols:
        raise ValueError(
            f"num_seams={num_seams} "
            f"must be smaller than width={cols}"
        )

    if (
        lung_mask is not None
        and lung_mask.shape != image.shape
    ):
        raise ValueError(
            f"lung_mask shape {lung_mask.shape} "
            f"!= image shape {image.shape}"
        )

    current = image.copy()

    coord_map = np.tile(
        np.arange(cols),
        (rows, 1)
    )

    current_mask = (
        lung_mask.copy().astype(bool)
        if lung_mask is not None
        else None
    )

    all_seams = []

    for k in range(num_seams):

        energy = energy_func(
            current,
            lung_mask=current_mask
        )

        seam = find_seam(
            energy
        )

        # 映射回原始图像坐标
        original_cols = [
            int(coord_map[i, seam[i]])
            for i in range(rows)
        ]

        all_seams.append(
            original_cols
        )

        # 同步删除
        coord_map = remove_seam(
            coord_map,
            seam
        )

        current = remove_seam(
            current,
            seam
        )

        if current_mask is not None:
            current_mask = remove_seam(
                current_mask.astype(
                    np.uint8
                ),
                seam
            ).astype(bool)

        if (
            verbose
            and (k + 1) % 50 == 0
        ):
            print(
                f"  {k + 1}/{num_seams} 条缝完成"
            )

    return (
        current,
        all_seams
    )


# =========================================================
# 5. Seam Carving with Mask Tracking
# =========================================================
def run_seam_carving_with_tracking(
    image,
    energy_func,
    num_seams,
    lung_mask=None,
    effective_mask=None,
    verbose=True
):
    """
    Seam Carving + Mask Tracking。

    与原 run_seam_carving 使用完全相同的：
    - energy_func
    - find_seam
    - remove_seam
    - coord_map

    额外同步追踪：
    1. lung_mask
    2. effective_mask

    这样可以得到真实 seam carving 后的：
    - Lung Visible Ratio
    - Lung Loss Ratio
    - Lung Area Preservation
    - EACR
    - Centroid Shift
    - Aspect Ratio Error

    Basic SC 也可以传 lung_mask：
    energy_basic 会通过 **kwargs 忽略该 mask，
    因此不会改变 Basic SC 的 seam 选择；
    mask 仅用于同步追踪。

    Parameters
    ----------
    image : ndarray
        原始二维图像。

    energy_func : callable
        energy_basic 或 energy_medical。

    num_seams : int
        删除 seam 数量。

    lung_mask : ndarray or None
        原始肺区 mask。

    effective_mask : ndarray or None
        原始有效解剖结构 mask。
        只用于 tracking，不直接参与能量函数。

    verbose : bool

    Returns
    -------
    current : ndarray
        seam carving 后图像。

    all_seams : list
        每条 seam 映射回原始图像的坐标。

    current_lung_mask : ndarray or None
        沿真实 seam 同步删除后的肺区 mask。

    current_effective_mask : ndarray or None
        沿真实 seam 同步删除后的有效解剖 mask。
    """
    if image.ndim != 2:
        raise ValueError(
            f"image must be 2-D, got {image.shape}"
        )

    if num_seams < 0:
        raise ValueError(
            "num_seams must be >= 0"
        )

    rows, cols = image.shape

    if num_seams >= cols:
        raise ValueError(
            f"num_seams={num_seams} "
            f"must be smaller than image width={cols}"
        )

    if (
        lung_mask is not None
        and lung_mask.shape != image.shape
    ):
        raise ValueError(
            f"lung_mask shape {lung_mask.shape} "
            f"!= image shape {image.shape}"
        )

    if (
        effective_mask is not None
        and effective_mask.shape != image.shape
    ):
        raise ValueError(
            f"effective_mask shape "
            f"{effective_mask.shape} "
            f"!= image shape {image.shape}"
        )

    current = image.copy()

    # 用于将每一步 seam 映射回原始图像
    coord_map = np.tile(
        np.arange(cols),
        (rows, 1)
    )

    current_lung_mask = (
        lung_mask.copy().astype(bool)
        if lung_mask is not None
        else None
    )

    current_effective_mask = (
        effective_mask.copy().astype(bool)
        if effective_mask is not None
        else None
    )

    all_seams = []

    for k in range(num_seams):

        # ---------------------------------
        # 1. Energy
        # ---------------------------------
        energy = energy_func(
            current,
            lung_mask=current_lung_mask
        )

        # ---------------------------------
        # 2. Find seam
        # ---------------------------------
        seam = find_seam(
            energy
        )

        # ---------------------------------
        # 3. Map seam to original coordinates
        # ---------------------------------
        original_cols = [
            int(coord_map[i, seam[i]])
            for i in range(rows)
        ]

        all_seams.append(
            original_cols
        )

        # ---------------------------------
        # 4. Remove seam from coordinate map
        # ---------------------------------
        coord_map = remove_seam(
            coord_map,
            seam
        )

        # ---------------------------------
        # 5. Remove seam from image
        # ---------------------------------
        current = remove_seam(
            current,
            seam
        )

        # ---------------------------------
        # 6. Track lung mask
        # ---------------------------------
        if current_lung_mask is not None:

            current_lung_mask = remove_seam(
                current_lung_mask.astype(
                    np.uint8
                ),
                seam
            ).astype(bool)

        # ---------------------------------
        # 7. Track effective anatomy mask
        # ---------------------------------
        if current_effective_mask is not None:

            current_effective_mask = remove_seam(
                current_effective_mask.astype(
                    np.uint8
                ),
                seam
            ).astype(bool)

        if (
            verbose
            and (k + 1) % 50 == 0
        ):
            print(
                f"  {k + 1}/{num_seams} 条缝完成"
            )

    return (
        current,
        all_seams,
        current_lung_mask,
        current_effective_mask
    )


# =========================================================
# 6. Lung Invasion Rate
# =========================================================
def count_lung_invasion(
    all_seams,
    lung_mask
):
    """
    统计 seam 侵入肺区的像素比例。

    返回单位：
    percent (%)

    Example:
        8.16 表示 8.16%
    """
    if lung_mask is None:
        return np.nan

    if len(all_seams) == 0:
        return 0.0

    seams_arr = np.asarray(
        all_seams,
        dtype=np.int64
    )

    if seams_arr.ndim != 2:
        raise ValueError(
            f"all_seams must be 2-D after conversion, "
            f"got shape {seams_arr.shape}"
        )

    num_seams, rows_per_seam = seams_arr.shape
    mask_rows, mask_cols = lung_mask.shape

    row_idx = np.tile(
        np.arange(rows_per_seam, dtype=np.int64),
        num_seams
    )

    col_idx = seams_arr.reshape(-1)

    valid = (
        (row_idx >= 0)
        & (row_idx < mask_rows)
        & (col_idx >= 0)
        & (col_idx < mask_cols)
    )

    total = int(
        seams_arr.size
    )

    if total == 0:
        return 0.0

    invaded = int(
        np.sum(
            lung_mask[
                row_idx[valid],
                col_idx[valid]
            ] > 0
        )
    )

    return float(
        invaded / total * 100.0
    )

# =========================================================
# 7. Lung Area Preservation
# =========================================================
def lung_area_preservation(
    original_mask,
    result_mask
):
    """
    肺区像素保留率。

    对 Seam Carving：
    result_mask 必须是沿真实 seam
    同步删除后的肺区 mask。

    Lung Area Preservation =
    transformed lung pixels /
    original lung pixels

    注意：
    数学上与 lung_visible_ratio 相同。
    为兼容旧代码保留此函数。

    论文中建议优先使用：
    Lung Visible Ratio
    Lung Loss Ratio
    """
    if (
        original_mask is None
        or result_mask is None
    ):
        return np.nan

    original_area = np.sum(
        original_mask > 0
    )

    if original_area == 0:
        return 0.0

    result_area = np.sum(
        result_mask > 0
    )

    return float(
        result_area / original_area
    )


# =========================================================
# 8. HU Distribution Similarity
# =========================================================
def hu_distribution_similarity(
    original_hu,
    result_hu,
    bins=100
):
    """
    HU 值分布相似度。

    使用：
    - HU range: -1000 ~ 400
    - histogram correlation

    返回：
    0 ~ 1
    越高越相似。
    """
    range_min = -1000
    range_max = 400

    hist_orig, _ = np.histogram(
        original_hu.flatten(),
        bins=bins,
        range=(
            range_min,
            range_max
        ),
        density=True
    )

    hist_res, _ = np.histogram(
        result_hu.flatten(),
        bins=bins,
        range=(
            range_min,
            range_max
        ),
        density=True
    )

    hist_orig = (
        hist_orig
        / (
            hist_orig.sum()
            + 1e-8
        )
    )

    hist_res = (
        hist_res
        / (
            hist_res.sum()
            + 1e-8
        )
    )

    # 防止常量直方图导致 corrcoef NaN
    if (
        np.std(hist_orig) < 1e-12
        or np.std(hist_res) < 1e-12
    ):
        return 0.0

    correlation = np.corrcoef(
        hist_orig,
        hist_res
    )[0, 1]

    if np.isnan(correlation):
        return 0.0

    return float(
        np.clip(
            correlation,
            0,
            1
        )
    )


# =========================================================
# 9. Lung Centroid Shift
# =========================================================
def lung_centroid_shift(
    original_mask,
    result_mask,
    scale_factor
):
    """
    肺区质心偏移。

    注意：
    这个指标更适合均匀缩放参考。

    对 Seam Carving 来说，
    由于变换是非均匀的，
    该指标只作为辅助评价，
    不建议作为论文核心指标。

    Parameters
    ----------
    original_mask : ndarray
    result_mask : ndarray
    scale_factor : float
        target_width / original_width

    Returns
    -------
    float
        pixels
    """
    if (
        original_mask is None
        or result_mask is None
    ):
        return np.nan

    if (
        original_mask.sum() == 0
        or result_mask.sum() == 0
    ):
        return 0.0

    cy_orig, cx_orig = (
        ndimage.center_of_mass(
            original_mask
        )
    )

    # 均匀缩放参考位置
    cx_expected = (
        cx_orig
        * scale_factor
    )

    cy_expected = cy_orig

    cy_res, cx_res = (
        ndimage.center_of_mass(
            result_mask
        )
    )

    shift = np.sqrt(
        (
            cx_res
            - cx_expected
        ) ** 2
        +
        (
            cy_res
            - cy_expected
        ) ** 2
    )

    return float(
        shift
    )


# =========================================================
# 10. Aspect Ratio Error
# =========================================================
def aspect_ratio_error(
    original_mask,
    result_mask
):
    """
    肺区 bounding region 的纵横比误差。

    越小表示全局肺区宽高比例变化越小。

    对 Seam Carving：
    作为辅助几何指标使用。
    """
    if (
        original_mask is None
        or result_mask is None
    ):
        return np.nan

    def get_aspect_ratio(mask):

        rows = np.any(
            mask,
            axis=1
        )

        cols = np.any(
            mask,
            axis=0
        )

        if (
            not rows.any()
            or not cols.any()
        ):
            return 1.0

        h = rows.sum()
        w = cols.sum()

        if h <= 0:
            return 1.0

        return float(
            w / h
        )

    ar_orig = get_aspect_ratio(
        original_mask
    )

    ar_result = get_aspect_ratio(
        result_mask
    )

    return float(
        abs(
            ar_orig
            - ar_result
        )
    )


# =========================================================
# 11. Effective Anatomical Coverage Ratio
# =========================================================
def effective_anatomical_coverage(
    effective_mask
):
    """
    Effective Anatomical Coverage Ratio
    EACR

    EACR =
    effective anatomical pixels /
    total preview pixels

    越高表示同样大小的预览图中，
    越多像素用于有效解剖结构，
    越少像素浪费在背景等无效区域。

    Parameters
    ----------
    effective_mask : ndarray
        当前变换后的有效解剖结构 mask。

    Returns
    -------
    float
        0 ~ 1
    """
    if effective_mask is None:
        return np.nan

    total_pixels = (
        effective_mask.size
    )

    if total_pixels == 0:
        return np.nan

    effective_pixels = np.sum(
        effective_mask > 0
    )

    return float(
        effective_pixels
        / total_pixels
    )


# =========================================================
# 12. Lung Visible Ratio
# =========================================================
def lung_visible_ratio(
    original_mask,
    transformed_mask
):
    """
    Lung Visible Ratio

    Lung Visible Ratio =
    transformed lung pixels /
    original lung pixels

    解释：
    真实保留下来的肺区比例。

    越高越好。
    """
    if (
        original_mask is None
        or transformed_mask is None
    ):
        return np.nan

    original_area = np.sum(
        original_mask > 0
    )

    if original_area == 0:
        return np.nan

    transformed_area = np.sum(
        transformed_mask > 0
    )

    return float(
        transformed_area
        / original_area
    )


# =========================================================
# 13. Lung Loss Ratio
# =========================================================
def lung_loss_ratio(
    original_mask,
    transformed_mask
):
    """
    Lung Loss Ratio

    Lung Loss Ratio =
    1 - Lung Visible Ratio

    越低越好。
    """
    visible = lung_visible_ratio(
        original_mask,
        transformed_mask
    )

    if np.isnan(visible):
        return np.nan

    return float(
        1.0 - visible
    )