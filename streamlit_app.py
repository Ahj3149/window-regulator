from __future__ import annotations

import io
import logging
import os
import sys
import textwrap
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any, Dict, Iterable, TYPE_CHECKING

import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
import reliability as rel
import streamlit as st

if TYPE_CHECKING:
    from streamlit.runtime.uploaded_file_manager import UploadedFile
else:  # pragma: no cover - runtime fallback for type checking only names
    UploadedFile = Any


DATA_COLUMN_NAME = "내구력 회수"
DATA_COLUMN_LABEL = "내구력 회수 (사이클)"

DISTRIBUTION_LABELS = {
    "Weibull_2P": "와이블 2모수",
    "Lognormal_2P": "로그정규 2모수",
    "Normal_2P": "정규 2모수",
    "Exponential_1P": "지수 1모수",
}

PARAM_LABELS = {
    "alpha": "규모 (alpha)",
    "beta": "형상 (beta)",
    "mu": "μ",
    "sigma": "σ",
    "Lambda": "λ",
    "mean": "평균",
    "std": "표준편차",
}

SHAPE_PARAM_LABELS = {
    "Weibull_2P": ("beta", "형상 (beta)"),
    "Lognormal_2P": ("sigma", "σ"),
}

SCALE_PARAM_LOOKUP = {
    "Weibull_2P": "alpha",
    "Lognormal_2P": "mu",
    "Normal_2P": "mu",
    "Exponential_1P": "Lambda",
}


class StreamlitLogHandler(logging.Handler):
    """Forward log records to the Streamlit UI."""

    def __init__(self) -> None:
        super().__init__()
        self.log_messages: list[tuple[str, str]] = []

    def emit(self, record: logging.LogRecord) -> None:
        msg = self.format(record)
        self.log_messages.append((record.levelname, msg))

    def display_logs(self) -> None:
        for level, msg in self.log_messages:
            if level == "INFO":
                st.info(msg)
            elif level == "WARNING":
                st.warning(msg)
            elif level in {"ERROR", "CRITICAL"}:
                st.error(msg)
            else:
                st.write(msg)


streamlit_handler = StreamlitLogHandler()
root_logger = logging.getLogger()
if not any(isinstance(handler, StreamlitLogHandler) for handler in root_logger.handlers):
    root_logger.addHandler(streamlit_handler)


def configure_matplotlib_font() -> None:
    """Ensure a font supporting Korean characters is used."""
    preferred_fonts = [
        "Malgun Gothic",
        "NanumGothic",
        "AppleGothic",
        "NanumGothicCoding",
        "Noto Sans CJK KR",
        "Noto Sans KR",
    ]
    available = {font.name for font in fm.fontManager.ttflist}
    for font_name in preferred_fonts:
        if font_name in available:
            plt.rcParams["font.family"] = font_name
            plt.rcParams["axes.unicode_minus"] = False
            plt.rcParams["pdf.fonttype"] = 42  # embed TrueType for PDF so 한글이 깨지지 않음
            plt.rcParams["ps.fonttype"] = 42
            logging.info("Matplotlib 폰트를 '%s'(으)로 설정했습니다.", font_name)
            break
    else:
        logging.warning("한글을 지원하는 폰트를 찾지 못했습니다. 기본 폰트로 표시됩니다.")


def setup_streamlit() -> None:
    """Configure the Streamlit page and file logging."""
    st.set_page_config(page_title="내구-필드 수명 분석", layout="wide")
    st.title("내구-필드 수명 분석")

    os.makedirs("results", exist_ok=True)
    log_file_path = os.path.join("results", "analysis_log.txt")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file_path, encoding="utf-8"),
            streamlit_handler,
            logging.StreamHandler(),
        ],
        force=True,
    )
    logging.info("Streamlit 환경이 초기화되었습니다.")

    configure_matplotlib_font()


@dataclass
class FitResults:
    name: str
    best_distribution_name: str
    best_distribution: Any
    results_table: pd.DataFrame
    fitters: Dict[str, Any]
    sample_stats: Dict[str, float]
    selection_reason: str

    def __getattr__(self, item: str) -> Any:
        try:
            return self.fitters[item]
        except KeyError as exc:
            raise AttributeError(item) from exc


def _extract_ci_bounds(fitter: Any, parameter: str) -> tuple[float, float] | None:
    """Return lower/upper confidence interval bounds for a fitter parameter."""
    tuple_attributes = [
        f"{parameter}_CI",
        f"{parameter}_ci",
        f"{parameter}_ConfidenceInterval",
    ]
    for attr in tuple_attributes:
        if hasattr(fitter, attr):
            ci_value = getattr(fitter, attr)
            if isinstance(ci_value, Iterable):
                ci_list = list(ci_value)
                if len(ci_list) >= 2:
                    try:
                        return float(ci_list[0]), float(ci_list[1])
                    except (TypeError, ValueError):
                        continue

    lower_candidates = [
        f"{parameter}_lower",
        f"{parameter}_Lower",
        f"{parameter}_lowerCL",
        f"{parameter}_lower_cl",
    ]
    upper_candidates = [
        f"{parameter}_upper",
        f"{parameter}_Upper",
        f"{parameter}_upperCL",
        f"{parameter}_upper_cl",
    ]

    lower = None
    for attr in lower_candidates:
        if hasattr(fitter, attr):
            lower = getattr(fitter, attr)
            break

    upper = None
    for attr in upper_candidates:
        if hasattr(fitter, attr):
            upper = getattr(fitter, attr)
            break

    if lower is not None and upper is not None:
        try:
            return float(lower), float(upper)
        except (TypeError, ValueError):
            return None

    return None


def format_distribution_name(dist_name: str) -> str:
    return DISTRIBUTION_LABELS.get(dist_name, dist_name)


def _fmt_int(value: float | int) -> str:
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return "-"


def _fmt_float(value: float, digits: int = 3) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "-"


def compute_sample_stats(series: pd.Series) -> Dict[str, float]:
    values = series.to_numpy(dtype=float, copy=False)
    return {
        "count": int(len(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values, ddof=1)) if len(values) > 1 else float("nan"),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "p10": float(np.percentile(values, 10)),
        "p90": float(np.percentile(values, 90)),
    }


def plot_distribution_fit(failures: np.ndarray, fitter: Any, dataset_label: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.hist(failures, bins="auto", density=True, color="#2878B5", alpha=0.6, label="관측 데이터")

    dist = getattr(fitter, "distribution", None)
    if dist and hasattr(dist, "PDF"):
        x_min, x_max = failures.min(), failures.max()
        pad = max((x_max - x_min) * 0.1, 1e-3)
        x = np.linspace(max(0, x_min - pad), x_max + pad, 200)
        try:
            y = np.asarray(dist.PDF(x))
            ax.plot(x, y, color="#D35400", linewidth=2, label="최적 분포 PDF")
        except Exception as exc:  # pragma: no cover - visualization only
            logging.warning("분포 PDF를 그리는 중 오류가 발생했습니다: %s", exc)

    ax.set_xlabel(DATA_COLUMN_LABEL)
    ax.set_ylabel("확률밀도")
    ax.set_title(f"{dataset_label} 고장 분포")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    return fig


def _df_to_markdown(df: pd.DataFrame, index_name: str | None = None) -> str:
    """Lightweight markdown table builder that avoids the tabulate dependency."""
    rounded = df.copy()
    float_cols = rounded.select_dtypes(include=["float", "float64"]).columns
    rounded[float_cols] = rounded[float_cols].round(3)
    if index_name:
        rounded = rounded.copy()
        rounded.insert(0, index_name, rounded.index)
    headers = list(rounded.columns)

    def fmt(value: Any) -> str:
        if isinstance(value, float):
            return f"{value:.3f}"
        return str(value)

    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for _, row in rounded.iterrows():
        lines.append("| " + " | ".join(fmt(v) for v in row) + " |")
    return "\n".join(lines)


def _add_wrapped_text(
    ax: plt.Axes,
    text: str,
    start_y: float,
    width: int = 90,
    line_height: float = 0.028,
    fontsize: int = 11,
    x: float = 0.05,
) -> float:
    y = start_y
    for line in textwrap.wrap(text, width=width) or [""]:
        ax.text(x, y, line, fontsize=fontsize, ha="left", va="top")
        y -= line_height
    return y


def _add_bullet_lines(
    ax: plt.Axes,
    lines: list[str],
    start_y: float,
    x: float = 0.05,
    bullet: str = "•",
    width: int = 90,
    line_height: float = 0.028,
    fontsize: int = 11,
) -> float:
    y = start_y
    for line in lines:
        wrapped = textwrap.wrap(line, width=width) or [""]
        for idx, segment in enumerate(wrapped):
            prefix = f"{bullet} " if idx == 0 else "   "
            ax.text(x, y, prefix + segment, fontsize=fontsize, ha="left", va="top")
            y -= line_height
    return y


def _add_section(
    ax: plt.Axes,
    title: str,
    lines: list[str],
    start_y: float,
    width: int = 90,
    line_height: float = 0.028,
    fontsize: int = 11,
) -> float:
    ax.text(0.05, start_y, title, fontsize=13, weight="bold", ha="left", va="top")
    y = start_y - line_height
    for line in lines:
        y = _add_wrapped_text(ax, line, y, width=width, line_height=line_height, fontsize=fontsize, x=0.06)
    return y - line_height * 0.4


def _read_csv_with_fallback(uploaded: UploadedFile) -> pd.DataFrame:
    raw_bytes = uploaded.getvalue()
    for encoding in ("utf-8-sig", "cp949", "euc-kr"):
        try:
            return pd.read_csv(io.BytesIO(raw_bytes), encoding=encoding)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(io.BytesIO(raw_bytes), encoding="utf-8", errors="ignore")


def _format_column_label(column: object) -> str:
    if isinstance(column, tuple):
        parts: list[str] = []
        for part in column:
            if part is None or pd.isna(part):
                continue
            text = str(part).strip()
            if text and text.lower() != "nan":
                parts.append(text)
        return " | ".join(parts)
    if column is None or pd.isna(column):
        return ""
    return str(column).strip()


def _get_excel_sheet_names(uploaded: UploadedFile | None) -> list[str]:
    if not uploaded:
        return []
    try:
        excel = pd.ExcelFile(io.BytesIO(uploaded.getvalue()))
    except Exception as exc:
        logging.warning("엑셀 시트 목록을 읽는 중 오류가 발생했습니다: %s", exc)
        st.warning("엑셀 시트 목록을 읽는 중 오류가 발생했습니다.")
        return []
    return excel.sheet_names


def _preview_excel_columns(
    uploaded: UploadedFile | None,
    sheet_name: str | None,
    header_rows: int,
) -> list[str]:
    if not uploaded or not sheet_name:
        return []
    header = list(range(header_rows)) if header_rows > 1 else 0
    try:
        df = pd.read_excel(
            io.BytesIO(uploaded.getvalue()),
            sheet_name=sheet_name,
            header=header,
            nrows=0,
        )
    except Exception as exc:
        logging.warning("엑셀 컬럼 미리보기에 실패했습니다: %s", exc)
        return []
    return [_format_column_label(col) for col in df.columns if _format_column_label(col)]


def load_and_prepare_data(
    test_file: UploadedFile | None, field_file: UploadedFile | None
) -> bool:
    """Load uploaded CSV files and cache the failure cycle count series."""
    if not (test_file and field_file):
        logging.warning("내구 시험 CSV와 필드 CSV를 모두 업로드해 주세요.")
        return False

    try:
        durability_df: pd.DataFrame = _read_csv_with_fallback(test_file)
        field_df: pd.DataFrame = _read_csv_with_fallback(field_file)
    except Exception as exc:  # pragma: no cover - guarded by user input
        logging.error("CSV 파일을 읽는 중 문제가 발생했습니다: %s", exc)
        return False

    required_column = DATA_COLUMN_NAME
    if required_column not in durability_df.columns or required_column not in field_df.columns:
        logging.error("CSV 파일에는 '%s' 컬럼이 모두 포함되어야 합니다.", required_column)
        return False

    durability_series: pd.Series = durability_df[required_column].dropna()
    field_series: pd.Series = field_df[required_column].dropna()

    st.session_state.durability_failures = durability_series
    st.session_state.field_failures = field_series

    logging.info("데이터가 정상적으로 불러와졌습니다.")

    with st.expander("원본 데이터 미리 보기"):
        col1, col2 = st.columns(2)
        with col1:
            st.write("**내구 시험 데이터**")
            st.dataframe(durability_df.head())
        with col2:
            st.write("**필드 데이터**")
            st.dataframe(field_df.head())

    return True


def analyse_single_dataset(data: pd.Series, name: str) -> FitResults:
    """Fit several lifetime distributions and report the best candidate."""
    st.header(f"1단계: '{name}' 분포 적합")
    failures = data.to_numpy(dtype=float, copy=False)
    stats = compute_sample_stats(data)
    st.caption(
        f"{name} 관측수 {stats['count']}건 · 평균 {_fmt_float(stats['mean'], 1)} · 중앙값 {_fmt_float(stats['median'], 1)} · "
        f"표준편차 {_fmt_float(stats['std'], 1)} · 10/90분위 ({_fmt_float(stats['p10'], 1)}, {_fmt_float(stats['p90'], 1)})"
    )

    with st.spinner(f"'{name}' 데이터에 후보 분포를 적합 중입니다..."):
        fitters: Dict[str, Any] = {
            "Weibull_2P": rel.Fitters.Fit_Weibull_2P(failures=failures, print_results=False),
            "Lognormal_2P": rel.Fitters.Fit_Lognormal_2P(
                failures=failures, print_results=False
            ),
            "Normal_2P": rel.Fitters.Fit_Normal_2P(failures=failures, print_results=False),
            "Exponential_1P": rel.Fitters.Fit_Exponential_1P(
                failures=failures, print_results=False
            ),
        }

        summary = pd.DataFrame(
            {
                "Log-Likelihood": [fit.loglik for fit in fitters.values()],
                "AICc": [fit.AICc for fit in fitters.values()],
                "BIC": [fit.BIC for fit in fitters.values()],
                "AD": [fit.AD for fit in fitters.values()],
            },
            index=list(fitters.keys()),
        )

    best_name = summary["AICc"].idxmin()
    best_fitter = fitters[best_name]

    sorted_aicc = summary.sort_values("AICc")
    runner_up = sorted_aicc.iloc[1] if len(sorted_aicc) > 1 else None
    reason_parts = [
        f"AICc가 가장 낮은 {format_distribution_name(best_name)}(AICc={summary.loc[best_name, 'AICc']:.3f})를 선정했습니다.",
        f"BIC={summary.loc[best_name, 'BIC']:.2f}, AD={summary.loc[best_name, 'AD']:.2f}로 적합도가 양호합니다.",
    ]
    if runner_up is not None:
        diff = runner_up["AICc"] - summary.loc[best_name, "AICc"]
        reason_parts.append(
            f"다음 후보({format_distribution_name(runner_up.name)}) 대비 AICc 차이는 {diff:.2f}입니다."
        )
    selection_reason = " ".join(reason_parts)

    st.subheader(f"'{name}' 분포 비교")
    display_df = summary.copy().round(3)
    display_df.insert(0, "분포", [format_distribution_name(idx) for idx in display_df.index])
    st.dataframe(display_df)
    st.info(selection_reason)

    with st.expander(f"'{name}' 최적 분포 파라미터"):
        rows = []
        for attr in ["alpha", "beta", "mu", "sigma", "Lambda", "mean", "std"]:
            if hasattr(best_fitter, attr):
                rows.append({"파라미터": PARAM_LABELS.get(attr, attr), "추정값": getattr(best_fitter, attr)})
        if rows:
            st.table(pd.DataFrame(rows))
        else:
            st.info("표시할 파라미터가 없습니다.")

    fig = plot_distribution_fit(failures, best_fitter, name)
    st.pyplot(fig)
    plt.close(fig)

    return FitResults(
        name=name,
        best_distribution_name=best_name,
        best_distribution=best_fitter,
        results_table=summary,
        fitters=fitters,
        sample_stats=stats,
        selection_reason=selection_reason,
    )


def compare_distributions(test_results: FitResults, field_results: FitResults) -> bool:
    """Compare shape parameters when distributions match."""
    st.header("2단계: 형상 모수 비교")

    if test_results.best_distribution_name != field_results.best_distribution_name:
        logging.warning(
            "내구와 필드의 최적 분포가 서로 다릅니다. (내구=%s, 필드=%s)",
            test_results.best_distribution_name,
            field_results.best_distribution_name,
        )
        st.warning(
            "내구와 필드의 최적 분포가 다릅니다. "
            f"내구 -> {format_distribution_name(test_results.best_distribution_name)}, "
            f"필드 -> {format_distribution_name(field_results.best_distribution_name)}."
        )
        return False

    dist_name = test_results.best_distribution_name
    logging.info("두 데이터셋의 최적 분포(%s)가 동일합니다.", dist_name)

    shape_param, shape_label = SHAPE_PARAM_LABELS.get(dist_name, (None, None))
    if not shape_param:
        st.info("해당 분포는 형상 모수를 비교하지 않습니다.")
        return True

    test_shape = float(getattr(test_results.best_distribution, shape_param))
    field_shape = float(getattr(field_results.best_distribution, shape_param))

    test_ci = _extract_ci_bounds(test_results.best_distribution, shape_param)
    field_ci = _extract_ci_bounds(field_results.best_distribution, shape_param)

    st.subheader(f"형상 모수 비교 ({shape_label})")
    comparison_df = pd.DataFrame(
        {
            "데이터셋": ["내구 시험", "필드"],
            "추정값": [test_shape, field_shape],
            "하한": [test_ci[0] if test_ci else np.nan, field_ci[0] if field_ci else np.nan],
            "상한": [test_ci[1] if test_ci else np.nan, field_ci[1] if field_ci else np.nan],
        }
    )
    st.table(comparison_df)

    shape_overlap = False
    if test_ci and field_ci:
        shape_overlap = not (test_ci[1] < field_ci[0] or field_ci[1] < test_ci[0])
    else:
        st.info("신뢰구간 정보가 부족하여 겹침 여부를 평가하기 어렵습니다.")

    if shape_overlap:
        st.success("형상 모수 신뢰구간이 겹칩니다.")
    else:
        st.warning("형상 모수 신뢰구간이 겹치지 않습니다.")

    return shape_overlap


def calculate_acceleration_factor(
    test_results: FitResults,
    field_results: FitResults,
    shape_is_valid: bool,
) -> float | None:
    """Compute acceleration factor depending on the selected model."""
    st.header("3단계: 가속계수")

    if not shape_is_valid:
        st.warning("형상 모수가 충분히 일치하지 않아 가속계수 해석 시 주의가 필요합니다.")

    if test_results.best_distribution_name != field_results.best_distribution_name:
        logging.error(
            "가속계수를 계산할 수 없습니다. 분포가 서로 다릅니다. (내구=%s, 필드=%s)",
            test_results.best_distribution_name,
            field_results.best_distribution_name,
        )
        st.error(
            "선택된 분포가 서로 달라 가속계수를 계산할 수 없습니다. 동일한 최적 분포가 나올 때까지 데이터를 확인해 주세요."
        )
        return None

    dist_name = test_results.best_distribution_name
    scale_param = SCALE_PARAM_LOOKUP.get(dist_name)
    if not scale_param:
        logging.error("가속계수를 계산할 수 없는 분포입니다: %s", dist_name)
        st.error(f"{format_distribution_name(dist_name)} 분포에서는 가속계수를 계산할 수 없습니다.")
        return None

    test_attr = getattr(test_results.best_distribution, scale_param, None)
    field_attr = getattr(field_results.best_distribution, scale_param, None)
    if test_attr is None or field_attr is None:
        logging.error(
            "필요한 파라미터 '%s'를 찾지 못했습니다 (내구=%s, 필드=%s).",
            scale_param,
            test_results.best_distribution_name,
            field_results.best_distribution_name,
        )
        st.error("필요한 파라미터가 없어 가속계수를 계산할 수 없습니다.")
        return None

    test_scale = float(test_attr)
    field_scale = float(field_attr)

    if dist_name == "Lognormal_2P":
        af = float(np.exp(field_scale - test_scale))
        formula_text = "AF = exp(μ_field - μ_test)"
    elif dist_name == "Exponential_1P":
        af = float(test_scale / field_scale)
        formula_text = "AF = λ_test / λ_field"
    else:
        af = float(field_scale / test_scale)
        formula_text = "AF = α_field / α_test"

    st.metric("가속계수 (AF)", f"{af:.3f}")
    st.caption(f"계산식: {formula_text}")
    logging.info("가속계를 계산했습니다: %.3f", af)
    return af


def build_markdown_report(
    test_results: FitResults,
    field_results: FitResults,
    af: float,
    shape_is_valid: bool,
    test_data: pd.Series,
    field_data: pd.Series,
) -> str:
    shape_summary = (
        "형상 모수 신뢰구간이 겹칩니다."
        if shape_is_valid
        else "형상 모수 신뢰구간이 충분히 겹치지 않습니다."
    )

    intro = textwrap.dedent(
        f"""
        # 내구-필드 수명 분석 종합 보고서

        ## 1. 분석 개요
        본 보고서는 내구 시험 데이터 {len(test_data):,}건과 필드 데이터 {len(field_data):,}건을 기반으로, 가속 시험이 실제 필드 수명과 얼마나 일치하는지 정량적으로 검증하기 위해 작성되었습니다. 두 데이터셋의 통계적 특성을 비교하고 최적 분포를 선정한 뒤, 형상 모수 일치 여부와 가속계수(Acceleration Factor, AF)를 도출하여 가속 시험의 유효성을 평가합니다.
        """
    ).strip()

    stats_table = pd.DataFrame(
        {
            "건수": [test_results.sample_stats["count"], field_results.sample_stats["count"]],
            "평균": [test_results.sample_stats["mean"], field_results.sample_stats["mean"]],
            "중앙값": [test_results.sample_stats["median"], field_results.sample_stats["median"]],
            "표준편차": [test_results.sample_stats["std"], field_results.sample_stats["std"]],
            "10/90 분위": [
                f"{test_results.sample_stats['p10']:.1f} / {test_results.sample_stats['p90']:.1f}",
                f"{field_results.sample_stats['p10']:.1f} / {field_results.sample_stats['p90']:.1f}",
            ],
        },
        index=["내구 시험", "필드"],
    )

    report = textwrap.dedent(
        f"""
        {intro}

        ## 2. 데이터 요약 및 기술 통계
        {_df_to_markdown(stats_table, index_name="구분")}

        ## 3. 수명 분포 분석
        ### 3.1 내구 시험 데이터 분포 적합 결과
        {_df_to_markdown(test_results.results_table, index_name="분포 모델")}

        최적 분포: {format_distribution_name(test_results.best_distribution_name)}
        선정 근거: {test_results.selection_reason}

        ### 3.2 필드 데이터 분포 적합 결과
        {_df_to_markdown(field_results.results_table, index_name="분포 모델")}

        최적 분포: {format_distribution_name(field_results.best_distribution_name)}
        선정 근거: {field_results.selection_reason}

        두 데이터셋이 동일한 분포를 따르는지는 형상 모수 비교를 통해 검증합니다.

        ## 4. 내구-필드 상관 분석
        - 형상 모수 비교 결과: {shape_summary}
        - 가속계수(AF): {af:.3f}
        - 해석: 내구 시험 1단위는 필드 사용 약 {af:.2f}단위와 동일한 손상을 의미합니다.

        ## 5. 결론
        - 모델 적합성: 내구와 필드 데이터 모두 {format_distribution_name(test_results.best_distribution_name)} 분포를 따릅니다.
        - 고장 메커니즘 동일성: 형상 모수 신뢰구간이 {'겹칩니다' if shape_is_valid else '충분히 겹치지 않습니다'}.
        - 정량적 상관관계: 가속계수(AF) = {af:.3f} (필드 대비 가속 배율).
        """
    ).strip()

    return report


def _table_figure(title: str, df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8.27, 5.0))
    ax.set_axis_off()
    ax.set_title(title, fontsize=14, weight="bold", pad=10)
    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        rowLabels=df.index,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.3)
    fig.tight_layout()
    return fig


def build_pdf_report(
    test_results: FitResults,
    field_results: FitResults,
    af: float,
    shape_is_valid: bool,
    test_data: pd.Series,
    field_data: pd.Series,
) -> bytes:
    test_failures = test_data.to_numpy(dtype=float, copy=False)
    field_failures = field_data.to_numpy(dtype=float, copy=False)

    buffer = io.BytesIO()
    with PdfPages(buffer) as pdf:
        # Page 1: Narrative
        fig, ax = plt.subplots(figsize=(8.27, 11.69))  # A4 portrait
        ax.axis("off")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.text(
            0.5,
            0.95,
            "내구-필드 수명 분석 종합 보고서",
            fontsize=18,
            weight="bold",
            ha="center",
            va="top",
        )

        y = 0.89
        overview = (
            f"본 보고서는 내구 시험 데이터 {len(test_failures):,}건과 필드 데이터 {len(field_failures):,}건을 기반으로 "
            "가속 시험 결과가 실제 필드 고장 패턴과 얼마나 일치하는지 정량적으로 검증합니다. "
            "데이터 요약 → 분포 적합 → 형상 모수 비교 → 가속계수 해석 순으로 구성됩니다."
        )
        y = _add_section(ax, "1. 분석 개요", [overview], start_y=y, width=90)

        stats_lines = [
            f"내구 시험 데이터: {len(test_failures):,}건",
            f"필드 데이터: {len(field_failures):,}건",
            f"최적 분포 (내구): {format_distribution_name(test_results.best_distribution_name)}",
            f"최적 분포 (필드): {format_distribution_name(field_results.best_distribution_name)}",
            f"형상 모수 신뢰구간: {'겹침' if shape_is_valid else '겹치지 않음/불충분'}",
            f"가속계수 (AF): {af:.3f}",
        ]
        y = _add_section(
            ax,
            "2. 핵심 요약",
            ["아래 항목은 분포 적합 및 가속계수 계산 결과를 간결하게 정리한 것입니다."],
            start_y=y,
            width=90,
        )
        y = _add_bullet_lines(ax, stats_lines, start_y=y + 0.01, width=90)

        y = _add_section(
            ax,
            "3. 가속계수 해석",
            [
                f"AF={af:.3f} → 내구 1회는 필드 약 {af:.2f}회와 동일한 응력/손상을 의미합니다.",
                "가속 시험의 결과를 실제 사용 수명으로 환산하는 데 활용할 수 있습니다.",
            ],
            start_y=y + 0.02,
            width=90,
        )

        y = _add_section(
            ax,
            "4. 형상 모수 해석",
            [
                "형상 모수는 시간 경과에 따른 고장률 변화를 설명합니다.",
                f"두 데이터셋의 형상 모수 신뢰구간은 {'충분히 겹쳐 고장 메커니즘이 동일함을 시사합니다.' if shape_is_valid else '충분히 겹치지 않아 추가 검토가 필요합니다.'}",
            ],
            start_y=y,
            width=90,
        )

        fig.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Page 2: Stats table
        stats_df = pd.DataFrame(
            {
                "건수": [test_results.sample_stats["count"], field_results.sample_stats["count"]],
                "평균": [test_results.sample_stats["mean"], field_results.sample_stats["mean"]],
                "중앙값": [test_results.sample_stats["median"], field_results.sample_stats["median"]],
                "표준편차": [test_results.sample_stats["std"], field_results.sample_stats["std"]],
                "10/90분위": [
                    f"{test_results.sample_stats['p10']:.1f} / {test_results.sample_stats['p90']:.1f}",
                    f"{field_results.sample_stats['p10']:.1f} / {field_results.sample_stats['p90']:.1f}",
                ],
            },
            index=["내구 시험", "필드"],
        ).round(2)
        stats_fig = _table_figure("데이터 요약 통계", stats_df)
        pdf.savefig(stats_fig, bbox_inches="tight")
        plt.close(stats_fig)

        # Page 3: Fit table (내구)
        test_table = test_results.results_table.copy().round(3)
        test_table.index = [format_distribution_name(idx) for idx in test_table.index]
        test_fig = _table_figure("내구 시험 분포 적합 결과", test_table)
        pdf.savefig(test_fig, bbox_inches="tight")
        plt.close(test_fig)

        # Page 4: Fit table (필드)
        field_table = field_results.results_table.copy().round(3)
        field_table.index = [format_distribution_name(idx) for idx in field_table.index]
        field_fig = _table_figure("필드 분포 적합 결과", field_table)
        pdf.savefig(field_fig, bbox_inches="tight")
        plt.close(field_fig)

        # Page 5: Distribution plots
        pdf.savefig(plot_distribution_fit(test_failures, test_results.best_distribution, "내구 시험"))
        pdf.savefig(plot_distribution_fit(field_failures, field_results.best_distribution, "필드"))
        plt.close("all")

    buffer.seek(0)
    return buffer.getvalue()


def generate_final_report(
    test_results: FitResults,
    field_results: FitResults,
    af: float,
    shape_is_valid: bool,
    test_data: pd.Series,
    field_data: pd.Series,
) -> None:
    """Compile markdown/PDF report and expose it for download."""
    st.header("4단계: 최종 보고서")

    report = build_markdown_report(
        test_results=test_results,
        field_results=field_results,
        af=af,
        shape_is_valid=shape_is_valid,
        test_data=test_data,
        field_data=field_data,
    )
    pdf_bytes = build_pdf_report(
        test_results=test_results,
        field_results=field_results,
        af=af,
        shape_is_valid=shape_is_valid,
        test_data=test_data,
        field_data=field_data,
    )

    try:
        with open(os.path.join("results", "final_report.pdf"), "wb") as f:
            f.write(pdf_bytes)
        logging.info("PDF 보고서를 results/final_report.pdf 로 저장했습니다.")
    except Exception as exc:  # pragma: no cover - file system best-effort
        logging.warning("PDF 저장 중 문제가 발생했습니다: %s", exc)

    st.markdown(report)
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            label="보고서 다운로드 (.md)",
            data=report,
            file_name="final_report.md",
            mime="text/markdown",
        )
    with col2:
        st.download_button(
            label="보고서 다운로드 (.pdf)",
            data=pdf_bytes,
            file_name="final_report.pdf",
            mime="application/pdf",
            key="pdf_download",
        )


@dataclass(frozen=True)
class Stage:
    name: str
    duration: int
    tests: tuple[str, ...]


@dataclass(frozen=True)
class ScheduledOccurrence:
    test_name: str
    start: date
    end: date
    group: str
    stage_name: str


OPERATING_TESTS = ("UNIT 고유저항 평가", "특성 평가")
STIFFNESS_TESTS = (
    "글라스 틸팅 강성 평가",
    "상하 유격 강성 평가",
    "모터 역전 확인 평가",
    "역전 방지부 하방향 강성 평가",
    "역전 방지부 상방향 강성 평가",
    "상승 구속 강성 평가",
)

INDIVIDUAL_TESTS = {
    "작동 이음 평가": 1,
    "케이블 앤드부 강성 평가": 1,
    "내식성 평가": 21,
    "크리프 벤치 평가": 20,
}

UNSCHEDULED_TESTS = {"작동력 평가"}

TEST_NUMBER = {
    "UNIT 고유저항 평가": "6.2.2.1",
    "작동 이음 평가": "6.2.2.2",
    "작동력 평가": "6.2.2.3",
    "특성 평가": "6.2.2.4",
    "세이프티 평가 (법규)": "6.2.2.5",
    "모터 소손 평가": "6.2.2.6",
    "글라스 틸팅 강성 평가": "6.2.3.1",
    "상하 유격 강성 평가": "6.2.3.2",
    "모터 역전 확인 평가": "6.2.3.4",
    "역전 방지부 하방향 강성 평가": "6.2.3.5",
    "역전 방지부 상방향 강성 평가": "6.2.3.6",
    "상승 구속 강성 평가": "6.2.3.7",
    "케이블 앤드부 강성 평가": "6.2.3.9",
    "내구력 평가": "6.2.4.1",
    "내진동 평가": "6.2.4.2",
    "내충격 평가": "6.2.4.3",
    "내열 평가": "6.2.4.4",
    "내한 평가": "6.2.4.5",
    "부가 내열 / 내한 평가": "6.2.4.6",
    "내식성 평가": "6.2.4.7",
    "내수성 평가": "6.2.4.8",
    "크리프 벤치 평가": "6.2.4.9",
    "모터 내한성 이음 평가": "6.2.5.1",
    "케이블 내한성 이탈 평가": "6.2.5.2",
}

OUTPUT_SECTIONS = [
    (
        "작동특성",
        [
            "UNIT 고유저항 평가",
            "작동 이음 평가",
            "작동력 평가",
            "특성 평가",
            "세이프티 평가 (법규)",
            "모터 소손 평가",
        ],
    ),
    ("강성시험", [*STIFFNESS_TESTS, "케이블 앤드부 강성 평가"]),
    (
        "내구/환경 평가",
        [
            "내구력 평가",
            "내진동 평가",
            "내충격 평가",
            "내열 평가",
            "내한 평가",
            "부가 내열 / 내한 평가",
            "내식성 평가",
            "내수성 평가",
            "크리프 벤치 평가",
        ],
    ),
    ("추가 평가", ["모터 내한성 이음 평가", "케이블 내한성 이탈 평가"]),
]


def _parse_holidays(raw_text: str) -> tuple[set[date], list[str]]:
    holidays: set[date] = set()
    invalid: list[str] = []
    tokens = [token.strip() for token in raw_text.replace(",", "\n").splitlines()]
    for token in tokens:
        if not token:
            continue
        try:
            parsed = datetime.strptime(token, "%Y-%m-%d").date()
        except ValueError:
            invalid.append(token)
            continue
        holidays.add(parsed)
    return holidays, invalid


def _is_non_working(day: date, holidays: set[date]) -> bool:
    return day.weekday() >= 5 or day in holidays


def _next_monday(day: date) -> date:
    days_ahead = (7 - day.weekday()) % 7
    if days_ahead == 0:
        days_ahead = 7
    return day + timedelta(days=days_ahead)


def _adjust_start(day: date, holidays: set[date], correction_mode: str) -> date:
    if correction_mode == "weekday":
        while _is_non_working(day, holidays):
            day += timedelta(days=1)
        return day

    while _is_non_working(day, holidays):
        day = _next_monday(day)
    return day


def _format_date(day: date | None, date_format: str) -> str:
    if not day:
        return "-"
    if date_format == "YY.MM.DD.":
        return day.strftime("%y.%m.%d.")
    return day.strftime("%Y-%m-%d")


def _format_duration(days: int | None, duration_format: str) -> str:
    if days is None:
        return "-"
    if duration_format == "N일":
        return f"{days}일"
    return f"D+{days}"


def _schedule_sequence(
    start_day: date,
    stages: list[Stage],
    group: str,
    holidays: set[date],
    correction_mode: str,
) -> tuple[list[ScheduledOccurrence], date]:
    occurrences: list[ScheduledOccurrence] = []
    current = start_day
    for stage in stages:
        stage_start = _adjust_start(current, holidays, correction_mode)
        stage_end = stage_start + timedelta(days=stage.duration)
        for test_name in stage.tests:
            occurrences.append(
                ScheduledOccurrence(
                    test_name=test_name,
                    start=stage_start,
                    end=stage_end,
                    group=group,
                    stage_name=stage.name,
                )
            )
        current = stage_end
    return occurrences, current


def _schedule_core(
    base_start: date,
    holidays: set[date],
    correction_mode: str,
) -> tuple[list[ScheduledOccurrence], date]:
    adjusted_start = _adjust_start(base_start, holidays, correction_mode)

    cross1 = [
        Stage("작동특성", 1, OPERATING_TESTS),
        Stage("세이프티 평가 (법규)", 1, ("세이프티 평가 (법규)",)),
        Stage("내구력 평가", 26, ("내구력 평가",)),
        Stage("세이프티 평가 (법규)", 1, ("세이프티 평가 (법규)",)),
        Stage("작동특성", 1, OPERATING_TESTS),
    ]
    cross2 = [
        Stage("작동특성", 1, OPERATING_TESTS),
        Stage("강성 평가(내구전)", 1, STIFFNESS_TESTS),
        Stage("내진동 평가", 14, ("내진동 평가",)),
        Stage("내구력 평가", 26, ("내구력 평가",)),
        Stage("강성 평가(내구후)", 1, STIFFNESS_TESTS),
        Stage("작동특성", 1, OPERATING_TESTS),
    ]
    cross3 = [
        Stage("작동특성", 1, OPERATING_TESTS),
        Stage("내열 평가", 2, ("내열 평가",)),
        Stage("내한 평가", 2, ("내한 평가",)),
        Stage("내충격 평가", 10, ("내충격 평가",)),
        Stage("내수성 평가", 1, ("내수성 평가",)),
    ]
    cross4 = [
        Stage("모터 내한성 이음 평가", 4, ("모터 내한성 이음 평가",)),
        Stage("부가 내열 / 내한 평가", 3, ("부가 내열 / 내한 평가",)),
        Stage("케이블 내한성 이탈 평가", 1, ("케이블 내한성 이탈 평가",)),
        Stage("모터 소손 평가", 1, ("모터 소손 평가",)),
    ]

    occurrences: list[ScheduledOccurrence] = []
    occ1, end1 = _schedule_sequence(adjusted_start, cross1, "교차#1", holidays, correction_mode)
    occ2, end2 = _schedule_sequence(adjusted_start, cross2, "교차#2", holidays, correction_mode)
    occ3, end3 = _schedule_sequence(adjusted_start, cross3, "교차#3", holidays, correction_mode)
    cross4_start = _adjust_start(end3, holidays, correction_mode)
    occ4, end4 = _schedule_sequence(cross4_start, cross4, "교차#4", holidays, correction_mode)

    occurrences.extend(occ1)
    occurrences.extend(occ2)
    occurrences.extend(occ3)
    occurrences.extend(occ4)

    for test_name, duration in INDIVIDUAL_TESTS.items():
        if test_name == "작동 이음 평가":
            continue
        start = adjusted_start
        end = start + timedelta(days=duration)
        occurrences.append(
            ScheduledOccurrence(
                test_name=test_name,
                start=start,
                end=end,
                group="개별",
                stage_name=test_name,
            )
        )

    max_end = max([end1, end2, end3, end4] + [occ.end for occ in occurrences])
    return occurrences, max_end


def _schedule_with_standalone(
    start_day: date,
    holidays: set[date],
    correction_mode: str,
    position: str,
) -> tuple[list[ScheduledOccurrence], date, date, date | None, date | None]:
    base_start = _adjust_start(start_day, holidays, correction_mode)

    def schedule_start() -> tuple[list[ScheduledOccurrence], date, date, date | None, date | None]:
        occurrences: list[ScheduledOccurrence] = []
        standalone_start = base_start
        standalone_end = standalone_start + timedelta(days=INDIVIDUAL_TESTS["작동 이음 평가"])
        occurrences.append(
            ScheduledOccurrence(
                test_name="작동 이음 평가",
                start=standalone_start,
                end=standalone_end,
                group="개별",
                stage_name="작동 이음 평가",
            )
        )
        other_start = _adjust_start(standalone_end, holidays, correction_mode)
        core_occ, core_end = _schedule_core(other_start, holidays, correction_mode)
        occurrences.extend(core_occ)
        final_end = max(core_end, standalone_end)
        return occurrences, base_start, final_end, standalone_start, standalone_end

    def schedule_end() -> tuple[list[ScheduledOccurrence], date, date, date | None, date | None]:
        core_occ, core_end = _schedule_core(base_start, holidays, correction_mode)
        standalone_start = _adjust_start(core_end, holidays, correction_mode)
        standalone_end = standalone_start + timedelta(days=INDIVIDUAL_TESTS["작동 이음 평가"])
        core_occ.append(
            ScheduledOccurrence(
                test_name="작동 이음 평가",
                start=standalone_start,
                end=standalone_end,
                group="개별",
                stage_name="작동 이음 평가",
            )
        )
        final_end = max(core_end, standalone_end)
        return core_occ, base_start, final_end, standalone_start, standalone_end

    if position == "start":
        return schedule_start()
    if position == "end":
        return schedule_end()

    start_schedule = schedule_start()
    end_schedule = schedule_end()
    if start_schedule[2] <= end_schedule[2]:
        return start_schedule
    return end_schedule


def _aggregate_occurrences(occurrences: list[ScheduledOccurrence]) -> dict[str, dict[str, date]]:
    aggregated: dict[str, dict[str, date]] = {}
    for occ in occurrences:
        if occ.test_name not in aggregated:
            aggregated[occ.test_name] = {"start": occ.start, "end": occ.end}
            continue
        aggregated[occ.test_name]["start"] = min(aggregated[occ.test_name]["start"], occ.start)
        aggregated[occ.test_name]["end"] = max(aggregated[occ.test_name]["end"], occ.end)
    return aggregated


def _build_section_tables(
    aggregated: dict[str, dict[str, date]],
    date_format: str,
    duration_format: str,
    include_no: bool,
    include_notes: bool,
) -> tuple[list[tuple[str, pd.DataFrame]], pd.DataFrame]:
    section_tables: list[tuple[str, pd.DataFrame]] = []
    combined_rows: list[dict[str, str]] = []

    for section_name, tests in OUTPUT_SECTIONS:
        rows: list[dict[str, str]] = []
        for test_name in tests:
            schedule = aggregated.get(test_name)
            start = schedule["start"] if schedule else None
            end = schedule["end"] if schedule else None
            duration_days = (end - start).days if start and end else None

            row: dict[str, str] = {
                "시험명": test_name,
                "소요일": _format_duration(duration_days, duration_format),
                "시작일": _format_date(start, date_format),
                "종료일": _format_date(end, date_format),
            }
            if include_no:
                row["No."] = TEST_NUMBER.get(test_name, "")
            if include_notes:
                row["비고"] = ""

            rows.append(row)
            combined_rows.append({"구분": section_name, **row})

        columns = ["시험명", "소요일", "시작일", "종료일"]
        if include_no:
            columns = ["No."] + columns
        if include_notes:
            columns = columns + ["비고"]
        df = pd.DataFrame(rows, columns=columns)
        section_tables.append((section_name, df))

    combined_columns = ["구분", "시험명", "소요일", "시작일", "종료일"]
    if include_no:
        combined_columns = ["구분", "No."] + combined_columns[1:]
    if include_notes:
        combined_columns = combined_columns + ["비고"]
    combined_df = pd.DataFrame(combined_rows, columns=combined_columns)
    return section_tables, combined_df


def render_schedule_planner() -> None:
    st.header("시험 일정 생성")
    st.write(
        "교차 시험 순서, 주말/공휴일 보정, 병행 규칙을 반영해 시험 일정을 자동으로 계산합니다."
    )

    with st.form("schedule_form"):
        start_date = st.date_input("시작일", value=date(2025, 12, 15))
        correction_label = st.selectbox(
            "주말/공휴일 보정 방식",
            ["다음 월요일(A 방식)", "다음 평일"],
            index=0,
        )
        holiday_text = st.text_area(
            "공휴일 입력 (YYYY-MM-DD, 줄/쉼표 구분)",
            placeholder="예) 2025-12-25, 2026-01-01",
        )
        standalone_position = st.selectbox(
            "작동 이음 평가 배치",
            ["자동(최소 종료일)", "시작(다른 시험 전)", "종료(다른 시험 후)"],
            index=0,
        )
        date_format = st.selectbox("날짜 표기", ["YY.MM.DD.", "YYYY-MM-DD"], index=0)
        duration_format = st.selectbox("소요일 표기", ["D+N", "N일"], index=0)
        include_no = st.checkbox("No. 열 포함", value=False)
        include_notes = st.checkbox("비고 열 포함", value=False)
        submitted = st.form_submit_button("일정 계산")

    if not submitted:
        return

    holidays, invalid = _parse_holidays(holiday_text)
    if invalid:
        st.warning("형식이 잘못된 공휴일 입력: " + ", ".join(invalid))

    correction_mode = "monday" if correction_label.startswith("다음 월요일") else "weekday"
    standalone_key = {
        "자동(최소 종료일)": "auto",
        "시작(다른 시험 전)": "start",
        "종료(다른 시험 후)": "end",
    }[standalone_position]

    occurrences, adjusted_start, final_end, standalone_start, standalone_end = _schedule_with_standalone(
        start_date, holidays, correction_mode, standalone_key
    )
    aggregated = _aggregate_occurrences(occurrences)
    for test_name in UNSCHEDULED_TESTS:
        aggregated.setdefault(test_name, {})

    section_tables, combined_df = _build_section_tables(
        aggregated=aggregated,
        date_format=date_format,
        duration_format=duration_format,
        include_no=include_no,
        include_notes=include_notes,
    )

    st.info(
        f"보정된 시작일: {_format_date(adjusted_start, date_format)} | "
        f"최종 종료일: {_format_date(final_end, date_format)}"
    )
    if standalone_start and standalone_end:
        st.caption(
            f"작동 이음 평가 단독 구간: {_format_date(standalone_start, date_format)} ~ "
            f"{_format_date(standalone_end, date_format)}"
        )

    for section_name, df in section_tables:
        st.subheader(section_name)
        st.dataframe(df.reset_index(drop=True), use_container_width=True)

    csv_data = combined_df.to_csv(index=False, encoding="utf-8-sig")
    st.download_button(
        label="일정표 다운로드 (CSV)",
        data=csv_data,
        file_name="test_schedule.csv",
        mime="text/csv",
    )


def render_capa_builder() -> None:
    st.header("CAPA 자동화")
    st.write("매트릭스를 롱 포맷으로 변환하고 표준시간/설비조건을 붙여 CAPA를 계산합니다.")

    plan_file = st.file_uploader(
        "PLAN 매트릭스 업로드 (xlsx)",
        type=["xlsx", "xls"],
        key="capa_plan_file",
    )
    plan_sheet_names = _get_excel_sheet_names(plan_file)
    if plan_sheet_names:
        plan_sheet = st.selectbox("PLAN 시트 선택", plan_sheet_names, key="capa_plan_sheet")
    else:
        plan_sheet = st.text_input("PLAN 시트명", value="", key="capa_plan_sheet_name")

    plan_header_rows = st.number_input(
        "PLAN 헤더 행 수",
        min_value=1,
        max_value=5,
        value=2,
        step=1,
        key="capa_plan_header_rows",
    )
    plan_id_column = st.text_input(
        "시험항목 컬럼명 (빈칸이면 첫 번째 컬럼 사용)",
        value="",
        key="capa_plan_id_column",
    )
    plan_source_label = st.text_input(
        "SourceSheet 라벨",
        value=plan_sheet or "PLAN",
        key="capa_plan_source_label",
    )
    keep_zero = st.checkbox("샘플 수 0 포함", value=False, key="capa_plan_keep_zero")

    plan_columns = _preview_excel_columns(plan_file, plan_sheet, int(plan_header_rows))
    if plan_columns:
        st.caption("PLAN 컬럼 미리보기: " + ", ".join(plan_columns[:12]))

    st.subheader("시험명 매칭키 (옵션)")
    mapping_file = st.file_uploader(
        "매칭키 파일 업로드 (xlsx)",
        type=["xlsx", "xls"],
        key="capa_mapping_file",
    )
    mapping_sheet_names = _get_excel_sheet_names(mapping_file)
    if mapping_sheet_names:
        mapping_sheet = st.selectbox(
            "매칭키 시트 선택",
            mapping_sheet_names,
            key="capa_mapping_sheet",
        )
    else:
        mapping_sheet = st.text_input("매칭키 시트명", value="", key="capa_mapping_sheet_name")
    mapping_raw_col = st.text_input(
        "원문 시험명 컬럼",
        value="",
        key="capa_mapping_raw_col",
    )
    mapping_key_col = st.text_input(
        "매칭키 컬럼",
        value="",
        key="capa_mapping_key_col",
    )
    mapping_columns = _preview_excel_columns(mapping_file, mapping_sheet, 1)
    if mapping_columns:
        st.caption("매칭키 컬럼 미리보기: " + ", ".join(mapping_columns[:12]))

    st.subheader("표준 공수 (옵션)")
    labor_file = st.file_uploader(
        "공수 표준 파일 업로드 (xlsx)",
        type=["xlsx", "xls"],
        key="capa_labor_file",
    )
    labor_sheet_names = _get_excel_sheet_names(labor_file)
    if labor_sheet_names:
        labor_sheet = st.selectbox("공수 표준 시트 선택", labor_sheet_names, key="capa_labor_sheet")
    else:
        labor_sheet = st.text_input("공수 표준 시트명", value="", key="capa_labor_sheet_name")
    labor_test_col = st.text_input(
        "시험항목/키 컬럼",
        value="",
        key="capa_labor_test_col",
    )
    labor_hr_col = st.text_input(
        "공수(hr/sample) 컬럼",
        value="",
        key="capa_labor_hr_col",
    )
    labor_columns = _preview_excel_columns(labor_file, labor_sheet, 1)
    if labor_columns:
        st.caption("공수 표준 컬럼 미리보기: " + ", ".join(labor_columns[:12]))

    st.subheader("설비 조건 (옵션)")
    machine_file = st.file_uploader(
        "설비 표준 파일 업로드 (xlsx)",
        type=["xlsx", "xls"],
        key="capa_machine_file",
    )
    machine_sheet_names = _get_excel_sheet_names(machine_file)
    if machine_sheet_names:
        machine_sheet = st.selectbox(
            "설비 표준 시트 선택",
            machine_sheet_names,
            key="capa_machine_sheet",
        )
    else:
        machine_sheet = st.text_input("설비 표준 시트명", value="", key="capa_machine_sheet_name")
    machine_test_col = st.text_input(
        "시험항목/키 컬럼",
        value="",
        key="capa_machine_test_col",
    )
    machine_time_col = st.text_input(
        "시험시간(hr/run) 컬럼",
        value="",
        key="capa_machine_time_col",
    )
    machine_parallel_col = st.text_input(
        "동시 시험 가능 회수 컬럼",
        value="",
        key="capa_machine_parallel_col",
    )
    machine_equip_col = st.text_input(
        "설비그룹 컬럼",
        value="",
        key="capa_machine_equip_col",
    )
    machine_columns = _preview_excel_columns(machine_file, machine_sheet, 1)
    if machine_columns:
        st.caption("설비 표준 컬럼 미리보기: " + ", ".join(machine_columns[:12]))

    st.subheader("설비 마스터 (옵션)")
    equip_file = st.file_uploader(
        "설비 마스터 파일 업로드 (xlsx)",
        type=["xlsx", "xls"],
        key="capa_equip_file",
    )
    equip_sheet_names = _get_excel_sheet_names(equip_file)
    if equip_sheet_names:
        equip_sheet = st.selectbox("설비 마스터 시트 선택", equip_sheet_names, key="capa_equip_sheet")
    else:
        equip_sheet = st.text_input("설비 마스터 시트명", value="", key="capa_equip_sheet_name")
    equip_group_col = st.text_input(
        "설비그룹 컬럼",
        value="",
        key="capa_equip_group_col",
    )
    equip_count_col = st.text_input(
        "보유대수 컬럼",
        value="",
        key="capa_equip_count_col",
    )
    equip_available_col = st.text_input(
        "가동가능시간 컬럼",
        value="",
        key="capa_equip_available_col",
    )
    equip_columns = _preview_excel_columns(equip_file, equip_sheet, 1)
    if equip_columns:
        st.caption("설비 마스터 컬럼 미리보기: " + ", ".join(equip_columns[:12]))

    output_name = st.text_input(
        "출력 파일명",
        value="capa_output.xlsx",
        key="capa_output_name",
    )

    if not st.button("CAPA 계산", key="capa_run"):
        return

    if not plan_file:
        st.error("PLAN 매트릭스 파일을 먼저 업로드해 주세요.")
        return

    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    if root_dir not in sys.path:
        sys.path.append(root_dir)

    try:
        from capa_auto_build import (
            apply_test_mapping,
            build_summaries,
            compute_metrics,
            read_equipment_master,
            read_labor_standard,
            read_machine_standard,
            read_mapping_table,
            read_plan_matrix,
        )
    except Exception as exc:
        logging.exception("CAPA 모듈 로딩 실패: %s", exc)
        st.error("CAPA 모듈을 불러오지 못했습니다. 파일 위치를 확인해 주세요.")
        return

    plan_bytes = plan_file.getvalue()
    plan_sheet = plan_sheet or None
    plan_source = plan_source_label.strip() or plan_sheet or "PLAN"
    plan_id_column = plan_id_column.strip() or None
    try:
        plan_long = read_plan_matrix(
            path=io.BytesIO(plan_bytes),
            sheet_name=plan_sheet,
            header_rows=int(plan_header_rows),
            id_column=plan_id_column,
            source_label=plan_source,
            keep_zero=keep_zero,
        )
    except Exception as exc:
        st.error(f"PLAN 매트릭스 처리 실패: {exc}")
        return

    mapping = None
    if mapping_file:
        if not mapping_raw_col.strip() or not mapping_key_col.strip():
            st.error("매칭키 파일의 컬럼명을 입력해 주세요.")
            return
        try:
            mapping = read_mapping_table(
                path=io.BytesIO(mapping_file.getvalue()),
                sheet_name=mapping_sheet or None,
                raw_col=mapping_raw_col.strip(),
                key_col=mapping_key_col.strip(),
            )
        except Exception as exc:
            st.error(f"매칭키 파일 처리 실패: {exc}")
            return

    plan_long = apply_test_mapping(plan_long, mapping)

    labor_std = None
    if labor_file:
        if not labor_test_col.strip() or not labor_hr_col.strip():
            st.error("공수 표준 파일의 컬럼명을 입력해 주세요.")
            return
        try:
            labor_std = read_labor_standard(
                path=io.BytesIO(labor_file.getvalue()),
                sheet_name=labor_sheet or None,
                test_col=labor_test_col.strip(),
                hr_col=labor_hr_col.strip(),
            )
        except Exception as exc:
            st.error(f"공수 표준 처리 실패: {exc}")
            return

    machine_std = None
    if machine_file:
        required_cols = [
            machine_test_col.strip(),
            machine_time_col.strip(),
            machine_parallel_col.strip(),
            machine_equip_col.strip(),
        ]
        if not all(required_cols):
            st.error("설비 표준 파일의 컬럼명을 입력해 주세요.")
            return
        try:
            machine_std = read_machine_standard(
                path=io.BytesIO(machine_file.getvalue()),
                sheet_name=machine_sheet or None,
                test_col=machine_test_col.strip(),
                time_col=machine_time_col.strip(),
                parallel_col=machine_parallel_col.strip(),
                equip_col=machine_equip_col.strip(),
            )
        except Exception as exc:
            st.error(f"설비 표준 처리 실패: {exc}")
            return

    equip_master = None
    if equip_file:
        required_cols = [
            equip_group_col.strip(),
            equip_count_col.strip(),
            equip_available_col.strip(),
        ]
        if not all(required_cols):
            st.error("설비 마스터 파일의 컬럼명을 입력해 주세요.")
            return
        try:
            equip_master = read_equipment_master(
                path=io.BytesIO(equip_file.getvalue()),
                sheet_name=equip_sheet or None,
                group_col=equip_group_col.strip(),
                count_col=equip_count_col.strip(),
                available_col=equip_available_col.strip(),
            )
        except Exception as exc:
            st.error(f"설비 마스터 처리 실패: {exc}")
            return

    plan_with_std = plan_long.copy()
    if labor_std is not None:
        plan_with_std = plan_with_std.merge(labor_std, on="TestKey", how="left")
    if machine_std is not None:
        plan_with_std = plan_with_std.merge(machine_std, on="TestKey", how="left")

    labor_summary = pd.DataFrame()
    capa_summary = pd.DataFrame()
    if labor_std is not None or machine_std is not None:
        plan_with_std = compute_metrics(plan_with_std)
        labor_summary, capa_summary = build_summaries(plan_with_std, equip_master)

    file_name = output_name.strip() or "capa_output.xlsx"
    if not file_name.lower().endswith(".xlsx"):
        file_name = f"{file_name}.xlsx"

    output_buffer = io.BytesIO()
    with pd.ExcelWriter(output_buffer, engine="openpyxl") as writer:
        plan_long.to_excel(writer, sheet_name="PLAN_Long", index=False)
        plan_with_std.to_excel(writer, sheet_name="PLAN_With_Std", index=False)
        if not labor_summary.empty:
            labor_summary.to_excel(writer, sheet_name="Labor_Summary", index=False)
        if not capa_summary.empty:
            capa_summary.to_excel(writer, sheet_name="CAPA_Summary", index=False)
    output_buffer.seek(0)

    st.success(f"완료: PLAN_Long {len(plan_long):,}행 생성")
    st.download_button(
        label="CAPA 결과 다운로드 (xlsx)",
        data=output_buffer.getvalue(),
        file_name=file_name,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


def main() -> None:
    setup_streamlit()

    mode = st.sidebar.radio(
        "기능 선택",
        ["내구-필드 수명 분석", "시험 일정 생성", "CAPA 자동화"],
        index=0,
    )
    if mode == "시험 일정 생성":
        render_schedule_planner()
        return
    if mode == "CAPA 자동화":
        render_capa_builder()
        return

    st.sidebar.header("데이터 입력")
    test_file = st.sidebar.file_uploader("내구 시험 CSV 업로드", type="csv")
    field_file = st.sidebar.file_uploader("필드 CSV 업로드", type="csv")

    if st.sidebar.button("데이터 불러오기 및 분석"):
        if load_and_prepare_data(test_file, field_file):
            test_results = analyse_single_dataset(st.session_state.durability_failures, "내구 시험")
            field_results = analyse_single_dataset(st.session_state.field_failures, "필드")

            shape_is_valid = compare_distributions(test_results, field_results)
            af = calculate_acceleration_factor(test_results, field_results, shape_is_valid)

            if af is not None:
                generate_final_report(
                    test_results=test_results,
                    field_results=field_results,
                    af=af,
                    shape_is_valid=shape_is_valid,
                    test_data=st.session_state.durability_failures,
                    field_data=st.session_state.field_failures,
                )
            else:
                logging.error("가속계수 계산에 실패하여 보고서를 생성하지 않습니다.")

    if st.sidebar.button("리셋", key="reset_button"):
        st.session_state.clear()
        st.rerun()

    with st.expander("로그"):
        streamlit_handler.display_logs()


if __name__ == "__main__":
    main()
