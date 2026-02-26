from __future__ import annotations

import io
import logging
import math
import os
import textwrap
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any, cast, Dict, Iterable, List, Set, Tuple, TYPE_CHECKING

import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
import reliability as rel
import scipy.optimize as opt
from scipy.stats import poisson
from nicegui import app, ui

# Type checking fallback for UploadedFile which is not a real type
if TYPE_CHECKING:
    from starlette.datastructures import UploadFile
else:
    UploadFile = Any


# --- Constants and Configuration ---

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

# --- Matplotlib Font Configuration ---

def configure_matplotlib_font() -> None:
    """Ensure a font supporting Korean characters is used."""
    preferred_fonts = [
        "Malgun Gothic", "NanumGothic", "AppleGothic", "NanumGothicCoding",
        "Noto Sans CJK KR", "Noto Sans KR",
    ]
    available = {font.name for font in fm.fontManager.ttflist}
    for font_name in preferred_fonts:
        if font_name in available:
            plt.rcParams["font.family"] = font_name
            plt.rcParams["axes.unicode_minus"] = False
            plt.rcParams["pdf.fonttype"] = 42
            plt.rcParams["ps.fonttype"] = 42
            logging.info("Matplotlib 폰트를 '%s'(으)로 설정했습니다.", font_name)
            ui.notify(f"Matplotlib 폰트를 '{font_name}'(으)로 설정했습니다.", type='info')
            return
    logging.warning("한글을 지원하는 폰트를 찾지 못했습니다. 기본 폰트로 표시됩니다.")
    ui.notify("한글 지원 폰트를 찾지 못했습니다. 일부 문자가 깨질 수 있습니다.", type='warning')


# --- Data Models ---

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

@dataclass(frozen=True)
class WarrantyAnalysisData:
    total_sales: int
    total_claims: int
    cutoff_month: pd.Period
    d_by_age: Dict[int, int]
    cohort_censored: List[Tuple[int, int]]
    triangle: pd.DataFrame


# --- Analysis & Plotting (largely unchanged backend logic) ---

def _extract_ci_bounds(fitter: Any, parameter: str) -> tuple[float, float] | None:
    """Return lower/upper confidence interval bounds for a fitter parameter."""
    tuple_attributes = [f"{parameter}_CI", f"{parameter}_ci", f"{parameter}_ConfidenceInterval"]
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

    lower_candidates = [f"{parameter}_lower", f"{parameter}_Lower", f"{parameter}_lowerCL", f"{parameter}_lower_cl"]
    upper_candidates = [f"{parameter}_upper", f"{parameter}_Upper", f"{parameter}_upperCL", f"{parameter}_upper_cl"]
    
    lower = next((getattr(fitter, attr) for attr in lower_candidates if hasattr(fitter, attr)), None)
    upper = next((getattr(fitter, attr) for attr in upper_candidates if hasattr(fitter, attr)), None)

    if lower is not None and upper is not None:
        try:
            return float(lower), float(upper)
        except (TypeError, ValueError):
            return None
    return None

def format_distribution_name(dist_name: str) -> str:
    return DISTRIBUTION_LABELS.get(dist_name, dist_name)

def _fmt_float(value: float, digits: int = 3) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "-"

def compute_sample_stats(series: pd.Series) -> Dict[str, float]:
    values = series.to_numpy(dtype=float, copy=False)
    return {
        "count": int(len(values)), "mean": float(np.mean(values)), "median": float(np.median(values)),
        "std": float(np.std(values, ddof=1)) if len(values) > 1 else float("nan"),
        "min": float(np.min(values)), "max": float(np.max(values)),
        "p10": float(np.percentile(values, 10)), "p90": float(np.percentile(values, 90)),
    }

def plot_distribution_fit(failures: np.ndarray, fitter: Any, dataset_label: str, ax: plt.Axes | None = None) -> plt.Figure:
    if ax is None:
        fig, ax = plt.subplots(figsize=(6.5, 4.2))
    else:
        fig = ax.figure
        ax.clear()
    ax.hist(failures, bins="auto", density=True, color="#2878B5", alpha=0.6, label="관측 데이터")
    dist = getattr(fitter, "distribution", None)
    if dist and hasattr(dist, "PDF"):
        x_min, x_max = failures.min(), failures.max()
        pad = max((x_max - x_min) * 0.1, 1e-3)
        x = np.linspace(max(0, x_min - pad), x_max + pad, 200)
        try:
            y = np.asarray(dist.PDF(x))
            ax.plot(x, y, color="#D35400", linewidth=2, label="최적 분포 PDF")
        except Exception as exc:
            logging.warning("분포 PDF를 그리는 중 오류가 발생했습니다: %s", exc)
    ax.set_xlabel(DATA_COLUMN_LABEL)
    ax.set_ylabel("확률밀도")
    ax.set_title(f"{dataset_label} 고장 분포")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    return fig

def _add_weibull_stats_box(ax: plt.Axes, fitter: Any, stats: Dict[str, float]) -> None:
    lines = [
        f"형상 모수: {_fmt_float(fitter.beta, 4)}",
        f"척도 모수: {_fmt_float(fitter.alpha, 4)}",
        f"평균: {_fmt_float(stats['mean'], 1)}",
        f"중앙값: {_fmt_float(stats['median'], 1)}",
        f"표준편차: {_fmt_float(stats['std'], 1)}",
        f"고장 수: {int(stats['count'])}",
        f"AD*: {_fmt_float(getattr(fitter, 'AD', float('nan')), 3)}",
    ]
    ax.text(
        0.98,
        0.05,
        "\n".join(lines),
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#555555", alpha=0.95),
    )

def plot_weibull_probability(failures: np.ndarray, fitter: Any, dataset_label: str, stats: Dict[str, float]) -> plt.Figure:
    fig = rel.Probability_plotting.Weibull_probability_plot(
        failures=failures,
        __fitted_dist_params=fitter,
        CI=0.95,
        CI_type="time",
        show_scatter_points=False,
        show_fitted_distribution=True,
    )
    x, y = rel.Probability_plotting.plotting_positions(failures=failures)
    plt.scatter(x, y, marker="o", s=18, color="#d9534f", edgecolors="none", zorder=3)
    ax = plt.gca()
    ax.set_title(f"{dataset_label}에 대한 확률 플롯\nWeibull 분포 - 95% CI\n전체 데이터 - ML 추정치")
    ax.set_xlabel(DATA_COLUMN_LABEL)
    ax.set_ylabel("누적 고장 확률")
    _add_weibull_stats_box(ax, fitter, stats)
    fig.set_size_inches(7.2, 4.2)
    fig.tight_layout()
    return fig

# ... (rest of the backend functions: _df_to_markdown, PDF/Markdown report builders, etc.)
# I will add the rest of the functions from the original file that are not UI-specific.
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

    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, row in rounded.iterrows():
        lines.append("| " + " | ".join(fmt(v) for v in row) + " |")
    return "\n".join(lines)


def build_markdown_report(
    test_results: FitResults, field_results: FitResults, af: float, shape_is_valid: bool,
    test_data: pd.Series, field_data: pd.Series
) -> str:
    shape_summary = "형상 모수 신뢰구간이 겹칩니다." if shape_is_valid else "형상 모수 신뢰구간이 충분히 겹치지 않습니다."
    intro = textwrap.dedent(f"""
        # 내구-필드 수명 분석 종합 보고서
        ## 1. 분석 개요
        본 보고서는 내구 시험 데이터 {len(test_data):,}건과 필드 데이터 {len(field_data):,}건을 기반으로, 가속 시험이 실제 필드 수명과 얼마나 일치하는지 정량적으로 검증하기 위해 작성되었습니다. 두 데이터셋의 통계적 특성을 비교하고 최적 분포를 선정한 뒤, 형상 모수 일치 여부와 가속계수(Acceleration Factor, AF)를 도출하여 가속 시험의 유효성을 평가합니다.
        """).strip()
    stats_table = pd.DataFrame({
            "건수": [test_results.sample_stats["count"], field_results.sample_stats["count"]],
            "평균": [test_results.sample_stats["mean"], field_results.sample_stats["mean"]],
            "중앙값": [test_results.sample_stats["median"], field_results.sample_stats["median"]],
            "표준편차": [test_results.sample_stats["std"], field_results.sample_stats["std"]],
            "10/90 분위": [
                f"{test_results.sample_stats['p10']:.1f} / {test_results.sample_stats['p90']:.1f}",
                f"{field_results.sample_stats['p10']:.1f} / {field_results.sample_stats['p90']:.1f}",
            ],
        }, index=["내구 시험", "필드"])
    report = textwrap.dedent(f"""
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
        """).strip()
    return report

def _add_wrapped_text(ax: plt.Axes, text: str, start_y: float, width: int = 90, line_height: float = 0.028, fontsize: int = 11, x: float = 0.05) -> float:
    y = start_y
    for line in textwrap.wrap(text, width=width) or [""]:
        ax.text(x, y, line, fontsize=fontsize, ha="left", va="top")
        y -= line_height
    return y

def _add_bullet_lines(ax: plt.Axes, lines: list[str], start_y: float, x: float = 0.05, bullet: str = "•", width: int = 90, line_height: float = 0.028, fontsize: int = 11) -> float:
    y = start_y
    for line in lines:
        wrapped = textwrap.wrap(line, width=width) or [""]
        for idx, segment in enumerate(wrapped):
            prefix = f"{bullet} " if idx == 0 else "   "
            ax.text(x, y, prefix + segment, fontsize=fontsize, ha="left", va="top")
            y -= line_height
    return y

def _add_section(ax: plt.Axes, title: str, lines: list[str], start_y: float, width: int = 90, line_height: float = 0.028, fontsize: int = 11) -> float:
    ax.text(0.05, start_y, title, fontsize=13, weight="bold", ha="left", va="top")
    y = start_y - line_height
    for line in lines:
        y = _add_wrapped_text(ax, line, y, width=width, line_height=line_height, fontsize=fontsize, x=0.06)
    return y - line_height * 0.4

def _table_figure(title: str, df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8.27, 5.0))
    ax.set_axis_off(); ax.set_title(title, fontsize=14, weight="bold", pad=10)
    table = ax.table(cellText=df.values, colLabels=df.columns, rowLabels=df.index, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.3)
    fig.tight_layout()
    return fig

def build_pdf_report(
    test_results: FitResults, field_results: FitResults, af: float, shape_is_valid: bool,
    test_data: pd.Series, field_data: pd.Series
) -> bytes:
    test_failures = test_data.to_numpy(dtype=float, copy=False)
    field_failures = field_data.to_numpy(dtype=float, copy=False)
    buffer = io.BytesIO()
    with PdfPages(buffer) as pdf:
        # Page 1: Narrative
        fig, ax = plt.subplots(figsize=(8.27, 11.69))  # A4 portrait
        ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.text(0.5, 0.95, "내구-필드 수명 분석 종합 보고서", fontsize=18, weight="bold", ha="center", va="top")
        y = 0.89
        overview = (f"본 보고서는 내구 시험 데이터 {len(test_failures):,}건과 필드 데이터 {len(field_failures):,}건을 기반으로 가속 시험 결과가 실제 필드 고장 패턴과 얼마나 일치하는지 정량적으로 검증합니다. 데이터 요약 → 분포 적합 → 형상 모수 비교 → 가속계수 해석 순으로 구성됩니다.")
        y = _add_section(ax, "1. 분석 개요", [overview], start_y=y, width=90)
        stats_lines = [
            f"내구 시험 데이터: {len(test_failures):,}건", f"필드 데이터: {len(field_failures):,}건",
            f"최적 분포 (내구): {format_distribution_name(test_results.best_distribution_name)}",
            f"최적 분포 (필드): {format_distribution_name(field_results.best_distribution_name)}",
            f"형상 모수 신뢰구간: {'겹침' if shape_is_valid else '겹치지 않음/불충분'}",
            f"가속계수 (AF): {af:.3f}",
        ]
        y = _add_section(ax, "2. 핵심 요약", ["아래 항목은 분포 적합 및 가속계수 계산 결과를 간결하게 정리한 것입니다."], start_y=y, width=90)
        y = _add_bullet_lines(ax, stats_lines, start_y=y + 0.01, width=90)
        y = _add_section(ax, "3. 가속계수 해석", [f"AF={af:.3f} → 내구 1회는 필드 약 {af:.2f}회와 동일한 응력/손상을 의미합니다.", "가속 시험의 결과를 실제 사용 수명으로 환산하는 데 활용할 수 있습니다."], start_y=y + 0.02, width=90)
        y = _add_section(ax, "4. 형상 모수 해석", ["형상 모수는 시간 경과에 따른 고장률 변화를 설명합니다.", f"두 데이터셋의 형상 모수 신뢰구간은 {'충분히 겹쳐 고장 메커니즘이 동일함을 시사합니다.' if shape_is_valid else '충분히 겹치지 않아 추가 검토가 필요합니다.'}"], start_y=y, width=90)
        fig.tight_layout(); pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        # Page 2: Stats table
        stats_df = pd.DataFrame({
                "건수": [test_results.sample_stats["count"], field_results.sample_stats["count"]],
                "평균": [test_results.sample_stats["mean"], field_results.sample_stats["mean"]],
                "중앙값": [test_results.sample_stats["median"], field_results.sample_stats["median"]],
                "표준편차": [test_results.sample_stats["std"], field_results.sample_stats["std"]],
                "10/90분위": [ f"{test_results.sample_stats['p10']:.1f} / {test_results.sample_stats['p90']:.1f}", f"{field_results.sample_stats['p10']:.1f} / {field_results.sample_stats['p90']:.1f}"],
            }, index=["내구 시험", "필드"]).round(2)
        pdf.savefig(_table_figure("데이터 요약 통계", stats_df), bbox_inches="tight"); plt.close("all")
        
        # Pages 3-4: Fit tables
        test_table = test_results.results_table.copy().round(3)
        test_table.index = [format_distribution_name(idx) for idx in test_table.index]
        pdf.savefig(_table_figure("내구 시험 분포 적합 결과", test_table), bbox_inches="tight"); plt.close("all")

        field_table = field_results.results_table.copy().round(3)
        field_table.index = [format_distribution_name(idx) for idx in field_table.index]
        pdf.savefig(_table_figure("필드 분포 적합 결과", field_table), bbox_inches="tight"); plt.close("all")
        
        # Page 5: Plots
        pdf.savefig(plot_distribution_fit(test_failures, test_results.best_distribution, "내구 시험")); plt.close("all")
        pdf.savefig(plot_distribution_fit(field_failures, field_results.best_distribution, "필드")); plt.close("all")

    buffer.seek(0)
    return buffer.getvalue()


# --- Scheduling Logic (largely unchanged) ---

OPERATING_TESTS = ("UNIT 고유저항 평가", "특성 평가")
STIFFNESS_TESTS = ("글라스 틸팅 강성 평가", "상하 유격 강성 평가", "모터 역전 확인 평가", "역전 방지부 하방향 강성 평가", "역전 방지부 상방향 강성 평가", "상승 구속 강성 평가")
INDIVIDUAL_TESTS = {"작동 이음 평가": 1, "케이블 앤드부 강성 평가": 1, "내식성 평가": 21, "크리프 벤치 평가": 20}
UNSCHEDULED_TESTS = {"작동력 평가"}
TEST_NUMBER = {
    "UNIT 고유저항 평가": "6.2.2.1", "작동 이음 평가": "6.2.2.2", "작동력 평가": "6.2.2.3", "특성 평가": "6.2.2.4",
    "세이프티 평가 (법규)": "6.2.2.5", "모터 소손 평가": "6.2.2.6", "글라스 틸팅 강성 평가": "6.2.3.1",
    "상하 유격 강성 평가": "6.2.3.2", "모터 역전 확인 평가": "6.2.3.4", "역전 방지부 하방향 강성 평가": "6.2.3.5",
    "역전 방지부 상방향 강성 평가": "6.2.3.6", "상승 구속 강성 평가": "6.2.3.7", "케이블 앤드부 강성 평가": "6.2.3.9",
    "내구력 평가": "6.2.4.1", "내진동 평가": "6.2.4.2", "내충격 평가": "6.2.4.3", "내열 평가": "6.2.4.4",
    "내한 평가": "6.2.4.5", "부가 내열 / 내한 평가": "6.2.4.6", "내식성 평가": "6.2.4.7", "내수성 평가": "6.2.4.8",
    "크리프 벤치 평가": "6.2.4.9", "모터 내한성 이음 평가": "6.2.5.1", "케이블 내한성 이탈 평가": "6.2.5.2",
}
OUTPUT_SECTIONS = [
    ("작동특성", ["UNIT 고유저항 평가", "작동 이음 평가", "작동력 평가", "특성 평가", "세이프티 평가 (법규)", "모터 소손 평가"]),
    ("강성시험", [*STIFFNESS_TESTS, "케이블 앤드부 강성 평가"]),
    ("내구/환경 평가", ["내구력 평가", "내진동 평가", "내충격 평가", "내열 평가", "내한 평가", "부가 내열 / 내한 평가", "내식성 평가", "내수성 평가", "크리프 벤치 평가"]),
    ("추가 평가", ["모터 내한성 이음 평가", "케이블 내한성 이탈 평가"]),
]

def _parse_holidays(raw_text: str) -> tuple[set[date], list[str]]:
    holidays: set[date] = set()
    invalid: list[str] = []
    tokens = [token.strip() for token in raw_text.replace(",", "\n").splitlines()]
    for token in filter(None, tokens):
        try:
            holidays.add(datetime.strptime(token, "%Y-%m-%d").date())
        except ValueError:
            invalid.append(token)
    return holidays, invalid

def _is_non_working(day: date, holidays: set[date]) -> bool:
    return day.weekday() >= 5 or day in holidays

def _next_monday(day: date) -> date:
    days_ahead = (7 - day.weekday()) % 7 or 7
    return day + timedelta(days=days_ahead)

def _adjust_start(day: date, holidays: set[date], correction_mode: str) -> date:
    if correction_mode == "weekday":
        while _is_non_working(day, holidays):
            day += timedelta(days=1)
    else: # monday
        while _is_non_working(day, holidays):
            day = _next_monday(day)
    return day

def _format_date(day: date | None, date_format: str) -> str:
    if not day: return "-"
    return day.strftime("%y.%m.%d.") if date_format == "YY.MM.DD." else day.strftime("%Y-%m-%d")

def _format_duration(days: int | None, duration_format: str) -> str:
    if days is None: return "-"
    return f"{days}일" if duration_format == "N일" else f"D+{days}"

def _schedule_sequence(start_day: date, stages: list[Stage], group: str, holidays: set[date], correction_mode: str) -> tuple[list[ScheduledOccurrence], date]:
    occurrences: list[ScheduledOccurrence] = []
    current = start_day
    for stage in stages:
        stage_start = _adjust_start(current, holidays, correction_mode)
        stage_end = stage_start + timedelta(days=stage.duration)
        for test_name in stage.tests:
            occurrences.append(ScheduledOccurrence(test_name, stage_start, stage_end, group, stage.name))
        current = stage_end
    return occurrences, current

def _schedule_core(base_start: date, holidays: set[date], correction_mode: str) -> tuple[list[ScheduledOccurrence], date]:
    adjusted_start = _adjust_start(base_start, holidays, correction_mode)
    cross1 = [Stage("작동특성", 1, OPERATING_TESTS), Stage("세이프티 평가 (법규)", 1, ("세이프티 평가 (법규)",)), Stage("내구력 평가", 26, ("내구력 평가",)), Stage("세이프티 평가 (법규)", 1, ("세이프티 평가 (법규)",)), Stage("작동특성", 1, OPERATING_TESTS)]
    cross2 = [Stage("작동특성", 1, OPERATING_TESTS), Stage("강성 평가(내구전)", 1, STIFFNESS_TESTS), Stage("내진동 평가", 14, ("내진동 평가",)), Stage("내구력 평가", 26, ("내구력 평가",)), Stage("강성 평가(내구후)", 1, STIFFNESS_TESTS), Stage("작동특성", 1, OPERATING_TESTS)]
    cross3 = [Stage("작동특성", 1, OPERATING_TESTS), Stage("내열 평가", 2, ("내열 평가",)), Stage("내한 평가", 2, ("내한 평가",)), Stage("내충격 평가", 10, ("내충격 평가",)), Stage("내수성 평가", 1, ("내수성 평가",))]
    cross4 = [Stage("모터 내한성 이음 평가", 4, ("모터 내한성 이음 평가",)), Stage("부가 내열 / 내한 평가", 3, ("부가 내열 / 내한 평가",)), Stage("케이블 내한성 이탈 평가", 1, ("케이블 내한성 이탈 평가",)), Stage("모터 소손 평가", 1, ("모터 소손 평가",))]

    occurrences: list[ScheduledOccurrence] = []
    occ1, end1 = _schedule_sequence(adjusted_start, cross1, "교차#1", holidays, correction_mode)
    occ2, end2 = _schedule_sequence(adjusted_start, cross2, "교차#2", holidays, correction_mode)
    occ3, end3 = _schedule_sequence(adjusted_start, cross3, "교차#3", holidays, correction_mode)
    cross4_start = _adjust_start(end3, holidays, correction_mode)
    occ4, end4 = _schedule_sequence(cross4_start, cross4, "교차#4", holidays, correction_mode)
    occurrences.extend(occ1 + occ2 + occ3 + occ4)

    for test_name, duration in INDIVIDUAL_TESTS.items():
        if test_name == "작동 이음 평가": continue
        occurrences.append(ScheduledOccurrence(test_name, adjusted_start, adjusted_start + timedelta(days=duration), "개별", test_name))
    
    max_end = max([end1, end2, end3, end4] + [occ.end for occ in occurrences])
    return occurrences, max_end

def _schedule_with_standalone(start_day: date, holidays: set[date], correction_mode: str, position: str) -> tuple[list[ScheduledOccurrence], date, date, date | None, date | None]:
    base_start = _adjust_start(start_day, holidays, correction_mode)
    
    def schedule_start():
        standalone_start = base_start
        standalone_end = standalone_start + timedelta(days=INDIVIDUAL_TESTS["작동 이음 평가"])
        occurrences = [ScheduledOccurrence("작동 이음 평가", standalone_start, standalone_end, "개별", "작동 이음 평가")]
        other_start = _adjust_start(standalone_end, holidays, correction_mode)
        core_occ, core_end = _schedule_core(other_start, holidays, correction_mode)
        occurrences.extend(core_occ)
        final_end = max(core_end, standalone_end)
        return occurrences, base_start, final_end, standalone_start, standalone_end

    def schedule_end():
        core_occ, core_end = _schedule_core(base_start, holidays, correction_mode)
        standalone_start = _adjust_start(core_end, holidays, correction_mode)
        standalone_end = standalone_start + timedelta(days=INDIVIDUAL_TESTS["작동 이음 평가"])
        core_occ.append(ScheduledOccurrence("작동 이음 평가", standalone_start, standalone_end, "개별", "작동 이음 평가"))
        final_end = max(core_end, standalone_end)
        return core_occ, base_start, final_end, standalone_start, standalone_end

    if position == "start": return schedule_start()
    if position == "end": return schedule_end()
    
    start_schedule, end_schedule = schedule_start(), schedule_end()
    return start_schedule if start_schedule[2] <= end_schedule[2] else end_schedule

def _aggregate_occurrences(occurrences: list[ScheduledOccurrence]) -> dict[str, dict[str, date]]:
    aggregated: dict[str, dict[str, date]] = {}
    for occ in occurrences:
        if occ.test_name not in aggregated:
            aggregated[occ.test_name] = {"start": occ.start, "end": occ.end}
        else:
            aggregated[occ.test_name]["start"] = min(aggregated[occ.test_name]["start"], occ.start)
            aggregated[occ.test_name]["end"] = max(aggregated[occ.test_name]["end"], occ.end)
    return aggregated

def _build_section_tables(aggregated: dict[str, dict[str, date]], date_format: str, duration_format: str, include_no: bool, include_notes: bool) -> tuple[list[tuple[str, pd.DataFrame]], pd.DataFrame]:
    section_tables: list[tuple[str, pd.DataFrame]] = []
    combined_rows: list[dict[str, str]] = []

    # Define base and final columns
    base_columns = ["시험명", "소요일", "시작일", "종료일"]
    final_columns = base_columns[:]
    if include_no:
        final_columns.insert(0, "No.")
    if include_notes:
        final_columns.append("비고")

    for section_name, tests in OUTPUT_SECTIONS:
        rows: list[dict[str, str]] = []
        for test_name in tests:
            schedule = aggregated.get(test_name)
            start, end = (schedule["start"], schedule["end"]) if schedule else (None, None)
            duration_days = (end - start).days if start and end else None
            row: dict[str, str] = {
                "시험명": test_name, "소요일": _format_duration(duration_days, duration_format),
                "시작일": _format_date(start, date_format), "종료일": _format_date(end, date_format),
            }
            if include_no: row["No."] = TEST_NUMBER.get(test_name, "")
            if include_notes: row["비고"] = ""
            rows.append(row)
            combined_rows.append({"구분": section_name, **row})

        df = pd.DataFrame(rows, columns=final_columns)
        section_tables.append((section_name, df))

    combined_columns = ["구분"] + final_columns
    combined_df = pd.DataFrame(combined_rows, columns=combined_columns)
    return section_tables, combined_df


# --- NiceGUI UI Implementation ---

# --- Global Setup ---
# This setup runs once when the script starts.
os.makedirs("results", exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s",
                    handlers=[logging.FileHandler(os.path.join("results", "analysis_log.txt"), encoding="utf-8"),
                              logging.StreamHandler()])
app.on_startup(configure_matplotlib_font)

@ui.page('/')
def main_page():
    # --- App State ---
    class AppState:
        test_file: UploadFile | None = None
        field_file: UploadFile | None = None
        durability_failures: pd.Series | None = None
        field_failures: pd.Series | None = None
        
        def reset(self):
            self.test_file = None
            self.field_file = None
            self.durability_failures = None
            self.field_failures = None

    state = AppState()

    # --- UI Layout ---
    ui.dark_mode().enable()
    with ui.header(elevated=True).classes('bg-primary text-white'):
        ui.label('내구 수명 분석 및 시험 계획 도구').classes('text-h5')

    with ui.tabs().classes('w-full') as tabs:
        one = ui.tab('내구-필드 수명 분석')
        two = ui.tab('시험 일정 생성')
        three = ui.tab('신뢰성 시험 설계 및 가속 분석')
        four = ui.tab('보증 데이터 분석')

    with ui.tab_panels(tabs, value=one).classes('w-full'):
        with ui.tab_panel(one):
            render_analysis_tab(state)
        with ui.tab_panel(two):
            render_schedule_planner()
        with ui.tab_panel(three):
            render_reliability_design_tab()
        with ui.tab_panel(four):
            render_warranty_analysis_tab()

def render_analysis_tab(state: 'AppState'):

    async def _get_upload_bytes(e: Any) -> tuple[bytes, str]:
        if hasattr(e, "file"):
            file_obj = e.file
            data = await file_obj.read()
            name = getattr(file_obj, "name", None) or getattr(e, "name", None) or "업로드된 파일"
            return data, name

        content = getattr(e, "content", None)
        name = getattr(e, "name", None) or "업로드된 파일"
        if content is None:
            raise ValueError("업로드 파일 내용을 찾을 수 없습니다.")
        if isinstance(content, (bytes, bytearray)):
            return bytes(content), name
        if hasattr(content, "read"):
            content.seek(0)
            data = content.read()
            if isinstance(data, str):
                data = data.encode("utf-8")
            return data, name
        raise ValueError("지원하지 않는 업로드 데이터 형식입니다.")

    def _coerce_numeric(series: pd.Series) -> pd.Series:
        if series.dtype == object:
            series = series.astype(str).str.replace(",", "", regex=False)
        return pd.to_numeric(series, errors="coerce").dropna()

    async def handle_upload(e: Any, is_test_file: bool):
        file_type = "내구 시험" if is_test_file else "필드"
        try:
            data_bytes, filename = await _get_upload_bytes(e)
            df = None
            # Try common encodings for CSV input
            for encoding in ("utf-8-sig", "cp949", "euc-kr"):
                try:
                    df = pd.read_csv(io.BytesIO(data_bytes), encoding=encoding)
                    break  # Success
                except (UnicodeDecodeError, pd.errors.ParserError, pd.errors.EmptyDataError):
                    continue
            
            if df is None:
                df = pd.read_csv(io.BytesIO(data_bytes), encoding="utf-8", errors="ignore")

            df.columns = df.columns.str.strip()
            if DATA_COLUMN_NAME not in df.columns:
                ui.notify(f"'{DATA_COLUMN_NAME}' 컬럼이 파일에 없습니다.", type='negative')
                return

            failures = _coerce_numeric(df[DATA_COLUMN_NAME])
            if failures.empty:
                ui.notify(f"{file_type} 파일에서 숫자 데이터를 찾지 못했습니다.", type='negative')
                return

            if is_test_file:
                state.durability_failures = failures
            else:
                state.field_failures = failures
            
            ui.notify(f"{file_type} 파일 '{filename}' 로드 성공.", type='positive')

        except Exception as err:
            ui.notify(f"{file_type} 파일 처리 중 오류: {err}", type='negative')
            logging.error("%s 파일 처리 오류: %s", file_type, err)

    async def run_analysis():
        if state.durability_failures is None or state.field_failures is None:
            ui.notify("내구 시험과 필드 CSV 파일을 모두 업로드해 주세요.", type='warning')
            return

        result_area.clear()
        with result_area:
            with ui.spinner(size='lg', color='primary'):
                ui.label("분석을 시작합니다...").classes('text-h6')

                try:
                    test_results = analyse_single_dataset(state.durability_failures, "내구 시험", result_area)
                except Exception as err:
                    logging.exception("내구 시험 분포 분석 중 오류")
                    ui.notify(f"내구 시험 분석 중 오류: {err}", type='negative')
                    return

                try:
                    field_results = analyse_single_dataset(state.field_failures, "필드", result_area)
                except Exception as err:
                    logging.exception("필드 분포 분석 중 오류")
                    ui.notify(f"필드 분석 중 오류: {err}", type='negative')
                    return

                try:
                    shape_is_valid = compare_distributions(test_results, field_results, result_area)
                    af = calculate_acceleration_factor(test_results, field_results, shape_is_valid, result_area)
                except Exception as err:
                    logging.exception("형상 모수/가속계수 계산 중 오류")
                    ui.notify(f"형상 모수/가속계수 계산 중 오류: {err}", type='negative')
                    return

                if af is not None:
                    try:
                        generate_final_report(test_results, field_results, af, shape_is_valid, state.durability_failures, state.field_failures, result_area)
                    except Exception as err:
                        logging.exception("최종 보고서 생성 중 오류")
                        ui.notify(f"최종 보고서 생성 중 오류: {err}", type='negative')
                else:
                    logging.error("가속계수 계산 실패로 보고서를 생성하지 않습니다.")
                    ui.notify("가속계수 계산 실패로 보고서를 생성하지 않습니다.", type='negative')

    # Main layout for this tab: 2 columns
    with ui.grid(columns=4).classes('w-full'):
        # Left column for inputs
        with ui.element('div').classes('col-span-1 p-4 bg-gray-100 dark:bg-gray-800 rounded-lg'):
            with ui.column():
                ui.label('데이터 입력').classes('text-h6 q-pb-md')
                ui.label("내구 시험 CSV 업로드")
                ui.upload(on_upload=lambda e: handle_upload(e, is_test_file=True), auto_upload=True, max_files=1).props('accept=.csv').classes('w-full')
                ui.label("필드 CSV 업로드")
                ui.upload(on_upload=lambda e: handle_upload(e, is_test_file=False), auto_upload=True, max_files=1).props('accept=.csv').classes('w-full')
                ui.button("수명 분석 시작", on_click=run_analysis).props('icon=analytics').classes('w-full q-mt-md')
                ui.button("리셋", on_click=lambda: (state.reset(), result_area.clear(), ui.notify("리셋되었습니다."))).props('icon=refresh color=accent').classes('w-full q-mt-sm')
        
        # Right column for results
        with ui.element('div').classes('col-span-3 p-4'):
            result_area = ui.column().classes('w-full')


def analyse_single_dataset(data: pd.Series, name: str, container: ui.column) -> FitResults:
    with container:
        ui.label(f"1단계: '{name}' 분포 적합").classes('text-h5 q-mt-md')
        failures = data.to_numpy(dtype=float, copy=False)
        stats = compute_sample_stats(data)
        ui.label(f"{name} 관측수 {stats['count']}건 · 평균 {_fmt_float(stats['mean'], 1)} · 중앙값 {_fmt_float(stats['median'], 1)} · 표준편차 {_fmt_float(stats['std'], 1)}").classes('text-caption')

        fitters = {
            "Weibull_2P": rel.Fitters.Fit_Weibull_2P(failures=failures, print_results=False),
            "Lognormal_2P": rel.Fitters.Fit_Lognormal_2P(failures=failures, print_results=False),
            "Normal_2P": rel.Fitters.Fit_Normal_2P(failures=failures, print_results=False),
            "Exponential_1P": rel.Fitters.Fit_Exponential_1P(failures=failures, print_results=False),
        }
        summary = pd.DataFrame({
                "Log-Likelihood": [fit.loglik for fit in fitters.values()], "AICc": [fit.AICc for fit in fitters.values()],
                "BIC": [fit.BIC for fit in fitters.values()], "AD": [fit.AD for fit in fitters.values()],
            }, index=[format_distribution_name(k) for k in fitters.keys()]).round(3)
        
        best_name_raw = summary["AICc"].idxmin()
        best_name = next(k for k, v in DISTRIBUTION_LABELS.items() if v == best_name_raw)
        best_fitter = fitters[best_name]

        reason = f"AICc가 가장 낮은 {best_name_raw}(AICc={summary.loc[best_name_raw, 'AICc']:.3f})를 선정했습니다."
        
        ui.label(f"'{name}' 분포 비교").classes('text-h6 q-mt-sm')
        ui.table.from_pandas(summary.reset_index().rename(columns={'index': '분포'}))
        ui.notify(reason, type='info')

        with ui.expansion(f"'{name}' 최적 분포 파라미터", icon='tune'):
            rows = [{"파라미터": PARAM_LABELS.get(attr, attr), "추정값": f"{getattr(best_fitter, attr):.3f}"} 
                    for attr in ["alpha", "beta", "mu", "sigma", "Lambda", "mean", "std"] if hasattr(best_fitter, attr)]
        ui.table(columns=[{'name': '파라미터', 'label': '파라미터', 'field': '파라미터'},
                              {'name': '추정값', 'label': '추정값', 'field': '추정값'}], rows=rows)
        
        if best_name == "Weibull_2P":
            with ui.pyplot(figsize=(7.2, 4.2)):
                plot_weibull_probability(failures, best_fitter, name, stats)
        else:
            with ui.pyplot(figsize=(6.5, 4.2)):
                plot_distribution_fit(failures, best_fitter, name, ax=plt.gca())

    return FitResults(name, best_name, best_fitter, summary, fitters, stats, reason)

def compare_distributions(test_results: FitResults, field_results: FitResults, container: ui.column) -> bool:
    with container:
        ui.label("2단계: 형상 모수 비교").classes('text-h5 q-mt-md')
        if test_results.best_distribution_name != field_results.best_distribution_name:
            msg = f"내구와 필드의 최적 분포가 다릅니다. 내구 -> {format_distribution_name(test_results.best_distribution_name)}, 필드 -> {format_distribution_name(field_results.best_distribution_name)}."
            ui.notify(msg, type='warning')
            return False

        dist_name = test_results.best_distribution_name
        ui.notify(f"두 데이터셋의 최적 분포({format_distribution_name(dist_name)})가 동일합니다.", type='positive')

        shape_param, shape_label = SHAPE_PARAM_LABELS.get(dist_name, (None, None))
        if not shape_param:
            ui.notify("해당 분포는 형상 모수를 비교하지 않습니다.", type='info')
            return True

        test_shape = float(getattr(test_results.best_distribution, shape_param))
        field_shape = float(getattr(field_results.best_distribution, shape_param))
        test_ci = _extract_ci_bounds(test_results.best_distribution, shape_param)
        field_ci = _extract_ci_bounds(field_results.best_distribution, shape_param)

        ui.label(f"형상 모수 비교 ({shape_label})").classes('text-h6 q-mt-sm')
        comp_df = pd.DataFrame({
            "데이터셋": ["내구 시험", "필드"], "추정값": [test_shape, field_shape],
            "하한": [test_ci[0] if test_ci else np.nan, field_ci[0] if field_ci else np.nan],
            "상한": [test_ci[1] if test_ci else np.nan, field_ci[1] if field_ci else np.nan],
        }).round(3)
        ui.table.from_pandas(comp_df)

        shape_overlap = False
        if test_ci and field_ci:
            shape_overlap = not (test_ci[1] < field_ci[0] or field_ci[1] < test_ci[0])
            ui.notify("형상 모수 신뢰구간이 겹칩니다." if shape_overlap else "형상 모수 신뢰구간이 겹치지 않습니다.", type='positive' if shape_overlap else 'warning')
        else:
            ui.notify("신뢰구간 정보가 부족하여 겹침 여부를 평가하기 어렵습니다.", type='info')
        
        return shape_overlap

def calculate_acceleration_factor(test_results: FitResults, field_results: FitResults, shape_is_valid: bool, container: ui.column) -> float | None:
    with container:
        ui.label("3단계: 가속계수").classes('text-h5 q-mt-md')
        if not shape_is_valid:
            ui.notify("형상 모수가 충분히 일치하지 않아 가속계수 해석 시 주의가 필요합니다.", type='warning')
        
        if test_results.best_distribution_name != field_results.best_distribution_name:
            ui.notify("분포가 서로 달라 가속계수를 계산할 수 없습니다.", type='negative')
            return None

        dist_name = test_results.best_distribution_name
        scale_param = SCALE_PARAM_LOOKUP.get(dist_name)
        if not scale_param:
            ui.notify(f"{format_distribution_name(dist_name)} 분포에서는 가속계수를 계산할 수 없습니다.", type='negative')
            return None
        
        test_scale = float(getattr(test_results.best_distribution, scale_param))
        field_scale = float(getattr(field_results.best_distribution, scale_param))

        if dist_name == "Lognormal_2P": af, formula = float(np.exp(field_scale - test_scale)), "exp(μ_field - μ_test)"
        elif dist_name == "Exponential_1P": af, formula = float(test_scale / field_scale), "λ_test / λ_field"
        else: af, formula = float(field_scale / test_scale), "α_field / α_test"
        
        with ui.card():
            ui.label(f"{af:.3f}").classes('text-h4')
            ui.label("가속계수 (AF)").classes('text-subtitle2')
            ui.label(f"계산식: AF = {formula}").classes('text-caption')
        
        logging.info("가속계수를 계산했습니다: %.3f", af)
        return af

def generate_final_report(test_results: FitResults, field_results: FitResults, af: float, shape_is_valid: bool, test_data: pd.Series, field_data: pd.Series, container: ui.column):
    with container:
        ui.label("4단계: 최종 보고서").classes('text-h5 q-mt-md')
        report_md = build_markdown_report(test_results, field_results, af, shape_is_valid, test_data, field_data)
        pdf_bytes = build_pdf_report(test_results, field_results, af, shape_is_valid, test_data, field_data)

        try:
            with open(os.path.join("results", "final_report.pdf"), "wb") as f:
                f.write(pdf_bytes)
            ui.notify("PDF 보고서를 results/final_report.pdf 로 저장했습니다.", type='positive')
        except Exception as e:
            ui.notify(f"PDF 저장 중 오류: {e}", type='warning')

        with ui.row():
            ui.button("보고서 다운로드 (.md)", on_click=lambda: app.download(report_md.encode(), 'final_report.md')).props('icon=download')
            ui.button("보고서 다운로드 (.pdf)", on_click=lambda: app.download(pdf_bytes, 'final_report.pdf')).props('icon=picture_as_pdf')
        
        ui.markdown(report_md).classes('q-mt-md')

def _read_csv_bytes(data_bytes: bytes) -> pd.DataFrame:
    for encoding in ("utf-8-sig", "cp949", "euc-kr"):
        try:
            return pd.read_csv(io.BytesIO(data_bytes), encoding=encoding)
        except (UnicodeDecodeError, pd.errors.ParserError, pd.errors.EmptyDataError):
            continue
    return pd.read_csv(io.BytesIO(data_bytes), encoding="utf-8", errors="ignore")

def _normalize_percent(value: float) -> float:
    if value is None:
        return float("nan")
    return value / 100 if value > 1 else value

def _prepare_alt_data(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    df = df.copy()
    df.columns = df.columns.str.strip()
    required = {"Stress(온도)", "Time(고장시간)", "Status(F/S)"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"필수 컬럼 누락: {', '.join(sorted(missing))}")

    stress = pd.to_numeric(df["Stress(온도)"], errors="coerce")
    time = pd.to_numeric(df["Time(고장시간)"], errors="coerce")
    status = df["Status(F/S)"].astype(str).str.strip().str.upper()
    status = status.replace({
        "FAIL": "F", "FAILURE": "F",
        "PASS": "S", "SUCCESS": "S", "C": "S", "CENSORED": "S",
    })

    valid_mask = stress.notna() & time.notna() & status.isin({"F", "S"}) & (time > 0) & (stress > -273.15)
    dropped = int((~valid_mask).sum())
    stress = stress[valid_mask]
    time = time[valid_mask]
    status = status[valid_mask]

    failures = time[status == "F"].to_numpy(dtype=float)
    failure_stress = stress[status == "F"].to_numpy(dtype=float)
    right_censored = time[status == "S"].to_numpy(dtype=float)
    right_censored_stress = stress[status == "S"].to_numpy(dtype=float)
    return failures, failure_stress, right_censored, right_censored_stress, dropped

def _c_to_k(value: float | np.ndarray) -> float | np.ndarray:
    return np.asarray(value, dtype=float) + 273.15

def _arrhenius_life(stress_k: float | np.ndarray, a: float, b: float) -> float | np.ndarray:
    return b * np.exp(a / np.asarray(stress_k, dtype=float))

def plot_arrhenius_exponential(stress_k: np.ndarray, a: float, b: float, use_stress_k: float) -> None:
    stress_k = np.asarray(stress_k, dtype=float)
    min_stress = float(np.min(np.append(stress_k, use_stress_k)))
    max_stress = float(np.max(np.append(stress_k, use_stress_k)))
    stress_line = np.linspace(min_stress, max_stress, 200)

    alpha_line = _arrhenius_life(stress_line, a, b)
    x_line = 1000.0 / stress_line
    y_line = np.log(alpha_line)

    stress_points = np.unique(stress_k)
    alpha_points = _arrhenius_life(stress_points, a, b)
    x_points = 1000.0 / stress_points
    y_points = np.log(alpha_points)

    alpha_use = _arrhenius_life(use_stress_k, a, b)
    x_use = 1000.0 / use_stress_k
    y_use = float(np.log(alpha_use))

    plt.plot(x_line, y_line, label="모델")
    plt.scatter(x_points, y_points, label="스트레스 레벨", zorder=3)
    plt.scatter([x_use], [y_use], color="red", label="사용 조건", zorder=4)
    plt.xlabel("1000 / T (1/K)")
    plt.ylabel("ln(특성수명)")
    plt.title("Arrhenius Plot (Weibull-Exponential)")
    plt.grid(True, alpha=0.3)
    plt.legend()

def _compute_acceleration_table(stress_k: np.ndarray, a: float, b: float, use_stress_k: float) -> pd.DataFrame:
    unique_stress = np.unique(stress_k)
    alpha_use = float(_arrhenius_life(use_stress_k, a, b))
    rows = []
    for stress in unique_stress:
        alpha_stress = float(_arrhenius_life(stress, a, b))
        af = alpha_use / alpha_stress
        rows.append({
            "Stress(℃)": round(float(stress) - 273.15, 3),
            "Stress(K)": round(float(stress), 3),
            "AF (Use/Stress)": round(float(af), 4),
        })
    return pd.DataFrame(rows)

def _binomial_cdf(n: int, r: int, R: float) -> float:
    return sum(math.comb(n, i) * (1 - R) ** i * R ** (n - i) for i in range(r + 1))

def _required_sample_size(R: float, CL: float, r: int, max_n: int = 100000) -> int | None:
    if not (0 < R < 1 and 0 < CL < 1) or r < 0:
        return None
    if r == 0:
        return int(math.ceil(math.log(1 - CL) / math.log(R)))
    target = 1 - CL
    n = max(1, r)
    while n <= max_n:
        if _binomial_cdf(n, r, R) <= target:
            return n
        n += 1
    return None

def render_reliability_design_tab():
    class AltState:
        df: pd.DataFrame | None = None
        filename: str | None = None

    alt_state = AltState()
    alt_result: ui.column | None = None
    sample_result: ui.column | None = None

    async def _get_upload_bytes(e: Any) -> tuple[bytes, str]:
        if hasattr(e, "file"):
            file_obj = e.file
            data = await file_obj.read()
            name = getattr(file_obj, "name", None) or getattr(e, "name", None) or "업로드된 파일"
            return data, name
        content = getattr(e, "content", None)
        name = getattr(e, "name", None) or "업로드된 파일"
        if content is None:
            raise ValueError("업로드 파일 내용을 찾을 수 없습니다.")
        if isinstance(content, (bytes, bytearray)):
            return bytes(content), name
        if hasattr(content, "read"):
            content.seek(0)
            data = content.read()
            if isinstance(data, str):
                data = data.encode("utf-8")
            return data, name
        raise ValueError("지원하지 않는 업로드 데이터 형식입니다.")

    async def handle_alt_upload(e: Any):
        try:
            data_bytes, filename = await _get_upload_bytes(e)
            df = _read_csv_bytes(data_bytes)
            alt_state.df = df
            alt_state.filename = filename
            ui.notify(f"ALT 파일 '{filename}' 로드 성공.", type='positive')
        except Exception as err:
            ui.notify(f"ALT 파일 처리 중 오류: {err}", type='negative')
            logging.error("ALT 파일 처리 오류: %s", err)

    def run_alt_analysis():
        if alt_result is None:
            return
        alt_result.clear()
        if alt_state.df is None:
            ui.notify("ALT CSV 파일을 업로드해 주세요.", type='warning')
            return
        if use_temp.value is None:
            ui.notify("사용 온도를 입력해 주세요.", type='warning')
            return
        try:
            failures, failure_stress, right_censored, right_censored_stress, dropped = _prepare_alt_data(alt_state.df)
            if dropped > 0:
                ui.notify(f"유효하지 않은 {dropped}개 행을 제외했습니다.", type='warning')
            if len(failures) < 2:
                ui.notify("고장 데이터가 최소 2건 필요합니다.", type='warning')
                return
            all_stress = np.concatenate([failure_stress, right_censored_stress]) if len(right_censored) else failure_stress
            unique_stress = np.unique(all_stress)
            if len(unique_stress) < 2:
                ui.notify("ALT 분석에는 2개 이상의 스트레스 레벨이 필요합니다.", type='warning')
                return

            stress_k_fail = _c_to_k(failure_stress)
            stress_k_rc = _c_to_k(right_censored_stress) if len(right_censored) else None
            use_stress_k = float(_c_to_k(use_temp.value))

            fit = rel.ALT_fitters.Fit_Weibull_Exponential(
                failures=failures,
                failure_stress=stress_k_fail,
                right_censored=right_censored if len(right_censored) else None,
                right_censored_stress=stress_k_rc if len(right_censored) else None,
                use_level_stress=use_stress_k,
                show_probability_plot=False,
                show_life_stress_plot=False,
                print_results=False,
            )

            dist_use = getattr(fit, "distribution_at_use_stress", None)
            mean_life = getattr(fit, "mean_life", None)
            b10_life = dist_use.quantile(0.1) if dist_use else None
            if mean_life is None and dist_use is not None:
                mean_life = dist_use.mean

            with alt_result:
                ui.label("ALT 분석 결과").classes('text-h6 q-mt-sm')
                ui.label(f"모델: Weibull-Exponential (Arrhenius)").classes('text-caption')
                ui.label(f"사용 온도: {use_temp.value} ℃").classes('text-caption')
                if b10_life is not None:
                    ui.label(f"B10 수명(사용 조건): {_fmt_float(b10_life, 2)}").classes('text-subtitle2')
                if mean_life is not None:
                    ui.label(f"평균 수명(사용 조건): {_fmt_float(mean_life, 2)}").classes('text-subtitle2')

                with ui.pyplot(figsize=(7.2, 4.2)):
                    plot_arrhenius_exponential(stress_k_fail, fit.a, fit.b, use_stress_k)

                af_table = _compute_acceleration_table(np.concatenate([stress_k_fail, stress_k_rc]) if stress_k_rc is not None else stress_k_fail, fit.a, fit.b, use_stress_k)
                ui.label("가속 계수 (AF)").classes('text-subtitle2 q-mt-md')
                ui.table.from_pandas(af_table).classes('w-full')

        except Exception as err:
            logging.exception("ALT 분석 오류")
            ui.notify(f"ALT 분석 중 오류: {err}", type='negative')

    def run_sample_size():
        if sample_result is None:
            return
        sample_result.clear()
        R = _normalize_percent(rel_input.value)
        CL = _normalize_percent(cl_input.value)
        r_value = int(r_input.value or 0)
        n_required = _required_sample_size(R, CL, r_value)
        if n_required is None:
            ui.notify("입력값을 확인해 주세요. (0<R<1, 0<CL<1, r>=0)", type='warning')
            return

        with sample_result:
            ui.label(f"최소 필요 시료 수: {n_required}개").classes('text-h6')
            ui.label(
                f"신뢰수준 {CL*100:.1f}%에서 신뢰도 {R*100:.1f}%를 보증하기 위해, "
                f"{r_value}개 불량 허용 기준 최소 {n_required}개의 시료가 필요합니다."
            ).classes('text-caption')

    with ui.grid(columns=2).classes('w-full'):
        with ui.card().classes('w-full'):
            with ui.column():
                ui.label("가속 수명 시험(ALT) 분석").classes('text-h5')
                ui.label("CSV 컬럼: Stress(온도), Time(고장시간), Status(F/S)").classes('text-body2')
                ui.label("온도는 ℃ 입력 기준이며, 내부에서 K로 변환됩니다.").classes('text-caption')
                ui.upload(on_upload=handle_alt_upload, auto_upload=True, max_files=1).props('accept=.csv').classes('w-full')
                use_temp = ui.number("사용 온도 (℃)", value=25, min=-50, max=200, step=1).classes('w-full')
                ui.button("ALT 분석 실행", on_click=run_alt_analysis).props('icon=analytics').classes('q-mt-md')
                alt_result = ui.column().classes('w-full')

        with ui.card().classes('w-full'):
            with ui.column():
                ui.label("필요 시료 수 계산기").classes('text-h5')
                rel_input = ui.number("목표 신뢰도 R (%)", value=90, min=1, max=99.9, step=0.1).classes('w-full')
                cl_input = ui.number("신뢰 수준 CL (%)", value=90, min=1, max=99.9, step=0.1).classes('w-full')
                r_input = ui.number("허용 고장 수 r", value=0, min=0, step=1).classes('w-full')
                ui.button("시료 수 계산", on_click=run_sample_size).props('icon=calculate').classes('q-mt-md')
                sample_result = ui.column().classes('w-full')

def _read_excel_bytes(data_bytes: bytes) -> pd.DataFrame:
    try:
        raw = pd.read_excel(io.BytesIO(data_bytes), header=None)
    except ImportError as err:
        raise ValueError("엑셀 파일 읽기를 위해 openpyxl이 필요합니다.") from err

    header_row = _find_header_row(raw)
    if header_row is not None:
        df = raw.iloc[header_row + 1 :].copy()
        df.columns = [str(col).strip() for col in raw.iloc[header_row].tolist()]
    else:
        df = pd.read_excel(io.BytesIO(data_bytes))

    df = df.dropna(axis=0, how="all").dropna(axis=1, how="all")
    df.columns = [str(col).strip() for col in df.columns]
    return df

def _find_header_row(raw: pd.DataFrame) -> int | None:
    max_rows = min(10, len(raw))
    for idx in range(max_rows):
        row = raw.iloc[idx].astype(str).str.strip()
        if row.str.contains("판매", regex=False).any():
            return idx
        if row.str.contains("Sales", case=False, regex=False).any():
            return idx
        if row.str.contains("행", regex=False).any():
            return idx
    return None

def _parse_sales_month(value: Any) -> pd.Period | None:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, pd.Period):
        return value.asfreq("M")
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return pd.Period(value, freq="M")
    if isinstance(value, (int, np.integer)):
        text = str(int(value))
        if len(text) == 6:
            return pd.Period(f"{text[:4]}-{text[4:]}", freq="M")
        return pd.Period(text, freq="M")
    if isinstance(value, (float, np.floating)):
        if not math.isfinite(float(value)):
            return None
        text = str(int(round(float(value))))
        if len(text) == 6:
            return pd.Period(f"{text[:4]}-{text[4:]}", freq="M")
        return pd.Period(text, freq="M")
    text = str(value).strip()
    try:
        return pd.Period(text, freq="M")
    except Exception:
        try:
            return pd.Period(pd.to_datetime(text), freq="M")
        except Exception:
            return None

def _select_nevada_columns(df: pd.DataFrame) -> tuple[Any, Any, list[Any]]:
    columns = [c for c in df.columns if not str(c).strip().lower().startswith("unnamed")]
    if len(columns) < 3:
        columns = list(df.columns)
    if not columns:
        raise ValueError("엑셀 컬럼을 찾지 못했습니다.")

    sales_candidates = [
        c for c in columns if any(k in str(c) for k in ["판매월", "Sales Month", "행 레이블", "행레이블", "Row"])
    ]
    sales_col = sales_candidates[0] if sales_candidates else columns[0]

    qty_candidates = [
        c for c in columns if any(k in str(c) for k in ["판매대수", "판매 수량", "판매수량", "Quantity", "Qty", "대수", "수량"])
    ]
    qty_col = qty_candidates[0] if qty_candidates else (columns[1] if len(columns) > 1 else columns[0])

    if sales_col == qty_col and len(columns) > 1:
        qty_col = columns[1]

    failure_cols = [c for c in columns if c not in {sales_col, qty_col}]
    if not failure_cols:
        raise ValueError("고장 데이터 컬럼을 찾지 못했습니다.")
    return sales_col, qty_col, failure_cols

def _coerce_count(value: Any) -> int:
    if value is None or pd.isna(value):
        return 0
    try:
        number = float(value)
        if math.isnan(number) or math.isinf(number):
            return 0
        count = int(round(number))
        return max(count, 0)
    except (TypeError, ValueError):
        return 0

def _build_nevada_triangle(df: pd.DataFrame, sales_col: Any, failure_cols: list[Any], column_labels: list[str]) -> pd.DataFrame:
    triangle = df[[sales_col] + failure_cols].copy()
    triangle = triangle.dropna(subset=[sales_col])
    triangle[sales_col] = triangle[sales_col].apply(_parse_sales_month)
    triangle = triangle[triangle[sales_col].notna()]
    triangle[sales_col] = triangle[sales_col].apply(lambda p: str(p))
    triangle = triangle.set_index(sales_col)
    triangle = triangle.apply(pd.to_numeric, errors="coerce")
    triangle.columns = column_labels
    return triangle


def plot_nevada_heatmap(triangle: pd.DataFrame) -> None:
    data = triangle.to_numpy(dtype=float)
    masked = np.ma.masked_invalid(data)
    cmap = plt.cm.viridis
    cmap.set_bad(color="#3a3a3a")
    plt.imshow(masked, aspect="auto", interpolation="nearest")
    plt.colorbar(label="고장 수량")
    plt.xticks(range(len(triangle.columns)), triangle.columns, rotation=45, ha="right")
    plt.yticks(range(len(triangle.index)), triangle.index)
    plt.title("Nevada Chart Heatmap")

def plot_cumulative_failure(dist: Any, failures: np.ndarray, total_units: int, max_time: float) -> None:
    if max_time <= 0:
        max_time = 1
    t = np.linspace(0, max_time, 200)
    plt.plot(t, dist.CDF(t), label="Weibull 추정")
    if failures.size > 0:
        failures_sorted = np.sort(failures)
        cum_rate = np.arange(1, failures_sorted.size + 1) / total_units
        plt.step(failures_sorted, cum_rate, where="post", label="실측 누적고장률")
    plt.xlabel("경과 개월")
    plt.ylabel("누적 고장률")
    plt.grid(True, alpha=0.3)
    plt.legend()

def _month_diff(start: pd.Period, end: pd.Period) -> int:
    return (end.year - start.year) * 12 + (end.month - start.month)

def _weibull_cdf(t: np.ndarray | float, beta: float, eta: float) -> np.ndarray | float:
    t_arr = np.asarray(t, dtype=float)
    return 1 - np.exp(-((t_arr / eta) ** beta))

def _weibull_time_at_reliability(beta: float, eta: float, reliability: float) -> float:
    return eta * (-math.log(reliability)) ** (1 / beta)

def _prepare_warranty_calendar_data(
    df: pd.DataFrame,
    cutoff_month: pd.Period | None,
    t_cohort_mode: str,
) -> WarrantyAnalysisData:
    sales_col, qty_col, claim_cols = _select_nevada_columns(df)

    claim_periods: dict[Any, pd.Period] = {}
    for col in claim_cols:
        period = _parse_sales_month(col)
        if period is not None:
            claim_periods[col] = period
    if not claim_periods:
        raise ValueError("클레임 월 컬럼(YYYY-MM/ YYYYMM)을 인식하지 못했습니다.")

    ordered_claim_cols = [col for col, _ in sorted(claim_periods.items(), key=lambda item: item[1])]
    cutoff = cutoff_month or max(claim_periods.values())

    triangle = _build_nevada_triangle(df, sales_col, ordered_claim_cols, [str(claim_periods[c]) for c in ordered_claim_cols])

    total_sales = 0
    total_claims = 0
    d_by_age: dict[int, int] = {}
    cohort_censored: list[tuple[int, int]] = []

    for _, row in df.iterrows():
        sale_period = _parse_sales_month(row.get(sales_col))
        if sale_period is None:
            continue
        sales_qty = _coerce_count(row.get(qty_col))
        if sales_qty <= 0:
            continue
        total_sales += sales_qty

        row_claims = 0
        for col in ordered_claim_cols:
            claim_period = claim_periods[col]
            age = _month_diff(sale_period, claim_period)
            if age < 0:
                continue
            val = pd.to_numeric(row.get(col, np.nan), errors="coerce")
            count = _coerce_count(val)
            if count > 0:
                d_by_age[age] = d_by_age.get(age, 0) + count
                row_claims += count

        total_claims += row_claims
        censored = sales_qty - row_claims
        if censored > 0:
            amax = _month_diff(sale_period, cutoff)
            if amax < 0:
                amax = 0
            T_cohort = amax if t_cohort_mode == "amax" else amax + 1
            cohort_censored.append((int(censored), int(T_cohort)))

    if total_sales <= 0:
        raise ValueError("총 판매 수량이 0입니다.")
    if total_claims <= 0:
        raise ValueError("관측된 클레임이 없습니다.")

    return WarrantyAnalysisData(
        total_sales=int(total_sales),
        total_claims=int(total_claims),
        cutoff_month=cutoff,
        d_by_age=d_by_age,
        cohort_censored=cohort_censored,
        triangle=triangle,
    )

def _initial_weibull_guess(data: WarrantyAnalysisData) -> tuple[float, float]:
    total_failures = sum(data.d_by_age.values())
    if total_failures <= 0:
        return 1.5, 1.0
    mean_age = sum((age + 0.5) * count for age, count in data.d_by_age.items()) / total_failures
    beta0 = 1.5
    eta0 = max(mean_age / math.gamma(1 + 1 / beta0), 1.0)
    return beta0, eta0

def _neg_log_likelihood(params: np.ndarray, data: WarrantyAnalysisData) -> float:
    log_beta, log_eta = params
    beta = math.exp(log_beta)
    eta = math.exp(log_eta)
    eps = 1e-12
    ll = 0.0
    for age, count in data.d_by_age.items():
        p = _weibull_cdf(age + 1, beta, eta) - _weibull_cdf(age, beta, eta)
        p = max(float(p), eps)
        ll += count * math.log(p)
    for censored_count, T_cohort in data.cohort_censored:
        sf = 1 - float(_weibull_cdf(T_cohort, beta, eta))
        sf = max(sf, eps)
        ll += censored_count * math.log(sf)
    return -ll

def _fit_weibull_mle(data: WarrantyAnalysisData) -> tuple[float, float, float]:
    beta0, eta0 = _initial_weibull_guess(data)
    result = opt.minimize(
        _neg_log_likelihood,
        x0=np.array([math.log(beta0), math.log(eta0)], dtype=float),
        args=(data,),
        method="L-BFGS-B",
    )
    if not result.success:
        raise ValueError(f"MLE 최적화 실패: {result.message}")
    beta = math.exp(result.x[0])
    eta = math.exp(result.x[1])
    loglik = -float(result.fun)
    return beta, eta, loglik

def _probability_plot_data(
    data: WarrantyAnalysisData,
    beta: float,
    eta: float,
    p_method: str,
    t_offset: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ages = sorted(data.d_by_age.keys())
    if not ages:
        return np.array([]), np.array([]), np.array([]), np.array([])
    x_pts: list[float] = []
    y_pts: list[float] = []
    d_cum = 0
    for age in ages:
        d_cum += data.d_by_age[age]
        if p_method == "median":
            p = (d_cum - 0.3) / (data.total_sales + 0.4)
        else:
            p = d_cum / data.total_sales
        p = min(max(p, 1e-6), 1 - 1e-6)
        t = age + t_offset
        x_pts.append(math.log(t))
        y_pts.append(math.log(-math.log(1 - p)))

    max_age = max(ages)
    t_line = np.linspace(t_offset, max_age + 1, 200)
    p_line = _weibull_cdf(t_line, beta, eta)
    p_line = np.clip(p_line, 1e-6, 1 - 1e-6)
    x_line = np.log(t_line)
    y_line = np.log(-np.log(1 - p_line))
    return np.asarray(x_pts), np.asarray(y_pts), x_line, y_line

def plot_nevada_probability_plot(
    data: WarrantyAnalysisData,
    beta: float,
    eta: float,
    p_method: str,
    t_offset: float,
    ax: plt.Axes | None = None,
) -> None:
    ax = ax or plt.gca()
    x_pts, y_pts, x_line, y_line = _probability_plot_data(data, beta, eta, p_method, t_offset)
    ax.plot(x_line, y_line, label="Weibull 적합선")
    ax.scatter(x_pts, y_pts, color="black", s=20, label="관측점")
    ax.set_xlabel("ln(t)")
    ax.set_ylabel("ln(-ln(1-p))")
    ax.set_title("Nevada Chart (Weibull Probability Plot)")
    ax.grid(True, which="both", linestyle="--", alpha=0.6)
    ax.legend()

def _forecast_table(beta: float, eta: float, n_at_risk: int, months: int) -> pd.DataFrame:
    rows = []
    for k in range(1, months + 1):
        p = float(_weibull_cdf(k, beta, eta) - _weibull_cdf(k - 1, beta, eta))
        expected = max(n_at_risk * p, 0.0)
        low = float(poisson.ppf(0.025, expected))
        high = float(poisson.ppf(0.975, expected))
        rows.append(
            {
                "월(k)": k,
                "예상 클레임": round(expected, 2),
                "CI 2.5%": low,
                "CI 97.5%": high,
            }
        )
    return pd.DataFrame(rows)

def _summary_report(
    data: WarrantyAnalysisData,
    beta: float,
    eta: float,
    t95: float,
    t99: float,
    loglik: float,
    n_at_risk: int,
) -> str:
    lines = [
        f"총 판매대수: {data.total_sales}",
        f"총 관측 클레임: {data.total_claims}",
        f"컷오프 월: {data.cutoff_month}",
        f"잔존대수(예측 기준): {n_at_risk}",
        f"Weibull β: {beta:.4f}",
        f"Weibull η: {eta:.2f}",
        f"t95: {t95:.2f} 개월",
        f"t99: {t99:.2f} 개월",
        f"log-likelihood: {loglik:.2f}",
    ]
    return "\n".join(f"- {line}" for line in lines)

def render_warranty_analysis_tab():
    class WarrantyState:
        df: pd.DataFrame | None = None
        filename: str | None = None

    state = WarrantyState()
    heatmap_container: ui.column | None = None
    result_container: ui.column | None = None

    async def _get_upload_bytes(e: Any) -> tuple[bytes, str]:
        if hasattr(e, "file"):
            file_obj = e.file
            data = await file_obj.read()
            name = getattr(file_obj, "name", None) or getattr(e, "name", None) or "업로드된 파일"
            return data, name
        content = getattr(e, "content", None)
        name = getattr(e, "name", None) or "업로드된 파일"
        if content is None:
            raise ValueError("업로드 파일 내용을 찾을 수 없습니다.")
        if isinstance(content, (bytes, bytearray)):
            return bytes(content), name
        if hasattr(content, "read"):
            content.seek(0)
            data = content.read()
            if isinstance(data, str):
                data = data.encode("utf-8")
            return data, name
        raise ValueError("지원하지 않는 업로드 데이터 형식입니다.")

    async def handle_warranty_upload(e: Any):
        try:
            data_bytes, filename = await _get_upload_bytes(e)
            df = _read_excel_bytes(data_bytes)
            state.df = df
            state.filename = filename
            ui.notify(f"엑셀 파일 '{filename}' 로드 성공.", type='positive')
        except Exception as err:
            ui.notify(f"엑셀 파일 처리 중 오류: {err}", type='negative')
            logging.error("엑셀 파일 처리 오류: %s", err)

    def run_warranty_analysis():
        if heatmap_container is None or result_container is None:
            return
        heatmap_container.clear()
        result_container.clear()
        if state.df is None:
            ui.notify("엑셀 파일을 업로드해 주세요.", type='warning')
            return

        try:
            cutoff_override = _parse_sales_month(cutoff_input.value)
            t_mode = "amax" if t_cohort_mode.value.startswith("amax") else "amax+1"
            warranty_data = _prepare_warranty_calendar_data(state.df, cutoff_override, t_mode)
        except Exception as err:
            ui.notify(f"데이터 변환 중 오류: {err}", type='negative')
            return

        with heatmap_container:
            ui.label("네바다 차트 히트맵").classes('text-h6')
            with ui.pyplot(figsize=(7.4, 4.2)):
                plot_nevada_heatmap(warranty_data.triangle)

        try:
            beta, eta, loglik = _fit_weibull_mle(warranty_data)
        except Exception as err:
            ui.notify(f"Weibull MLE 실패: {err}", type='negative')
            return

        horizon = int(forecast_months.value or 12)
        horizon = max(horizon, 1)
        t_offset = 0.5 if t_offset_mode.value.startswith("age+0.5") else 1.0
        p_method = "median" if p_method_mode.value.startswith("Median") else "cumulative"

        t95 = _weibull_time_at_reliability(beta, eta, 0.95)
        t99 = _weibull_time_at_reliability(beta, eta, 0.99)
        n_at_risk = sum(count for count, _ in warranty_data.cohort_censored)

        forecast_df = _forecast_table(beta, eta, n_at_risk, horizon)
        summary_md = _summary_report(warranty_data, beta, eta, t95, t99, loglik, n_at_risk)

        fig, ax = plt.subplots(figsize=(7.0, 4.2))
        plot_nevada_probability_plot(warranty_data, beta, eta, p_method, t_offset, ax=ax)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
        plt.close(fig)
        png_bytes = buf.getvalue()

        with heatmap_container:
            ui.separator()
            ui.label("요약 리포트").classes('text-subtitle2 q-mt-md')
            ui.markdown(summary_md)

        with result_container:
            ui.label("요약 리포트").classes('text-h6 q-mt-sm')
            ui.markdown(summary_md)

            csv_bytes = forecast_df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
            with ui.row():
                ui.button(
                    "예측표 다운로드 (CSV)",
                    on_click=lambda: app.download(csv_bytes, "warranty_forecast.csv"),
                ).props('icon=download')
                ui.button(
                    "네바다 차트 다운로드 (PNG)",
                    on_click=lambda: app.download(png_bytes, "nevada_chart.png"),
                ).props('icon=download')

            ui.label("월별 클레임 예측표 (95% CI)").classes('text-subtitle2 q-mt-md')
            ui.table.from_pandas(forecast_df).classes('w-full')

            ui.label("네바다 차트 (Weibull Probability Plot)").classes('text-subtitle2 q-mt-md')
            with ui.pyplot(figsize=(7.4, 4.2)):
                plot_nevada_probability_plot(warranty_data, beta, eta, p_method, t_offset)

    with ui.grid(columns=2).classes('w-full'):
        with ui.card().classes('w-full'):
            with ui.column():
                ui.label("보증 데이터 분석 (Nevada Chart)").classes('text-h5')
                ui.label("A열: 판매월, B열: 판매수량, C열~: 클레임 월(YYYY-MM/ YYYYMM)").classes('text-body2')
                ui.upload(on_upload=handle_warranty_upload, auto_upload=True, max_files=1).props('accept=.xlsx,.xls,.xlsm').classes('w-full')
                cutoff_input = ui.input("컷오프 월 (YYYY-MM, 비워두면 자동)")
                t_cohort_mode = ui.select(
                    ["amax+1 (기본)", "amax"],
                    value="amax+1 (기본)",
                    label="T_cohort 설정",
                ).classes('w-full')
                p_method_mode = ui.select(
                    ["누적 실패율 (기본)", "Median Rank"],
                    value="누적 실패율 (기본)",
                    label="관측 p 계산",
                ).classes('w-full')
                t_offset_mode = ui.select(
                    ["age+0.5 (기본)", "age+1"],
                    value="age+0.5 (기본)",
                    label="t 계산 방식",
                ).classes('w-full')
                forecast_months = ui.number("예측 기간 (개월)", value=12, min=1, max=60, step=1).classes('w-full')
                ui.button("보증 분석 실행", on_click=run_warranty_analysis).props('icon=analytics').classes('q-mt-md')
                heatmap_container = ui.column().classes('w-full')

        with ui.card().classes('w-full'):
            result_container = ui.column().classes('w-full')

def render_schedule_planner():
    result_container = ui.column().classes('w-full')
    
    with ui.card().classes('w-full'):
        with ui.column():
            ui.label("시험 일정 생성").classes('text-h5')
            ui.label("교차 시험 순서, 주말/공휴일 보정, 병행 규칙을 반영해 시험 일정을 자동으로 계산합니다.").classes('text-body1 q-mb-md')
            
            ui.label("시작일")
            start_date = ui.date(value=date(2025, 12, 15))
            
            ui.label("주말/공휴일 보정 방식")
            correction_label = ui.select(["다음 월요일(A 방식)", "다음 평일"], value="다음 월요일(A 방식)")
            
            holiday_text = ui.textarea("공휴일 입력 (YYYY-MM-DD, 줄/쉼표 구분)", placeholder="예) 2025-12-25, 2026-01-01")
            
            ui.label("작동 이음 평가 배치")
            standalone_position = ui.select(["자동(최소 종료일)", "시작(다른 시험 전)", "종료(다른 시험 후)"], value="자동(최소 종료일)")
            
            ui.label("날짜 표기")
            date_format = ui.select(["YY.MM.DD.", "YYYY-MM-DD"], value="YY.MM.DD.")
            
            ui.label("소요일 표기")
            duration_format = ui.select(["D+N", "N일"], value="D+N")
        
            with ui.row():
                include_no = ui.checkbox("No. 열 포함", value=False)
                include_notes = ui.checkbox("비고 열 포함", value=False)

    def on_calculate():
        result_container.clear()
        holidays, invalid = _parse_holidays(holiday_text.value)
        if invalid:
            ui.notify("형식이 잘못된 공휴일 입력: " + ", ".join(invalid), type='warning')

        correction_mode = "monday" if correction_label.value.startswith("다음 월요일") else "weekday"
        standalone_key = {"자동(최소 종료일)": "auto", "시작(다른 시험 전)": "start", "종료(다른 시험 후)": "end"}[standalone_position.value]

        occurrences, adjusted_start, final_end, standalone_start, _ = _schedule_with_standalone(
            datetime.strptime(start_date.value, '%Y-%m-%d').date(), holidays, correction_mode, standalone_key
        )
        aggregated = _aggregate_occurrences(occurrences)
        for test_name in UNSCHEDULED_TESTS:
            aggregated.setdefault(test_name, {})

        section_tables, combined_df = _build_section_tables(
            aggregated=aggregated, date_format=date_format.value, duration_format=duration_format.value,
            include_no=include_no.value, include_notes=include_notes.value
        )
        
        with result_container:
            ui.notify(f"보정된 시작일: {_format_date(adjusted_start, date_format.value)} | 최종 종료일: {_format_date(final_end, date_format.value)}", type='info')
            
            for section_name, df in section_tables:
                ui.label(section_name).classes('text-h6 q-mt-md')
                ui.table.from_pandas(df.reset_index(drop=True)).classes('w-full')

            csv_data = combined_df.to_csv(index=False, encoding="utf-8-sig")
            ui.button("일정표 다운로드 (CSV)", on_click=lambda: app.download(csv_data.encode('utf-8-sig'), 'test_schedule.csv')).props('icon=download')

    ui.button("일정 계산", on_click=on_calculate).props('icon=event').classes('q-mt-md')


ui.run(title="내구 수명 분석 도구")
