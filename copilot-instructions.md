# Copilot Instructions for Durability Analysis Project

## 프로젝트 개요
- 내구력 시험 데이터 분석 및 보고서 자동화 도구
- 주요 UI: NiceGUI 기반 웹앱, Streamlit 기반 대시보드, Gemini 일정 계산기
- 데이터 분석, 통계 모델링(Weibull, Lognormal 등), PDF 리포트 생성

## 주요 컴포넌트
- `nicegui_app.py`, `새 폴더/nicegui_app.py`: NiceGUI로 내구력 데이터 업로드, 분포 추정, PDF 리포트 생성
- `streamlit_app.py`, `새 폴더/streamlit_app.py`: Streamlit 대시보드, 데이터 시각화 및 통계 분석
- `gemini/app.py`: 시험 일정 계산, 휴일/근무일 로직 포함
- `results/`: 분석 로그(`analysis_log.txt`), 최종 리포트(`final_report.pdf`)

## 데이터 흐름 및 통합
- 데이터 업로드 → 분포 추정(Weibull 등) → 시각화/리포트(PDF)
- 공통 상수/라벨(`DATA_COLUMN_NAME`, `DISTRIBUTION_LABELS`, `PARAM_LABELS`)은 각 앱에서 동일하게 사용
- 분석 결과는 PDF로 저장, 로그는 `results/`에 기록

## 개발 워크플로우
- **실행**: `python nicegui_app.py` 또는 `streamlit run streamlit_app.py`
- **PDF 리포트**: `matplotlib.backends.backend_pdf.PdfPages` 활용
- **로그**: `results/analysis_log.txt`에 기록
- **휴일/근무일 계산**: `gemini/app.py`의 `KR_HOLIDAYS`, `is_workday`, `get_next_workday` 참고

## 프로젝트별 관례
- 한글 라벨 및 주석 사용(분포, 파라미터 등)
- 타입 체크 시 `TYPE_CHECKING` 분기, 런타임 fallback 패턴
- 업로드 파일 타입: NiceGUI/Streamlit별로 다름, 타입 분기 처리
- 분포/파라미터 라벨은 딕셔너리로 관리, UI에 직접 사용

## 외부 의존성
- `matplotlib`, `numpy`, `pandas`, `reliability`, `nicegui`, `streamlit`
- PDF 생성: `matplotlib.backends.backend_pdf.PdfPages`

## 예시 패턴
```python
# 분포 라벨 관리
DISTRIBUTION_LABELS = {
    "Weibull_2P": "와이불 2모수",
    ...
}
# PDF 저장
with PdfPages(filename) as pdf:
    pdf.savefig(fig)
```

## 참고 파일
- 주요 로직: `nicegui_app.py`, `streamlit_app.py`, `gemini/app.py`
- 결과물: `results/analysis_log.txt`, `results/final_report.pdf`

---
이 문서가 불명확하거나 추가할 내용이 있다면 피드백을 남겨주세요.
