# Thyroid Cancer Diagnosis Classification  
**갑상선암 진단 분류 AI 모델 개발 프로젝트**

---

## 🩺 개요

본 프로젝트는 **데이콘(DACON)** 에서 주관한 **'갑상선암 진단 분류 해커톤'** 참가를 통해 진행된 AI 분류 모델 개발 과제입니다.

갑상선암은 조기 진단이 중요한 질환으로, 양성과 악성을 빠르게 구분하는 것이 환자의 치료 방향 결정과 예후 개선에 매우 중요합니다.  
이번 프로젝트에서는 갑상선 관련 정형 데이터를 바탕으로, **양성(Benign)** 과 **악성(Malignant)** 여부를 판별하는 **이진 분류 모델**을 개발합니다.

---

## 🎯 주제

> 갑상선 관련 건강 데이터를 기반으로  
> **양성인지 악성인지 분류하는 AI 모델을 개발하라!**

---

## 🧠 배경

AI 기술은 의료 분야에서 점점 더 중요해지고 있으며, 특히 머신러닝을 활용한 **질병 조기 감지 및 예측 기술**은 진단 정확도를 높이고 빠른 의사결정을 돕는 핵심 도구로 떠오르고 있습니다.

이번 해커톤은 실제 갑상선 관련 데이터를 기반으로 하여, **정확한 이진 분류 모델을 설계**하는 데 목표를 두고 있습니다.  
이를 통해 참가자들은 **의료 데이터 분석 경험**과 **모델링 실전 역량**을 동시에 키울 수 있습니다.

---

## 📌 대회 정보

- **대회명**: 갑상선암 진단 분류 해커톤 : 양성과 악성, AI로 정확히 구분하라!
- **주최/주관**: DACON (데이콘)
- **대회 기간**: 2025.05.07 ~ 2025.06.30 09:59
- **참가 대상**: 데이콘 회원 누구나 (입문자 대상 해커톤)
- **평가 지표**: F1 Score
- **상금**: 데이스쿨 Pro 구독권
- **링크**: [Overview](https://dacon.io/competitions/official/236488/overview/description)

> 본 해커톤은 교육 목적으로 기획된 비공식 대회로,  
> 이력 노출 및 랭킹 포인트는 반영되지 않습니다.

---

## 📂 프로젝트 구조 예시

```bash
ThyroidCancerDiagnosis/
├── data/ # 원본 데이터 (GitHub에 업로드 X)
├── notebooks/ # EDA, 모델 실험용 Jupyter 노트북
├── src/ # 학습 및 추론 코드 (Python)
├── models/ # 학습된 모델 파일
├── results/ # 예측 결과 및 그래프
├── requirements.txt # 설치 패키지 목록
└── README.md # 프로젝트 설명 문서
```

---

## ⚙️ 실행 방법

### 1. 환경설정

```bash
pip install -r requirements.txt
```

### 2. 데이터 다운로드 및 준비

- [datasets](https://dacon.io/competitions/official/236488/data) 데이콘 대회 페이지에서 데이터셋 다운로드

- 'data/' 폴더에 데이터 압축 해제 및 저장 (GitHub에는 포함하지 않음)

### 3. 모델 학습

```bash
python src/train.py
```

### 4. 예측 및 평가

```bash
python src/predict.py
```

---

## 📈 기술 스택

- Python, Pandas, NumPy

- Scikit-learn, XGBoost, LightGBM

- Matplotlib, Seaborn

- Jupyter Notebook

- Git, VS Code