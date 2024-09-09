# SW_AIchallenge
## 팀 구성
**팀장** [전지훈](https://github.com/Chun-Jihun)   
**팀원** [서상혁](https://github.com/tiemhub)   
**팀원** [이강혁](https://github.com/KH4901)   
**팀원** [허홍준](https://github.com/HongJuneHu)   

## 목차
1. [개요](#개요)   
2. [학습 데이터셋](#학습-데이터셋)   
3. [아키텍쳐](#아키텍쳐)   
4. [사용 패키지](#사용-패키지)   
5. [Prompt Engineering](#Prompt-Engineering)   
6. [사용법](#사용법)   
7. [출력 예시](#출력예시)   
   
## 개요
### 연세대학교 미래캠퍼스 디지털헬스케어사업단 생성 AI 챌린지
고령 만성질환자(고혈압, 당뇨, 심장질환) 대상의 챗봇 서비스 개발
![image](https://github.com/user-attachments/assets/1bf93724-dc6f-40d5-80cf-12c5dc17d0a9)

## 학습 데이터셋
8명의 데이터로 구성되어 있으며 고혈압, 당뇨, 비만 등 여러 시나리오를 고려하여 임의로 제작한 데이터이다. 4개의 csv파일(patients, vital, reports, inbody)로 구성되어있고 각 환자는 0~7의 id로 부여되어있다.
* patients.csv   
1인당 1개의 record, 성별과 출생년도가 기록되어있다.   
* vital.csv   
1인당 5개의 record, 측정시간, 혈압약 복용여부, 공복여부, 수축기 혈압, 이완기 혈압, 혈중 산소포화도, 체온, 몸무게, 키, 혈당, 맥박수 정보가 기록되어있다.   
* inbody.csv   
1인당 2~6개의 record, 측정일, 측정시간, 체지방량, 체수분, 제지방량, 골격근량, 근육량, 체질량지수, 체지방률, 복부지방률, 골무기질량, 체세포량, 기초대사량, 내장지방레벨이 기록되어있다.   
* reports.csv   
설문지에 대한 답변이 기록되어있다.

## 아키텍쳐
![image](https://github.com/user-attachments/assets/7f69a2a6-5aee-4893-b7d3-592bf495c43b)

## 사용 패키지
### Langchain
### RAG([Retrieval-Augmented Generation](https://arxiv.org/pdf/2005.11401))
![image](https://github.com/user-attachments/assets/7f8aad9d-49ae-4cea-8be8-09c1eb9a8ee1)
![image](https://github.com/user-attachments/assets/09665547-5abe-4be8-a2f4-2139665bda3b)
#### 작동원리
1. 도큐멘트 검색: 질문에 관련된 정보를 실시간으로 검색합니다.
2. 텍스트 생성: 검색된 정보를 활용해 답변을 생성합니다.

#### 장점
- 높은 정확성: 검색된 정보를 활용해 더 정확하고 관련성 높은 답변을 제공합니다.
- 파라메틱 메모리 한계 극복: LLM은 학습된 고정된 지식에 의존하지만, RAG는 외부 정보 검색을 통해 최신 정보와 지식을 반영함으로써 LLM의 고정된 지식 한계를 보완합니다.
- 정보 기반 생성: 실시간으로 검색된 정보와 LLM의 지식을 결합해 신뢰성 있는 답변을 제공합니다.

## Prompt Engineering
![image](https://github.com/user-attachments/assets/a660488c-aeda-4422-a199-ecf098400644)

## 사용법
### installation
```
conda create -n medical_chat python=3.11
git clone https://github.com/Chun-Jihun/SW_AIchallenge.git
conda activate medical_chat
cd SW_AIchallenge
pip install -r requirements.txt
```
### usage
```
python main.py
```

## 출력예시
```
환자의 건강 정보를 분석한 결과는 다음과 같습니다.

1. 건강 상태 분석 :
   - 환자는 64세 여성으로, 당뇨병 전단계와 고혈압이 우려됩니다. 체지방률이 증가하고 있으며, 운동 부족이 관찰됩니다.

2. 주요 건강 문제 :
   - 혈당 변동성이 크고, 당뇨병 전단계 가능성이 있습니다.
   - 고혈압 약물 복용이 불규칙하여 혈압 조절이 불안정합니다.
   - 체지방률이 높아 심혈관 질환 위험이 증가합니다.

3. 건강 관리 주의사항 :
   - 정기적인 혈당 및 혈압 측정을 권장합니다.
   - 규칙적인 운동을 통해 체중과 체지방률을 관리해야 합니다.
   - 고혈압 약물 복용을 지속적으로 관리하고, 의사와 상담이 필요합니다.

이 정보를 바탕으로, 환자는 의료 제공자와 상담하여 추가적인 조치를 취하는 것이 중요합니다.
```

