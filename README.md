# SW_AIchallenge
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
![image](https://github.com/user-attachments/assets/0c2060e6-457b-40b4-bc71-fdaff08e5f20)
