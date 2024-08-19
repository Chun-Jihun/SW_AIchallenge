import pandas as pd
import numpy as np
import os

def preprocess():
    # CSV 파일 경로
    inbody_file_path = './data/inbody.csv'
    patients_file_path = './data/patients.csv'
    reports_file_path = './data/reports.xlsx'
    vital_file_path = './data/vital.csv'

    # CSV 읽어오기
    inbody_df = pd.read_csv(inbody_file_path, encoding='cp949')
    patients_df = pd.read_csv(patients_file_path, encoding='cp949')
    report_df = pd.read_excel(reports_file_path)
    vital_df = pd.read_csv(vital_file_path, encoding='cp949')

    inbody_columns = {
        'date': '측정일자',
        'time': '측정시간',
        'body_fat_mass': '체지방량(Kg)',
        'total_body_water': '체수분(L)',
        'fat_free_mass': '제지방량(Kg)',
        'skeletal_muscle_mass': '골격근량(Kg)',
        'soft_lean_mass': '근육량(Kg)',
        'bmi': '체질량지수(kg/m^2)',
        'percent_body_fat': '체지방률(%)',
        'waist_hip_ratio': '복부지방률(허리둘레/엉덩이둘레)',
        'bmc': '골무기질량(Kg)',
        'bcm': '체세포량(Kg)',
        'bmr': '기초대사량(kcal)',
        'visceral_fat_level': '내장지방레벨'

    }
    patients_columns = {
        'pat_sex': '성별',
        'pat_birth': '출생년도'
    }
    report_columns = {
        'time': '측정일자',
        'report.disease_parent.stroke': '환자부모_뇌졸중',
        'report.disease_parent.ami': '환자부모_심근경색',
        'report.disease_parent.hbp': '환자부모_고혈압',
        'report.disease_parent.dm': '환자부모_당뇨병',
        'report.disease_parent.etc': '환자부모_기타(암_포함)',
        'report.disease_history.stroke': '환자_뇌졸중',
        'report.disease_history.ami': '환자_심근경색',
        'report.disease_history.hbp': '환자_고혈압',
        'report.disease_history.dm': '환자_당뇨병',
        'report.disease_history.etc': '환자_기타(암_포함)',
        'report.disease_history.covid': '환자_COVID-19(감염횟수)',
        'report.exercise.indoor': '실내운동',
        'report.exercise.outdoor': '실외운동',
        'report.exercise.favorite': '평소_즐겨하는_운동_종목',
        'report.exercise.exercise_hard_period': '숨_많이_차는_고강도_운동_시행하는_일주일_기준_일수',
        'report.exercise.exercise_time': '평균_하루_운동시간',
        'report.exercise.exercise_strength_period': '최근 1주일 동안 근력 운동한 일수',
        'report.smoking.1': '흡연여부',
        'report.smoking.2-1': '과거_흡연_년수',
        'report.smoking.2-2': '금연_전_주간_흡연일수',
        'report.smoking.2-3': '현재_흡연_년수',
        'report.smoking.2-4': '현재_흡연중이며_한_주_흡연량',
        'report.smoking.3': '전자담배_사용여부',
        'report.smoking.4': '전자담배_사용경험이_있으며_최근_한달_내_전자담배_사용여부',
        'report.drinking.habit.unit': '지난_1년간_음주_횟수_기준',
        'report.drinking.habit.count': '지난_1년간_마신_횟수',
        'report.drinking.usual.category': '술_종류',
        'report.drinking.usual.unit': '지난_1년간_음주량_단위',
        'report.drinking.usual.count': '술_종류와_음주량_단위_기준_평소_음주_횟수',
        'report.drinking.max.category': '지난_1년간_가장_많이_마셨을_때_술_종류',
        'report.drinking.max.unit': '지난_1년간_가장_많이_마셨을_때_음주량_단위',
        'report.drinking.max.count': '지난_1년간_가장_많이_마셨을_때_술_종류와_음주량_단위_기준_마신_횟수',
        'report.disease_set.1': '가슴과_관련된_증상',
        'report.disease_set.2': '근력과_관련된_증상',
        'report.disease_set.3': '기침과_관련된_증상',
        'report.disease_set.4': '두통과_관련된_증상',
        'report.disease_set.5': '소변과_관련된_증상',
        'report.disease_set.6': '식욕과_관련된_증상',
        'report.disease_set.7': '심장과_관련된_증상',
        'report.disease_set.8': '의식과_관련된_증상',
        'report.disease_set.9': '체중과_관련된_증상',
        'report.disease_set.10': '열과_관련된_증상',
        'report.disease_set.11': '피로감과_관련된_증상',
        'report.disease_set.12': '피부모양과_관련된_증상',
        'report.disease_set.13': '피부색과_관련된_증상',
        'report.disease_set.14': '호흡과_관련된_증상'
    }
    vital_columns = {
        'time': '측정일자',
        'antihypertensives': '혈압약_복용여부',
        'fasting': '공복여부',
        'sbp': '수축기_혈압(mmHg)',
        'dbp': '이완기_혈압(mmHg)',
        'spo2': '혈중_산소포화도(%)',
        'temp': '체온(°C)',
        'weight': '몸무게(Kg)',
        'height': '키(cm)',
        'glucose': '혈당',
        'pulse': '맥박수(1분_기준_횟수)'
    }
    inbody_df.rename(columns=inbody_columns, inplace=True)
    patients_df.rename(columns=patients_columns, inplace=True)
    report_df.rename(columns=report_columns, inplace=True)
    vital_df.rename(columns=vital_columns, inplace=True)

    # 특정 열에서 값 변경
    patients_df['출생년도'] = patients_df['출생년도'].replace({'F': '여성', 'M': '남성'})
    vital_df['혈압약_복용여부'] = vital_df['혈압약_복용여부'].replace({0: '미복용', 1: '복용'})
    vital_df['공복여부'] = vital_df['공복여부'].replace({0: '공복아님', 1: '공복'})

    bool_replace = {False: '없음', True: '있음'}
    report_df['환자부모_뇌졸중'] = report_df['환자부모_뇌졸중'].replace(bool_replace)
    report_df['환자부모_심근경색'] = report_df['환자부모_심근경색'].replace(bool_replace)
    report_df['환자부모_고혈압'] = report_df['환자부모_고혈압'].replace(bool_replace)
    report_df['환자부모_당뇨병'] = report_df['환자부모_당뇨병'].replace(bool_replace)
    report_df['환자부모_기타(암_포함)'] = report_df['환자부모_기타(암_포함)'].replace(bool_replace)
    report_df['환자_뇌졸중'] = report_df['환자_뇌졸중'].replace(bool_replace)
    report_df['환자_심근경색'] = report_df['환자_심근경색'].replace(bool_replace)
    report_df['환자_고혈압'] = report_df['환자_고혈압'].replace(bool_replace)
    report_df['환자_당뇨병'] = report_df['환자_당뇨병'].replace(bool_replace)
    report_df['환자_기타(암_포함)'] = report_df['환자_기타(암_포함)'].replace(bool_replace)
    report_df['실내운동'] = report_df['실내운동'].replace({False: '안함', True: '함'})
    report_df['실외운동'] = report_df['실외운동'].replace({False: '안함', True: '함'})
    report_df['평소_즐겨하는_운동_종목'] = report_df['평소_즐겨하는_운동_종목'].replace({np.nan: '없음'})
    report_df['숨_많이_차는_고강도_운동_시행하는_일주일_기준_일수'] = report_df['숨_많이_차는_고강도_운동_시행하는_일주일_기준_일수'].astype(str) + "일"
    report_df['평균_하루_운동시간'] = (report_df['평균_하루_운동시간'] * 60).astype(int).astype(str) + "분"
    report_df['최근 1주일 동안 근력 운동한 일수'] = report_df['최근 1주일 동안 근력 운동한 일수'].astype(str) + "일"

    report_df['흡연여부'] = report_df['흡연여부'].replace({1: '아니오', 2: '예, 지금은 끊었음', 3: '예, 현재도 흡연중'})
    report_df['금연_전_주간_흡연일수'] = report_df['금연_전_주간_흡연일수'].replace({0: '0일', 1: '1~2일', 2: '3~4일', 3: '매일'})
    report_df['현재_흡연중이며_한_주_흡연량'] = report_df['현재_흡연중이며_한_주_흡연량'].replace({0: '0일', 1: '1~2일', 2: '3~4일', 3: '매일'})
    report_df['전자담배_사용여부'] = report_df['전자담배_사용여부'].replace({0: '사용한 적 없음', 1: '사용한 적 있음', 2: '사용한 적 없음'})
    report_df['전자담배_사용경험이_있으며_최근_한달_내_전자담배_사용여부'] = report_df['전자담배_사용경험이_있으며_최근_한달_내_전자담배_사용여부'].replace(
        {0: '사용한 적 없음', 1: '사용한 적 없음', 2: '월 1~2일', 3: '월 3~9일', 4: '월 10~29일', 5: '매일'})

    # 두 열을 하나로 합치기
    # NaN 값을 0으로 대체한 후 정수로 변환
    report_df['술_종류와_음주량_단위_기준_평소_음주_횟수'] = report_df['술_종류와_음주량_단위_기준_평소_음주_횟수'].fillna(0).astype(int)
    report_df['지난_1년간_가장_많이_마셨을_때_술_종류와_음주량_단위_기준_마신_횟수'] = report_df['지난_1년간_가장_많이_마셨을_때_술_종류와_음주량_단위_기준_마신_횟수'].fillna(
        0).astype(int)

    # 두 열을 하나로 합치기 (숫자를 정수형으로 변환하여 결합)
    report_df['지난_1년간_음주_빈도'] = np.where(
        report_df['지난_1년간_음주_횟수_기준'].isin(["don't drink", '4번']),
        report_df['지난_1년간_음주_횟수_기준'],
        report_df['지난_1년간_음주_횟수_기준'] + ' 기준 ' + report_df['지난_1년간_마신_횟수'].astype(int).astype(str) + '번'
    )

    report_df['지난_1년간_평균_음주량'] = report_df['술_종류'] + report_df['술_종류와_음주량_단위_기준_평소_음주_횟수'].astype(int).astype(str) + \
                                 report_df['지난_1년간_음주량_단위'].astype(str)
    report_df['지난_1년간_최대_음주량'] = report_df['지난_1년간_가장_많이_마셨을_때_술_종류'] + report_df[
        '지난_1년간_가장_많이_마셨을_때_술_종류와_음주량_단위_기준_마신_횟수'].astype(int).astype(str) + report_df['지난_1년간_가장_많이_마셨을_때_음주량_단위'].astype(
        str)

    # 불필요한 열 삭제
    report_df.drop([
        '지난_1년간_음주_횟수_기준',
        '지난_1년간_마신_횟수',
        '술_종류',
        '지난_1년간_음주량_단위',
        '술_종류와_음주량_단위_기준_평소_음주_횟수',
        '지난_1년간_가장_많이_마셨을_때_술_종류',
        '지난_1년간_가장_많이_마셨을_때_음주량_단위',
        '지난_1년간_가장_많이_마셨을_때_술_종류와_음주량_단위_기준_마신_횟수'
    ], axis=1, inplace=True)


    # 매핑 함수 정의
    def map_symptoms_columns(df, column_mappings):
        def symptoms_to_text(symptoms, mapping):
            symptoms_text = []
            for num, text in mapping.items():
                if str(num) in str(symptoms):
                    symptoms_text.append(text)
            return ', '.join(symptoms_text)

        for column, mapping in column_mappings.items():
            df[column] = df[column].apply(lambda x: symptoms_to_text(x, mapping))

        return df


    # 각 열에 대한 매핑 사전 정의
    column_mappings = {
        '가슴과_관련된_증상': {
            1: '가슴이 아픕니다',
            2: '가슴이 답답합니다',
            3: '유방이 아픕니다',
            4: '해당 사항 없음'
        },
        '근력과_관련된_증상': {
            1: '근력이 감소하였습니다',
            2: '근력이 평소와 다릅니다',
            3: '움직일 때 힘듭니다',
            4: '해당 사항 없음'
        },
        '기침과_관련된_증상': {
            1: '기침을 자주합니다',
            2: '재채기가 심합니다',
            3: '해당 사항 없음'
        },
        '두통과_관련된_증상': {
            1: '머리가 어지럽습니다',
            2: '머리가 아픕니다',
            3: '해당 사항 없음'
        },
        '소변과_관련된_증상': {
            1: '평소에 소변을 자주 봅니다',
            2: '야간에 소변을 자주 봅니다',
            3: '소변량이 적습니다',
            4: '소변량이 많습니다',
            5: '해당 사항 없음'
        },
        '식욕과_관련된_증상': {
            1: '마실 것을 많이 마십니다',
            2: '음식을 먹는 것이 어렵습니다',
            3: '음식을 많이 먹습니다',
            4: '식욕이 없습니다',
            5: '해당 사항 없음'
        },
        '심장과_관련된_증상': {
            1: '가슴이 두근거립니다',
            2: '심박수가 불규칙합니다',
            3: '해당 사항 없음'
        },
        '의식과_관련된_증상': {
            1: '의식이 흐려졌습니다',
            2: '발작을 일으켰습니다',
            3: '기운이 없고 피곤합니다',
            4: '갑자기 쓰러졌습니다',
            5: '뇌졸중이 있습니다',
            6: '해당 사항 없음'
        },
        '체중과_관련된_증상': {
            1: '체중이 증가하였습니다',
            2: '체중이 감소하였습니다',
            3: '해당 사항 없음'
        },
        '열과_관련된_증상': {
            1: '피부가 뜨겁습니다',
            2: '열이 많이 납니다',
            3: '열이 조금 납니다',
            4: '몸이 으슬으슬합니다',
            5: '해당 사항 없음'
        },
        '피로감과_관련된_증상': {
            1: '아무것도 하기 싫습니다',
            2: '힘이 하나도 나지 않습니다',
            3: '피로가 계속 쌓입니다',
            4: '많이 지쳐있는 상태입니다',
            5: '움직이는 것이 힘듭니다',
            6: '의욕이 없습니다',
            7: '해당 사항 없음'
        },
        '피부모양과_관련된_증상': {
            1: '피부가 부었습니다',
            2: '피부가 눌려있습니다',
            3: '피부 모양이 이상합니다',
            4: '피부가 처졌습니다',
            5: '피부가 함몰되었습니다',
            6: '멍들었습니다',
            7: '해당 사항 없음'
        },
        '피부색과_관련된_증상': {
            1: '피부색이 변하였습니다',
            2: '피부가 갈색으로 변했습니다',
            3: '피부가 착색되었습니다',
            4: '피부가 창백합니다',
            5: '황달이 있습니다',
            6: '해당 사항 없음'
        },
        '호흡과_관련된_증상': {
            1: '숨쉴 때 쇳소리가 납니다',
            2: '폐렴이 있습니다',
            3: '딸꾹질이 심합니다',
            4: '호흡하기가 어렵습니다',
            5: '해당 사항 없음'
        },
    }

    # 매핑 함수 호출
    report_df = map_symptoms_columns(report_df, column_mappings)

    # 데이터 저장 경로 설정
    default_path = './data/col_value_change/'
    output_inbody_path = default_path + 'inbody.csv'
    output_patients_path = default_path + 'patients.csv'
    output_report_path = default_path + 'reports.csv'
    output_vital_path = default_path + 'vital.csv'

    # 디렉토리 생성
    os.makedirs(os.path.dirname(default_path), exist_ok=True)

    # 데이터프레임 저장
    inbody_df.to_csv(output_inbody_path, index=False, encoding='cp949')
    patients_df.to_csv(output_patients_path, index=False, encoding='cp949')
    report_df.to_csv(output_report_path, index=False, encoding='cp949')
    vital_df.to_csv(output_vital_path, index=False, encoding='cp949')