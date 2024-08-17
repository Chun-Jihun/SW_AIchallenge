from collections import defaultdict
import json
import pandas as pd

#csv파일 json형식으로 불러오는 함수
def csv_files_to_json(file_paths, encoding='cp949'):
    # Initialize a default dictionary to store dictionaries for each ID
    json_data = defaultdict(lambda: defaultdict(list))

    # Iterate through each CSV file
    for file_path in file_paths:
        # Extract the file name without extension to use as a key
        file_key = file_path.split('/')[-1].split('.')[0]

        # Read the CSV file into a DataFrame with cp949 encoding
        df = pd.read_csv(file_path, encoding=encoding)

        # Group the data by 'id'
        for id, group in df.groupby('id'):
            # Convert the group data to a dictionary without the 'id' column
            record = group.drop(columns=['id']).to_dict(orient='records')
            # Append the record under the corresponding file name for the given id
            json_data[str(id)][file_key].extend(record)

    # Convert the default dictionary to a regular dictionary and then to a JSON string
    # Ensure that ASCII encoding is disabled to properly display Korean characters
    json_str = json.dumps(dict(json_data), ensure_ascii=False, indent=4)

    return json_str

#user_id 입력받는 함수
def input_user_id():
    user_id = input('검색하고자 하는 환자 ID를 입력해주세요(종료 : -1) : ')
    while (user_id not in json_str.keys() and user_id != -1):
        if user_id == '-1':
            return -1
        print('잘못된 입력입니다. 다시 입력해주세요.')
        print('현재 존재하는 환자 ID 목록:', ', '.join(json_str.keys()))
        user_id = input('검색하고자 하는 환자 ID를 입력해주세요(종료 : -1) : ')
        return user_id
    return user_id

#csv읽어오기
file_paths = ['./data/col_value_change/inbody.csv', './data/col_value_change/patients.csv', './data/col_value_change/reports.csv', './data/col_value_change/vital.csv']
json_str = csv_files_to_json(file_paths)
json_str = json.loads(json_str)

#검색하고싶은 user_id 입력
user_id = input_user_id()

#종료가 아니라면 기능 출력
if user_id != '-1':
    #메뉴 변수 초기화
    menu = 0
    while(True):
        # 검색하고 싶은 user_id에 따른 데이터 가져오기
        data = json_str[user_id]

        #메뉴 출력
        print('1. 건강정보\n2. 추천운동\n3. 식단관리\n4. 환자ID 재검색\n5. 종료')
        print('------------------------------------------')
        menu = int(input('보시고 싶은 메뉴를 입력하세요 : '))
        while(menu < 1 or menu > 5):
            print('잘못된 입력입니다. 다시 입력해주세요 : ')
            menu = input('보시고 싶은 메뉴를 입력하세요 : ')

        #건강정보 출력
        if menu == 1:
            print('------------------------------------------')
            print('건강정보 출력')
            print('------------------------------------------')

        #추천운동 출력
        elif menu == 2:
            print('------------------------------------------')
            print('추천운동 출력')
            print('------------------------------------------')

        #식단관리방법 출력
        elif menu == 3:
            print('------------------------------------------')
            print('식단관리방법 출력')
            print('------------------------------------------')

        #환자ID재검색
        elif menu == 4:
            user_id = input_user_id()
            if user_id == '-1':
                break

        #종료 입력시 반복문 탈출
        elif menu == 5:
            break

    print('프로그램을 종료합니다.')
