import openai
import os
import pandas as pd
import time
from dotenv import load_dotenv
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from collections import defaultdict
import json
from langchain_text_splitters import RecursiveJsonSplitter, CharacterTextSplitter
from langchain_unstructured import UnstructuredLoader
import logging

# 로거의 레벨을 WARNING 이상으로 설정 (INFO 메시지를 출력하지 않도록 함)
logger = logging.getLogger('pikepdf')
logger.setLevel(logging.WARNING)
logger = logging.getLogger("Assitant")
logging.getLogger("openai").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)

#api키 불러오기
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

#임베딩함수 초기화
embedding_function = OpenAIEmbeddings()

#llm모델 초기화
llm = ChatOpenAI(model='gpt-4o-mini', temperature=0.0)

#splitter 초기화
splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=2000,
        chunk_overlap=500,
    )

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
        patient_data = json_str[user_id]

        # 딕셔너리인 patient_data를 JSON 문자열로 변환
        patient_data_str = json.dumps(patient_data, ensure_ascii=False, indent=4)

        # 변환된 JSON 문자열을 splitter로 나눔
        patient_docs = splitter.create_documents(texts=[patient_data_str])

        # 벡터스토어 생성
        patient_vectorstore = FAISS.from_documents(patient_docs, embedding_function)
        # retriever
        patient_retriever = patient_vectorstore.as_retriever()

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
            loader = UnstructuredLoader('data/docs/exercise.pdf')

            # The splitter can also output documents
            docs = loader.load_and_split(text_splitter=splitter)

            # 벡터스토어 생성
            exercise_vectorstore = FAISS.from_documents(docs, embedding_function)
            # retriever
            exercise_retriever = exercise_vectorstore.as_retriever()
            exercise_prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        """
                        당신은 전문적인 의사이며 주어진 환자 정보를 토대로 추천하는 운동에 대해 친절하고 정확하게 설명해야한다. 추천하는 운동은 주어지는 문서의 내용을 참조한다. 환자의 생체 신호가 시간에 따른 변화에서 유의미한 정보가 있다면 참고하며 없는 내용이거나 잘 모르는 내용을 생성하지않는다.
                        --------
                        환자 정보 : 
                        {patient_info}

                        문서 : 
                        {exercise}
                        """,
                    ),
                    ("human", "{question}")
                ]
            )

            exercise_chain = (
                    {"patient_info": patient_retriever, "exercise": exercise_retriever, "question": RunnablePassthrough()}
                    | exercise_prompt
                    | llm
                    | StrOutputParser()
            )

            exercise_response = exercise_chain.invoke('추천하는 운동을 간략하게 정리해')
            print(exercise_response)
            print('------------------------------------------')

        #식단관리방법 출력
        elif menu == 3:
            print('------------------------------------------')
            loader = UnstructuredLoader('data/docs/recipe.pdf')

            # The splitter can also output documents
            docs = loader.load_and_split(text_splitter=splitter)

            # 벡터스토어 생성
            recipe_vectorstore = FAISS.from_documents(docs, embedding_function)
            # retriever
            recipe_retriever = recipe_vectorstore.as_retriever()
            recipe_prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        """
                        당신은 전문적인 의사이며 주어진 환자 정보를 토대로 식단을 관리하는 방법에 대해 친절하고 정확하게 설명해야한다. 식단을 관리하는 방법은 주어지는 문서의 내용을 참조한다. 환자의 생체 신호가 시간에 따른 변화에서 유의미한 정보가 있다면 참고하며 없는 내용이거나 잘 모르는 내용을 생성하지않는다.
                        --------
                        환자 정보 : 
                        {patient_info}
                        
                        문서 : 
                        {recipe}
                        """,
                    ),
                    ("human", "{question}")
                ]
            )

            recipe_chain = (
                    {"patient_info": patient_retriever,"recipe" : recipe_retriever, "question": RunnablePassthrough()}
                    | recipe_prompt
                    | llm
                    | StrOutputParser()
            )

            recipe_response = recipe_chain.invoke('식단을 추천해서 간략하게 정리해')
            print(recipe_response)
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
