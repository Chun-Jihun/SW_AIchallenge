import openai
import os
import pandas as pd
import time
from dotenv import load_dotenv
from langchain_community.document_loaders import UnstructuredFileLoader, PyPDFLoader, TextLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders import PyPDFLoader
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

def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)

def DM_response(patient_data):
    DMdoc_loader = PyPDFLoader("./data/docs/당뇨병_가이드라인.pdf")
    DMdocs = DMdoc_loader.load_and_split(text_splitter=splitter)
    DMvectorstore = FAISS.from_documents(DMdocs, embedding_function)
    DMretriever = DMvectorstore.as_retriever()

    DM_prompt = ChatPromptTemplate.from_messages([
        ("system",
         """
         너는 당뇨병 전문 의사야.
         환자의 정보들을 기반으로 당뇨병에 대한 환자의 건강상태에 대해서 말해 가족 중에 당뇨병이 있는 사람도 있고 없는 사람도 있으니 해당 정보도 유심히 봐야 해.
         내가 주는 docs 파일을 기반으로 이 환자에 특화적인 조언도 추가 해
         이를 기반으로 아래처럼 말해라.
         예시) ~ 환자는 가족 중에 당뇨병에 대한 이력이 ~ 하고 현재 환자는 당뇨병이 ~ 한 상태이다.
         -------
         docs : 
         {DMdocs}
         """
         ),
        ("human", "{patient_info}")
    ])

    final_chain = (
            {
                "DMdocs": DMretriever,
                "patient_info": patient_data,
            }
            | DM_prompt
            | llm
            | StrOutputParser()
    )

    print('당뇨 분석 중...')
    response = final_chain.invoke('당뇨 분석')
    return response

def HBP_response(patient_data):
    HBPdoc_loader = PyPDFLoader("./data/docs/고혈압_가이드라인.pdf")
    HBPdocs = HBPdoc_loader.load_and_split(text_splitter=splitter)
    HBPvectorstore = FAISS.from_documents(HBPdocs, embedding_function)
    HBPretriever = HBPvectorstore.as_retriever()

    HBP_prompt = ChatPromptTemplate.from_messages([
        ("system",
         """
         너는 고혈압 전문 의사야.
         환자의 정보들을 기반으로 고혈압에 대한 환자의 건강상태에 대해서 말해 가족 중에 고혈압이 있는 사람도 있고 없는 사람도 있으니 해당 정보도 유심히 봐야 해.
         내가 주는 docs 파일을 기반으로 이 환자에 특화적인 조언도 추가 해
         이를 기반으로 아래처럼 말해라.
         예시) ~ 환자는 가족 중에 고혈압에 대한 이력이 ~ 하고 현재 환자는 고혈압이 ~ 한 상태이다.
         -------
         docs : 
         {HBPdocs}
         """
         ),
        ("human", "{patient_info}")
    ])

    final_chain = (
            {
                "HBPdocs": HBPretriever,
                "patient_info": patient_data,
            }
            | HBP_prompt
            | llm
            | StrOutputParser()
    )

    print('고혈압 분석 중...')
    response = final_chain.invoke('고혈압 분석')
    return response

def HD_response(patient_data):
    HDdoc_loader_1 = PyPDFLoader("./data/docs/심부전_가이드라인.pdf")
    HDdocs_1 = HDdoc_loader_1.load_and_split(text_splitter=splitter)

    HDdoc_loader_2 = PyPDFLoader("./data/docs/심장질환_가이드라인.pdf")
    HDdocs_2 = HDdoc_loader_2.load_and_split(text_splitter=splitter)

    HDdocs = HDdocs_1 + HDdocs_2
    HDvectorstore = FAISS.from_documents(HDdocs, embedding_function)
    HDretriever = HDvectorstore.as_retriever()

    HD_prompt = ChatPromptTemplate.from_messages([
        ("system",
         """
         너는 심장질환 전문 의사야.
         환자의 정보들을 기반으로 심장질환에 대한 환자의 건강상태에 대해서 말해 가족 중에 심장질환이 있는 사람도 있고 없는 사람도 있으니 해당 정보도 유심히 봐야 해.
         내가 주는 docs 파일을 기반으로 이 환자에 특화적인 조언도 추가 해
         이를 기반으로 아래처럼 말해라.
         예시) ~ 환자는 가족 중에 심장질환에 대한 이력이 ~ 하고 현재 환자는 심장질환이 ~ 한 상태이다.
         -------
         docs : 
         {HDdocs}
         """
         ),
        ("human", "{patient_info}")
    ])

    final_chain = (
            {
                "HDdocs": HDretriever,
                "patient_info": patient_data,
            }
            | HD_prompt
            | llm
            | StrOutputParser()
    )

    print('심장질환 분석 중...')
    response = final_chain.invoke('심장질환 분석')
    return response

def Cancer_response(patient_data):
    Cancerdoc_loader = TextLoader("./data/docs/암_지침서.txt")
    Cancerdocs = Cancerdoc_loader.load_and_split(text_splitter=splitter)
    Cancervectorstore = FAISS.from_documents(Cancerdocs, embedding_function)
    Cancerretriever = Cancervectorstore.as_retriever()

    Cancer_prompt = ChatPromptTemplate.from_messages([
        ("system",
         """
         너는 암 전문 의사야.
         환자의 정보들을 기반으로 암에 대한 환자의 건강상태에 대해서 말해 가족 중에 암이 있는 사람도 있고 없는 사람도 있으니 해당 정보도 유심히 봐야 해.
         내가 주는 docs 파일을 기반으로 이 환자에 특화적인 조언도 추가 해.
         암 관련 얘기만 해라.
         이를 기반으로 아래처럼 말해라.
         예시) ~ 환자는 가족 중에 암에 대한 이력이 ~ 하고 현재 환자는 암이 ~ 한 상태이다.
         -------
         docs : 
         {Cancerdocs}
         """
         ),
        ("human", "{patient_info}")
    ])

    final_chain = (
            {
                "Cancerdocs": Cancerretriever,
                "patient_info": patient_data,
            }
            | Cancer_prompt
            | llm
            | StrOutputParser()
    )

    print('암 분석 중...')
    response = final_chain.invoke('암 분석')
    return response

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
    #처음 실행 확인
    first = True
    while(True):
        if first or menu == 4:
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
            dm_response = DM_response(patient_retriever)
            hbp_response = HBP_response(patient_retriever)
            hd_response = HD_response(patient_retriever)
            cancer_response = Cancer_response(patient_retriever)

        #메뉴 출력
        print('------------------------------------------')
        print('1. 건강정보\n2. 추천운동\n3. 식단관리\n4. 환자ID 재검색\n5. 종료')
        print('------------------------------------------')
        menu = int(input('보시고 싶은 메뉴의 번호를 입력하세요 : '))
        while(menu < 1 or menu > 5):
            print('잘못된 입력입니다. 다시 입력해주세요 : ')
            menu = input('보시고 싶은 메뉴의 번호를 입력하세요 : ')

        #건강정보 출력
        if menu == 1:
            print('------------------------------------------')
            total_query = f"""
                As a medical expert, your task is to provide a comprehensive analysis of the patient’s health information, including important considerations based on the patient's medical conditions, lifestyle factors, and other relevant health data. Using the provided context, your goal is to:
    
                Thoroughly analyze the patient's health status.
                Identify key health issues and concerns that need to be addressed.
                Highlight important precautions and considerations for managing the patient's health to ensure safety and effectiveness.
                Each observation and recommendation should be supported by clear explanations, referencing clinical guidelines or research when applicable. If there is insufficient information, acknowledge this and suggest a method for the patient to obtain the necessary details, such as consulting with a healthcare provider.
    
                Do not include personal opinions, speculative advice, or unsupported claims in your responses. Your role is to deliver expert, evidence-based insights and safety guidelines tailored to the patient’s unique health needs.
                PLEASE SPEAK KOREAN
                --------
                Patient's diabetes condition :
                {dm_response}
                Patient's heart disease condition :
                {hd_response}
                Patient's hypertension condition :
                {hbp_response} 
                Patient's cancer condition :
                {cancer_response}
            """

            total_prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        total_query,
                    ),
                    ("human", "{patient_info}")
                ]
            )

            total_chain = (
                    {"patient_info": patient_retriever}
                    | total_prompt
                    | llm
                    | StrOutputParser()
            )
            print("종합 소견 출력 중...")
            total_response = total_chain.invoke("건강정보에 대한 종합적인 소견을 간략하게 정리해")
            print(total_response)
            print('------------------------------------------')
            first = False

        #추천운동 출력
        elif menu == 2:
            print('------------------------------------------')
            loader = PyPDFLoader('./data/docs/exercise.pdf')

            # The splitter can also output documents
            docs = loader.load_and_split(text_splitter=splitter)

            # 벡터스토어 생성
            exercise_vectorstore = FAISS.from_documents(docs, embedding_function)
            # retriever
            exercise_retriever = exercise_vectorstore.as_retriever()

            exercise_query = """
                        As a medical expert, your task is to provide tailored exercise recommendations, including important precautions, based on the patient's medical conditions, fitness level, lifestyle factors, and other relevant health data.
                        Using the provided context, your goal is to:
                        Analyze the patient's condition comprehensively.
                        Recommend appropriate exercises tailored to the patient's needs.
                        Highlight important precautions and considerations during exercise to ensure safety and effectiveness.
                        Each recommendation and precaution should be supported by clear explanations, referencing clinical guidelines or research when applicable. If there is insufficient information, acknowledge this and suggest a method for the patient to obtain the necessary details, such as consulting with a healthcare provider.

                        Do not include personal opinions, speculative advice, or unsupported claims in your responses. Your role is to deliver expert, evidence-based exercise recommendations and safety guidelines tailored to the patient's unique health needs.
                        PLEASE SPEAK KOREAN
                        --------
                        exercise document : 
                        {exercise}
                        
                        """
            exercise_query += f"""
                Patient's diabetes condition :
                {dm_response}
                patient's heart disease condition :
                {hd_response}
                patient's hypertension condition :
                {hbp_response} 
                patient's cancer condition :
                {cancer_response}
            """

            exercise_prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        exercise_query,
                    ),
                    ("human", "{patient_info}")
                ]
            )

            exercise_chain = (
                    {"exercise": exercise_retriever, "patient_info": patient_retriever}
                    | exercise_prompt
                    | llm
                    | StrOutputParser()
            )
            print("추천 운동 검색 중...")
            exercise_response = exercise_chain.invoke("추천 운동을 간략하게 정리해")
            print(exercise_response)
            print('------------------------------------------')
            first = False

        #식단관리방법 출력
        elif menu == 3:
            print('------------------------------------------')
            loader = PyPDFLoader('data/docs/recipe.pdf')

            # The splitter can also output documents
            docs = loader.load_and_split(text_splitter=splitter)

            # 벡터스토어 생성
            recipe_vectorstore = FAISS.from_documents(docs, embedding_function)
            # retriever
            recipe_retriever = recipe_vectorstore.as_retriever()

            recipe_query = """
                        As a medical expert, your task is to provide tailored dietary recommendations based on the patient's medical conditions, nutritional needs, lifestyle factors, and other relevant health data. Using the provided context, your goal is to:

                        Comprehensively analyze the patient's condition.
                        Recommend an appropriate diet plan tailored to the patient's needs.
                        Highlight important precautions and considerations related to diet to ensure safety and effectiveness.
                        Each recommendation and precaution should be supported by clear explanations, referencing clinical guidelines or research when applicable. If there is insufficient information, acknowledge this and suggest a method for the patient to obtain the necessary details, such as consulting with a healthcare provider or a registered dietitian.

                        Do not include personal opinions, speculative advice, or unsupported claims in your responses. Your role is to deliver expert, evidence-based dietary recommendations and safety guidelines tailored to the patient's unique health needs.
                        PLEASE SPEAK KOREAN
                        --------
                        recipe document : 
                        {recipe}
                        """
            recipe_query += f"""
                            Patient's diabetes condition :
                            {dm_response}
                            patient's heart disease condition :
                            {hd_response}
                            patient's hypertension condition :
                            {hbp_response} 
                            patient's cancer condition :
                            {cancer_response}
                        """

            recipe_prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        recipe_query,
                    ),
                    ("human", "{patient_info}")
                ]
            )

            recipe_chain = (
                    {"recipe" : recipe_retriever, "patient_info": patient_retriever}
                    | recipe_prompt
                    | llm
                    | StrOutputParser()
            )

            print('추천 식단 검색 중...')
            recipe_response = recipe_chain.invoke('추천 식단 관리 ')
            print(recipe_response)
            print('------------------------------------------')
            first = False

        #환자ID재검색
        elif menu == 4:
            user_id = input_user_id()
            first = False
            if user_id == '-1':
                break

        #종료 입력시 반복문 탈출
        elif menu == 5:
            break

    print('프로그램을 종료합니다.')
