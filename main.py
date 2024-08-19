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
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.embeddings import CacheBackedEmbeddings
from collections import defaultdict
import json
from langchain_text_splitters import RecursiveJsonSplitter, CharacterTextSplitter
from langchain_unstructured import UnstructuredLoader
from langchain.storage import LocalFileStore
# from langchain.globals import set_llm_cache
# from langchain_community.cache import InMemoryCache
import logging
import os
from preprocess import preprocess

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 로거의 레벨을 WARNING 이상으로 설정 (INFO 메시지를 출력하지 않도록 함)
logger = logging.getLogger('pikepdf')
logger.setLevel(logging.WARNING)
logger = logging.getLogger("Assitant")
logging.getLogger("openai").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)

#api키 불러오기
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# SQLite Cache 초기화
# set_llm_cache(InMemoryCache())

#데이터 전처리코드 실행
preprocess()

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

# @st.cache_data(show_spinner="Embedding file...")
def embed_file(file_path):
    with open(file_path, "rb") as f:
        file_content = f.read()
    cache_dir = LocalFileStore(f"./.cache/embeddings/{file_path}")
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=2000,
        chunk_overlap=500,
    )
    if file_path[-3:] == 'txt':
        loader = TextLoader(file_path)
    else:
        loader = PyPDFLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    embeddings = OpenAIEmbeddings()
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    retriever = vectorstore.as_retriever()
    return retriever

def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)

#csv파일 json형식으로 불러오는 함수
def csv_files_to_json(file_paths, encoding='cp949'):
    # Initialize a default dictionary to store dictionaries for each ID
    json_data = defaultdict(lambda: defaultdict(list))

    # Iterate through each CSV files
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
    DMretriever = embed_file("./data/docs/당뇨병_가이드라인.pdf")

    DM_prompt = ChatPromptTemplate.from_messages([
        ("system",
        """
        As a medical expert, your task is to provide a focused analysis of the patient’s health information specifically related to diabetes. The analysis should be concise and meet the following criteria:

        Diabetes Status Analysis: Provide a brief and focused analysis of the patient's diabetes status, summarizing key findings without directly mentioning specific dates or individual data points.

        Key Diabetes-Related Issues and Concerns: Identify major concerns related to the patient's diabetes management, without referencing personal data directly. Focus on general trends or risks based on the analysis.

        ENSURE that all information is strictly related to diabetes, with no inclusion of precautions, considerations, recommendations, or conclusions.
        PLEASE SPEAK KOREAN
        -------
        docs : 
        {DMdocs}
        """
         ),
        ("human", "{patient_info}")
    ])

    final_chain = (
            {
                "DMdocs": DMretriever | RunnableLambda(format_docs),
                "patient_info": RunnablePassthrough()
            }
            | DM_prompt
            | llm
            | StrOutputParser()
    )

    print('질환 분석 중(1/4)...')
    response = final_chain.invoke(patient_data)
    return response

def HBP_response(patient_data):
    HBPretriever = embed_file("./data/docs/고혈압_가이드라인.pdf")

    HBP_prompt = ChatPromptTemplate.from_messages([
        ("system",
        """
        As a medical expert, your task is to provide a focused analysis of the patient’s health information specifically related to hypertension. The analysis should be concise and meet the following criteria:

        Hypertension Status Analysis: Summarize the patient’s current blood pressure status, highlighting key findings without directly mentioning specific dates or individual data points.

        Key Hypertension-Related Issues and Concerns: Identify major concerns related to the patient’s blood pressure management, focusing on general trends or risks based on the analysis.

        ENSURE that the analysis is strictly related to hypertension, with no inclusion of precautions, considerations, recommendations, or conclusions.
        PLEASE SPEAK KOREAN
        -------
        docs : 
        {HBPdocs}
        """
        ),
        ("human", "{patient_info}")
    ])

    final_chain = (
            {
                "HBPdocs": HBPretriever | RunnableLambda(format_docs),
                "patient_info": RunnablePassthrough(),
            }
            | HBP_prompt
            | llm
            | StrOutputParser()
    )

    print('질환 분석 중(2/4)...')
    response = final_chain.invoke(patient_data)
    return response

def HD_response(patient_data):
    # HDdoc_loader_1 = PyPDFLoader("./data/docs/심부전_가이드라인.pdf")
    # HDdocs_1 = HDdoc_loader_1.load_and_split(text_splitter=splitter)
    #
    # HDdoc_loader_2 = PyPDFLoader("./data/docs/심장_가이드라인.pdf")
    # HDdocs_2 = HDdoc_loader_2.load_and_split(text_splitter=splitter)
    #
    # HDdocs = HDdocs_1 + HDdocs_2
    # HDvectorstore = FAISS.from_documents(HDdocs, embedding_function)
    HDretriever = embed_file("./data/docs/심장_가이드라인.pdf")

    HD_prompt = ChatPromptTemplate.from_messages([
        ("system",
        """
        As a medical expert, your task is to provide a focused analysis of the patient’s health information specifically related to heart disease. The analysis should be concise and meet the following criteria:

        Heart Disease Status Analysis: Summarize the patient’s current heart disease status, highlighting key findings without directly mentioning specific dates or individual data points.

        Key Heart Disease-Related Issues and Concerns: Identify major concerns related to the patient’s heart disease management, focusing on general trends or risks based on the analysis.

        ENSURE that the analysis is strictly related to heart disease, with no inclusion of precautions, considerations, recommendations, or conclusions.
        PLEASE SPEAK KOREAN
        -------
        docs : 
        {HDdocs}
        """
        ),
        ("human", "{patient_info}")
    ])

    final_chain = (
            {
                "HDdocs": HDretriever | RunnableLambda(format_docs),
                "patient_info": RunnablePassthrough(),
            }
            | HD_prompt
            | llm
            | StrOutputParser()
    )

    print('질환 분석 중(3/4)...')
    response = final_chain.invoke(patient_data)
    return response

def Cancer_response(patient_data):
    Cancerretriever = embed_file("./data/docs/암_지침서.txt")

    Cancer_prompt = ChatPromptTemplate.from_messages([
        ("system",
        """
        As a medical expert, your task is to provide a focused analysis of the patient’s health information specifically related to cancer. The analysis should be concise and meet the following criteria:

        Cancer Status Analysis: Provide a brief and focused analysis of the patient’s current cancer status, summarizing key findings without directly mentioning specific dates or individual data points.

        Key Cancer-Related Issues and Concerns: Identify major concerns related to the patient’s cancer management, without referencing personal data directly. Focus on general trends or risks based on the analysis.

        ENSURE that all information is strictly related to cancer, with no inclusion of precautions, considerations, recommendations, or conclusions.
        PLEASE SPEAK KOREAN
        -------
        docs : 
        {Cancerdocs}
        """
        ),
        ("human", "{patient_info}")
    ])

    final_chain = (
            {
                "Cancerdocs": Cancerretriever | RunnableLambda(format_docs),
                "patient_info": RunnablePassthrough(),
            }
            | Cancer_prompt
            | llm
            | StrOutputParser()
    )

    print('질환 분석 중(4/4)...')
    response = final_chain.invoke(patient_data)
    return response

#csv읽어오기
file_paths = ['./data/col_value_change/inbody.csv', './data/col_value_change/patients.csv', './data/col_value_change/reports.csv', './data/col_value_change/vital.csv']
json_str = csv_files_to_json(file_paths)
json_str = json.loads(json_str)

#검색하고싶은 user_id 입력
user_id = input_user_id()




#종료가 아니라면 기능 출력
if user_id != -1:
    menu = 4
    while(True):
        if menu == 4:
            # 검색하고 싶은 user_id에 따른 데이터 가져오기
            patient_data = json_str[user_id]
            age = 2024 - patient_data['patients'][0]['출생년도']
            patient_data['patients'][0]['나이'] = age

            dm_response = DM_response(str(patient_data))
            hbp_response = HBP_response(str(patient_data))
            hd_response = HD_response(str(patient_data))
            cancer_response = Cancer_response(str(patient_data))

            patient_data['patients'][0]['당뇨 분석 결과'] = dm_response
            patient_data['patients'][0]['고혈압 분석 결과'] = hbp_response
            patient_data['patients'][0]['심장질환 분석 결과'] = hd_response
            patient_data['patients'][0]['암 분석 결과'] = cancer_response

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

            system_message = f"""
                "As a medical expert, your task is to provide a comprehensive analysis of the patient’s health information, including important considerations based on the patient's age, gender, age group, medical conditions, lifestyle factors, and other relevant health data. Using the provided context, your goal is to:

                1. Thoroughly analyze the patient's health status.

                2. Identify key health issues and concerns that need to be addressed.

                3. Highlight important precautions and considerations for managing the patient's health to ensure safety and effectiveness.

                4. Each observation and recommendation should be supported by clear explanations, referencing clinical guidelines or research when applicable. If there is insufficient information, acknowledge this and suggest a method for the patient to obtain the necessary details, such as consulting with a healthcare provider.

                5. Do not include personal opinions, speculative advice, or unsupported claims in your responses. Your role is to deliver expert, evidence-based insights and safety guidelines tailored to the patient’s unique health needs.

                Please consider readability by ensuring that each sentence does not exceed 55 characters and the response is within 10 lines."
                PLEASE SPEAK KOREAN
                --------
            """

            total_prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        system_message,
                    ),
                    ("human", "{patient_info}")
                ]
            )

            total_chain = (
                    total_prompt
                    | llm
                    | StrOutputParser()
            )
            print("종합 소견 출력 중...")
            total_response = total_chain.invoke({"patient_info": str(patient_data)})
            print(total_response)
            first = False

        #추천운동 출력
        elif menu == 2:
            print('------------------------------------------')
            loader = PyPDFLoader('data/docs/exercise.pdf')

            # The splitter can also output documents
            docs = loader.load_and_split(text_splitter=splitter)

            # 벡터스토어 생성
            exercise_vectorstore = FAISS.from_documents(docs, embedding_function)
            # retriever
            exercise_retriever = exercise_vectorstore.as_retriever()

            system_message = """
                Your role is to act as a medical expert and provide exercise recommendations tailored to the patient's medical conditions, fitness level, lifestyle factors, and other relevant health data provided.

                1. Carefully consider the patient’s HEALTH CONDITONS and MEDICAL HISTORY to recommend exercises that are appropriate for them.

                2. Incorporate the patient’s EXERCISE PREFERENCES into your recommendations, but ensure that these exercises are safe and effective for their specific health situation.

                3. Clearly outline IMPORTANT PRECAUTIONS AND CONSIDERATIONS during exercise to ensure the patient's safety and the effectiveness of the exercise.

                4. Support each recommendation and precaution with CLINICAL GUIDELINES or research. When necessary, suggest that the patient consult with a healthcare provider for further information.

                5. Avoid including PERSONAL CONCLUSIONS or SPECULATIVE ADVICE. Provide EVIDENCE-BASED exercise recommendations and safety guidelines tailored to the patient’s unique health needs.
                
                Please consider readability by ensuring that each sentence does not exceed 55 characters and the response is within 10 lines."
                PLEASE SPEAK KOREAN
                --------
                exercise document : 
                {exercise}
                        
            """

            exercise_prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        system_message,
                    ),
                    ("human", "{patient_info}")
                ]
            )

            exercise_chain = (
                    {
                        "exercise": exercise_retriever | RunnableLambda(format_docs),
                        "patient_info": RunnablePassthrough()
                        }
                    | exercise_prompt
                    | llm
                    | StrOutputParser()
            )
            print("추천 운동 검색 중...")
            exercise_response = exercise_chain.invoke(str(patient_data))
            print(exercise_response)
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

            system_message = """
                "Your role is to act as a medical expert and provide tailored dietary recommendations based on the patient's medical conditions, nutritional needs, lifestyle factors, and other relevant health data provided.

                1. ANALYZE THE PATIENT'S SPECIFIC HEALTH CONDITIONS and MEDICAL HISTORY in relation to their dietary needs to provide a comprehensive understanding.

                2. Recommend a DIET PLAN that is SPECIFICALLY TAILORED to the patient's unique health needs, considering their medical conditions and FOOD PREFERENCD.

                3. Clearly outline IMPORTANT DIETARY PRECAUTIONS AND CONSIDERATIONS, including specific foods to avoid and eating habits to prevent, ensuring the patient's safety and the effectiveness of the diet plan.

                4. Each recommendation and precaution should be SUPPORTED BY CLINICAL GUIDELINES OR RESEARCH when applicable. If information is insufficient, suggest that the patient CONSULT A HEALTHCARE PROVIDER OR REGISTERED DIETITIAN for further advice.

                5. FOCUS SOLELY ON DIET-RELATED INFORMATION. Do not include recommendations or advice related to exercise, lifestyle habits, or any other non-dietary factors.

                6. Avoid including PERSONAL OPINIONS, SPECULATIVE ADVICE, OR UNSUPPORTED CLAIMS. Provide EVIDENCE-BASED DIETARY RECOMMENDATIONS and SAFETY GUIDELINES tailored to the patient's specific health needs.
                
                Please consider readability by ensuring that each sentence does not exceed 55 characters and the response is within 10 lines."
                PLEASE SPEAK KOREAN
                --------
                recipe document : 
                {recipe}
            """

            recipe_prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        system_message,
                    ),
                    ("human", "{patient_info}")
                ]
            )

            recipe_chain = (
                    {
                        "recipe" : recipe_retriever | format_docs,
                        "patient_info": RunnablePassthrough()
                        }
                    | recipe_prompt
                    | llm
                    | StrOutputParser()
            )

            print('추천 식단 검색 중...')
            recipe_response = recipe_chain.invoke(str(patient_data))
            print(recipe_response)
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
