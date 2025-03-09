# firm-agent
- 사내 근무지원시스템 AI FirmAgent
- 회사의 업무범위를 입력하면 범위와 조직에 맞는 "도구" "멀티턴" "멀티쿼리" 를 정제해주는 agent

## 데모
- 회사 부서별 업무가 작성되어있는 조직이 사용하는 업무 도구 문서(Profile of Task and Tools, 이하 POTAT)문서를 엑셀로 작성해서 입력하면, 필요한 도구로
```
<입력예시>
유저입력 : 사내 탬플릿과 문서양식을 어디서 확인해
유저입력 : 사내어린이집은 어디에서 도움받지?

<출력예시>
생성된 계획: {
  "plan": {
    "Childcare Support": [
      "사내 어린이집 운영에 대한 정보는 어디에서 확인할 수 있나요?",
      "사내 어린이집 지원을 받으려면 어떻게 해야 하나요?"
    ]
  },
  "selected_tools": [
    "Childcare Support"
  ]
}

FirmAgent 정제 질문 : 생성된 계획: {
  "plan": {
    "Internal Comms": [
      "사내 템플릿과 문서 양식을 확인할 수 있는 시스템이나 위치를 안내해 주세요."
    ]
  },
  "selected_tools": [
    "Internal Comms"
  ]
}
```
- POTAT 문서를 사내 타팀에 전달하여, 취합 후, 고성능 LLM 에 입력하면 FirmAgent 가 "멀티턴", "멀티쿼리", "툴 셀렉" 3가지 분리해 유저 입력 문맥과 행간의 의미를 파악 하여 질문을 정제하고, 적절한 정보와 기능을 갖춘 후방시스템에 넘겨주는 **"분기처리기"** 역할을 수행하게 됩니다

## POTAT 문서 예시
- 아래 예시는 "네이버 주식회사" 의 2023년 사업보고서를 기반으로 POTAT 문서를 작성한 예시 입니다.
![image.png](22a87cda-4070-4f12-8283-dcf20094a87a.png)

## 기획
- 회사에서 "업무" 이외의 다른 일들이 많은데, 각각의 개개인이 핵심 업무에 집중할수 있도록 비핵심업무에 대한 자동화를 해주는 AI Agent 입니다.
- 예시 1) HR팀 케이스
  - HR팀의 핵심업무 : 채용관리, 인재관리, 조직개발, 인사평가 및 성과보상 등등..
  - HR팀의 비 핵심 업무 : 신규입사자 문의, 사내생활관련 문의대응, 복지제도안내, 등등.. >> 자동화 목표
- 예시 2) Biz팀 케이스
  - 핵심업무 : 신규 고객 발굴, CRM, 세일즈 전략 수립 및 실행, 매출 및 수익 관리
  - 비핵심업무 : 행사 및 마케팅지원, CS VoC 처리 >> 자동화
- 예시 3) 웹 백엔드 개발자 케이스
  - 서버&DB 아키설계, 비즈니스 로직 구현, 성능&리소스 최적화, api 개발, 보안, test, CICD
  - 내부문의대응(데이터추출, 보정), 기획서 검토, 데이터 지표 Aggregate
- 이외 모든 직군이 겪을수 있는 비 핵심 업무
  - 형식적인 보고서 작성 : 사내표준문서양식 (ppt, 설문조사 등..) 찾아서 보고서 수정
  - 출장비, 업무경비 정산 및 영수증처리
  - 비품요청

### 용어집
- 멀티턴 (Multi-turn): 여러 번의 질문과 응답을 주고받으며 대화를 이어가는 방식.
- 멀티쿼리 (Multi-query): 여러 개의 검색 또는 요청을 동시에 실행하는 방식.
- 툴 셀렉 (Tool Select): 요청에 맞는 가장 적합한 툴을 자동으로 선택하는 과정.


```python
!pip install langchain langchain_openai langchain_community pypdf faiss-cpu --quiet
```


```python
# 필요한 라이브러리 임포트
import os
import json
from typing import List, Dict, Any, Optional, Tuple
from IPython.display import display, Markdown

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
```

#### OPENAI Key 는 파일시스템에서 읽어오게 처리하였습니다.
- colab 환경의 경우 하드코딩으로 변경해주세요!


```python
# OpenAI API 키 설정
OPENAI_API_KEY = ""

#OPENAI_APIKEY=""
with open("../secrets/apikey.txt", "r") as file:
    OPENAI_API_KEY = file.read().strip()
    os.environ["OPENAI_API_KEY"] = file.read().strip()

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
```

## 1. PDF 파일 다운로드


```python
# URL에서 PDF 파일 다운로드
import urllib.request

"""
- 사실 사내 챗봇을 만들려면, 사내 데이터 각 부서가 어떤 업무를 담당하고 있는지에 대한 데이터가 필요합니다.
- 그런데 그런 데이터는 외부인이 구할수 없기 때문에 아쉬운대로 최대한 비슷한 네이버의 사업보고서, IR 자료등을 데이터셋으로 넣었습니다.
- 참고링크 : https://www.navercorp.com/investment/irReport
"""

# 데이터셋
list_of_downloads_pdf = {
    "https://www.navercorp.com/api/article/download/3216dc37-0b95-4f12-ab59-e475fa22ec03": "2024년 3분기 검토보고서",
    "https://www.navercorp.com/api/article/download/7dee92cb-9c55-4476-8477-d0600b799125": "2023년 NAVER 사업보고서",
    "https://www.navercorp.com/api/article/download/439815bb-16ac-4b81-b364-6227ec28a3d8": "2023 통합보고서",
    "https://www.navercorp.com/api/article/download/8b7f2fa1-c6bd-4ed8-9478-25a6ada62c26": "2023 TCFD 보고서",
    "https://www.navercorp.com/api/article/download/f06ed790-683a-4463-aa75-7ea2127c4c04": "2024년 4분기 NAVER 실적발표",
    "https://www.navercorp.com/api/download/776d0bff-78de-4f85-9fbe-e687cf7c5f42": "CEO 주주서한 - AI 시대 속 네이버의 경쟁력",
    "https://www.navercorp.com/api/download/eee8efa9-61ae-496a-ba5e-0a950acb3a2f": "CEO 주주서한 - 네이버 커머스의 현재와 미래"
}
## 중간에 두개 뺴놓은건 너무 커서 임시로 빼움

# PDF 다운로드 함수
def download_pdf_from_url(url: str, output_filename: str):
    print(f"PDF 다운로드 시작: {url}")
    urllib.request.urlretrieve(url, filename=output_filename)
    print(f"PDF 다운로드 완료: {output_filename}")
```


```python
list_pdf_files = []
for key, value in list_of_downloads_pdf.items():
    list_pdf_files.append(value+".pdf")

print("\n".join(list_pdf_files))
```

    2024년 3분기 검토보고서.pdf
    2023년 NAVER 사업보고서.pdf
    2023 통합보고서.pdf
    2023 TCFD 보고서.pdf
    2024년 4분기 NAVER 실적발표.pdf
    CEO 주주서한 - AI 시대 속 네이버의 경쟁력.pdf
    CEO 주주서한 - 네이버 커머스의 현재와 미래.pdf



```python
# 순회하며 다운로드
for key, value in list_of_downloads_pdf.items():
    #print(f"TASK 데이터셋 로딩 URL: {(str(key)[:10])}, Description: {value}")
    download_pdf_from_url(url=key,output_filename=value+".pdf")
```

    PDF 다운로드 시작: https://www.navercorp.com/api/article/download/3216dc37-0b95-4f12-ab59-e475fa22ec03
    PDF 다운로드 완료: 2024년 3분기 검토보고서.pdf
    PDF 다운로드 시작: https://www.navercorp.com/api/article/download/7dee92cb-9c55-4476-8477-d0600b799125
    PDF 다운로드 완료: 2023년 NAVER 사업보고서.pdf
    PDF 다운로드 시작: https://www.navercorp.com/api/article/download/439815bb-16ac-4b81-b364-6227ec28a3d8
    PDF 다운로드 완료: 2023 통합보고서.pdf
    PDF 다운로드 시작: https://www.navercorp.com/api/article/download/8b7f2fa1-c6bd-4ed8-9478-25a6ada62c26
    PDF 다운로드 완료: 2023 TCFD 보고서.pdf
    PDF 다운로드 시작: https://www.navercorp.com/api/article/download/f06ed790-683a-4463-aa75-7ea2127c4c04
    PDF 다운로드 완료: 2024년 4분기 NAVER 실적발표.pdf
    PDF 다운로드 시작: https://www.navercorp.com/api/download/776d0bff-78de-4f85-9fbe-e687cf7c5f42
    PDF 다운로드 완료: CEO 주주서한 - AI 시대 속 네이버의 경쟁력.pdf
    PDF 다운로드 시작: https://www.navercorp.com/api/download/eee8efa9-61ae-496a-ba5e-0a950acb3a2f
    PDF 다운로드 완료: CEO 주주서한 - 네이버 커머스의 현재와 미래.pdf



```python
# PDF 파일로부터 벡터 DB 생성 함수
def create_vectorstore_from_pdf(pdf_path: str, db_name: str) -> FAISS:
    print(f"PDF 로딩 시작: {pdf_path}")
    print(f"모든 pdf 파일을 하나의 벡터스토어로 합치도록 코드를 변경하였으므로 개별파일의 벡터화는 스킵함")
    return None
    # PDF 로드 및 분할
    loader = PyPDFLoader(pdf_path)
    doc_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=100)
    docs = loader.load_and_split(doc_splitter)

    print(f"PDF 로딩 완료: {len(docs)}개 청크 생성됨")

    # 임베딩 및 벡터스토어 생성
    embedding = OpenAIEmbeddings(model="text-embedding-3-large")
    vectorstore = FAISS.from_documents(docs, embedding)

    # 벡터스토어 저장
    persist_directory = f"./DB/{db_name}"
    os.makedirs(persist_directory, exist_ok=True)
    vectorstore.save_local(persist_directory)

    print(f"{db_name} 벡터스토어 생성 완료")
    return vectorstore
```

이 함수는 PDF 파일을 로드하여 검색 가능한 벡터 데이터베이스를 생성합니다.
1. PDF 파일 로딩: PyPDFLoader를 사용하여 PDF 파일의 텍스트를 추출합니다.
2. 텍스트 분할: RecursiveCharacterTextSplitter를 사용하여 추출된 텍스트를 300자 크기의
  청크로 분할하며, 인접 청크 간에 100자의 중복을 허용합니다. 이 중복은 문맥 연속성을
  유지하는 데 중요합니다.
3. 벡터 임베딩 생성: OpenAI의 text-embedding-3-large 모델을 사용하여 각 텍스트 청크를
  고차원 벡터 공간에 표현합니다. 이 임베딩은 의미적 유사도 검색의 기반이 됩니다.
4. FAISS 벡터스토어 생성: Facebook AI의 FAISS 라이브러리를 사용하여 임베딩된 벡터들을
  효율적으로 저장하고 검색할 수 있는 인덱스를 구축합니다.
5. 벡터스토어 저장: 생성된 벡터스토어를 로컬 디렉토리에 저장하여 나중에 다시 로드할 수
  있게 합니다. 이는 매번 임베딩을 다시 계산하지 않아도 되므로 시간을 절약할 수 있습니다.
각 단계마다 진행 상황을 출력하여 처리 과정을 모니터링할 수 있습니다.
함수는 최종적으로 생성된 FAISS 벡터스토어 객체를 반환하며, 이는 이후 유사도 기반
검색에 사용됩니다.


```python
# 이제 다운로드된 PDF 파일을 사용하여 벡터스토어 생성
db_list = []
# 순회하며 벡터스토어 생성
for key, value in list_of_downloads_pdf.items():
    db = create_vectorstore_from_pdf((value+".pdf"),value)
    db_list.append(db)
```

    PDF 로딩 시작: 2024년 3분기 검토보고서.pdf
    모든 pdf 파일을 하나의 벡터스토어로 합치도록 코드를 변경하였으므로 개별파일의 벡터화는 스킵함
    PDF 로딩 시작: 2023년 NAVER 사업보고서.pdf
    모든 pdf 파일을 하나의 벡터스토어로 합치도록 코드를 변경하였으므로 개별파일의 벡터화는 스킵함
    PDF 로딩 시작: 2023 통합보고서.pdf
    모든 pdf 파일을 하나의 벡터스토어로 합치도록 코드를 변경하였으므로 개별파일의 벡터화는 스킵함
    PDF 로딩 시작: 2023 TCFD 보고서.pdf
    모든 pdf 파일을 하나의 벡터스토어로 합치도록 코드를 변경하였으므로 개별파일의 벡터화는 스킵함
    PDF 로딩 시작: 2024년 4분기 NAVER 실적발표.pdf
    모든 pdf 파일을 하나의 벡터스토어로 합치도록 코드를 변경하였으므로 개별파일의 벡터화는 스킵함
    PDF 로딩 시작: CEO 주주서한 - AI 시대 속 네이버의 경쟁력.pdf
    모든 pdf 파일을 하나의 벡터스토어로 합치도록 코드를 변경하였으므로 개별파일의 벡터화는 스킵함
    PDF 로딩 시작: CEO 주주서한 - 네이버 커머스의 현재와 미래.pdf
    모든 pdf 파일을 하나의 벡터스토어로 합치도록 코드를 변경하였으므로 개별파일의 벡터화는 스킵함



```python
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import numpy as np
import faiss

def create_vectorstore_from_all_pdfs(pdf_paths: list[str], db_name: str) -> FAISS:
    print(f"PDF 파일 로딩 시작: {len(pdf_paths)}개 파일 처리")

    # PDF에서 로딩된 모든 문서 청크를 저장할 리스트
    all_docs = []

    # 각 PDF 파일을 처리
    for pdf_path in pdf_paths:
        print(f"로딩 중: {pdf_path}")

        # PDF 로드 및 분할
        loader = PyPDFLoader(pdf_path)
        doc_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=100)
        docs = loader.load_and_split(doc_splitter)

        print(f"{pdf_path} 로딩 완료: {len(docs)}개 청크 생성됨")
        all_docs.extend(docs)  # 각 PDF 파일에서 생성된 청크들을 모두 합침

    print(f"모든 PDF 로딩 완료: 총 {len(all_docs)}개 청크 생성됨")

    # 임베딩 및 벡터스토어 생성
    embedding = OpenAIEmbeddings(model="text-embedding-3-large")
    vectorstore = FAISS.from_documents(all_docs, embedding)

    # 벡터스토어 저장
    persist_directory = f"./DB/{db_name}"
    os.makedirs(persist_directory, exist_ok=True)
    vectorstore.save_local(persist_directory)

    print(f"{db_name} 벡터스토어 생성 완료")
    return vectorstore

```


```python
merged_db = create_vectorstore_from_all_pdfs(list_pdf_files,"naver_vector")

```

    PDF 파일 로딩 시작: 7개 파일 처리
    로딩 중: 2024년 3분기 검토보고서.pdf
    2024년 3분기 검토보고서.pdf 로딩 완료: 325개 청크 생성됨
    로딩 중: 2023년 NAVER 사업보고서.pdf
    2023년 NAVER 사업보고서.pdf 로딩 완료: 3783개 청크 생성됨
    로딩 중: 2023 통합보고서.pdf
    2023 통합보고서.pdf 로딩 완료: 1468개 청크 생성됨
    로딩 중: 2023 TCFD 보고서.pdf
    2023 TCFD 보고서.pdf 로딩 완료: 218개 청크 생성됨
    로딩 중: 2024년 4분기 NAVER 실적발표.pdf
    2024년 4분기 NAVER 실적발표.pdf 로딩 완료: 47개 청크 생성됨
    로딩 중: CEO 주주서한 - AI 시대 속 네이버의 경쟁력.pdf
    CEO 주주서한 - AI 시대 속 네이버의 경쟁력.pdf 로딩 완료: 42개 청크 생성됨
    로딩 중: CEO 주주서한 - 네이버 커머스의 현재와 미래.pdf
    CEO 주주서한 - 네이버 커머스의 현재와 미래.pdf 로딩 완료: 34개 청크 생성됨
    모든 PDF 로딩 완료: 총 5917개 청크 생성됨


    /var/folders/35/31j0q5116x1cqdmnp56bw1xc0000gp/T/ipykernel_12346/660574234.py:30: LangChainDeprecationWarning: The class `OpenAIEmbeddings` was deprecated in LangChain 0.0.9 and will be removed in 1.0. An updated version of the class exists in the :class:`~langchain-openai package and should be used instead. To use it run `pip install -U :class:`~langchain-openai` and import as `from :class:`~langchain_openai import OpenAIEmbeddings``.
      embedding = OpenAIEmbeddings(model="text-embedding-3-large")


    naver_vector 벡터스토어 생성 완료



```python
## 각 부서가 무슨일을 하는지-부서의 데이터셋을 구할수 없는 시점이라, 회사 외부 자료를 기준으로 하나의 백터DB 애서만 처리할 수 있도록 수정하였습니다!
```

## 2. 도구 생성


```python
from dataclasses import dataclass

@dataclass
class Tool:
    tool_name: str
    description: str
    vectorstore: any = None

    def __str__(self):
        return f"{self.tool_name}: {self.description}, db={self.vectorstore}"
```


```python
# 테스트로 도구 인스턴스 생성해보기
test_tool = Tool("test", "테스트 도구입니다")
print(test_tool)
```

    test: 테스트 도구입니다, db=None


### 도구 시트 다운로드 받기
- 스프레드시트 뷰어 링크 : https://docs.google.com/spreadsheets/d/1Qy-K7CDvWLwfmt7LBQMt5KRYYnSRpaNk/edit?usp=sharing&ouid=103304252528783415009&rtpof=true&sd=true
- 다운로드 링크 : https://docs.google.com/uc?export=download&id=1Qy-K7CDvWLwfmt7LBQMt5KRYYnSRpaNk&confirm=t


```python
! pip install openpyxl
```

    Requirement already satisfied: openpyxl in /Users/kep-dong22/modu-llm/venv/lib/python3.10/site-packages (3.1.5)
    Requirement already satisfied: et-xmlfile in /Users/kep-dong22/modu-llm/venv/lib/python3.10/site-packages (from openpyxl) (2.0.0)



```python
# 도구 엑셀시트 로드 
TOOLS_SHEET_URL = "https://docs.google.com/uc?export=download&id=1Qy-K7CDvWLwfmt7LBQMt5KRYYnSRpaNk&confirm=t"
TOOL_SHEET_FILENAME="TOOLS_SHEET.xlsx"
urllib.request.urlretrieve(TOOLS_SHEET_URL, filename=TOOL_SHEET_FILENAME)
```




    ('TOOLS_SHEET.xlsx', <http.client.HTTPMessage at 0x125bec040>)




```python
import pandas as pd

def convert_excel_to_tools(excel_file):
    # Read the Excel file into a pandas DataFrame
    df = pd.read_excel(excel_file)

    tools = []

    # Iterate over each row in the DataFrame
    for _, row in df.iterrows():
        # Extract 'tool_name' from 'keyword' column
        tool_name = row['keyword']
        
        # Create the description by joining required columns with a comma
        description = ', '.join([str(row[col]) for col in ['업무영역', '업무구분', 'TASK', '시스템&솔루션', 'keyword']])
        vectorstore = merged_db
        # Create Tool object
        tool = Tool(tool_name=tool_name, description=description,vectorstore=merged_db)
        tools.append(tool)
    
    return tools
```


```python
tools_list = convert_excel_to_tools(TOOL_SHEET_FILENAME)
tools = tools_list
for tool in tools:
    print(str(tool)[:70])
    #print(type(tool))


print(type(tools))

#import numpy as np
#tools = np.array(tools_list)

# # 도구 목록 확인
# for name, tool in tools.items():
#     print(f"{name}: {tool.description}")
```

    Fund & Investment: 자금운용 및 조달, 투자 및 자금 관리, 자금 흐름 관리, 금융기관 협업, 투자 리스크 분석
    Forex & Exchange: 자금운용 및 조달, 외환 및 환율 관리, 환율 변동성 분석, 외화 자금 운용, 외환관리 시스템
    Corporate Reporting: 재무회계, 법인 결산 및 재무 보고, 월별, 분기별, 연간 결산 / 재무제표 작성, ER
    Tax Filing & Strategy: 세무 관리, 세금 신고 및 절세 전략, 법인세, 부가세 신고 / 절세 방안 마련, 세
    Recruitment & Mgmt: 채용 및 온보딩, 인재 채용 및 관리, 채용 공고, 지원자 관리, 입사 프로세스 운영, A
    Payroll & Benefits: 급여 및 복지 관리, 급여 정산 및 복지제도, 급여 계산, 4대 보험 신고, 연말정산, 급
    Executive Scheduling: 일정 및 의전 관리, 임원 일정 관리, 회의 일정 조율, 출장 예약, VIP 응대, 그
    Internal Comms: 문서 및 정보 관리, 내부 커뮤니케이션 관리, 내부 보고서 작성, 메일 응대, 문서 보안 유지, 
    Strategy Planning: 사업 기획 및 분석, 중장기 전략 수립, 시장 분석, 경쟁사 벤치마킹, 사업 타당성 평가, 
    Budgeting & Execution: 예산 및 비용 관리, 연간 예산 편성 및 집행, 부서별 예산 수립, 예산 사용 모니터
    Network & Security: IT 인프라 관리, 네트워크 및 보안 관리, 기업 내부 IT 인프라 운영, 데이터 보안 정
    System Maintenance: 시스템 운영 및 유지보수, 업무 시스템 유지보수, ERP, 그룹웨어, 사내 시스템 유지보수
    Job Posting & Mgmt: 인재 채용 및 온보딩, 채용 공고 및 후보자 관리, 채용 공고 등록, 지원자 관리, 면접 
    Onboarding & Training: 인재 채용 및 온보딩, 온보딩 및 초기 교육, 신입사원 온보딩 프로그램 운영, 사내 
    Payroll Processing: 보상 및 급여 관리, 급여 계산 및 지급, 월급, 인센티브, 상여금 정산 및 지급, 세금 
    Benefits Management: 복리후생 및 직원 혜택, 복지제도 운영 및 관리, 사내 복지제도 기획 및 운영(건강검진,
    Performance Review: 성과 관리 및 평가, 연간 평가 및 피드백, 연말 성과 평가 진행, 360도 피드백 시스템
    Promotions & Rewards: 승진 및 보상 관리, 승진 심사 및 보상 정책, 승진 심사 기준 설정, 보상 정책 수립
    Leadership Training: 사내 교육 및 개발, 리더십 및 직무 교육, 사내 리더십 과정 운영, 직무 스킬 향상 교
    Career Mentoring: 커리어 개발, 멘토링 및 경력 개발 프로그램, 사내 멘토링 프로그램 운영, 개인별 경력 개발 
    Attendance Mgmt: 근태 및 근무제 관리, 근태 및 출퇴근 관리, 출퇴근 기록 관리, 원격 근무제 운영, 유연 근무
    Culture & Feedback: 조직문화 및 커뮤니케이션, 직원 만족도 조사 및 문화 개선, 조직문화 진단, 직원 만족도 
    Childcare Support: 복리후생 관리, 사내 보육 지원, 사내 어린이집 운영, 보육 지원금 지급, 어린이집 입소 신
    Employee Loan: 복리후생 관리, 임직원 대출 지원, 사내 대출 신청 접수 및 심사, 대출 한도 및 이자 관리, 상환
    Benefits Inquiry: 직원 복지 및 지원, 복지 정책 상담, 복지제도 관련 문의 접수, 직원 복지 상담, 복지 항목
    Funeral Benefits: 경조금 및 지원금 관리, 경조사 지원금 운영, 경조금 신청 접수, 지급 심사 및 승인, 경조사
    Anti-Abuse & Spam: devops, 안티어뷰징&안티스팸, 어뷰징과 스팸을 필터링하는 부서, 내부개발시스템, Ant
    Search Quality Eval: devops, 검색 품질 평가(Search Quality Evaluation), 검색 결
    Anti-Scraping & Sec: devops, 안티스크래핑&보안(Anti-Scraping & Security), 검색 시
    Search Penalty Sys: devops, 검색 패널티 시스템(Search Penalty System), 저품질 콘텐츠
    Search Reliability: devops, 검색 신뢰성 향상(Search Reliability Enhancement),
    NLU & QA: devops, 자연어 이해 및 질의 응답 (Natural Language Understanding & QA)
    Ranking & Personalization: devops, 검색 순위 및 추천 알고리즘 (Search Ranking & P
    Real-time Trend Analysis: 사일로(기획+개발+비즈), 실시간 검색 트렌드 분석 (Real-time Sear
    Keyword Marketing & Ads: 사일로(기획+개발+비즈), 검색 키워드 마케팅 및 광고 최적화 (Search Ke
    New Business Dev: 전략, 신사업개발, 신규사업개발부서, 내부개발시스템, New Business Dev, db=<
    Ad Client Mgmt: 마케팅, 광고주관리, 광고주 및 잠재광고주 고객 관리, 프로모션 기획 및 광고단가문의대응, 내부개
    BM & Cost Management: 사업, BM 및 원가관리, 광고BM관리 및 광고노출원가 및 단가산정, 내부개발시스템, 
    Contract & Settling: 사업, 계약 및 정산관리, 인바운드 아웃바운드 계약관리 및 정, 내부개발시스템, Cont
    Ad & Client Portal: 기획, 광고 및 광고주 포털 기획, 광고 및 광고주 포털 기획, 내부개발시스템, Ad & 
    Ad Exposure Dev: devops, 광고노출시스템 개발, 서비스 페이지에서 노출될 광고 서비스 개발 및 운영, 내부개
    Ad Client Portal Dev: devops, 광고주 포털 개발, 광고주가 네이버에서 광고를 올리기 위해 접속하는 포털
    Search System Dev: devops, 검색시스템제공, 검색기반기술을 활용하여 타사에 검색시스템을 구축해주는 사업 진
    Search SRE: devops, 검색 SRE, 검색서비스 인프라관리 및 개발을 위한 부서, 내부개발시스템, Search S
    Ad Product Planning: 광고운영, 광고 상품 기획 및 운영, 광고 상품 개발, 운영 전략 수립, 광고 운영 시스
    Ad Performance Opt: 광고운영, 성과 분석 및 최적화, 광고 성과 모니터링, 최적화 방안 도출, 광고 데이터 분
    Ad Client Support: 광고운영, 광고주 지원 및 관리, 광고주 컨설팅, 캠페인 성과 보고, CRM, 광고주 포털,
    Product Categorization: 상품 카테고리 관리, 상품 분류 체계 관리, 상품 카테고리 정의 및 운영, 상품 관
    Category Trend Analysis: 상품 카테고리 관리, 카테고리별 트렌드 분석, 인기 상품 및 트렌드 분석, 데이터
    Seller Policy Dev: 판매 정책, 입점 및 판매 정책 수립, 입점 가이드라인 설정, 판매 정책 개발, ERP, C
    Seller Support: 판매자 지원, 셀러 지원 및 관리, 판매자 교육, 입점 프로세스 지원, 셀러 관리 시스템, Sel
    Marketplace Mgmt: 판매자 지원, 마켓플레이스 운영, 상품 등록 및 품질 관리, 상품 데이터 관리 시스템, Mar
    Review & Rating Mgmt: 판매자 지원, 리뷰 및 평점 관리, 리뷰 모니터링, 평점 시스템 최적화, 리뷰 관리 솔
    Membership Product Dev: 멤버십 운영, 멤버십 상품 기획, 혜택 및 상품 기획, 멤버십 관리 시스템, Mem
    Membership Data Analysis: 멤버십 운영, 멤버십 데이터 분석, 사용자 분석 및 리텐션 전략, 데이터 분석 
    Membership Promo Dev: 멤버십 운영, 멤버십 프로모션 기획, 할인 및 쿠폰 프로모션 운영, 쿠폰 및 포인트 시
    Order & Return Mgmt: 고객 지원, 주문 및 반품 관리, 고객 문의 대응, 반품 프로세스 최적화, CS 관리 시
    User Review Analysis: 고객 지원, 사용자 리뷰 분석, 고객 피드백 분석 및 서비스 개선, NLP 기반 분석 
    Claims & Dispute Mgmt: 고객 지원, 클레임 및 분쟁 처리, 구매자/판매자 분쟁 해결 지원, 클레임 처리 시스
    Logistics Partner Mgmt: 배송 및 물류, 물류 파트너 관리, 배송 업체 계약 및 운영 관리, WMS(창고관리
    Delivery Optimization: 배송 및 물류, 배송 최적화, 실시간 배송 모니터링 및 개선, 실시간 트래킹 시스템,
    Returns & Exchanges: 배송 및 물류, 반품 및 교환 프로세스, 반품 및 교환 기준 운영, 반품 관리 시스템, 
    Purchase Pattern Analysis: 데이터 기반 최적화, 구매 패턴 분석, 사용자 행동 분석 및 AI 추천, AI
    Ad Efficiency Analysis: 데이터 기반 최적화, 광고 효율 분석, 광고 성과 모니터링 및 인사이트 도출, BI
    Payment System Ops: 전자결제(PG), 결제 시스템 운영, 온라인 및 오프라인 결제 프로세스 관리, PG 시스템
    Payment Data Settling: 전자결제(PG), 결제 데이터 정산, 거래 정산, 정산 오류 검출, 정산 시스템, E
    Refund & Claim Mgmt: 전자결제(PG), 환불 및 클레임 처리, 결제 취소, 환불 요청 처리, CS 관리 시스템
    Simplified Payment Ops: 간편결제, 간편결제 서비스 운영, QR 결제, NFC 결제, 정기결제 서비스 운영,
    Cross-border Payments: 간편결제, 해외 결제 확장, 글로벌 결제 네트워크 연동, 해외 PG 시스템, Cros
    Merchant Support: 간편결제, 가맹점 지원, 가맹점 등록 및 결제 서비스 제공, 가맹점 관리 시스템, POS 연동
    Open API Management: 금융 플랫폼, 오픈 API 운영, 핀테크 API 제공 및 연동 지원, API Gatewa
    Data Analysis & Insights: 금융 플랫폼, 데이터 분석 및 인사이트, 사용자 금융 데이터 분석, BI 시스템
    Financial Product Intermediary: 금융 플랫폼, 금융상품 중개, 대출, 보험, 투자상품 연계 및 운영,
    User Authentication: 보안 및 인증, 사용자 인증, FIDO, OTP, 바이오 인증 등 보안 인증 관리, 인증
    Fraud Detection: 보안 및 인증, 이상거래 탐지, AI 기반 사기 탐지 및 분석, Fraud Detection S
    Credit Risk Analysis: 리스크 관리, 신용평가 및 리스크 분석, 머신러닝 기반 신용평가 운영, 신용평가 모델,
    Fraud Detection: 리스크 관리, 부정거래 감지, 비정상적인 금융 거래 패턴 분석, AML(자금세탁방지) 시스템, 
    Loan Application & Approval: 디지털 대출, 대출 신청 및 심사, 온라인 대출 신청 및 자동 심사 프로세
    Loan Arrears Mgmt: 디지털 대출, 대출 연체 관리, 연체 고객 모니터링 및 채권 관리, 채권 관리 시스템(Deb
    Real-time Transfer Ops: 송금 및 정산, 실시간 송금 서비스 운영, P2P 송금, 해외 송금, 실시간 정산 
    Virtual Account Ops: 송금 및 정산, 가상계좌 운영, 가상계좌 기반 결제 및 정산, 가상계좌 시스템(Virtu
    Points Accumulation & Usage: 포인트 및 리워드, 포인트 적립 및 사용, 결제 기반 리워드 운영, 포인트
    Affiliate Points Integration: 포인트 및 리워드, 제휴사 포인트 연동, 다양한 제휴사 리워드 프로그램과
    Payment Issue Resolution: 고객 지원, 결제 오류 및 문의 처리, 결제 오류 분석 및 고객 대응, 고객센터
    Data Center Ops: 클라우드 인프라, 데이터센터 운영, 데이터센터 구축 및 유지보수, DCIM(데이터센터 인프라 관
    Server Management: 클라우드 인프라, 서버 운영 및 관리, 가상화 서버 및 Bare Metal 서버 운영, 서버
    Storage Management: 클라우드 인프라, 스토리지 관리, 객체 스토리지, 블록 스토리지 운영, 클라우드 스토리지 
    Network Operations: 클라우드 인프라, 네트워크 운영, 클라우드 네트워크 구축 및 유지보수, SDN(소프트웨어 
    Container Service Ops: 클라우드 플랫폼, 컨테이너 서비스 운영, Kubernetes 기반 컨테이너 관리, K
    Serverless Computing: 클라우드 플랫폼, 서버리스 컴퓨팅, FaaS(Function as a Service) 
    DBaaS Operations: 클라우드 플랫폼, 데이터베이스 서비스(DBaaS), 클라우드 기반 DBMS 운영, Manage
    AI/ML Platform Ops: 클라우드 플랫폼, AI/ML 플랫폼 운영, 클라우드 기반 머신러닝 학습 환경 제공, AI 
    Account & Access Control: 클라우드 보안, 계정 및 접근 제어, IAM(Identity Access Man
    Data Encryption: 클라우드 보안, 데이터 암호화, 저장 및 전송 데이터 암호화 관리, Encryption Mana
    Security Monitoring: 클라우드 보안, 보안 모니터링, 이상 트래픽 감지 및 대응, SIEM(보안 정보 및 이벤
    DDoS Protection: 클라우드 보안, DDoS 방어, 클라우드 DDoS 방어 시스템 운영, DDoS Protectio
    Speech-to-Text: AI 서비스, 음성 인식(Speech-to-Text), AI 기반 음성 데이터 처리, STT(음성
    Natural Language Processing: AI 서비스, 자연어 처리(NLP), 챗봇 및 문서 요약 AI 운영, NL
    Image & Video Recognition: AI 서비스, 이미지 및 영상 인식, OCR, 얼굴 인식 서비스 운영, AI 
    Text-to-Speech AI: AI 서비스, AI 음성 합성(Text-to-Speech), TTS 서비스 운영, TTS A
    Corporate Email Service: 협업 솔루션, 기업용 이메일 서비스, 클라우드 이메일 운영 및 보안 관리, 메일 
    Video Conference: 협업 솔루션, 화상회의 솔루션, 클라우드 기반 영상 회의 시스템 운영, Video Confer
    File Sharing & Storage: 협업 솔루션, 파일 공유 및 스토리지, 기업용 클라우드 저장소 운영, 클라우드 드라
    Workflow Automation: 협업 솔루션, 업무 자동화, 워크플로우 및 자동화 툴 제공, RPA(로봇 프로세스 자동화
    CI/CD Pipeline Dev: DevOps, CI/CD 파이프라인 구축, 자동 배포 및 빌드 시스템 운영, Jenkins
    Monitoring & Logging: DevOps, 모니터링 및 로깅, 클라우드 기반 시스템 모니터링, Prometheus,
    IaC Implementation: DevOps, IaC(Infrastructure as Code), Terraform 및 A
    Data Lake Operations: 빅데이터 분석, 데이터 레이크 운영, 대량 데이터 저장 및 분석 환경 구축, Data 
    Real-time Data Streaming: 빅데이터 분석, 실시간 데이터 스트리밍, 실시간 데이터 수집 및 처리, Kafk
    Data Visualization: 빅데이터 분석, 데이터 시각화, BI 및 대시보드 서비스 운영, Tableau, Looke
    Blockchain Node Ops: 블록체인 서비스, 블록체인 노드 운영, 블록체인 네트워크 유지보수, Hyperledger
    Smart Contract Dev: 블록체인 서비스, 스마트 컨트랙트 개발, 블록체인 기반 계약 자동화, Solidity, S
    Cloud Tech Support: 고객 지원, 클라우드 기술 지원, 고객 문의 및 기술 문제 해결, 고객센터 CRM, AI 
    Cloud Guide Support: 고객 지원, 클라우드 사용 가이드 제공, 사용 매뉴얼 및 교육 콘텐츠 제작, 클라우드 포
    <class 'list'>


## 3. 계획 테스트


```python
# 도구 선택 계획 생성기
class ToolPlanGenerator:
    def __init__(self, tools: Dict[str, Tool], model_name="gpt-4o", temperature=0):
        self.tools = tools
        self.llm = ChatOpenAI(temperature=temperature, model_name=model_name, max_tokens=1500)

        # 도구 설명 구성
        tool_descriptions = "\n".join([f"- {tool.tool_name}: {tool.description}" for tool in self.tools])

        self.template = """
        당신은 사용자 질문을 분석하여 적절한 도구와 쿼리를 결정하는 전문가입니다.

        # 사용 가능한 도구
        {tool_descriptions}

        # 이전 대화 맥락
        이전 질문: {prev_query}
        이전 답변: {prev_response}

        # 현재 질문
        사용자 질문: {current_query}

        # 분석 지침
        1. 사용자의 질문을 분석하고, 이전 대화 맥락도 고려하세요.
        2. 질문에 여러 개의 요청이 포함되어 있다면, 각각에 대해 별도의 쿼리를 생성하세요.
        3. 각 쿼리가 어떤 도구를 사용해야 하는지 결정하세요.
        4. 질문이 어떤 도구와도 관련이 없다면 'no_tool'을 선택하세요.
        5. 마크다운 형식으로 작성하지 마세요.

        # 출력 형식
        반드시 다음 JSON 형식으로 출력하세요:
        {{
          "plan": {{
            "도구이름1": ["쿼리1", "쿼리2", ...],
            "도구이름2": ["쿼리3", "쿼리4", ...],
            ...
          }}
        }}
        """

        self.prompt = PromptTemplate(
            input_variables=["tool_descriptions", "prev_query", "prev_response", "current_query"],
            template=self.template
        )

    def generate_plan(self, prev_query: str, prev_response: str, current_query: str) -> Dict:
        # 도구 설명 구성
        tool_descriptions = "\n".join([f"- {tool.tool_name}: {tool.description}" for tool in self.tools])

        try:
            # 프롬프트 준비 및 실행
            formatted_prompt = self.prompt.format(
                tool_descriptions=tool_descriptions,
                prev_query=prev_query,
                prev_response=prev_response,
                current_query=current_query
            )

            # LLM 호출
            llm_response = self.llm.invoke(formatted_prompt)
            llm_content = llm_response.content
            
            # print("1------")
            # print(llm_response)
            # print(type(llm_response))
            # print("2------")
            # JSON 파싱
            try:
                plan_json = json.loads(llm_content) # plan_json 형식 검증
                # print("3------")
                # print(type(plan_json))
                plan_keys = plan_json["plan"].keys()
                # print(type(plan_keys))
                # print("3------")
                plan_json.update({"selected_tools": list(plan_keys)})   
                if "plan" not in plan_json:
                    print(f"응답에 'plan' 키가 없습니다: {plan_json}")
                    return {"plan": {"no_tool": [current_query]}}
                # print(plan_keys)
                print(f"agent-planner 응답 : {plan_json}")
                return plan_json
            except json.JSONDecodeError as json_err:
                print(f"JSON 파싱 오류: {str(json_err)}")
                print(f"LLM 원본 응답: {llm_content}")
                return {"plan": {"no_tool": [current_query]}}
        except Exception as e:
            print(f"계획 생성 중 오류 발생: {str(e)}")
            return {"plan": {"no_tool": [current_query]}}
```


```python
# 계획 생성기 테스트
planner = ToolPlanGenerator(tools)
## print(planner)
test_plan = planner.generate_plan("", "", "사내복지대출은 어디서 물어봐야해?")
#print("생성된 계획:")
print(json.dumps(test_plan, indent=2, ensure_ascii=False))
```

    agent-planner 응답 : {'plan': {'Employee Loan': ['사내복지대출 관련 문의는 어디서 해야 하는지 알려주세요.']}, 'selected_tools': ['Employee Loan']}
    {
      "plan": {
        "Employee Loan": [
          "사내복지대출 관련 문의는 어디서 해야 하는지 알려주세요."
        ]
      },
      "selected_tools": [
        "Employee Loan"
      ]
    }


## 4. 도구 선택 능력 + 멀티턴 + 멀티 쿼리 시스템


```python
# 멀티 도구 RAG 시스템
class MultiToolRAG:
    def __init__(self, tools: Dict[str, Tool], model_name="gpt-4o", temperature=0.2):
        self.tools = tools
        self.tool_planner = ToolPlanGenerator(tools, model_name)
        self.llm = ChatOpenAI(temperature=temperature, model_name=model_name)

        self.prev_query = ""
        self.prev_response = ""

        # 응답 생성 프롬프트
        self.response_template = """
        당신은 사용자 질문에 대한 답변을 제공하는 AI 어시스턴트입니다.

        사용자 질문: {query}

        다음은 질문에 관련된 정보입니다:
        {context}

        # 지침
        1. 제공된 정보를 바탕으로 사용자 질문에 답변하세요.
        2. 사용자 질문에 여러 질의가 있다면 각각에 대해 답변하세요.
        3. 제공된 정보가 충분하지 않으면 솔직히 모른다고 답변하세요.
        4. 답변은 한국어로 작성하세요.
        """

        self.response_prompt = PromptTemplate(
            input_variables=["query", "context"],
            template=self.response_template
        )

    def update_conversation(self, query: str, response: str):
        self.prev_query = query
        self.prev_response = response

    def search_with_tool(self, tool_name: str, queries: List[str], num_docs=3) -> List[Document]:
        # 선택된 도구로 문서 검색
        tool = "no_tool"
        #tool = self.tools.get(tool_name)
        #if not tool or tool_name == "no_tool" or not tool.vectorstore:
        #return []  # no_tool이거나 벡터스토어가 없는 경우 빈 리스트 반환

        all_docs = []
        seen_contents = set()

        for query in queries:
            try:
                docs = tool.vectorstore.similarity_search(query, k=num_docs)
                print(f"{tool_name} 도구로 '{query}' 검색 완료: {len(docs)}개 문서 찾음")

                # 중복 제거
                for doc in docs:
                    if doc.page_content not in seen_contents:
                        seen_contents.add(doc.page_content)
                        # 메타데이터에 도구 및 쿼리 정보 추가
                        if not hasattr(doc, 'metadata') or doc.metadata is None:
                            doc.metadata = {}
                        doc.metadata['tool'] = tool_name
                        doc.metadata['query'] = query
                        all_docs.append(doc)
            except Exception as e:
                print(f"{tool_name} 도구로 '{query}' 검색 중 오류 발생: {str(e)}")

        return all_docs

    def query(self, current_query: str) -> Dict:
        # 도구 계획 생성
        plan = self.tool_planner.generate_plan(self.prev_query, self.prev_response, current_query)
        print(f"생성된 계획: {json.dumps(plan, indent=2, ensure_ascii=False)}")

        # 모든 도구에서 검색 수행
        all_docs = []
        for tool_name, queries in plan.get("plan", {}).items():
            tool_docs = self.search_with_tool(tool_name, queries)
            all_docs.extend(tool_docs)
            print(f"{tool_name} 도구에서 {len(tool_docs)}개 문서 검색됨")

        # 검색 결과가 없고 no_tool이 아닌 경우, no_tool 추가
        if not all_docs and "no_tool" not in plan.get("plan", {}):
            #print("검색 결과가 없어 no_tool 사용")
            plan["plan"]["no_tool"] = [current_query]

        # 컨텍스트 구성
        if all_docs:
            context = "\n\n".join([f"[{doc.metadata.get('tool', 'unknown')}] 문서 {i+1}:\n{doc.page_content}"
                                 for i, doc in enumerate(all_docs)])
        else:
            context = "관련 문서가 검색되지 않았습니다."

        # 응답 생성
        formatted_prompt = self.response_prompt.format(
            query=current_query,
            context=context
        )

        result = self.llm.invoke(formatted_prompt)
        response = result.content

        # 대화 기록 업데이트
        self.update_conversation(current_query, response)

        # 결과 반환
        return {
            "query": current_query,
            "result": response,
            "plan": plan,
            "source_documents": all_docs
        }
```


```python
# RAG 시스템 초기화
rag_system = MultiToolRAG(tools)
```


```python
# 대화 예시 1: 일본 ICT 관련 질문
query1 = "어뷰징 신고는 어디서 해"
result1 = rag_system.query(query1)

print(f"\n질문: {query1}")
print(f"답변:\n{result1['result']}")
```

    agent-planner 응답 : {'plan': {'Anti-Abuse & Spam': ['어뷰징 신고 처리 방법']}, 'selected_tools': ['Anti-Abuse & Spam']}
    생성된 계획: {
      "plan": {
        "Anti-Abuse & Spam": [
          "어뷰징 신고 처리 방법"
        ]
      },
      "selected_tools": [
        "Anti-Abuse & Spam"
      ]
    }
    Anti-Abuse & Spam 도구로 '어뷰징 신고 처리 방법' 검색 중 오류 발생: 'str' object has no attribute 'vectorstore'
    Anti-Abuse & Spam 도구에서 0개 문서 검색됨
    
    질문: 어뷰징 신고는 어디서 해
    답변:
    죄송하지만, 어뷰징 신고에 대한 구체적인 정보를 제공할 수 없습니다. 일반적으로 어뷰징 신고는 해당 서비스나 플랫폼의 고객 지원 센터나 신고 기능을 통해 할 수 있습니다. 사용하는 서비스의 공식 웹사이트나 앱에서 신고 절차를 확인해 보시기 바랍니다.



```python
# 대화 예시 2: 멀티쿼리 및 멀티턴
query2 = "음 그렇다면 사내복지기금 정보는 어디서 물어봐?"
result2 = rag_system.query(query2)

print(f"\n질문: {query2}")
print(f"답변:\n{result2['result']}")
```

    agent-planner 응답 : {'plan': {'Benefits Inquiry': ['사내복지기금 정보 문의']}, 'selected_tools': ['Benefits Inquiry']}
    생성된 계획: {
      "plan": {
        "Benefits Inquiry": [
          "사내복지기금 정보 문의"
        ]
      },
      "selected_tools": [
        "Benefits Inquiry"
      ]
    }
    Benefits Inquiry 도구로 '사내복지기금 정보 문의' 검색 중 오류 발생: 'str' object has no attribute 'vectorstore'
    Benefits Inquiry 도구에서 0개 문서 검색됨
    
    질문: 음 그렇다면 사내복지기금 정보는 어디서 물어봐?
    답변:
    죄송하지만, 제공된 정보로는 사내복지기금에 대한 구체적인 내용을 알 수 없습니다. 사내복지기금에 대한 정보는 일반적으로 회사의 인사부서나 복지 담당 부서에 문의하시면 자세한 안내를 받을 수 있습니다. 회사의 내부 포털이나 인트라넷에 관련 정보가 게시되어 있을 수도 있으니 참고하시기 바랍니다.



```python
# 대화 예시 3: 도구와 관련 없는 질문
query3 = "너는 누구니?"
result3 = rag_system.query(query3)

print(f"\n질문: {query3}")
print(f"답변:\n{result3['result']}")
```

    agent-planner 응답 : {'plan': {'no_tool': ['너는 누구니?']}, 'selected_tools': ['no_tool']}
    생성된 계획: {
      "plan": {
        "no_tool": [
          "너는 누구니?"
        ]
      },
      "selected_tools": [
        "no_tool"
      ]
    }
    no_tool 도구로 '너는 누구니?' 검색 중 오류 발생: 'str' object has no attribute 'vectorstore'
    no_tool 도구에서 0개 문서 검색됨
    
    질문: 너는 누구니?
    답변:
    저는 사용자 질문에 대한 답변을 제공하는 AI 어시스턴트입니다. 더 궁금한 점이 있으면 언제든지 물어보세요!


## 5. 챗봇 UI


```python
!pip install gradio --quiet
```


```python
import gradio as gr

# 챗봇의 응답을 처리하는 함수 (qa_chain 함수는 미리 정의되어 있어야 합니다)
def respond(message, chat_history):
    # 메시지 처리: qa_chain 또는 rag_system.query 함수 호출
    result = rag_system.query(message)
    bot_message = result['result']

    # 채팅 기록 업데이트: (사용자 메시지, 챗봇 응답) 튜플 추가
    chat_history.append((message, bot_message))
    return "", chat_history

# Gradio Blocks 인터페이스 생성
with gr.Blocks() as demo:
    # 챗봇 채팅 기록 표시 (좌측 상단 레이블 지정)
    chatbot = gr.Chatbot(label="사내 업무지원 AI 챗봇 ")
    # 사용자 입력 텍스트박스 (하단 레이블 지정)
    msg = gr.Textbox(label="질문해주세요!")
    # 입력창과 채팅 기록 모두 초기화할 수 있는 ClearButton
    clear = gr.ClearButton([msg, chatbot])

    # 사용자가 텍스트박스에 입력 후 제출하면 respond 함수 호출
    msg.submit(respond, inputs=[msg, chatbot], outputs=[msg, chatbot])

# 인터페이스 실행 (debug=True로 실행하면 디버깅 정보를 확인할 수 있습니다)
demo.launch(debug=True)
```

    /Users/kep-dong22/modu-llm/venv/lib/python3.10/site-packages/gradio/components/chatbot.py:285: UserWarning: You have not specified a value for the `type` parameter. Defaulting to the 'tuples' format for chatbot messages, but this is deprecated and will be removed in a future version of Gradio. Please set type='messages' instead, which uses openai-style dictionaries with 'role' and 'content' keys.
      warnings.warn(


    * Running on local URL:  http://127.0.0.1:7860
    
    To create a public link, set `share=True` in `launch()`.



<div><iframe src="http://127.0.0.1:7860/" width="100%" height="500" allow="autoplay; camera; microphone; clipboard-read; clipboard-write;" frameborder="0" allowfullscreen></iframe></div>


    agent-planner 응답 : {'plan': {'Benefits Inquiry': ['복지제도 관련 정보를 알아보는 방법', '복지제도에 대한 문의를 어디서 할 수 있는지']}, 'selected_tools': ['Benefits Inquiry']}
    생성된 계획: {
      "plan": {
        "Benefits Inquiry": [
          "복지제도 관련 정보를 알아보는 방법",
          "복지제도에 대한 문의를 어디서 할 수 있는지"
        ]
      },
      "selected_tools": [
        "Benefits Inquiry"
      ]
    }
    Benefits Inquiry 도구로 '복지제도 관련 정보를 알아보는 방법' 검색 중 오류 발생: 'str' object has no attribute 'vectorstore'
    Benefits Inquiry 도구로 '복지제도에 대한 문의를 어디서 할 수 있는지' 검색 중 오류 발생: 'str' object has no attribute 'vectorstore'
    Benefits Inquiry 도구에서 0개 문서 검색됨
    agent-planner 응답 : {'plan': {'Childcare Support': ['사내 어린이집 운영에 대한 정보는 어디에서 확인할 수 있나요?', '사내 어린이집 지원을 받으려면 어떻게 해야 하나요?']}, 'selected_tools': ['Childcare Support']}
    생성된 계획: {
      "plan": {
        "Childcare Support": [
          "사내 어린이집 운영에 대한 정보는 어디에서 확인할 수 있나요?",
          "사내 어린이집 지원을 받으려면 어떻게 해야 하나요?"
        ]
      },
      "selected_tools": [
        "Childcare Support"
      ]
    }
    Childcare Support 도구로 '사내 어린이집 운영에 대한 정보는 어디에서 확인할 수 있나요?' 검색 중 오류 발생: 'str' object has no attribute 'vectorstore'
    Childcare Support 도구로 '사내 어린이집 지원을 받으려면 어떻게 해야 하나요?' 검색 중 오류 발생: 'str' object has no attribute 'vectorstore'
    Childcare Support 도구에서 0개 문서 검색됨
    agent-planner 응답 : {'plan': {'Internal Comms': ['사내 템플릿과 문서 양식을 확인할 수 있는 시스템이나 위치를 안내해 주세요.']}, 'selected_tools': ['Internal Comms']}
    생성된 계획: {
      "plan": {
        "Internal Comms": [
          "사내 템플릿과 문서 양식을 확인할 수 있는 시스템이나 위치를 안내해 주세요."
        ]
      },
      "selected_tools": [
        "Internal Comms"
      ]
    }
    Internal Comms 도구로 '사내 템플릿과 문서 양식을 확인할 수 있는 시스템이나 위치를 안내해 주세요.' 검색 중 오류 발생: 'str' object has no attribute 'vectorstore'
    Internal Comms 도구에서 0개 문서 검색됨

