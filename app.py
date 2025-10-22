import streamlit as st
import tempfile
import os
from langchain_groq import ChatGroq
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader, TextLoader

# 세션 상태 초기화
if "history" not in st.session_state:
    st.session_state.history = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

# 페이지 설정
st.set_page_config(
    page_title="RAG 챗봇 - 중간고사 프로젝트",
    page_icon="🤖",
    layout="wide"
)

# 모델 초기화
@st.cache_resource
def load_models():
    # API 키를 직접 코드에 작성
    groq_api_key = "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"
    
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=groq_api_key,
        temperature=0.1
    )
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return llm, embeddings

def process_uploaded_file(uploaded_file):
    """업로드된 파일 처리"""
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name
    
    try:
        # 파일 타입에 따른 로더 선택
        if uploaded_file.name.endswith('.pdf'):
            loader = PyPDFLoader(tmp_file_path)
        else:
            loader = TextLoader(tmp_file_path, encoding='utf-8')
        
        documents = loader.load()
        
        # 텍스트 분할
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        texts = text_splitter.split_documents(documents)
        
        return texts
    except Exception as e:
        st.error(f"파일 처리 중 오류 발생: {str(e)}")
        return []
    finally:
        # 임시 파일 정리
        if os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)

def main():
    st.title("🤖 RAG 챗봇 - 중간고사 프로젝트")
    st.markdown("---")
    
    # 사이드바 - 파일 업로드
    with st.sidebar:
        st.header("📁 문서 업로드")
        uploaded_file = st.file_uploader(
            "TXT 또는 PDF 파일을 업로드하세요",
            type=['txt', 'pdf'],
            help="문서를 업로드하면 AI가 해당 내용을 기반으로 답변합니다."
        )
        
        if uploaded_file and st.button("문서 처리 시작"):
            with st.spinner("문서 처리 중... 잠시만 기다려주세요."):
                try:
                    llm, embeddings = load_models()
                    texts = process_uploaded_file(uploaded_file)
                    
                    if texts:
                        # 벡터 저장소 생성
                        st.session_state.vectorstore = Chroma.from_documents(
                            documents=texts,
                            embedding=embeddings
                        )
                        
                        # RAG 체인 생성
                        st.session_state.qa_chain = RetrievalQA.from_chain_type(
                            llm=llm,
                            chain_type="stuff",
                            retriever=st.session_state.vectorstore.as_retriever(search_kwargs={"k": 3}),
                            return_source_documents=True
                        )
                        
                        st.success(f"✅ 문서 처리가 완료되었습니다! ({len(texts)}개의 텍스트 조각)")
                        st.session_state.history.append({
                            'role': 'assistant',
                            'content': f'문서 "{uploaded_file.name}"이(가) 성공적으로 처리되었습니다. 이제 문서 내용에 대해 질문해주세요!'
                        })
                    else:
                        st.error("❌ 문서에서 텍스트를 추출할 수 없습니다.")
                        
                except Exception as e:
                    st.error(f"❌ 문서 처리 중 오류 발생: {str(e)}")
        
        st.markdown("---")
        st.info("""
        **📋 사용 방법:**
        1. 왼쪽에서 문서 파일 업로드
        2. '문서 처리 시작' 버튼 클릭
        3. 아래 채팅창에서 질문 입력
        4. AI가 문서 기반 답변 제공
        
        **🛠️ 기술 스택:**
        - **Langchain**: RAG 시스템 구축
        - **Groq API**: 고속 AI 모델
        - **ChromaDB**: 벡터 데이터베이스
        - **Streamlit**: 웹 인터페이스
        """)
        
        # 현재 상태 표시
        st.markdown("---")
        st.subheader("시스템 상태")
        if st.session_state.qa_chain:
            st.success("✅ 문서 로드 완료")
        else:
            st.warning("⚠️ 문서를 업로드해주세요")

    # 채팅 인터페이스
    st.subheader("💬 문서 질의 응답")
    
    # 채팅 기록 표시
    for msg in st.session_state.history:
        with st.chat_message(msg['role']):
            st.markdown(msg['content'])
    
    # 사용자 입력
    user_input = st.chat_input("문서에 대해 궁금한 것을 질문해보세요...")
    
    if user_input:
        # 사용자 메시지 추가 및 표시
        st.session_state.history.append({
            'role': 'user',
            'content': user_input
        })
        
        with st.chat_message("user"):
            st.markdown(user_input)
        
        # 응답 생성
        if st.session_state.qa_chain:
            with st.chat_message("assistant"):
                with st.spinner("답변 생성 중..."):
                    try:
                        result = st.session_state.qa_chain.invoke({"query": user_input})
                        
                        # 응답 표시
                        response_text = result["result"]
                        st.markdown(response_text)
                        
                        # 채팅 기록에 추가
                        st.session_state.history.append({
                            'role': 'assistant',
                            'content': response_text
                        })
                        
                        # 참고 문서 표시
                        with st.expander("🔍 참고 문서 보기"):
                            st.write("다음 문서 조각을 참고하여 답변을 생성했습니다:")
                            for i, doc in enumerate(result["source_documents"]):
                                st.markdown(f"**참고 {i+1}:**")
                                st.text(doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content)
                                st.caption(f"출처: {doc.metadata.get('source', 'Unknown')}")
                                st.markdown("---")
                                
                    except Exception as e:
                        error_msg = f"답변 생성 중 오류가 발생했습니다: {str(e)}"
                        st.error(error_msg)
                        st.session_state.history.append({
                            'role': 'assistant',
                            'content': error_msg
                        })
        else:
            warning_msg = "⚠️ 먼저 문서를 업로드하고 처리해주세요!"
            st.warning(warning_msg)
            st.session_state.history.append({
                'role': 'assistant',
                'content': warning_msg
            })

    # 대화 기록 관리
    if st.session_state.history and st.sidebar.button("대화 기록 지우기"):
        st.session_state.history = []
        st.rerun()

    # 푸터
    st.markdown("---")
    st.caption("중간고사 프로젝트 - Langchain RAG 챗봇 | Streamlit Cloud 배포")

if __name__ == "__main__":

    main()

