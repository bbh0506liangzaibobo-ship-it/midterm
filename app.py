import streamlit as st
import tempfile
import os
import requests
import json
import numpy as np
from groq import Groq
from pypdf import PdfReader

# Groq 클라이언트 초기화
client = Groq(api_key="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")

# 간단한 텍스트 분할
def split_text(text, chunk_size=1000, chunk_overlap=200):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - chunk_overlap
    return chunks

# 업로드된 파일 처리
def process_uploaded_file(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name
    
    try:
        reader = PdfReader(tmp_file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        
        # 텍스트 분할
        chunks = split_text(text)
        
        return chunks, text
    finally:
        os.unlink(tmp_file_path)

# 간단한 의미론적 검색
def semantic_search(query, chunks):
    # 간단한 키워드 매칭 (실제 프로젝트에서는 임베딩 모델 사용 가능)
    query_words = set(query.lower().split())
    scored_chunks = []
    
    for i, chunk in enumerate(chunks):
        chunk_words = set(chunk.lower().split())
        score = len(query_words.intersection(chunk_words))
        scored_chunks.append((score, chunk, i))
    
    # 관련성 기준 정렬
    scored_chunks.sort(reverse=True)
    return [chunk for _, chunk, _ in scored_chunks[:3]]

# 답변 생성
def generate_response(query, relevant_chunks):
    context = "\n\n".join(relevant_chunks)
    
    prompt = f"""
    다음 문서 내용을 참고하여 질문에 답변해주세요:

    문서 내용:
    {context}

    질문: {query}

    답변:
    """
    
    try:
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.1-8b-instant",
            temperature=0.1,
            max_tokens=1024
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"답변 생성 중 오류 발생: {str(e)}"

# Streamlit 앱
def main():
    st.set_page_config(
        page_title="RAG 챗봇 - 중간고사 프로젝트",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🤖 RAG 챗봇 - 중간고사 프로젝트")
    st.markdown("---")
    
    # 세션 상태 초기화
    if "history" not in st.session_state:
        st.session_state.history = []
    if "chunks" not in st.session_state:
        st.session_state.chunks = None
    if "raw_text" not in st.session_state:
        st.session_state.raw_text = None
    
    # 사이드바 - 문서 업로드
    with st.sidebar:
        st.header("📁 문서 업로드")
        uploaded_file = st.file_uploader(
            "PDF 파일을 업로드하세요", 
            type=['pdf'],
            help="PDF 문서를 업로드하면 AI가 해당 내용을 기반으로 답변합니다."
        )
        
        if uploaded_file and st.button("문서 처리 시작"):
            with st.spinner("문서 처리 중... 잠시만 기다려주세요."):
                chunks, raw_text = process_uploaded_file(uploaded_file)
                st.session_state.chunks = chunks
                st.session_state.raw_text = raw_text
                st.success(f"✅ 문서 처리가 완료되었습니다! ({len(chunks)}개의 텍스트 조각)")
                
                # 처리 완료 메시지 추가
                st.session_state.history.append({
                    'role': 'assistant', 
                    'content': f'문서 "{uploaded_file.name}"이(가) 성공적으로 처리되었습니다. 이제 문서 내용에 대해 질문해주세요!'
                })
        
        st.markdown("---")
        st.info("""
        **📋 사용 방법:**
        1. 왼쪽에서 PDF 문서 업로드
        2. '문서 처리 시작' 버튼 클릭
        3. 아래 채팅창에서 질문 입력
        4. AI가 문서 기반 답변 제공
        
        **🛠️ 기술 스택:**
        - **Groq API**: 고속 AI 모델
        - **PyPDF**: 문서 처리
        - **Streamlit**: 웹 인터페이스
        """)
        
        # 시스템 상태 표시
        st.markdown("---")
        st.subheader("시스템 상태")
        if st.session_state.chunks:
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
        # 사용자 메시지 추가
        st.session_state.history.append({'role': 'user', 'content': user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
        
        # 응답 생성
        if st.session_state.chunks:
            with st.chat_message("assistant"):
                with st.spinner("답변 생성 중..."):
                    # 관련 문서 검색
                    relevant_chunks = semantic_search(user_input, st.session_state.chunks)
                    # 답변 생성
                    response = generate_response(user_input, relevant_chunks)
                    
                    st.markdown(response)
                    st.session_state.history.append({'role': 'assistant', 'content': response})
                    
                    # 참고 문서 표시
                    with st.expander("🔍 참고 문서 보기"):
                        st.write("다음 문서 조각을 참고하여 답변을 생성했습니다:")
                        for i, chunk in enumerate(relevant_chunks):
                            st.markdown(f"**참고 {i+1}:**")
                            st.text(chunk[:300] + "..." if len(chunk) > 300 else chunk)
                            st.markdown("---")
        else:
            warning_msg = "⚠️ 먼저 문서를 업로드하고 처리해주세요!"
            st.warning(warning_msg)
            st.session_state.history.append({'role': 'assistant', 'content': warning_msg})

    # 대화 기록 관리
    if st.session_state.history and st.sidebar.button("대화 기록 지우기"):
        st.session_state.history = []
        st.rerun()

    # 푸터
    st.markdown("---")
    st.caption("중간고사 프로젝트 - RAG 챗봇 | Streamlit Cloud 배포")

if __name__ == "__main__":
    main()
