import re
import asyncio
from langchain_ollama import ChatOllama
from langchain_community.chat_models import ChatLlamaCpp
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

# 전역 변수 (실제 서비스에서는 상태 관리를 별도 저장소로 처리하는 것이 좋습니다)
store = {}  # 채팅 기록을 저장할 딕셔너리
tasks = {}  # 진행 중인 작업을 저장하는 딕셔너리
