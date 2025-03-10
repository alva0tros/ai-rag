# app/services/prompt_service.py
import logging
from langchain_ollama import ChatOllama

logger = logging.getLogger(__name__)


class ImagePrompt:
    """프롬프트 처리 서비스: 번역, 개선, 확장 기능 제공"""

    def __init__(self):
        self._llm = None

    @property
    def llm(self):
        """LLM 인스턴스 가져오기"""
        if self._llm is None:
            self._llm = ChatOllama(
                model="phi4:latest", temperature=0.7, num_predict=512
            )
        return self._llm

    async def translate_and_enhance(self, message: str) -> str:
        """
        메시지를 영어로 번역하고 필요시 내용을 풍부하게 만듦

        Args:
            message: 원본 사용자 메시지

        Returns:
            str: 번역 및 향상된 영어 프롬프트
        """
        try:
            if len(message) < 20:  # 짧은 메시지 기준 (조정 가능)
                prompt = f"""
                아래 한국어 메시지를 상세한 이미지 생성용 영어 프롬프트로 변환해주세요.
                메시지가 짧거나 구체적이지 않은 경우, 이미지 생성에 도움될 디테일을 추가해주세요.
                결과는 영어로만 작성하고, 설명 없이 프롬프트 텍스트만 반환해주세요.
                
                메시지: {message}
                """
            else:
                prompt = f"""
                아래 한국어 메시지를 이미지 생성용 영어 프롬프트로 번역해주세요.
                결과는 영어로만 작성하고, 설명 없이 프롬프트 텍스트만 반환해주세요.
                
                메시지: {message}
                """

            response = await self.llm.ainvoke(prompt)
            enhanced_prompt = response.content.strip()

            logger.info(f"Enhanced prompt: {enhanced_prompt[:100]}...")
            return enhanced_prompt

        except Exception as e:
            logger.error(f"Error enhancing prompt: {str(e)}")
            # 오류 발생 시 기본 번역만 시도 (실제 구현 시 더 강건한 방식 사용)
            return f"Generate an image of {message}"


# 싱글톤 인스턴스
image_prompt = ImagePrompt()
