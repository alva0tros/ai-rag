"""
대화 제목 생성을 위한 유틸리티 모듈
다양한 언어에 최적화된 자연스러운 제목 생성을 지원합니다.
"""

import re
import logging

# 로거 설정
logger = logging.getLogger(__name__)

class AdvancedTitleGenerator:
    """다양한 언어에 최적화된 고급 제목 생성기"""
    
    def __init__(self):
        """필요한 모델과 처리기 초기화"""
        # 한국어 처리기 초기화
        try:
            # KoNLPy 가져오기 (필요시 설치)
            from konlpy.tag import Okt
            from konlpy.tag import Kkma
            self.okt = Okt()  # 빠른 형태소 분석
            self.kkma = Kkma()  # 정확한 구문 분석
            self.korean_enabled = True
            logger.info("한국어 처리기 초기화 성공")
        except Exception as e:
            logger.warning(f"한국어 처리기 초기화 실패: {e}")
            self.korean_enabled = False
        
        # spaCy 모델 초기화
        try:
            # spaCy 가져오기 (필요시 설치)
            import spacy
            import pytextrank
            
            # 다국어 지원 모델 로드 (가벼운 버전)
            self.nlp = spacy.load("xx_ent_wiki_sm")
            # TextRank 파이프라인 추가
            self.nlp.add_pipe("textrank")
            self.spacy_enabled = True
            logger.info("spaCy 모델 초기화 성공")
        except Exception as e:
            logger.warning(f"spaCy 모델 초기화 실패: {e}")
            self.spacy_enabled = False
    
    def is_korean(self, text: str) -> bool:
        """텍스트가 주로 한국어인지 확인"""
        korean_char_count = len(re.findall(r'[ㄱ-ㅎㅏ-ㅣ가-힣]', text))
        return korean_char_count > len(text) * 0.3  # 30% 이상이 한글이면 한국어로 간주
    
    def detect_language(self, text: str) -> str:
        """간단한 언어 감지 로직"""
        if self.is_korean(text):
            return "ko"
        
        # 기본값은 영어
        return "en"
    
    def remove_command_endings(self, text: str) -> str:
        """명령어 종결 표현 제거 (예: ~해줘, ~할래?)"""
        patterns = [
            r'(해)\s*(줘|주세요|볼래|볼까|보자)\s*$',
            r'(알려)\s*(줘|주세요|달라|다오)\s*$',
            r'(설명)\s*(해줘|해주세요|해봐|했으면)\s*$',
            r'(만들어)\s*(줘|주세요|볼래|볼까)\s*$',
            r'(보여)\s*(줘|주세요|달라|다오)\s*$',
            r'(\?|\.|!|습니다|입니다|어요|에요|거야|니다|세요|까요|실래|겠어|겠니)\s*$'
        ]
        
        result = text
        for pattern in patterns:
            result = re.sub(pattern, '', result).strip()
        
        return result
    
    def extract_key_phrases_korean(self, text: str, max_length: int = 30) -> str:
        """한국어 텍스트에서 핵심 구문 추출"""
        if not self.korean_enabled:
            # 한국어 처리기 사용 불가능한 경우 대체 로직
            return self.extract_key_phrases_generic(text, max_length)
        
        try:
            # 명령형 종결어 제거
            cleaned_text = self.remove_command_endings(text)
            
            # 너무 짧은 경우 그대로 반환
            if len(cleaned_text) <= max_length:
                return cleaned_text
            
            # 명사 추출 (주제어 후보)
            nouns = self.okt.nouns(cleaned_text)
            
            # 중요 명사구 추출
            noun_phrases = []
            for noun in nouns:
                if len(noun) > 1:  # 한 글자 명사 제외
                    # 명사를 포함한 구문 찾기
                    idx = cleaned_text.find(noun)
                    if idx >= 0:
                        # 명사 앞뒤 문맥 포함 (최대 5글자)
                        start = max(0, idx - 5)
                        end = min(len(cleaned_text), idx + len(noun) + 5)
                        phrase = cleaned_text[start:end]
                        noun_phrases.append((phrase, idx))  # 위치 정보와 함께 저장
            
            # 위치순으로 정렬하여 원래 문장 순서 유지
            noun_phrases.sort(key=lambda x: x[1])
            
            # 명사구가 있으면 처리
            if noun_phrases:
                # 첫 1-2개 명사구 선택
                if len(noun_phrases) >= 2:
                    # 두 명사구를 합쳐 제목 생성
                    title_parts = [noun_phrases[0][0], noun_phrases[1][0]]
                    # 중복 제거
                    if title_parts[0] in title_parts[1]:
                        title = title_parts[1]
                    elif title_parts[1] in title_parts[0]:
                        title = title_parts[0]
                    else:
                        title = " ".join(title_parts)
                else:
                    title = noun_phrases[0][0]
                
                # 길이 제한
                if len(title) > max_length:
                    # 조사 경계로 자르기
                    parts = re.split(r'(에서|에게|으로|로|을|를|이|가|은|는|의|와|과|에)', title)
                    title = "".join(parts[:min(4, len(parts))])
                    
                    if len(title) > max_length:
                        title = title[:max_length] + "..."
                
                return title.strip()
            
            # 문장 구문 분석으로 핵심 구조 추출
            try:
                # Kkma로 문장 분석
                pos = self.kkma.pos(cleaned_text[:100])  # 처리 속도를 위해 앞부분만
                
                # 주어-목적어 구조 찾기
                subject = None
                object = None
                
                for word, tag in pos:
                    if tag.startswith('N') and subject is None:
                        subject = word
                    elif tag.startswith('JK') and object is None:
                        object = word
                
                if subject and object:
                    return f"{subject} {object}"
            except Exception as e:
                logger.debug(f"Kkma 분석 실패: {e}")
            
            # 기본 방식으로 폴백
            return cleaned_text[:max_length] + "..." if len(cleaned_text) > max_length else cleaned_text
            
        except Exception as e:
            logger.error(f"한국어 핵심 구문 추출 오류: {e}")
            return text[:max_length] + "..." if len(text) > max_length else text
    
    def extract_key_phrases_spacy(self, text: str, max_length: int = 30) -> str:
        """spaCy를 사용한 핵심 구문 추출"""
        if not self.spacy_enabled:
            # spaCy 사용 불가능한 경우 대체 로직
            return self.extract_key_phrases_generic(text, max_length)
        
        try:
            # TextRank로 핵심 구문 추출
            doc = self.nlp(text)
            
            # 핵심 구문 추출
            phrases = [p.text for p in doc._.phrases[:2]]
            
            if phrases:
                title = " ".join(phrases)
                if len(title) > max_length:
                    title = title[:max_length] + "..."
                return title
            
            # 명사구 추출
            noun_chunks = list(doc.noun_chunks)
            if noun_chunks:
                # 가장 중요한 명사구 2개 선택
                chunks = [chunk.text for chunk in noun_chunks[:2]]
                title = " ".join(chunks)
                if len(title) > max_length:
                    title = title[:max_length] + "..."
                return title
            
            # 개체명 추출
            entities = list(doc.ents)
            if entities:
                # 개체명 텍스트 추출
                entity_texts = [ent.text for ent in entities[:2]]
                title = " ".join(entity_texts)
                if len(title) > max_length:
                    title = title[:max_length] + "..."
                return title
            
            # 기본 방식으로 폴백
            return text[:max_length] + "..." if len(text) > max_length else text
            
        except Exception as e:
            logger.error(f"spaCy 핵심 구문 추출 오류: {e}")
            return text[:max_length] + "..." if len(text) > max_length else text
    
    def extract_key_phrases_generic(self, text: str, max_length: int = 30) -> str:
        """일반적인 방법으로 핵심 구문 추출"""
        # 특수 문자 및 과도한 공백 제거
        cleaned_text = re.sub(r'\s+', ' ', text).strip()
        
        # 명령형 종결어 제거
        cleaned_text = self.remove_command_endings(cleaned_text)
        
        # 짧은 텍스트는 그대로 반환
        if len(cleaned_text) <= max_length:
            return cleaned_text
        
        # 문장 분리
        sentences = re.split(r'[.?!。？！\n]+', cleaned_text)
        first_sentence = sentences[0].strip()
        
        if len(first_sentence) <= max_length:
            return first_sentence
        
        # 첫 문장이 길면 중요 단어 포함 부분만 추출
        words = first_sentence.split()
        if len(words) > 4:
            title = " ".join(words[:4])
            if len(title) > max_length:
                title = title[:max_length] + "..."
            return title
        
        # 단순 절삭
        return first_sentence[:max_length] + "..."
    
    def generate_title(self, text: str, max_length: int = 30) -> str:
        """다국어 지원 제목 생성"""
        # 입력 텍스트 정제
        text = text.strip()
        
        # 텍스트가 너무 짧으면 그대로 반환
        if len(text) <= max_length:
            return text
        
        # 언어 감지
        language = self.detect_language(text)
        
        # 언어별 처리
        if language == "ko" and self.korean_enabled:
            return self.extract_key_phrases_korean(text, max_length)
        elif self.spacy_enabled:
            return self.extract_key_phrases_spacy(text, max_length)
        else:
            return self.extract_key_phrases_generic(text, max_length)


# 싱글톤 인스턴스 생성 (성능을 위해 초기화 비용 절감)
_title_generator = None

def get_title_generator():
    """싱글톤 인스턴스 반환"""
    global _title_generator
    if _title_generator is None:
        _title_generator = AdvancedTitleGenerator()
    return _title_generator