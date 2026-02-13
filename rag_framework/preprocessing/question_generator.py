"""Question generation module for creating multi-hop questions."""

from typing import List, Dict, Optional
from ..clients.llm_client import BaseLLMClient


class QuestionGenerator:
    """Generate multi-hop questions from document structure."""

    def __init__(self, llm_client: BaseLLMClient):
        """Initialize question generator.
        
        Args:
            llm_client: LLM client for generating questions
        """
        self.llm_client = llm_client

    def generate_1hop_questions(
        self, content: str, num_questions: int = 5
    ) -> List[Dict[str, str]]:
        """Generate 1-hop questions (single fact retrieval).
        
        Args:
            content: Source content
            num_questions: Number of questions to generate
            
        Returns:
            List of Q&A dictionaries
        """
        prompt = f"""Generate {num_questions} simple factual questions that can be answered directly from the following text. 
Each question should require only one piece of information from the text to answer.

Text:
{content}

Format your response as:
Q1: [question]
A1: [answer]
Q2: [question]
A2: [answer]
...
"""
        response = self.llm_client.generate(prompt, temperature=0.7)
        return self._parse_qa_response(response)

    def generate_2hop_questions(
        self, content: str, num_questions: int = 5
    ) -> List[Dict[str, str]]:
        """Generate 2-hop questions (requires connecting two pieces of info).
        
        Args:
            content: Source content
            num_questions: Number of questions to generate
            
        Returns:
            List of Q&A dictionaries
        """
        prompt = f"""Generate {num_questions} questions that require connecting TWO different pieces of information from the text.
Each question should require reasoning across two related facts.

Text:
{content}

Format your response as:
Q1: [question]
A1: [answer]
Q2: [question]
A2: [answer]
...
"""
        response = self.llm_client.generate(prompt, temperature=0.7)
        return self._parse_qa_response(response)

    def generate_3hop_questions(
        self, content: str, num_questions: int = 5
    ) -> List[Dict[str, str]]:
        """Generate 3-hop questions (requires connecting three pieces of info).
        
        Args:
            content: Source content
            num_questions: Number of questions to generate
            
        Returns:
            List of Q&A dictionaries
        """
        prompt = f"""Generate {num_questions} complex questions that require connecting THREE different pieces of information from the text.
Each question should require multi-step reasoning across three related facts.

Text:
{content}

Format your response as:
Q1: [question]
A1: [answer]
Q2: [question]
A2: [answer]
...
"""
        response = self.llm_client.generate(prompt, temperature=0.7)
        return self._parse_qa_response(response)

    def generate_all_questions(
        self, hierarchical_structure: Dict, questions_per_hop: int = 3
    ) -> Dict[str, List[Dict[str, str]]]:
        """Generate all types of questions from hierarchical structure.
        
        Args:
            hierarchical_structure: Document structure from PDFProcessor
            questions_per_hop: Number of questions per hop type per section
            
        Returns:
            Dictionary with question types as keys
        """
        # Combine all content
        all_content = []
        if "sections" in hierarchical_structure:
            for section in hierarchical_structure["sections"]:
                if section.get("content"):
                    all_content.append(section["content"])
                for subsection in section.get("subsections", []):
                    if subsection.get("content"):
                        all_content.append(subsection["content"])
        
        combined_content = " ".join(all_content)
        
        # Generate questions
        questions = {
            "1-hop": self.generate_1hop_questions(combined_content, questions_per_hop),
            "2-hop": self.generate_2hop_questions(combined_content, questions_per_hop),
            "3-hop": self.generate_3hop_questions(combined_content, questions_per_hop),
        }
        
        return questions

    def _parse_qa_response(self, response: str) -> List[Dict[str, str]]:
        """Parse Q&A format response.
        
        Args:
            response: LLM response in Q/A format
            
        Returns:
            List of Q&A dictionaries
        """
        qa_pairs = []
        lines = response.strip().split('\n')
        
        current_question = None
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Match question pattern
            if line.startswith('Q') and ':' in line:
                current_question = line.split(':', 1)[1].strip()
            # Match answer pattern
            elif line.startswith('A') and ':' in line and current_question:
                answer = line.split(':', 1)[1].strip()
                qa_pairs.append({
                    "question": current_question,
                    "answer": answer
                })
                current_question = None
        
        return qa_pairs
