"""
Summarizer Agent - Summarizes long contexts while preserving key information
"""

from typing import Dict, Any, List, Optional
import json
import re
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

class SummarizerAgent:
    """
    Specialized agent for summarizing long contexts and documents
    """
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key
        self.llm = None
        if api_key:
            self.llm = ChatOpenAI(
                model="gemini-2.5-flash",
                api_key=api_key,
                base_url='https://ai-gateway.andrew.cmu.edu/',
                temperature=0.1
            )
    
    def process(self, text: str, max_length: int = 1000, 
                preserve_keys: List[str] = None) -> Dict[str, Any]:
        """
        Summarize text while preserving key information
        
        Args:
            text: Text to summarize
            max_length: Maximum length of summary
            preserve_keys: Specific information to preserve (numbers, dates, etc.)
        
        Returns:
            Summary and extracted key information
        """
        if not text:
            return {"summary": "", "key_info": {}}
        
        # First, extract key information
        key_info = self._extract_key_info(text, preserve_keys)
        
        # If text is already short enough, return as-is with key info
        if len(text) < max_length:
            return {
                "summary": text,
                "key_info": key_info,
                "compression_ratio": 1.0
            }
        
        # Use LLM for intelligent summarization if available
        if self.llm:
            summary = self._llm_summarize(text, max_length, key_info)
        else:
            # Fallback to extractive summarization
            summary = self._extractive_summarize(text, max_length)
        
        return {
            "summary": summary,
            "key_info": key_info,
            "compression_ratio": len(summary) / len(text),
            "original_length": len(text),
            "summary_length": len(summary)
        }
    
    def _extract_key_info(self, text: str, preserve_keys: List[str] = None) -> Dict:
        """Extract key information from text"""
        key_info = {
            "numbers": [],
            "dates": [],
            "entities": [],
            "action_items": []
        }
        
        # Extract numbers (including currency)
        number_pattern = r'\$?\d+(?:,\d{3})*(?:\.\d+)?%?'
        key_info["numbers"] = re.findall(number_pattern, text)
        
        # Extract dates
        date_pattern = r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}-\d{2}-\d{2}'
        key_info["dates"] = re.findall(date_pattern, text)
        
        # Extract potential entities (capitalized words)
        entity_pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b'
        key_info["entities"] = list(set(re.findall(entity_pattern, text)))
        
        # Look for action items
        action_patterns = [
            r'(?:should|must|need to|will|plan to) ([^\.]+)',
            r'action item[s]?:? ([^\.]+)',
            r'next step[s]?:? ([^\.]+)'
        ]
        
        for pattern in action_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            key_info["action_items"].extend(matches)
        
        # Filter based on preserve_keys if provided
        if preserve_keys:
            filtered = {}
            for key in preserve_keys:
                if key in key_info:
                    filtered[key] = key_info[key]
            return filtered
        
        return key_info
    
    def _llm_summarize(self, text: str, max_length: int, key_info: Dict) -> str:
        """Use LLM to generate intelligent summary"""
        prompt = ChatPromptTemplate.from_template("""
        Summarize the following text, preserving all important information.
        
        KEY INFORMATION TO PRESERVE:
        - Numbers and amounts: {numbers}
        - Dates: {dates}
        - Important entities: {entities}
        - Action items: {action_items}
        
        TEXT TO SUMMARIZE:
        {text}
        
        REQUIREMENTS:
        1. Keep all numbers and dates exactly as they appear
        2. Preserve the meaning and relationships between key points
        3. Maximum length: {max_length} characters
        4. Be concise but complete
        
        SUMMARY:
        """)
        
        chain = prompt | self.llm
        result = chain.invoke({
            "numbers": ", ".join(key_info.get("numbers", [])[:10]),
            "dates": ", ".join(key_info.get("dates", [])[:5]),
            "entities": ", ".join(key_info.get("entities", [])[:10]),
            "action_items": "\n".join(key_info.get("action_items", [])[:5]),
            "text": text[:5000] if len(text) > 5000 else text,  # Truncate for LLM
            "max_length": max_length
        })
        
        return result.content
    
    def _extractive_summarize(self, text: str, max_length: int) -> str:
        """Simple extractive summarization fallback"""
        sentences = text.split('. ')
        
        if not sentences:
            return text
        
        # Score sentences by importance (position + keyword density)
        scored_sentences = []
        for i, sentence in enumerate(sentences):
            score = 1.0 / (i + 1)  # Earlier sentences more important
            
            # Boost sentences with key information
            if re.search(r'\$\d+', sentence):  # Has money
                score *= 1.5
            if re.search(r'\d{4}-\d{2}-\d{2}', sentence):  # Has date
                score *= 1.3
            if re.search(r'should|must|need|important', sentence.lower()):
                score *= 1.2
            
            scored_sentences.append((score, sentence))
        
        # Sort by score and select until max_length
        scored_sentences.sort(reverse=True)
        
        summary = ""
        for _, sentence in scored_sentences:
            if len(summary) + len(sentence) <= max_length:
                if summary:
                    summary += ". "
                summary += sentence
            else:
                break
        
        return summary
    
    def summarize_conversation(self, conversation: List[Dict], 
                                max_turns: int = 5) -> Dict[str, Any]:
        """
        Summarize a conversation history
        
        Args:
            conversation: List of conversation turns
            max_turns: Maximum turns to keep in summary
        
        Returns:
            Summarized conversation with key points
        """
        if len(conversation) <= max_turns:
            return {
                "summary": conversation,
                "compressed": False
            }
        
        # Keep recent turns, summarize older ones
        recent = conversation[-max_turns:]
        older = conversation[:-max_turns]
        
        # Summarize older turns
        older_text = "\n".join([
            f"User: {t.get('user', '')}\nAgent: {t.get('agent', '')}"
            for t in older if isinstance(t, dict)
        ])
        
        if older_text:
            older_summary = self.process(older_text, max_length=500)
            return {
                "summary": {
                    "older_turns_summary": older_summary["summary"],
                    "key_info": older_summary["key_info"],
                    "recent_turns": recent
                },
                "compressed": True,
                "original_turns": len(conversation),
                "compressed_turns": max_turns + 1
            }
        
        return {
            "summary": conversation,
            "compressed": False
        }
    
    def truncate_to_fit(self, text: str, token_limit: int, 
                         token_count_func=None) -> str:
        """
        Truncate text to fit within token limit
        
        Args:
            text: Text to truncate
            token_limit: Maximum tokens
            token_count_func: Function to count tokens (if None, use character estimate)
        
        Returns:
            Truncated text
        """
        if token_count_func:
            current_tokens = token_count_func(text)
        else:
            # Rough estimate: 4 chars per token
            current_tokens = len(text) / 4
        
        if current_tokens <= token_limit:
            return text
        
        # Need to truncate
        ratio = token_limit / current_tokens
        target_length = int(len(text) * ratio * 0.9)  # Slightly conservative
        
        # Try to truncate at sentence boundaries
        sentences = text.split('. ')
        truncated = ""
        for sentence in sentences:
            if len(truncated) + len(sentence) <= target_length:
                if truncated:
                    truncated += ". "
                truncated += sentence
            else:
                break
        
        return truncated