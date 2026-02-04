from typing import List, Dict, Any, Optional
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputToolsParser
from langchain_core.utils.function_calling import convert_to_openai_tool

from RAG.index import create_vector_store
from RAG.retriever import create_retriever, format_docs
from tools.user_profile import get_user_info, save_user_preference
from utils.logger import AgentLogger

class CoFinaAgent:
    def __init__(self, api_key: str, logger: Optional[AgentLogger] = None):
        self.logger = logger or AgentLogger()
        self.api_key = api_key
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            model="gemini-2.5-pro",
            api_key=api_key,
            base_url='https://ai-gateway.andrew.cmu.edu/',
            temperature=0.3
        )
        
        # Tools
        self.tools = [get_user_info, save_user_preference]
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        
        # Initialize RAG components
        try:
            self.vector_store = create_vector_store(api_key)
            self.retriever = create_retriever(self.vector_store)
        except Exception as e:
            self.logger.log_step("error", f"Failed to init RAG: {e}")
            self.retriever = None

    def retrieve_context(self, query: str) -> str:
        """Retrieve relevant context from the vector store."""
        if not self.retriever:
            return ""
            
        docs = self.retriever.invoke(query)
        self.logger.log_retrieval(query, docs)
        return format_docs(docs), docs

    def run(self, user_query: str, user_id: str = "default_user") -> str:
        """Run the ReAct loop."""
        self.logger.start_turn(user_query)
        messages = []
        
        # 1. Get User Profile (Implicit first step)
        profile_tool = get_user_info
        profile_result = profile_tool.invoke({"user_id": user_id})
        self.logger.log_step("context", f"Loaded profile: {profile_result}")
        
        # 2. Retrieve Documents
        context, docs = self.retrieve_context(user_query)
        
        # 3. Construct System Prompt
        system_prompt = f"""You are CoFina, an advanced financial assistant.
        
        USER CONTEXT:
        {profile_result}
        
        RETRIEVED KNOWLEDGE:
        {context}
        
        INSTRUCTIONS:
        1. Answer the user's question based on the RETRIEVED KNOWLEDGE and their USER CONTEXT.
        2. If the user expresses a preference (e.g., "I want low risk"), use the 'save_user_preference' tool.
        3. Be helpful, concise, and professional.
        4. If you use information from the retrieved documents, cite the source filename.
        """
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_query)
        ]
        
        # 4. ReAct Loop (Max 5 turns)
        max_steps = 5
        current_step = 0
        
        while current_step < max_steps:
            print(f"DEBUG: Step {current_step + 1}/{max_steps}")
            
            # Plan/Act
            response = self.llm_with_tools.invoke(messages)
            messages.append(response)
            
            # If tool call
            if response.tool_calls:
                print(f"DEBUG: Tool calls detected: {len(response.tool_calls)}")
                self.logger.log_step("decision", "tool_call", {"calls": response.tool_calls})
                
                for tool_call in response.tool_calls:
                    tool_name = tool_call["name"]
                    tool_args = tool_call["args"]
                    
                    # Execute tool
                    if tool_name == "save_user_preference":
                        result = save_user_preference.invoke(tool_args)
                    elif tool_name == "get_user_info":
                        result = get_user_info.invoke(tool_args)
                    else:
                        result = f"Error: Tool {tool_name} not found"
                        
                    self.logger.log_step("tool_output", result, {"tool": tool_name})
                    
                    messages.append(ToolMessage(
                        tool_call_id=tool_call["id"], 
                        content=str(result),
                        name=tool_name
                    ))
            else:
                print("DEBUG: Final answer candidate generated. Verifying...")
                # Final Answer
                from agent.verifier import verify_response
                
                # Check for empty context or no docs
                # Safer extraction that handles formatting variations
                try:
                    # We want to pass the FULL context (User + Retrieved) to the verifier
                    # so that it doesn't think user profile info is a hallucination.
                    if "USER CONTEXT:" in messages[0].content:
                        # Extract everything between USER CONTEXT and INSTRUCTIONS
                        full_context_str = messages[0].content.split("USER CONTEXT:")[1].split("INSTRUCTIONS:")[0].strip()
                    else:
                        full_context_str = ""
                        
                    # Also trying old retrieval as fallback if something is weird
                    if not full_context_str and "RETRIEVED KNOWLEDGE:" in messages[0].content:
                        full_context_str = messages[0].content.split("RETRIEVED KNOWLEDGE:")[1].split("INSTRUCTIONS:")[0].strip()
                        
                except Exception as e:
                    print(f"DEBUG: Context extraction failed: {e}")
                    full_context_str = ""
                
                print(f"DEBUG: Verifying against context length: {len(full_context_str)}")
                
                verification = verify_response(
                    question=user_query,
                    answer=response.content,
                    context=full_context_str,
                    api_key=self.api_key
                )
                
                print(f"DEBUG: Verification result: {verification}")
                self.logger.log_step("verification", verification)
                
                action = verification.get("action", "accept")
                score = verification.get("score", 0.0)
                
                if action == "accept" or score >= 0.8:
                    print("DEBUG: Accepted.")
                    self.logger.end_turn(response.content)
                    return response.content
                
                elif action == "retry" and current_step < max_steps - 1:
                    print("DEBUG: Retrying...")
                    # Retry with feedback
                    retry_msg = f"Your previous answer had a low groundedness score ({score}). Reason: {verification['reason']}. Please try again, strictly strictly adhering to the context."
                    messages.append(HumanMessage(content=retry_msg))
                    self.logger.log_step("retry", str(retry_msg))
                    current_step += 1
                    continue
                    
                else: # Refuse or run out of retries
                    print("DEBUG: Refusing.")
                    refusal_msg = f"I apologize, but I couldn't verify the accuracy of my response based on the available documents (Verification Score: {score}). {verification['reason']}"
                    self.logger.end_turn(refusal_msg)
                    return refusal_msg
            
            current_step += 1
            
        return "I apologize, but I couldn't complete the request within the step limit."
