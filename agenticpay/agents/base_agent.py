"""Agent Base Class"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union
from agenticpay.models.base_llm import BaseLLM
from agenticpay.models.base_vlm import BaseVLM


class BaseAgent(ABC):
    """Agent base class
    
    Base class for all Agents, defines the basic interface and behavior of Agents.
    """
    
    def __init__(
        self,
        model: Union[BaseLLM, BaseVLM],
        role_description: str,
        name: str,
    ):
        """Initialize Agent
        
        Args:
            model: LLM or VLM interface (supports both BaseLLM and BaseVLM)
            role_description: Role description (used in prompt)
            name: Agent name
        """
        self.model = model
        self.role_description = role_description
        self.name = name
        self.context: Dict[str, Any] = {}
        self.initialized = False
        # Check if the model is a VLM
        self.is_vlm = isinstance(model, BaseVLM)
    
    def initialize(self, context: Dict[str, Any]):
        """Initialize Agent context
        
        Args:
            context: Agent's context information
        """
        self.context = context
        self.initialized = True
    
    @abstractmethod
    def respond(
        self,
        conversation_history: List[Dict[str, Any]],
        current_state: Dict[str, Any],
    ) -> str:
        """Generate response
        
        Args:
            conversation_history: Conversation history
            current_state: Current state
            
        Returns:
            Agent's response text
        """
        pass
    
    def _build_prompt(
        self,
        conversation_history: List[Dict[str, Any]],
        current_state: Dict[str, Any],
    ) -> str:
        """Build prompt
        
        Args:
            conversation_history: Conversation history
            current_state: Current state
            
        Returns:
            Complete prompt string
        """
        # Base prompt template
#         prompt = f"""You are {self.name}, {self.role_description}

# Context Information:
# {self._format_context()}

# Current Negotiation State:
# {self._format_state(current_state)}

# Conversation History:
# {self._format_history(conversation_history)}

# Please respond naturally as {self.name} would.
# """
        prompt = f"""You are {self.name}, {self.role_description}

Context Information:
{self._format_context()}

Conversation History:
{self._format_history(conversation_history)}

Please respond naturally as {self.name} would.
"""
        return prompt
    
    def _format_context(self) -> str:
        """Format context information"""
        if not self.context:
            return "No context available."
        
        context_str = ""
        for key, value in self.context.items():
            if isinstance(value, dict):
                context_str += f"- {key}:\n"
                for sub_key, sub_value in value.items():
                    context_str += f"  * {sub_key}: {sub_value}\n"
            elif hasattr(value, 'get_description'):
                # Handle objects like UserProfile that have get_description method
                context_str += f"- {key}: {value.get_description()}\n"
            else:
                context_str += f"- {key}: {value}\n"
        return context_str
    
    def _format_state(self, state: Dict[str, Any]) -> str:
        """Format state information"""
        if not state:
            return "No state information available."
        
        state_str = ""
        for key, value in state.items():
            state_str += f"- {key}: {value}\n"
        return state_str
    
    def _format_history(self, history: List[Dict[str, Any]]) -> str:
        """Format conversation history"""
        if not history:
            return "No conversation history yet."
        
        history_str = ""
        for msg in history:
            role = msg.get('role', 'unknown').upper()
            content = msg.get('content', '')
            round_num = msg.get('round', 0)
            thread_label = msg.get('thread_label')
            thread_prefix = f"[{thread_label}]" if thread_label else ""
            history_str += f"[Round {round_num}]{thread_prefix} {role}: {content}\n"
        return history_str

