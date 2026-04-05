"""Seller Agent Implementation"""

import re
from typing import Dict, List, Any, Optional, Union
from agenticpay.agents.base_agent import BaseAgent
from agenticpay.models.base_llm import BaseLLM
from agenticpay.models.base_vlm import BaseVLM
from loguru import logger


class SellerAgent(BaseAgent):
    """Seller Agent
    
    Represents the seller, negotiates with the buyer based on product information and market conditions.
    """
    
    def __init__(
        self,
        model: Union[BaseLLM, BaseVLM],
        name: str = "Seller",
        role_description: str = "You are a seller trying to maximize profit while being reasonable. You are professional, friendly, and want to close a deal that benefits both parties.",
        seller_min_price: Optional[float] = None,
        system_prompt_suffix: Optional[str] = None,
    ):
        """Initialize Seller Agent
        
        Args:
            model: LLM or VLM interface (supports both BaseLLM and BaseVLM)
            name: Agent name
            role_description: Role description
            seller_min_price: Minimum acceptable selling price for seller (bottom price, confidential information)
            system_prompt_suffix: Additional text to append to system prompt (e.g., personality profile)
        """
        super().__init__(model, role_description, name)
        self.seller_min_price = seller_min_price
        self.system_prompt_suffix = system_prompt_suffix
    
    def respond(
        self,
        conversation_history: List[Dict[str, Any]],
        current_state: Dict[str, Any],
    ) -> str:
        """Generate Seller response
        
        Args:
            conversation_history: Conversation history
            current_state: Current state
            
        Returns:
            Seller's response text
        """
        if not self.initialized:
            raise ValueError("Agent not initialized. Call initialize() first.")
        
        prompt = self._build_prompt(conversation_history, current_state)
        
        # Get seller's minimum acceptable price (bottom price)
        min_price = self.seller_min_price or self.context.get('min_price', 'unknown')
        initial_price = self.context.get('initial_price', 'unknown')
        product_info = self.context.get('product_info', {})
        available_products = self.context.get('available_products', [])
        
        # Format available products information
        available_products_info = ""
        if available_products:
            available_products_info = "\n\nAVAILABLE PRODUCTS IN YOUR INVENTORY:\n"
            for i, prod in enumerate(available_products, 1):
                available_products_info += f"{i}. {prod.get('name', 'Unknown')} - "
                available_products_info += f"Brand: {prod.get('brand', 'N/A')}, "
                available_products_info += f"Price: ${prod.get('price', 0):.2f}, "
                available_products_info += f"Features: {', '.join(prod.get('features', []))}\n"
            available_products_info += "\nYou can suggest other products from your inventory if they better match the buyer's needs.\n"
        
        # Add Seller-specific guidance
#         seller_guidance = f"""

# IMPORTANT REMINDERS:
# - Your initial asking price is ${initial_price}
# - Your minimum acceptable price (confidential - do not reveal this to the buyer) is ${min_price}
# - Current product information: {product_info}
# {available_products_info}
# - Consider the environment factors: {self.context.get('environment_info', {})}
# - Be professional and try to find a win-win solution
# - Highlight the value and quality of your product
# - If the buyer's needs might be better met by another product in your inventory, you can suggest it
# - Be willing to negotiate but don't go below your minimum acceptable price
# - Try to negotiate the price as high as possible, but ensure the deal is successful in the end
# - **CRITICAL: Each conversation you MUST make one price offer, you MUST use the format: ### SELLER_PRICE($X) ###**
# - Example: "I can offer ### SELLER_PRICE($150) ### for this product"
# - Example: "How about ### SELLER_PRICE($130.00) ###?"
# - This specific format is required for the system to correctly extract your offer price
# - Consider market conditions and seasonality
# - NEVER reveal your minimum acceptable price to the buyer - keep it confidential
# - Keep communication short and concise.

# Now, respond as {self.name}:
# """
        # Add personality profile if provided
        personality_section = ""
        if self.system_prompt_suffix:
            personality_section = f"\n{self.system_prompt_suffix}\n"
        
        seller_guidance = f"""
IMPORTANT REMINDERS:
- Your minimum acceptable price (confidential) is ${min_price}. Never reveal it.
- Current product information: {product_info}
{available_products_info}
- Consider the environment factors: {self.context.get('environment_info', {})}.
{personality_section}
- **CRITICAL: In each turn, you MUST make exactly ONE price offer for the product using the format:**
  ### SELLER_PRICE($X) ###
- **IMPORTANT: SELLER_PRICE($X) must be the TOTAL PRICE for the entire order/transaction, NOT a per-unit price.**
  - If selling multiple units/items, $X should be the total amount the buyer will pay.
  - Example: For 10,000 units at $0.40 each, use ### SELLER_PRICE($4000) ###, NOT ### SELLER_PRICE($0.40) ###
- Example: "I can offer ### SELLER_PRICE($15) ### for this product."
- Example: "How about ### SELLER_PRICE($13.00) ###?"
- This specific format is required for the system to correctly extract your offer price.
- NEVER reveal your minimum acceptable price to the buyer.

MENTAL MODELING INSTRUCTION:
Before composing your negotiation message, you MUST first perform internal mental modeling of the negotiation.
Think privately about the following three aspects:
1. [Opponent Reservation Price]: Based on the conversation so far, what is the buyer's likely maximum acceptable price? Provide a specific estimated price range and a confidence score (0-100%).
2. [Opponent Strategy]: What negotiation tactic or strategy is the buyer currently using? (e.g., aggressive lowballing, anchoring low, comparison shopping threat, value questioning, etc.)
3. [My Strategy]: What is your current negotiation strategy and why? (e.g., holding firm on value, slow concession, urgency creation, bundle offer, etc.)

You MUST format your entire output as follows (do NOT skip either block):
<mental_model>
[Opponent Reservation Price]: <your estimate and confidence score>
[Opponent Strategy]: <your inference about the buyer's tactic>
[My Strategy]: <your chosen tactic and reasoning>
</mental_model>
<message>
[Your actual negotiation message to the buyer, following all IMPORTANT rules above]
</message>
"""

        full_prompt = prompt + seller_guidance

        # logger.info(f"Seller prompt: {full_prompt}")


        # Extract images from current_state if VLM is used
        images = None
        if self.is_vlm:
            # Check for images in current_state (e.g., product images)
            images = current_state.get('images') or current_state.get('product_images')
            # Also check in context
            if images is None:
                images = self.context.get('images') or self.context.get('product_images')
        
        # Generate response: VLM supports images, LLM doesn't
        if self.is_vlm and images is not None:
            response = self.model.generate(
                full_prompt, 
                images=images,
                temperature=0.0,
                max_tokens=2048  # Increased to accommodate mental model + message
            )
        else:
            response = self.model.generate(
                full_prompt, 
                temperature=0.0,
                max_tokens=2048  # Ensure complete response generation
            )
        
        # Remove <think>...</think> tags (used by reasoning models like DeepSeek)
        response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL | re.IGNORECASE)

        # Extract <mental_model> block — log it but do NOT include in returned message
        mental_model_match = re.search(r'<mental_model>(.*?)</mental_model>', response, flags=re.DOTALL | re.IGNORECASE)
        if mental_model_match:
            mental_model_content = mental_model_match.group(1).strip()
            logger.info(f"\n{'='*50}\n[{self.name} MENTAL MODEL]\n{mental_model_content}\n{'='*50}")

        # Extract <message> block — this is the only part that enters conversation history
        message_match = re.search(r'<message>(.*?)</message>', response, flags=re.DOTALL | re.IGNORECASE)
        if message_match:
            final_message = message_match.group(1).strip()
        else:
            # Fallback: strip mental_model tags and use remainder as message
            logger.warning(f"[{self.name}] Output did not follow <mental_model>/<message> format. Using fallback.")
            final_message = re.sub(r'<mental_model>.*?</mental_model>', '', response, flags=re.DOTALL | re.IGNORECASE).strip()

        return final_message

