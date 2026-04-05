"""Buyer Agent Implementation"""

import re
from typing import Dict, List, Any, Optional, Union
from agenticpay.agents.base_agent import BaseAgent
from agenticpay.models.base_llm import BaseLLM
from agenticpay.models.base_vlm import BaseVLM
from agenticpay.utils.user_profile import UserProfile, StylePreference, ShoppingHabit
from loguru import logger


class BuyerAgent(BaseAgent):
    """Buyer Agent
    
    Represents the buyer, negotiates with the seller based on user requirements and budget.
    """
    
    def __init__(
        self,
        model: Union[BaseLLM, BaseVLM],
        name: str = "Buyer",
        role_description: str = "You are a buyer looking for a good deal. You are polite, strategic, and want to get the best price within your budget.",
        buyer_max_price: Optional[float] = None,
        system_prompt_suffix: Optional[str] = None,
    ):
        """Initialize Buyer Agent
        
        Args:
            model: LLM or VLM interface (supports both BaseLLM and BaseVLM)
            name: Agent name
            role_description: Role description
            buyer_max_price: Maximum acceptable purchase price for buyer (bottom price, confidential information)
            system_prompt_suffix: Additional text to append to system prompt (e.g., personality profile)
        """
        super().__init__(model, role_description, name)
        self.buyer_max_price = buyer_max_price
        self.system_prompt_suffix = system_prompt_suffix
    
    def respond(
        self,
        conversation_history: List[Dict[str, Any]],
        current_state: Dict[str, Any],
    ) -> str:
        """Generate Buyer response
        
        Args:
            conversation_history: Conversation history
            current_state: Current state
            
        Returns:
            Buyer's response text
        """
        if not self.initialized:
            raise ValueError("Agent not initialized. Call initialize() first.")
        
        prompt = self._build_prompt(conversation_history, current_state)
        
        # Get buyer's maximum acceptable price (bottom price)
        max_price = self.buyer_max_price or self.context.get('max_price', 'unknown')
        
        # Get product information (similar to seller_agent)
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
        
        # Get user profile information
        user_profile = self.context.get('user_profile')
        
        # Build user preference-related guidance
        preference_guidance = ""
        if user_profile:
            preference_guidance = "\nUSER PREFERENCES:\n"
            
            # If user_profile is a string (text description), use it directly
            if isinstance(user_profile, str):
                preference_guidance += f"- {user_profile}\n"
            # If user_profile is a UserProfile object, use the original logic
            elif isinstance(user_profile, UserProfile):
                # Style preference guidance
                if user_profile.style_preference:
                    style = user_profile.style_preference
                    if style == StylePreference.SIMPLE:
                        preference_guidance += "- Style preference: You prefer SIMPLE/MINIMALIST styles. Focus on clean, simple designs without excessive decoration.\n"
                    elif style == StylePreference.BUSINESS:
                        preference_guidance += "- Style preference: You prefer BUSINESS/PROFESSIONAL styles. Focus on formal, professional-looking items suitable for work.\n"
                    elif style == StylePreference.TRADITIONAL:
                        preference_guidance += "- Style preference: You prefer TRADITIONAL/CLASSIC styles. Focus on timeless, classic designs with traditional elements.\n"
                
                # Shopping habit guidance
                if user_profile.shopping_habit:
                    habit = user_profile.shopping_habit
                    if habit == ShoppingHabit.COMPARE:
                        preference_guidance += "- Shopping habit: You like to COMPARE PRICES and shop around. You may mention that you're comparing options, ask for better deals, or reference other sellers. Take your time in negotiations.\n"
                    elif habit == ShoppingHabit.DIRECT:
                        preference_guidance += "- Shopping habit: You prefer DIRECT PURCHASES. You value efficiency and may be willing to pay a fair price quickly if the deal is reasonable. Don't waste too much time haggling.\n"
        
        # Add Buyer-specific guidance
#         buyer_guidance = f"""

# IMPORTANT REMINDERS:
# - Your maximum acceptable price (confidential - do not reveal this to the seller) is ${max_price}
# - You want to negotiate a fair price that fits your needs
# - Consider the environment factors: {self.context.get('environment_info', {})}
# - Be polite but firm in your negotiations
# - Try to find a win-win solution
# - If the seller's price is too high, suggest a reasonable counter-offer
# - Try to negotiate the price as low as possible, but ensure the deal is successful in the end
# - **CRITICAL: Each conversation you MUST make one price offer, you MUST use the format: ### BUYER_PRICE($X) ###**
# - Example: "I can offer ### BUYER_PRICE($100) ### for this product"
# - Example: "How about ### BUYER_PRICE($120.50) ###?"
# - This specific format is required for the system to correctly extract your offer price
# - NEVER reveal your maximum acceptable price to the seller - keep it confidential
# - Keep communication short and concise.

# DEAL AGREEMENT INSTRUCTION:
# - If you decide to accept the deal and want to make a transaction, you MUST include the exact phrase "MAKE_DEAL" in your response
# - This phrase should appear when you are ready to finalize the agreement
# - Example: "That sounds good! I accept your offer. MAKE_DEAL"
# - Only use "MAKE_DEAL" when you are genuinely ready to complete the transaction
# {preference_guidance}
# Now, respond as {self.name}:
# """

        # Add personality profile if provided
        personality_section = ""
        if self.system_prompt_suffix:
            personality_section = f"\n{self.system_prompt_suffix}\n"
        
#         buyer_guidance = f"""
# IMPORTANT:
# - Your top price is ${max_price} (confidential, do not reveal).
# - Current product information: {product_info}
# {available_products_info}
# - Consider the environment: {self.context.get('environment_info', {})}.
# {personality_section}
# - **CRITICAL: In each turn, you MUST make exactly ONE price offer for the product using the format:**
#   ### BUYER_PRICE($X) ###
# - **IMPORTANT: BUYER_PRICE($X) must be the TOTAL PRICE for the entire order/transaction, NOT a per-unit price.**
#   - If ordering multiple units/items, $X should be the total amount you will pay.
#   - Example: For 10,000 units at $0.40 each, use ### BUYER_PRICE($4000) ###, NOT ### BUYER_PRICE($0.40) ###
# - Example: "I can offer ### BUYER_PRICE($10) ### for this product."
# - Example: "How about ### BUYER_PRICE($12.50) ###?"
# - This specific format is required for the system to correctly extract your offer price.
# - NEVER reveal your maximum acceptable price to the seller.
# - Keep communication short (150 words or less), clear, and focused on negotiation.

# DEAL AGREEMENT INSTRUCTION:
# - Only finalize the transaction when you believe the price is reasonably balanced.
# - If you decide to accept the deal, you MUST include the exact phrase "MAKE_DEAL" in your response.
# - Example: "That sounds acceptable to me. MAKE_DEAL"

# {preference_guidance}

# Now, respond as {self.name}:
# """

        buyer_guidance = f"""
IMPORTANT:
- Your top price is ${max_price} (confidential, do not reveal).
- Current product information: {product_info}
{available_products_info}
- Consider the environment: {self.context.get('environment_info', {})}.
{personality_section}
- **CRITICAL: In each turn, you MUST make exactly ONE price offer for the product using the format:**
  ### BUYER_PRICE($X) ###
- **IMPORTANT: BUYER_PRICE($X) must be the TOTAL PRICE for the entire order/transaction, NOT a per-unit price.**
  - If ordering multiple units/items, $X should be the total amount you will pay.
  - Example: For 10,000 units at $0.40 each, use ### BUYER_PRICE($4000) ###, NOT ### BUYER_PRICE($0.40) ###
- Example: "I can offer ### BUYER_PRICE($10) ### for this product."
- Example: "How about ### BUYER_PRICE($12.50) ###?"
- This specific format is required for the system to correctly extract your offer price.
- NEVER reveal your maximum acceptable price to the seller.

DEAL AGREEMENT INSTRUCTION:
- Only finalize the transaction when you believe the price is reasonably balanced.
- If you decide to accept the deal, you MUST include the exact phrase "MAKE_DEAL" in your response.
- Example: "That sounds acceptable to me. MAKE_DEAL"

{preference_guidance}

MENTAL MODELING INSTRUCTION:
Before composing your negotiation message, you MUST first perform internal mental modeling of the negotiation.
Think privately about the following three aspects:
1. [Opponent Reservation Price]: Based on the conversation so far, what is the seller's likely minimum acceptable price? Provide a specific estimated price range and a confidence score (0-100%).
2. [Opponent Strategy]: What negotiation tactic or strategy is the seller currently using? (e.g., anchoring high, slow concession, urgency creation, value emphasis, etc.)
3. [My Strategy]: What is your current negotiation strategy and why? (e.g., aggressive lowballing, gradual concession, value questioning, walking-away threat, etc.)

You MUST format your entire output as follows (do NOT skip either block):
<mental_model>
[Opponent Reservation Price]: <your estimate and confidence score>
[Opponent Strategy]: <your inference about the seller's tactic>
[My Strategy]: <your chosen tactic and reasoning>
</mental_model>
<message>
[Your actual negotiation message to the seller, following all IMPORTANT rules above]
</message>
"""

        full_prompt = prompt + buyer_guidance

        # logger.info(f"Buyer prompt: {full_prompt}")
        
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
                max_tokens=2048  # Ensure complete response generation
            )
        else:
            response = self.model.generate(
                full_prompt, 
                temperature=0.0,
                max_tokens=2048  # Increased to accommodate mental model + message
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

