"""Buyer Agent Implementation"""

import re
from typing import Dict, List, Any, Optional, Union
from agenticpay.agents.base_agent import BaseAgent
from agenticpay.agents.structured_message import extract_message_content
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
        role_description: str = "You are a buyer looking for a good deal.",
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
        self.last_selected_seller: Optional[int] = None
    
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
        self.last_selected_seller = None
        
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

        num_sellers = current_state.get("num_sellers")
        if num_sellers is None:
            num_sellers = self.context.get("num_sellers", 1)
        try:
            num_sellers = int(num_sellers)
        except (TypeError, ValueError):
            num_sellers = 1
        num_sellers = max(1, num_sellers)

        task_instruction = current_state.get("instruction", "")
        task_instruction_section = ""
        if task_instruction:
            task_instruction_section = f"\nTASK INSTRUCTION:\n- {task_instruction}\n"
        contract_config = (
            self.context.get("contract_config")
            or self.context.get("environment_info", {}).get("contract_config")
            or current_state.get("contract_config")
        )
        seller_contract_configs = (
            self.context.get("seller_contract_configs")
            or self.context.get("environment_info", {}).get("seller_contract_configs")
            or current_state.get("contract_configs")
        )
        message_reasoning_rule = (
            "MESSAGE RULE: Before giving your <contract>, briefly explain your reasoning. "
            "When relevant, mention what you observe in the product text or image."
        )

        selected_seller_rules = f"""
- MULTI-SELLER ROUTING: You are negotiating in a multi-seller setting.
- You MUST choose exactly one seller for this turn and output it in a dedicated `<selected_seller>` block.
- `<selected_seller>` must contain exactly one integer from `1` to `{num_sellers}`.
- If there is only one seller in the task, you MUST output `1`.
- Base your choice on the conversation history and current offers.
- Do NOT put any extra text inside `<selected_seller>`.
"""
        selected_seller_format = f"""<selected_seller>
[one integer from 1 to {num_sellers}]
</selected_seller>
"""
        
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

        if seller_contract_configs:
            config_summaries = {}
            for seller_id, cfg in seller_contract_configs.items():
                buyer_preferences = cfg.get("buyer_preferences", {})
                config_summaries[seller_id] = {
                    "field_descriptions": cfg.get("field_descriptions", {}),
                    "continuous_bounds": cfg.get("continuous_bounds", {}),
                    "discrete_options": cfg.get("discrete_options", {}),
                    "buyer_preferences": buyer_preferences,
                    "weight_descriptions": buyer_preferences.get("weight_descriptions", {}),
                    "contrainfo": cfg.get("contrainfo", {}),
                }
            contract_rules = f"""
CONTRACT INFORMATION:
- You are negotiating with multiple sellers. Each seller may have a different contract_config, but all sellers use the same JSON schema:
  {{
    "price": <number>,
    "continuous_terms": <object>,
    "discrete_terms": <object>
  }}
- Seller-specific public fields and your private buyer utility information: {config_summaries}
- No contract field has a default value or is pre-agreed at the start of negotiation.
- The `<contract>` JSON in your message is your current complete proposal for the selected seller, with every field filled by you.
- Each `<message>` must contain exactly one `<contract>...</contract>` JSON block following the selected seller's schema/options.
- Never reveal your confidential reservation value directly.
"""
            deal_instruction = """
DEAL AGREEMENT INSTRUCTION:
- A deal is finalized only when you and the selected seller output compatible contract JSONs.
- If you accept, repeat the accepted final contract JSON exactly in `<contract>...</contract>`.
"""
            message_instruction = f"[Your actual negotiation message to the selected seller. {message_reasoning_rule} You can discuss any contract terms, but must include exactly one valid <contract>...</contract> JSON block for that seller.]"
        elif contract_config:
            continuous_bounds = contract_config.get("continuous_bounds", {})
            discrete_options = contract_config.get("discrete_options", {})
            field_descriptions = contract_config.get("field_descriptions", {})
            buyer_preferences = contract_config.get("buyer_preferences", {})
            weight_descriptions = buyer_preferences.get("weight_descriptions", {})
            contra_info = contract_config.get("contrainfo") or self.context.get("environment_info", {}).get("contrainfo", {})
            contract_rules = f"""
CONTRACT INFORMATION:
- The negotiable contract has this JSON schema:
  {{
    "price": <number>,
    "continuous_terms": <object>,
    "discrete_terms": <object>
  }}
- Field meanings: {field_descriptions}
- Valid continuous terms and bounds: {continuous_bounds}
- Valid discrete terms and options: {discrete_options}
- Your private utility information: {buyer_preferences}
- Meaning of your private utility information: {weight_descriptions}
- Additional contract background (`contrainfo`): {contra_info}
- No contract field has a default value or is pre-agreed at the start of negotiation.
- The `<contract>` JSON in your message is your current complete proposal, with every field filled by you.
- Each `<message>` must contain exactly one `<contract>...</contract>` JSON block following the schema above.
- Never reveal your confidential reservation value directly.
"""
            deal_instruction = """
DEAL AGREEMENT INSTRUCTION:
- A deal is finalized only when both sides output compatible contract JSONs.
- If you accept, repeat the accepted final contract JSON exactly in `<contract>...</contract>`.
"""
            message_instruction = f"[Your actual negotiation message to the seller. {message_reasoning_rule} You can discuss any contract terms, but must include exactly one valid <contract>...</contract> JSON block.]"
        else:
            contract_rules = """
- **CRITICAL: In each turn, you MUST include exactly ONE ### BUYER_PRICE($X) ### inside `<message>` — including when you accept or confirm a deal.** There are no exceptions: if you only say you accept without this tag, the environment keeps your previous numeric offer and the deal will fail to register as agreed.
- When you accept the seller's price, set $X to the total you agree to pay (typically the seller's last stated price in ### SELLER_PRICE($Y) ### that you are accepting). When counter-offering, $X is your new offer.
- **IMPORTANT: BUYER_PRICE($X) must be the TOTAL PRICE for the entire order/transaction, NOT a per-unit price.**
  - If ordering multiple units/items, $X should be the total amount you will pay.
  - Example: For 10,000 units at $0.40 each, use ### BUYER_PRICE($4000) ###, NOT ### BUYER_PRICE($0.40) ###
- Example: "I can offer ### BUYER_PRICE($10) ### for this product."
- Example: "How about ### BUYER_PRICE($12.50) ###?"
- Example (accepting their price): "Deal — I'll take it at ### BUYER_PRICE($6.50) ###."
- This specific format is required for the system to correctly extract your offer price.
"""
            deal_instruction = """
DEAL AGREEMENT INSTRUCTION:
- Only finalize the transaction when you believe the price is reasonably balanced.
- When you accept, state it clearly in `<message>` and include ### BUYER_PRICE($X) ### with $X equal to the agreed total you will pay (same as the price you are accepting). Do not use any special deal-finalization keyword beyond that.
- Wrong: accepting verbally with no ### BUYER_PRICE(...) ### — the environment keeps your previous numeric offer and agreement may not register.
- Right: "I accept your offer at ### BUYER_PRICE($6.50) ###."
"""
            message_instruction = "[Your actual negotiation message to the seller. Must include exactly one ### BUYER_PRICE($X) ### and obey all IMPORTANT / DEAL AGREEMENT rules above.]"

        buyer_guidance = f"""
IMPORTANT:
- Your top price is ${max_price} (confidential, do not reveal).
- Current product information: {product_info}
{available_products_info}
- Consider the environment: {self.context.get('environment_info', {})}.
{personality_section}
{task_instruction_section}
{contract_rules}
- NEVER reveal your maximum acceptable price to the seller.
{selected_seller_rules}

{deal_instruction}

{preference_guidance}

MENTAL MODELING INSTRUCTION:
Before replying, briefly state your internal assessment for evaluation:
1. [Opponent Reservation Price]: Output exactly `price=<number>; confidence=<0-100>` for your estimate of the opponent's reservation price and confidence score.
2. [Opponent Strategy]: Your inference about the opponent's tactic.
3. [My Strategy]: Your chosen tactic and reasoning.

You MUST format your entire output exactly as follows:
<mental_model>
[Opponent Reservation Price]: price=<number>; confidence=<0-100>
[Opponent Strategy]: <your inference about the opponent's tactic>
[My Strategy]: <your chosen tactic and reasoning>
</mental_model>
{selected_seller_format}<message>
{message_instruction}
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
            round_num = current_state.get("current_round")
            if round_num is None:
                round_num = current_state.get("round")
            round_str = str(round_num) if round_num is not None else "?"
            logger.info(
                f"\n{'='*50}\n[{self.name} MENTAL MODEL | round {round_str}]\n{mental_model_content}\n{'='*50}"
            )

        selected_seller_match = re.search(r'<selected_seller>\s*(\d+)\s*</selected_seller>', response, flags=re.DOTALL | re.IGNORECASE)
        if selected_seller_match:
            parsed_selected_seller = int(selected_seller_match.group(1))
            if 1 <= parsed_selected_seller <= num_sellers:
                self.last_selected_seller = parsed_selected_seller

        # Extract <message> block — this is the only part that enters conversation history
        final_message = extract_message_content(response)
        if final_message is None:
            # No <message> tag: strip structured blocks and use remainder
            logger.warning(f"[{self.name}] Output did not follow the expected structured format. Using fallback.")
            final_message = re.sub(r'<mental_model>.*?</mental_model>', '', response, flags=re.DOTALL | re.IGNORECASE)
            final_message = re.sub(r'<selected_seller>.*?</selected_seller>', '', final_message, flags=re.DOTALL | re.IGNORECASE).strip()

        return final_message

