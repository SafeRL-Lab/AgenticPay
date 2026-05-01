"""Seller Agent Implementation"""

import re
from typing import Dict, List, Any, Optional, Union
from agenticpay.agents.base_agent import BaseAgent
from agenticpay.agents.structured_message import extract_message_content
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
        role_description: str = "You are a seller looking to make a good deal.",
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
        self.last_selected_buyer: Optional[int] = None
    
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
        
        self.last_selected_buyer = None
        # Keep initial_price out of the seller-visible prompt. The seller should
        # reason from product_info and its private reservation price instead of
        # being anchored by an environment-side initial quote.
        original_context = self.context
        filtered_context = {k: v for k, v in self.context.items() if k != "initial_price"}
        self.context = filtered_context
        try:
            prompt = self._build_prompt(conversation_history, current_state)
        finally:
            self.context = original_context
        
        # Get seller's minimum acceptable price (bottom price)
        min_price = self.seller_min_price or self.context.get('min_price', 'unknown')
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

        num_buyers = current_state.get("num_buyers", 1)
        try:
            num_buyers = int(num_buyers)
        except (TypeError, ValueError):
            num_buyers = 1
        num_buyers = max(1, num_buyers)
        task_instruction = current_state.get("instruction", "")
        task_instruction_section = ""
        if task_instruction:
            task_instruction_section = f"\nTASK INSTRUCTION:\n- {task_instruction}\n"
        contract_config = (
            self.context.get("contract_config")
            or self.context.get("environment_info", {}).get("contract_config")
            or current_state.get("contract_config")
        )
        buyer_contract_configs = (
            self.context.get("buyer_contract_configs")
            or self.context.get("environment_info", {}).get("buyer_contract_configs")
        )
        message_reasoning_rule = (
            "MESSAGE RULE: Before giving your <contract>, briefly explain your reasoning. "
            "When relevant, mention what you observe in the product text or image."
        )
        selected_buyer_rules = f"""
- MULTI-BUYER ROUTING: You are negotiating with multiple potential buyers.
- You MUST choose exactly one buyer for this turn and output it in a dedicated `<selected_buyer>` block.
- `<selected_buyer>` must contain exactly one integer from `1` to `{num_buyers}`.
- Base your choice on the conversation history and current offers.
- Do NOT put any extra text inside `<selected_buyer>`.
"""
        selected_buyer_format = f"""<selected_buyer>
[one integer from 1 to {num_buyers}]
</selected_buyer>
"""
        if buyer_contract_configs:
            config_summaries = {}
            for buyer_id, cfg in buyer_contract_configs.items():
                seller_preferences = cfg.get("seller_preferences", {})
                config_summaries[buyer_id] = {
                    "field_descriptions": cfg.get("field_descriptions", {}),
                    "continuous_bounds": cfg.get("continuous_bounds", {}),
                    "discrete_options": cfg.get("discrete_options", {}),
                    "seller_preferences": seller_preferences,
                    "weight_descriptions": seller_preferences.get("weight_descriptions", {}),
                    "contrainfo": cfg.get("contrainfo", {}),
                }
            contract_rules = f"""
- CONTRACT INFORMATION:
  - You are negotiating with multiple buyers. Each buyer may have a different contract_config, but all buyers use the same JSON schema:
    {{
      "price": <number>,
      "continuous_terms": <object>,
      "discrete_terms": <object>
    }}
  - Buyer-specific public fields and your private seller utility information: {config_summaries}
  - No contract field has a default value or is pre-agreed at the start of negotiation.
  - The `<contract>` JSON in your message is your current complete proposal for the selected buyer, with every field filled by you.
  - Each `<message>` must contain exactly one `<contract>...</contract>` JSON block following the selected buyer's schema/options.
"""
            structure_message_line = f"[Your actual negotiation message to the selected buyer. {message_reasoning_rule} You can discuss any contract terms, but must include exactly one valid <contract>...</contract> JSON block for that buyer.]"
        elif contract_config:
            continuous_bounds = contract_config.get("continuous_bounds", {})
            discrete_options = contract_config.get("discrete_options", {})
            field_descriptions = contract_config.get("field_descriptions", {})
            seller_preferences = contract_config.get("seller_preferences", {})
            weight_descriptions = seller_preferences.get("weight_descriptions", {})
            contra_info = contract_config.get("contrainfo") or self.context.get("environment_info", {}).get("contrainfo", {})
            contract_rules = f"""
- CONTRACT INFORMATION:
  - The negotiable contract has this JSON schema:
    {{
      "price": <number>,
      "continuous_terms": <object>,
      "discrete_terms": <object>
    }}
  - Field meanings: {field_descriptions}
  - Valid continuous terms and bounds: {continuous_bounds}
  - Valid discrete terms and options: {discrete_options}
  - Your private utility information: {seller_preferences}
  - Meaning of your private utility information: {weight_descriptions}
  - Additional contract background (`contrainfo`): {contra_info}
  - No contract field has a default value or is pre-agreed at the start of negotiation.
  - The `<contract>` JSON in your message is your current complete proposal, with every field filled by you.
  - Each `<message>` must contain exactly one `<contract>...</contract>` JSON block following the schema above.
"""
            structure_message_line = f"[Your actual negotiation message to the buyer. {message_reasoning_rule} You can discuss any contract terms, but must include exactly one valid <contract>...</contract> JSON block.]"
        else:
            contract_rules = """
- **CRITICAL: In each turn, you MUST include exactly ONE ### SELLER_PRICE($X) ### inside `<message>` — including when you accept the buyer's offer or confirm terms.** There are no exceptions: without this tag each round, the environment may keep an outdated price and agreement checks will be wrong.
- When you accept the buyer's price, set $X to the total sale price you agree to (typically the buyer's last ### BUYER_PRICE($Y) ### that you are accepting). When counter-offering, $X is your asking price.
- **IMPORTANT: SELLER_PRICE($X) must be the TOTAL PRICE for the entire order/transaction, NOT a per-unit price.**
  - If selling multiple units/items, $X should be the total amount the buyer will pay.
  - Example: For 10,000 units at $0.40 each, use ### SELLER_PRICE($4000) ###, NOT ### SELLER_PRICE($0.40) ###
- Example: "I can offer ### SELLER_PRICE($15) ### for this product."
- Example: "How about ### SELLER_PRICE($13.00) ###?"
- Example (accepting their offer): "You've got a deal at ### SELLER_PRICE($6.50) ### — I'll ship it today."
- This specific format is required for the system to correctly extract your offer price.
"""
            structure_message_line = "[Your actual negotiation message to the buyer. Must include exactly one ### SELLER_PRICE($X) ### and obey all IMPORTANT rules above.]"

        if num_buyers >= 2:
            structure_tail = f"""{task_instruction_section}
{selected_buyer_rules}
You MUST format your entire output as follows (do NOT skip any block):
<mental_model>
[Opponent Reservation Price]: price=<number>; confidence=<0-100>
[Opponent Strategy]: <your inference about the opponent's tactic>
[My Strategy]: <your chosen tactic and reasoning>
</mental_model>
{selected_buyer_format}<message>
{structure_message_line}
</message>
"""
        else:
            structure_tail = f"""{task_instruction_section}
You MUST format your entire output as follows (do NOT skip either block):
<mental_model>
[Opponent Reservation Price]: price=<number>; confidence=<0-100>
[Opponent Strategy]: <your inference about the opponent's tactic>
[My Strategy]: <your chosen tactic and reasoning>
</mental_model>
<message>
{structure_message_line}
</message>
"""
        
        seller_guidance = f"""
IMPORTANT REMINDERS:
- Your minimum acceptable price (confidential) is ${min_price}. Never reveal it.
- Current product information: {product_info}
{available_products_info}
- Consider the environment factors: {self.context.get('environment_info', {})}.
{personality_section}
{contract_rules}
- NEVER reveal your minimum acceptable price to the buyer.

MENTAL MODELING INSTRUCTION:
Before replying, briefly state your internal assessment for evaluation:
1. [Opponent Reservation Price]: Output exactly `price=<number>; confidence=<0-100>` for your estimate of the opponent's reservation price and confidence score.
2. [Opponent Strategy]: Your inference about the opponent's tactic.
3. [My Strategy]: Your chosen tactic and reasoning.

{structure_tail}
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
            round_num = current_state.get("current_round")
            if round_num is None:
                round_num = current_state.get("round")
            round_str = str(round_num) if round_num is not None else "?"
            logger.info(
                f"\n{'='*50}\n[{self.name} MENTAL MODEL | round {round_str}]\n{mental_model_content}\n{'='*50}"
            )

        selected_buyer_match = re.search(r'<selected_buyer>\s*(\d+)\s*</selected_buyer>', response, flags=re.DOTALL | re.IGNORECASE)
        if selected_buyer_match:
            parsed = int(selected_buyer_match.group(1))
            if 1 <= parsed <= num_buyers:
                self.last_selected_buyer = parsed

        # Extract <message> block — this is the only part that enters conversation history
        final_message = extract_message_content(response)
        if final_message is None:
            # No <message> tag: strip structured blocks and use remainder
            logger.warning(f"[{self.name}] Output did not follow <mental_model>/<message> format. Using fallback.")
            final_message = re.sub(r'<mental_model>.*?</mental_model>', '', response, flags=re.DOTALL | re.IGNORECASE)
            final_message = re.sub(r'<selected_buyer>.*?</selected_buyer>', '', final_message, flags=re.DOTALL | re.IGNORECASE).strip()

        return final_message

