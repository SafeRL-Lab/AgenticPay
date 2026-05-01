"""Task14 Scenario 10: AmeriMist Food Color (two listings) - Sequential Two-Buyer Two-Seller Negotiation (image + text)

Same AmeriMist lemon-yellow airbrush food color from two marketplace listings. Visible product info has no
per-seller identity; two sellers each have a different confidential floor. Two buyers each pick one seller per
round (structured routing).
Category: Grocery & Gourmet Food
"""

import os
import sys
import json
import time
import random
import argparse
from pathlib import Path
from datetime import datetime

# Add project path (4 levels up from script to reach repo root AgenticPayGym)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, project_root)

from agenticpay.envs.multi_buyer_multi_seller.Task3_sequential_two_buyer_two_seller_negotiation import Task3SequentialTwoBuyerTwoSellerNegotiation
from agenticpay.agents.buyer_agent import BuyerAgent
from agenticpay.agents.seller_agent import SellerAgent
from agenticpay.models.openai_vlm import OpenAIVLM
from agenticpay.examples.config import reward_weights, max_rounds, price_tolerance, OPENAI_API_KEY


def get_model_name(model):
    """Extract model name from model object
    
    Args:
        model: Model object (CustomLLM, VLLMLLM, etc.)
    
    Returns:
        str: Model name
    """
    if hasattr(model, 'model'):
        return model.model
    elif hasattr(model, 'model_id'):
        return model.model_id
    elif hasattr(model, 'model_path'):
        # Extract model name from path
        model_path = model.model_path
        return os.path.basename(model_path) if model_path else str(model)
    else:
        # Fallback to string representation, but try to extract model name
        model_str = str(model)
        # Try to extract model name from string like "CustomLLM(model=qwen3-8b)"
        if "model=" in model_str:
            try:
                return model_str.split("model=")[1].split(")")[0]
            except:
                return model_str
        else:
            return model_str


def _run_buyer_routing(buyer, combined_history: list, observation: dict, routing_instruction: str):
    """Structured ``<selected_seller>`` + retries + random fallback (aligned with Task5)."""
    max_selection_retries = 2
    retry_count = 0
    inst = routing_instruction
    buyer_response = None
    selected_seller = None
    while True:
        buyer_response = buyer.respond(
            conversation_history=combined_history,
            current_state={
                **observation,
                "instruction": inst,
                "num_sellers": 2,
            },
        )
        selected_seller = buyer.last_selected_seller
        if selected_seller is not None:
            break
        if retry_count >= max_selection_retries:
            break
        retry_count += 1
        print(
            f"\n[Warning] Missing <selected_seller>; retrying buyer response "
            f"({retry_count}/{max_selection_retries})..."
        )
        inst = (
            routing_instruction
            + " IMPORTANT: You MUST include a valid <selected_seller> block with only 1 or 2."
        )
    if selected_seller is None:
        selected_seller = random.choice([1, 2])
        print(
            f"\n[Warning] Failed to parse <selected_seller> after retries; "
            f"randomly selecting Seller {selected_seller}."
        )
    return buyer_response, selected_seller


def main(model_name=None):
    """Main function: Demonstrates sequential multi-buyer multi-seller negotiation flow
    
    Args:
        model_name: Optional model name. If None, uses default model.
    """
    
    print("Initializing model...")
    
    # Check API key
    api_key = os.getenv("OPENAI_API_KEY") or OPENAI_API_KEY
    if not api_key:
        print("Warning: OPENAI_API_KEY not set. Please set it to use OpenAI models.")
        print("You can set it with: export OPENAI_API_KEY='your-key-here'")
        return
    
    model_name = model_name or "gpt-5.4"
    model = OpenAIVLM(model=model_name, api_key=api_key)
    
    print(f"✓ Successfully initialized: {model}")

    print("Creating agents...")
    product_request = (
        "I want AmeriColor AmeriMist lemon yellow food color—new. "
        "I also prefer the spray pump and collar ring look intact with no cracked plastic around the nozzle base."
    )
    shared_contract_fields = {
        "contrainfo": {
            "product_request": product_request,
            "initial_contract_status": (
                "No price, delivery time, return policy, packaging option, or user product preference match "
                "has been selected or agreed before negotiation starts."
            ),
            "contract_completion_requirement": (
                "A valid offer must explicitly fill price, continuous_terms.delivery_days, "
                "discrete_terms.return_policy, discrete_terms.packaging, and "
                "discrete_terms.user_product_preference."
            ),
        },
        "field_descriptions": {
            "price": "Total US dollars paid for this 0.65 oz AmeriMist lemon yellow bottle.",
            "continuous_terms.delivery_days": "Days until delivery after the deal closes.",
            "discrete_terms.return_policy": (
                "`30_days`: return-eligible groceries policy; `none`: final sale (common for opened consumables)."
            ),
            "discrete_terms.packaging": (
                "`protective`: extra padding against leaks/crush; `standard`: normal pouch/box."
            ),
            "discrete_terms.user_product_preference": (
                "Match to the buyer's stated sprayer-condition check (pump/collar intact; no obvious cracked plastic "
                "around the nozzle base). `strong_match` / `partial_match` / `mismatch_or_uncertain`."
            ),
        },
        "continuous_bounds": {"delivery_days": {"min": 1, "max": 7}},
        "discrete_options": {
            "return_policy": ["30_days", "none"],
            "packaging": ["protective", "standard"],
            "user_product_preference": ["strong_match", "partial_match", "mismatch_or_uncertain"],
        },
    }
    buyer1_preferences = {
        "v_base": 4.92,
        "weight_descriptions": {
            "v_base": "Reservation value before shipping/return/packaging ($); lower paid price helps.",
            "continuous_weights.delivery_days": "$/day; negative if slower hurts.",
            "discrete_weights.return_policy": "$ utility per option.",
            "discrete_weights.packaging": "$ utility per option.",
            "discrete_weights.user_product_preference": "$ utility per preference match tier.",
        },
        "continuous_weights": {"delivery_days": -0.29},
        "discrete_weights": {
            "return_policy": {"30_days": 1.05, "none": -1.22},
            "packaging": {"protective": 0.95, "standard": -0.35},
            "user_product_preference": {
                "strong_match": 0.22,
                "partial_match": 0.09,
                "mismatch_or_uncertain": -0.18,
            },
        },
    }
    buyer2_preferences = json.loads(json.dumps(buyer1_preferences))
    buyer2_preferences["v_base"] = 5.11
    buyer2_preferences["continuous_weights"]["delivery_days"] = -0.19
    buyer2_preferences["discrete_weights"]["return_policy"] = {"30_days": 0.88, "none": -1.05}
    buyer2_preferences["discrete_weights"]["packaging"] = {"protective": 0.78, "standard": -0.22}
    seller1_preferences = {
        "c_base": 4.36,
        "weight_descriptions": {
            "c_base": "Fulfillment floor before terms ($); higher received price helps.",
            "continuous_weights.delivery_days": "$/day; positive if slower shipping saves you hassle/cost.",
            "discrete_weights.return_policy": "$ utility per option.",
            "discrete_weights.packaging": "$ utility per option.",
            "discrete_weights.user_product_preference": (
                "$ utility per preference match tier; stronger match implies small nonzero cost."
            ),
        },
        "continuous_weights": {"delivery_days": 0.19},
        "discrete_weights": {
            "return_policy": {"30_days": -1.38, "none": 0.96},
            "packaging": {"protective": -0.76, "standard": 0.29},
            "user_product_preference": {
                "strong_match": -0.06,
                "partial_match": -0.03,
                "mismatch_or_uncertain": 0.008,
            },
        },
    }
    seller2_preferences = json.loads(json.dumps(seller1_preferences))
    seller2_preferences["c_base"] = 3.90
    seller2_preferences["continuous_weights"]["delivery_days"] = 0.23
    seller2_preferences["discrete_weights"]["return_policy"] = {"30_days": -1.22, "none": 0.81}
    seller2_preferences["discrete_weights"]["packaging"] = {"protective": -0.9, "standard": 0.33}

    buyer_seller_contract_configs = {
        "b1s1": {**shared_contract_fields, "buyer_preferences": buyer1_preferences, "seller_preferences": seller1_preferences},
        "b1s2": {**shared_contract_fields, "buyer_preferences": buyer1_preferences, "seller_preferences": seller2_preferences},
        "b2s1": {**shared_contract_fields, "buyer_preferences": buyer2_preferences, "seller_preferences": seller1_preferences},
        "b2s2": {**shared_contract_fields, "buyer_preferences": buyer2_preferences, "seller_preferences": seller2_preferences},
    }
    buyer1_max_price = buyer1_preferences["v_base"]
    buyer2_max_price = buyer2_preferences["v_base"]
    seller1_min_price = seller1_preferences["c_base"]
    seller2_min_price = seller2_preferences["c_base"]

    buyer1 = BuyerAgent(model=model, name="Buyer1", buyer_max_price=buyer1_max_price)
    buyer2 = BuyerAgent(model=model, name="Buyer2", buyer_max_price=buyer2_max_price)
    seller1 = SellerAgent(model=model, name="Seller1", seller_min_price=seller1_min_price)
    seller2 = SellerAgent(model=model, name="Seller2", seller_min_price=seller2_min_price)
    
    # Create environment
    print("Creating sequential multi-buyer multi-seller negotiation environment...")
    env = Task3SequentialTwoBuyerTwoSellerNegotiation(
        buyer1_agent=buyer1,
        buyer2_agent=buyer2,
        seller1_agent=seller1,
        seller2_agent=seller2,
        max_rounds=max_rounds,
        initial_seller1_price=6.25,  # Opening ask — same food color, listing 1 (list price)
        initial_seller2_price=6.89,  # Opening ask — same food color, listing 2
        buyer1_max_price=buyer1_max_price,  # Buyer1 max willingness (confidential)
        buyer2_max_price=buyer2_max_price,  # Buyer2 max willingness (confidential)
        seller1_min_price=seller1_min_price,  # Seller1 minimum acceptable (confidential)
        seller2_min_price=seller2_min_price,  # Seller2 minimum acceptable (confidential)
        environment_info={
            "platform": "Amazon",
            "market_type": "B2C",
            "buyer_seller_contract_configs": buyer_seller_contract_configs,
        },
        price_tolerance=price_tolerance,
        reward_weights=reward_weights,  # Reward weights configuration
    )
    
    user_profile = None
    print(f"User Profile: {user_profile}")

    user_requirement = product_request
    print(f"Using default requirement: {user_requirement}")
    
    # Reset environment
    print("\n" + "="*60)
    print("Starting new sequential negotiation with two buyers and two sellers (food color, two listings)...")
    print("="*60)
    
    product_image_url = "https://m.media-amazon.com/images/I/41p+jdUZTJL.jpg"

    observation, info = env.reset(
        user_requirement=user_requirement,
        product_info={
            "name": "AmeriColor AmeriMist - Lemon Yellow Airbrush Food Color, 0.65 oz",
            "condition": "New",
            "brand": "AmeriColor",
            "color": "Lemon Yellow",
            "size": "0.65 oz",
            "original_price": 6.25,
            "product_category": "Grocery & Gourmet Food › Pantry Staples › Cooking & Baking › Food Coloring",
            "average_rating": 5.0,
            "total_reviews": 1,
            "asin": "B00FBPHZKC",
            "full_description": "Super-strength concentrated airbrush food color; works on icings and non-dairy whipped toppings; reduces over-spray and water spots.",
            "image_url": product_image_url,
        },
        user_profile=user_profile,
    )
    
    # Start negotiation loop
    done = False
    start_time = time.time()
    
    # Initialize results dictionary
    results = {
        "task": "Task14_s10_smokehouse_treat_negotiation",
        "timestamp": datetime.now().isoformat(),
        "user_requirement": user_requirement,
        "user_profile": user_profile,
        "status": "unknown",
        "success": False,
        "error": None,
    }
    
    while not done:
        # Each round, each buyer chooses one seller to negotiate with
        # Let buyers decide which seller to negotiate with and provide negotiation message
        
        # Build combined conversation history for buyer1 (includes both sellers' conversations)
        combined_history_b1 = []
        # Add seller1 messages with prefix
        for msg in observation.get("conversation_history_b1s1", []):
            combined_history_b1.append({**msg, "thread_label": "Talk with Seller 1"})
        for msg in observation.get("conversation_history_b1s2", []):
            combined_history_b1.append({**msg, "thread_label": "Talk with Seller 2"})
        
        # Build combined conversation history for buyer2 (includes both sellers' conversations)
        combined_history_b2 = []
        # Add seller1 messages with prefix
        for msg in observation.get("conversation_history_b2s1", []):
            combined_history_b2.append({**msg, "thread_label": "Talk with Seller 1"})
        for msg in observation.get("conversation_history_b2s2", []):
            combined_history_b2.append({**msg, "thread_label": "Talk with Seller 2"})
        
        routing_instruction = (
            "You are negotiating with two sellers. Each round, choose exactly ONE seller "
            "and output that choice in a dedicated <selected_seller> block containing only "
            "the digit 1 or 2. Then put only your negotiation text in <message>, including "
            "one complete <contract>...</contract> JSON block for the selected seller."
        )
        buyer1_response, buyer1_selected_seller = _run_buyer_routing(
            buyer1, combined_history_b1, observation, routing_instruction
        )
        buyer2_response, buyer2_selected_seller = _run_buyer_routing(
            buyer2, combined_history_b2, observation, routing_instruction
        )
        
        print(f"\n[Buyer 1 chooses to negotiate with Seller {buyer1_selected_seller} this round]")
        print(f"[Buyer 2 chooses to negotiate with Seller {buyer2_selected_seller} this round]")
        
        # Use buyer's full response as the negotiation message
        buyer1_action = buyer1_response
        buyer2_action = buyer2_response
        
        # Get the conversation history for each buyer-seller pair
        # Create updated conversation histories that include buyers' responses
        # So sellers can see buyers' messages before responding
        if buyer1_selected_seller == 1:
            conversation_history_b1s1 = observation["conversation_history_b1s1"].copy()
            if buyer1_action:
                current_round = observation.get("current_round", 0)
                conversation_history_b1s1.append({
                    "role": "buyer",
                    "content": buyer1_action,
                    "round": current_round
                })
        else:
            conversation_history_b1s2 = observation["conversation_history_b1s2"].copy()
            if buyer1_action:
                current_round = observation.get("current_round", 0)
                conversation_history_b1s2.append({
                    "role": "buyer",
                    "content": buyer1_action,
                    "round": current_round
                })
        
        if buyer2_selected_seller == 1:
            conversation_history_b2s1 = observation["conversation_history_b2s1"].copy()
            if buyer2_action:
                current_round = observation.get("current_round", 0)
                conversation_history_b2s1.append({
                    "role": "buyer",
                    "content": buyer2_action,
                    "round": current_round
                })
        else:
            conversation_history_b2s2 = observation["conversation_history_b2s2"].copy()
            if buyer2_action:
                current_round = observation.get("current_round", 0)
                conversation_history_b2s2.append({
                    "role": "buyer",
                    "content": buyer2_action,
                    "round": current_round
                })
        
        # Get the selected sellers' responses (sellers can now see buyers' messages)
        seller1_action_buyer1 = None
        seller1_action_buyer2 = None
        seller2_action_buyer1 = None
        seller2_action_buyer2 = None
        
        if buyer1_selected_seller == 1:
            seller1_action_buyer1 = seller1.respond(
                conversation_history=conversation_history_b1s1,
                current_state={
                    **observation,
                    "contract_config": env._build_role_contract_config("seller", 1, 1),
                },
            )
        elif buyer1_selected_seller == 2:
            seller2_action_buyer1 = seller2.respond(
                conversation_history=conversation_history_b1s2,
                current_state={
                    **observation,
                    "contract_config": env._build_role_contract_config("seller", 1, 2),
                },
            )

        if buyer2_selected_seller == 1:
            seller1_action_buyer2 = seller1.respond(
                conversation_history=conversation_history_b2s1,
                current_state={
                    **observation,
                    "contract_config": env._build_role_contract_config("seller", 2, 1),
                },
            )
        elif buyer2_selected_seller == 2:
            seller2_action_buyer2 = seller2.respond(
                conversation_history=conversation_history_b2s2,
                current_state={
                    **observation,
                    "contract_config": env._build_role_contract_config("seller", 2, 2),
                },
            )
        
        # Execute step with selected sellers and actions
        observation, reward, terminated, truncated, info = env.step(
            buyer1_selected_seller=buyer1_selected_seller,
            buyer2_selected_seller=buyer2_selected_seller,
            buyer1_action=buyer1_action,
            buyer2_action=buyer2_action,
            seller1_action_buyer1=seller1_action_buyer1,
            seller1_action_buyer2=seller1_action_buyer2,
            seller2_action_buyer1=seller2_action_buyer1,
            seller2_action_buyer2=seller2_action_buyer2
        )
        done = terminated or truncated
        
        # Render current state (includes all print information)
        env.render()
        
        # Flush output to ensure complete display
        sys.stdout.flush()
        
        # Display step rewards for each round with detailed calculation
        if ('step_buyer1_reward' in info or 'step_buyer2_reward' in info or
            'step_seller1_reward' in info or 'step_seller2_reward' in info):
            print(f"\n[Step Rewards] ", end="")
            if 'step_buyer1_reward' in info:
                print(f"Buyer1: {info['step_buyer1_reward']:.3f}", end="")
            if 'step_buyer2_reward' in info:
                if 'step_buyer1_reward' in info:
                    print(f" | ", end="")
                print(f"Buyer2: {info['step_buyer2_reward']:.3f}", end="")
            if 'step_seller1_reward' in info:
                if 'step_buyer1_reward' in info or 'step_buyer2_reward' in info:
                    print(f" | ", end="")
                print(f"Seller1: {info['step_seller1_reward']:.3f}", end="")
            if 'step_seller2_reward' in info:
                if 'step_buyer1_reward' in info or 'step_buyer2_reward' in info or 'step_seller1_reward' in info:
                    print(f" | ", end="")
                print(f"Seller2: {info['step_seller2_reward']:.3f}", end="")
            print()
            
            # Display detailed calculation with weights
            round_cost = -info['round']
            weights = env.reward_weights
            
            # Buyer1 step reward details
            if 'step_buyer1_reward' in info:
                buyer_price = None
                if info.get('buyer1_selected_seller') == 1:
                    buyer_price = info.get('b1s1_buyer_price')
                elif info.get('buyer1_selected_seller') == 2:
                    buyer_price = info.get('b1s2_buyer_price')
                
                if buyer_price is not None and env.buyer1_max_price is not None:
                    buyer_savings = env.buyer1_max_price - buyer_price
                    weighted_savings = buyer_savings * weights["buyer_savings"]
                    weighted_round_cost = round_cost * weights["time_cost"]
                    print(f"  Buyer1 Step Reward = buyer_savings({buyer_savings:.2f} * {weights['buyer_savings']:.2f}) + round_cost({round_cost:.2f} * {weights['time_cost']:.2f}) = {info['step_buyer1_reward']:.2f} (buyer1_max={env.buyer1_max_price}, buyer_price={buyer_price:.2f}, round={info['round']})")
                else:
                    weighted_round_cost = round_cost * weights["time_cost"]
                    print(f"  Buyer1 Step Reward = round_cost({round_cost:.2f} * {weights['time_cost']:.2f}) = {weighted_round_cost:.2f} (buyer_price not specified, round={info['round']})")
            
            # Buyer2 step reward details
            if 'step_buyer2_reward' in info:
                buyer_price = None
                if info.get('buyer2_selected_seller') == 1:
                    buyer_price = info.get('b2s1_buyer_price')
                elif info.get('buyer2_selected_seller') == 2:
                    buyer_price = info.get('b2s2_buyer_price')
                
                if buyer_price is not None and env.buyer2_max_price is not None:
                    buyer_savings = env.buyer2_max_price - buyer_price
                    weighted_savings = buyer_savings * weights["buyer_savings"]
                    weighted_round_cost = round_cost * weights["time_cost"]
                    print(f"  Buyer2 Step Reward = buyer_savings({buyer_savings:.2f} * {weights['buyer_savings']:.2f}) + round_cost({round_cost:.2f} * {weights['time_cost']:.2f}) = {info['step_buyer2_reward']:.2f} (buyer2_max={env.buyer2_max_price}, buyer_price={buyer_price:.2f}, round={info['round']})")
                else:
                    weighted_round_cost = round_cost * weights["time_cost"]
                    print(f"  Buyer2 Step Reward = round_cost({round_cost:.2f} * {weights['time_cost']:.2f}) = {weighted_round_cost:.2f} (buyer_price not specified, round={info['round']})")
            
            # Seller1 step reward details
            if 'step_seller1_reward' in info:
                seller1_price = None
                # Get price from whichever buyer selected seller1
                if info.get('buyer1_selected_seller') == 1 and info.get('b1s1_seller_price') is not None:
                    seller1_price = info.get('b1s1_seller_price')
                elif info.get('buyer2_selected_seller') == 1 and info.get('b2s1_seller_price') is not None:
                    seller1_price = info.get('b2s1_seller_price')
                # If both selected seller1, prefer higher price
                if (info.get('buyer1_selected_seller') == 1 and info.get('buyer2_selected_seller') == 1 and
                    info.get('b1s1_seller_price') is not None and info.get('b2s1_seller_price') is not None):
                    seller1_price = max(info.get('b1s1_seller_price'), info.get('b2s1_seller_price'))
                
                if seller1_price is not None and env.seller1_min_price is not None:
                    seller1_profit = seller1_price - env.seller1_min_price
                    weighted_seller1_profit = seller1_profit * weights["seller_profit"]
                    weighted_round_cost = round_cost * weights["time_cost"]
                    print(f"  Seller1 Step Reward = seller_profit({seller1_profit:.2f} * {weights['seller_profit']:.2f}) + round_cost({round_cost:.2f} * {weights['time_cost']:.2f}) = {info['step_seller1_reward']:.2f} (seller1_price={seller1_price:.2f}, seller1_min={env.seller1_min_price}, round={info['round']})")
                else:
                    weighted_round_cost = round_cost * weights["time_cost"]
                    print(f"  Seller1 Step Reward = round_cost({round_cost:.2f} * {weights['time_cost']:.2f}) = {weighted_round_cost:.2f} (seller1_price not specified, round={info['round']})")
            
            # Seller2 step reward details
            if 'step_seller2_reward' in info:
                seller2_price = None
                # Get price from whichever buyer selected seller2
                if info.get('buyer1_selected_seller') == 2 and info.get('b1s2_seller_price') is not None:
                    seller2_price = info.get('b1s2_seller_price')
                elif info.get('buyer2_selected_seller') == 2 and info.get('b2s2_seller_price') is not None:
                    seller2_price = info.get('b2s2_seller_price')
                # If both selected seller2, prefer higher price
                if (info.get('buyer1_selected_seller') == 2 and info.get('buyer2_selected_seller') == 2 and
                    info.get('b1s2_seller_price') is not None and info.get('b2s2_seller_price') is not None):
                    seller2_price = max(info.get('b1s2_seller_price'), info.get('b2s2_seller_price'))
                
                if seller2_price is not None and env.seller2_min_price is not None:
                    seller2_profit = seller2_price - env.seller2_min_price
                    weighted_seller2_profit = seller2_profit * weights["seller_profit"]
                    weighted_round_cost = round_cost * weights["time_cost"]
                    print(f"  Seller2 Step Reward = seller_profit({seller2_profit:.2f} * {weights['seller_profit']:.2f}) + round_cost({round_cost:.2f} * {weights['time_cost']:.2f}) = {info['step_seller2_reward']:.2f} (seller2_price={seller2_price:.2f}, seller2_min={env.seller2_min_price}, round={info['round']})")
                else:
                    weighted_round_cost = round_cost * weights["time_cost"]
                    print(f"  Seller2 Step Reward = round_cost({round_cost:.2f} * {weights['time_cost']:.2f}) = {weighted_round_cost:.2f} (seller2_price not specified, round={info['round']})")
        
        if done:
            print("\n" + "="*60)
            print("Negotiation Ended")
            print("="*60)
            print(f"Status: {info['status']}")
            if info.get('selected_buyer') and info.get('selected_seller'):
                print(f"Selected Deal: Buyer {info['selected_buyer']} - Seller {info['selected_seller']}")
                print(f"Final Deal Price: ${info.get('final_deal_price', 0):.2f}")
                pair_key = f"b{info['selected_buyer']}s{info['selected_seller']}"
                agreed_contract = info.get(f"{pair_key}_agreed_contract")
                if agreed_contract is not None:
                    print(f"Final Contract: {agreed_contract}")
            print(f"Buyer1-Seller1 Prices: Buyer=${info.get('b1s1_buyer_price', 0) or 0:.2f} | Seller=${info.get('b1s1_seller_price', 0) or 0:.2f}")
            print(f"Buyer1-Seller2 Prices: Buyer=${info.get('b1s2_buyer_price', 0) or 0:.2f} | Seller=${info.get('b1s2_seller_price', 0) or 0:.2f}")
            print(f"Buyer2-Seller1 Prices: Buyer=${info.get('b2s1_buyer_price', 0) or 0:.2f} | Seller=${info.get('b2s1_seller_price', 0) or 0:.2f}")
            print(f"Buyer2-Seller2 Prices: Buyer=${info.get('b2s2_buyer_price', 0) or 0:.2f} | Seller=${info.get('b2s2_seller_price', 0) or 0:.2f}")
            # Print score calculations after Step Rewards
            env._print_global_score_details()
            env._print_buyer_score_details()
            env._print_seller_score_details()
            
            # current_round has been incremented to reflect the completed round
            actual_rounds = info['round']
            print(f"Total Rounds: {actual_rounds}")
            print(f"Global Reward: {reward:.3f}")
            if 'buyer1_reward' in info:
                print(f"Buyer1 Reward: {info['buyer1_reward']:.3f}")
            if 'buyer2_reward' in info:
                print(f"Buyer2 Reward: {info['buyer2_reward']:.3f}")
            if 'seller1_reward' in info:
                print(f"Seller1 Reward: {info['seller1_reward']:.3f}")
            if 'seller2_reward' in info:
                print(f"Seller2 Reward: {info['seller2_reward']:.3f}")
            if 'global_score' in info:
                print(f"GlobalScore: {info['global_score']:.3f}")
            if 'buyer_score' in info:
                print(f"BuyerScore: {info['buyer_score']:.3f}")
            if 'seller_score' in info:
                print(f"SellerScore: {info['seller_score']:.3f}")
            if info.get('termination_reason'):
                print(f"Reason: {info['termination_reason']}")
            print("="*60)
            
            # Collect results
            elapsed_time = time.time() - start_time
            product_info = info.get('product_info', {})
            results.update({
                "status": info.get('status', 'unknown'),
                "success": terminated,
                "selected_buyer": info.get('selected_buyer'),
                "selected_seller": info.get('selected_seller'),
                "final_deal_price": info.get('final_deal_price'),
                "b1s1_buyer_price": info.get('b1s1_buyer_price'),
                "b1s1_seller_price": info.get('b1s1_seller_price'),
                "b1s2_buyer_price": info.get('b1s2_buyer_price'),
                "b1s2_seller_price": info.get('b1s2_seller_price'),
                "b2s1_buyer_price": info.get('b2s1_buyer_price'),
                "b2s1_seller_price": info.get('b2s1_seller_price'),
                "b2s2_buyer_price": info.get('b2s2_buyer_price'),
                "b2s2_seller_price": info.get('b2s2_seller_price'),
                "b1s1_agreed_contract": info.get('b1s1_agreed_contract'),
                "b1s1_buyer_utility": info.get('b1s1_buyer_utility'),
                "b1s1_seller_utility": info.get('b1s1_seller_utility'),
                "b1s1_z_max": info.get('b1s1_z_max'),
                "b1s2_agreed_contract": info.get('b1s2_agreed_contract'),
                "b1s2_buyer_utility": info.get('b1s2_buyer_utility'),
                "b1s2_seller_utility": info.get('b1s2_seller_utility'),
                "b1s2_z_max": info.get('b1s2_z_max'),
                "b2s1_agreed_contract": info.get('b2s1_agreed_contract'),
                "b2s1_buyer_utility": info.get('b2s1_buyer_utility'),
                "b2s1_seller_utility": info.get('b2s1_seller_utility'),
                "b2s1_z_max": info.get('b2s1_z_max'),
                "b2s2_agreed_contract": info.get('b2s2_agreed_contract'),
                "b2s2_buyer_utility": info.get('b2s2_buyer_utility'),
                "b2s2_seller_utility": info.get('b2s2_seller_utility'),
                "b2s2_z_max": info.get('b2s2_z_max'),
                # current_round has been incremented to reflect the completed round
                "total_rounds": info.get('round', 0),
                "total_reward": float(reward) if reward is not None else None,
                "buyer1_reward": info.get('buyer1_reward'),
                "buyer2_reward": info.get('buyer2_reward'),
                "seller1_reward": info.get('seller1_reward'),
                "seller2_reward": info.get('seller2_reward'),
                "global_score": info.get('global_score'),
                "buyer_score": info.get('buyer_score'),
                "seller_score": info.get('seller_score'),
                "termination_reason": info.get('termination_reason'),
                "elapsed_time": elapsed_time,
                "buyer1_max_price": buyer1_max_price,
                "buyer2_max_price": buyer2_max_price,
                "seller1_min_price": seller1_min_price,
                "seller2_min_price": seller2_min_price,
                "buyer_seller_contract_configs": buyer_seller_contract_configs,
                "product_info": product_info,
                "model": get_model_name(model),
            })
            break
    
    # Close environment
    env.close()
    print("\nSequential multi-buyer multi-seller negotiation completed!")
    
    # Ensure elapsed_time is set even if negotiation didn't complete normally
    if "elapsed_time" not in results:
        results["elapsed_time"] = time.time() - start_time
    
    # Save results to file
    try:
        # Create results directory structure
        results_dir = Path(project_root) / "agenticpay" / "results" / "multi_buyer_multi_seller"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Get model name for directory (sanitize for filesystem)
        model_name = get_model_name(model)
        model_name_safe = model_name.replace("/", "_").replace("\\", "_").replace(":", "_")
        model_dir = results_dir / model_name_safe
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Create timestamped subdirectory for this run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = model_dir / f"batch_evaluation_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=True)
        
        # Save summary JSON
        summary_file = run_dir / "summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # Save output text
        output_file = run_dir / "Task14_s10_smokehouse_treat_output.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("Task14 Scenario 10: AmeriMist Food Color (two listings) - Sequential Two-Buyer Two-Seller Negotiation Results\n")
            f.write("Category: Grocery & Gourmet Food\n")
            f.write("="*80 + "\n\n")
            f.write(f"Timestamp: {results['timestamp']}\n")
            f.write(f"Model: {results['model']}\n")
            f.write(f"User Requirement: {results['user_requirement']}\n")
            f.write(f"User Profile: {results['user_profile']}\n\n")
            f.write(f"Status: {results['status']}\n")
            f.write(f"Success: {results['success']}\n")
            f.write(f"Total Rounds: {results['total_rounds']}\n")
            elapsed_time = results.get('elapsed_time', 0)
            f.write(f"Elapsed Time: {elapsed_time:.2f}s\n\n")
            if results.get('selected_buyer') and results.get('selected_seller'):
                f.write(f"Selected Deal: Buyer {results['selected_buyer']} - Seller {results['selected_seller']}\n")
                f.write(f"Final Deal Price: ${results.get('final_deal_price', 0):.2f}\n\n")
                pk = f"b{results['selected_buyer']}s{results['selected_seller']}"
                fc = results.get(f"{pk}_agreed_contract")
                if fc is not None:
                    f.write(f"Final Contract: {fc}\n\n")
            f.write("Final Prices:\n")
            f.write(f"  Buyer1-Seller1: Buyer=${results['b1s1_buyer_price']:.2f} | Seller=${results['b1s1_seller_price']:.2f}" if results.get('b1s1_buyer_price') is not None and results.get('b1s1_seller_price') is not None else "  Buyer1-Seller1: Not specified")
            f.write("\n")
            f.write(f"  Buyer1-Seller2: Buyer=${results['b1s2_buyer_price']:.2f} | Seller=${results['b1s2_seller_price']:.2f}" if results.get('b1s2_buyer_price') is not None and results.get('b1s2_seller_price') is not None else "  Buyer1-Seller2: Not specified")
            f.write("\n")
            f.write(f"  Buyer2-Seller1: Buyer=${results['b2s1_buyer_price']:.2f} | Seller=${results['b2s1_seller_price']:.2f}" if results.get('b2s1_buyer_price') is not None and results.get('b2s1_seller_price') is not None else "  Buyer2-Seller1: Not specified")
            f.write("\n")
            f.write(f"  Buyer2-Seller2: Buyer=${results['b2s2_buyer_price']:.2f} | Seller=${results['b2s2_seller_price']:.2f}" if results.get('b2s2_buyer_price') is not None and results.get('b2s2_seller_price') is not None else "  Buyer2-Seller2: Not specified")
            f.write("\n\n")
            f.write("Contract Utilities:\n")
            for pair_key, label in (
                ("b1s1", "Buyer1-Seller1"),
                ("b1s2", "Buyer1-Seller2"),
                ("b2s1", "Buyer2-Seller1"),
                ("b2s2", "Buyer2-Seller2"),
            ):
                z_max = results.get(f"{pair_key}_z_max")
                buyer_utility = results.get(f"{pair_key}_buyer_utility")
                seller_utility = results.get(f"{pair_key}_seller_utility")
                agreed_contract = results.get(f"{pair_key}_agreed_contract")
                if z_max is not None:
                    f.write(f"  {label} Z_max: {z_max:.3f}\n")
                if buyer_utility is not None:
                    f.write(f"  {label} Buyer Utility: {buyer_utility:.3f}\n")
                if seller_utility is not None:
                    f.write(f"  {label} Seller Utility: {seller_utility:.3f}\n")
                if agreed_contract is not None:
                    f.write(f"  {label} Agreed Contract: {agreed_contract}\n")
            f.write("\n")
            product_info = results.get('product_info', {})
            f.write("Product:\n")
            f.write(f"  Name: {product_info.get('name', 'N/A')}\n")
            f.write(f"  Brand: {product_info.get('brand', 'N/A')}\n")
            f.write(f"  Price: ${product_info.get('price', product_info.get('original_price', 0)):.2f}\n")
            f.write("\n")
            f.write("Rewards:\n")
            if results.get('total_reward') is not None:
                f.write(f"  Total Reward: {results['total_reward']:.3f}\n")
            if results.get('buyer1_reward') is not None:
                f.write(f"  Buyer1 Reward: {results['buyer1_reward']:.3f}\n")
            if results.get('buyer2_reward') is not None:
                f.write(f"  Buyer2 Reward: {results['buyer2_reward']:.3f}\n")
            if results.get('seller1_reward') is not None:
                f.write(f"  Seller1 Reward: {results['seller1_reward']:.3f}\n")
            if results.get('seller2_reward') is not None:
                f.write(f"  Seller2 Reward: {results['seller2_reward']:.3f}\n")
            f.write("\n")
            f.write("Scores:\n")
            if results.get('global_score') is not None:
                f.write(f"  Global Score: {results['global_score']:.3f}\n")
            if results.get('buyer_score') is not None:
                f.write(f"  Buyer Score: {results['buyer_score']:.3f}\n")
            if results.get('seller_score') is not None:
                f.write(f"  Seller Score: {results['seller_score']:.3f}\n")
            f.write("\n")
            if results.get('termination_reason'):
                f.write(f"Termination Reason: {results['termination_reason']}\n")
            if results.get('error'):
                f.write(f"\nError: {results['error']}\n")
        
        print(f"\nResults saved to: {run_dir}")
        print(f"  - Summary JSON: {summary_file}")
        print(f"  - Output Text: {output_file}")
    except Exception as e:
        print(f"\nWarning: Failed to save results: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Task14 Scenario 10: AmeriMist Food Color (two listings) - Sequential Two-Buyer Two-Seller Negotiation")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name to use (e.g., 'gemini-3-pro-all', 'gpt-5.2', 'claude-sonnet-4-5-20250929'). If not provided, uses default model."
    )
    args = parser.parse_args()
    main(model_name=args.model)

