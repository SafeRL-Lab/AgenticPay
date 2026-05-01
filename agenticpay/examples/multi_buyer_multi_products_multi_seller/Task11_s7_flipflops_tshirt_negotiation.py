"""Task11 Scenario 7: Flip-flops & Marvel tee — Sequential 2x2 (2 products, image + text)

Same two-SKU cart, two offers, no per-seller identity in the listing; floors differ per buyer/seller.
``<selected_seller>`` routing aligned with Task5_s1. Category: Clothing & Fashion
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

from agenticpay.envs.multi_buyer_multi_products_multi_seller.Task3_sequential_two_buyer_two_seller_two_product_negotiation import (
    Task3SequentialTwoBuyerTwoSellerTwoProductNegotiation,
)
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
    """Structured ``<selected_seller>`` + retries + random fallback (aligned with Task5_s1)."""
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
    """Main function: Demonstrates sequential multi-buyer multi-seller multi-product negotiation flow
    
    Args:
        model_name: Optional model name. If None, uses default model.
    """
    
    print("Initializing model...")
    
    # OpenVLM via OpenAI-compatible API (product images passed to VLM)
    api_key = os.getenv("OPENAI_API_KEY") or OPENAI_API_KEY
    openvlm_base_url = os.getenv("OPENAI_URL") or os.getenv("OPENVLM_BASE_URL", "http://localhost:8000/v1")
    openvlm_model = os.getenv("OPENVLM_MODEL", "openvlm")
    
    model = OpenAIVLM(
        model=model_name or openvlm_model,
        api_key=api_key,
        base_url=openvlm_base_url,
    )
    
    print(f"✓ Successfully initialized: {model}")
    
    # Create Agents (set their respective bottom prices, this information is confidential, unknown to each other)
    # buyer_max_price and seller_min_price: bundle totals (Flip Flops + T-Shirt); confidential ZOPA sits below summed public list prices in product_info.
    print("Creating agents...")
    product_request = (
        "I want black flip-flops size 44 and the Captain America tee together. "
        "I also prefer the flip-flop sole sidewall to show a clean molded edge without obvious glue lines."
    )
    shared_contract_fields = {
        "contrainfo": {
            "product_request": product_request,
            "initial_contract_status": (
                "No total bundle price, delivery time, return policy, gift-wrap option, or user product preference match "
                "has been selected or agreed before negotiation starts."
            ),
            "contract_completion_requirement": (
                "A valid offer must explicitly fill price, continuous_terms.delivery_days, "
                "discrete_terms.return_policy, discrete_terms.gift_wrap, and discrete_terms.user_product_preference. "
                "The price is the total bundle price for both products (flip-flops + t-shirt)."
            ),
        },
        "field_descriptions": {
            "price": "The total amount of money the buyer pays for both products together, measured in US dollars.",
            "continuous_terms.delivery_days": (
                "How many days the seller can take to deliver the complete two-product bundle after the deal is made."
            ),
            "discrete_terms.return_policy": (
                "The return rule for the whole bundle. `30_days` means the buyer can return the bundle within "
                "30 days; `none` means the sale is final and returns are not allowed."
            ),
            "discrete_terms.gift_wrap": (
                "`yes` means the seller uses gift-style or retail packaging suitable for apparel; "
                "`no` means normal poly-mailer shipping."
            ),
            "discrete_terms.user_product_preference": (
                "How well the listings match the buyer's stated preference that the flip-flop sole sidewall show a clean "
                "molded edge without obvious glue lines. `strong_match` when clearly satisfied; `partial_match` when "
                "ambiguous or partly satisfied; `mismatch_or_uncertain` when not satisfied or unconfirmable."
            ),
        },
        "continuous_bounds": {"delivery_days": {"min": 1, "max": 7}},
        "discrete_options": {
            "return_policy": ["30_days", "none"],
            "gift_wrap": ["yes", "no"],
            "user_product_preference": ["strong_match", "partial_match", "mismatch_or_uncertain"],
        },
    }
    buyer1_preferences = {
        "v_base": 31.68,
        "weight_descriptions": {
            "v_base": (
                "Your private maximum value for the two-product apparel bundle before delivery, return, and gift-wrap terms, measured in dollars. "
                "A lower price is better for you because every dollar paid reduces your utility by 1 dollar."
            ),
            "continuous_weights.delivery_days": (
                "How much each additional delivery day changes your utility, measured in dollars per day. "
                "A negative number means slower delivery is worse for you."
            ),
            "discrete_weights.return_policy": (
                "How much each return-policy option changes your utility, measured in dollars. "
                "Positive numbers are good for you; negative numbers are bad for you."
            ),
            "discrete_weights.gift_wrap": (
                "How much each gift-wrap option changes your utility, measured in dollars. "
                "Positive numbers are good for you; negative numbers are bad for you."
            ),
            "discrete_weights.user_product_preference": (
                "How much each match level for your stated sole-edge preference changes your utility, measured in dollars. "
                "Positive numbers are good for you; negative numbers are bad for you."
            ),
        },
        "continuous_weights": {"delivery_days": -0.42},
        "discrete_weights": {
            "return_policy": {"30_days": 1.3, "none": -1.5},
            "gift_wrap": {"yes": 1.0, "no": -0.25},
            "user_product_preference": {
                "strong_match": 0.24,
                "partial_match": 0.09,
                "mismatch_or_uncertain": -0.18,
            },
        },
    }
    buyer2_preferences = json.loads(json.dumps(buyer1_preferences))
    buyer2_preferences["v_base"] = 33.48
    buyer2_preferences["continuous_weights"]["delivery_days"] = -0.28
    buyer2_preferences["discrete_weights"]["return_policy"] = {"30_days": 1.05, "none": -1.15}
    buyer2_preferences["discrete_weights"]["gift_wrap"] = {"yes": 0.75, "no": -0.12}
    seller1_preferences = {
        "c_base": 28.48,
        "weight_descriptions": {
            "c_base": (
                "Your private minimum cost for fulfilling the two-product apparel bundle before delivery, return, and gift-wrap terms, measured in dollars. "
                "A higher price is better for you because every dollar received increases your utility by 1 dollar."
            ),
            "continuous_weights.delivery_days": (
                "How much each additional delivery day changes your utility, measured in dollars per day. "
                "A positive number means more delivery flexibility is better for you."
            ),
            "discrete_weights.return_policy": (
                "How much each return-policy option changes your utility, measured in dollars. "
                "Positive numbers are good for you; negative numbers are bad for you."
            ),
            "discrete_weights.gift_wrap": (
                "How much each gift-wrap option changes your utility, measured in dollars. "
                "Positive numbers are good for you; negative numbers are bad for you."
            ),
            "discrete_weights.user_product_preference": (
                "How much committing to each match level on the buyer's sole molding preference changes your utility, measured in dollars. "
                "Stronger commitments carry a small nonzero risk or handling cost."
            ),
        },
        "continuous_weights": {"delivery_days": 0.32},
        "discrete_weights": {
            "return_policy": {"30_days": -1.7, "none": 1.05},
            "gift_wrap": {"yes": -0.95, "no": 0.18},
            "user_product_preference": {
                "strong_match": -0.065,
                "partial_match": -0.032,
                "mismatch_or_uncertain": 0.008,
            },
        },
    }
    seller2_preferences = json.loads(json.dumps(seller1_preferences))
    seller2_preferences["c_base"] = 25.41
    seller2_preferences["continuous_weights"]["delivery_days"] = 0.36
    seller2_preferences["discrete_weights"]["return_policy"] = {"30_days": -1.45, "none": 0.88}
    seller2_preferences["discrete_weights"]["gift_wrap"] = {"yes": -0.82, "no": 0.14}

    buyer_seller_contract_configs = {
        "b1s1": {
            **shared_contract_fields,
            "buyer_preferences": buyer1_preferences,
            "seller_preferences": seller1_preferences,
        },
        "b1s2": {
            **shared_contract_fields,
            "buyer_preferences": buyer1_preferences,
            "seller_preferences": seller2_preferences,
        },
        "b2s1": {
            **shared_contract_fields,
            "buyer_preferences": buyer2_preferences,
            "seller_preferences": seller1_preferences,
        },
        "b2s2": {
            **shared_contract_fields,
            "buyer_preferences": buyer2_preferences,
            "seller_preferences": seller2_preferences,
        },
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
    print("Creating sequential multi-buyer multi-seller multi-product negotiation environment...")
    env = Task3SequentialTwoBuyerTwoSellerTwoProductNegotiation(
        buyer1_agent=buyer1,
        buyer2_agent=buyer2,
        seller1_agent=seller1,
        seller2_agent=seller2,
        max_rounds=max_rounds,
        initial_seller1_price=38.0,  # Initial total price offered by seller1 (Flip Flops + T-Shirt)
        initial_seller2_price=36.0,  # Initial total price offered by seller2 (competitive pricing)
        buyer1_max_price=buyer1_max_price,  # Buyer1 total max price (confidential)
        buyer2_max_price=buyer2_max_price,  # Buyer2 total max price (confidential)
        seller1_min_price=seller1_min_price,  # Seller1 total min price (confidential)
        seller2_min_price=seller2_min_price,  # Seller2 total min price (confidential)
        environment_info={
            "platform": "Amazon",
            "market_type": "B2C",
            "note": "Multiple third-party offers for the same two-SKU cart; prices are bundle totals.",
            "seller1_listing_age": "2 days",
            "seller2_listing_age": "1 week",
            "buyer_seller_contract_configs": buyer_seller_contract_configs,
        },
        price_tolerance=price_tolerance,
        reward_weights=reward_weights,  # Reward weights configuration
    )
    
    user_profile = None
    print(f"User Profile: {user_profile}")
    
    # Define two products with their individual prices (Product 1 from Task10_s7_sandals, Product 2 from sampled_products2.jsonl line 7)
    # Product 1: N/C Mens Flip Flops
    # Product 2: Marvel Avengers Captain America T-Shirt
    product_info = {
        "products": [
            {
                "name": "N/C Mens Flip Flops Thong Sandals Yoga Foam Slippers 44 R011 Black",
                "brand": "N/C",
                "price": 17.99,
                "list_price": 17.99,
                "condition": "New",
                "size": "44",
                "color": "R011 Black",
                "product_category": "Clothing, Shoes & Jewelry › Men › Shoes › Sandals",
                "average_rating": 4.0,
                "total_reviews": 0,
                "asin": "B0989VY7D8",
                "full_description": "Men's flip flops: the skin contact part is made of cloth so you can walk without friction or sharpness. Even if used for a long time, it will not cause blisters. Antiskid comfort with good spiral antiskid pattern at the bottom. Arch support provides good walking stability and keeps your feet comfortable. Upper material: mesh. Sole material: PVC. Waterproof and suitable for all seasons. Perfect for beach, pool, and casual wear. Sizes: 39-45.",
                "image_url": "https://m.media-amazon.com/images/I/61vR1ZJ9u3S.jpg",
            },
            {
                "name": "Marvel Avengers: Endgame Captain America America's Language T-Shirt",
                "brand": "Marvel",
                "price": 22.99,
                "list_price": 22.99,
                "condition": "New",
                "product_category": "Clothing, Shoes & Jewelry › Novelty & More › Clothing › Novelty › Women › Tops & Tees › T-Shirts",
                "average_rating": 5,
                "total_reviews": 2,
                "asin": "B07XPR3R7N",
                "full_description": "Team up with what is left of the Avengers to fix the damage that Thanos has caused to the universe. You'll find the perfect gear within this collection of Officially Licensed Marvel Avengers: Endgame tee shirts, sweatshirts, and hoodies!",
                "small_description": ["Officially Licensed Marvel Apparel", "19MARF00431A-001", "Lightweight, Classic fit, Double-needle sleeve and bottom hem"],
                "image_url": "https://m.media-amazon.com/images/I/A13usaonutL.png",
            },
        ]
    }
    
    # Calculate total product price
    total_product_price = sum(p["price"] for p in product_info["products"])
    print(f"\nProducts (Flip Flops + T-Shirt Package):")
    for i, p in enumerate(product_info["products"], 1):
        print(f"  {i}. {p['name']}: ${p['price']:.2f}")
    print(f"  Total Package Price: ${total_product_price:.2f}")
    
    user_requirement = product_request
    print(f"Using default requirement: {user_requirement}")
    
    # Reset environment
    print("\n" + "="*60)
    print("Starting new sequential negotiation for Flip Flops + Marvel T-Shirt package...")
    print("Two buyers competing with two sellers for Flip Flops and Marvel Avengers T-Shirt")
    print("="*60)
    
    observation, info = env.reset(
        user_requirement=user_requirement,
        product_info=product_info,
        user_profile=user_profile,  # Pass user profile
    )
    
    # Start negotiation loop
    done = False
    start_time = time.time()
    
    # Initialize results dictionary
    results = {
        "task": "Task11_s7_flipflops_tshirt_negotiation",
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
        for msg in observation.get("conversation_history_b1s1", []):
            combined_history_b1.append({**msg, "thread_label": "Talk with Seller 1"})
        for msg in observation.get("conversation_history_b1s2", []):
            combined_history_b1.append({**msg, "thread_label": "Talk with Seller 2"})

        combined_history_b2 = []
        for msg in observation.get("conversation_history_b2s1", []):
            combined_history_b2.append({**msg, "thread_label": "Talk with Seller 1"})
        for msg in observation.get("conversation_history_b2s2", []):
            combined_history_b2.append({**msg, "thread_label": "Talk with Seller 2"})

        routing_instruction = (
            "You are negotiating with two sellers. Each round, choose exactly ONE seller "
            "and output that choice in a dedicated <selected_seller> block containing only "
            "the digit 1 or 2. Then put only your negotiation text in <message>, including "
            "one complete <contract>...</contract> JSON block for the selected seller. "
            "The contract price is the **total** for both products in the cart."
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
                    print(f"  Buyer1 Step Reward = buyer_savings({buyer_savings:.2f} * {weights['buyer_savings']:.2f}) + round_cost({round_cost:.2f} * {weights['time_cost']:.2f}) = {info['step_buyer1_reward']:.2f} (buyer1_max={env.buyer1_max_price}, buyer_total_price={buyer_price:.2f}, round={info['round']})")
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
                    print(f"  Buyer2 Step Reward = buyer_savings({buyer_savings:.2f} * {weights['buyer_savings']:.2f}) + round_cost({round_cost:.2f} * {weights['time_cost']:.2f}) = {info['step_buyer2_reward']:.2f} (buyer2_max={env.buyer2_max_price}, buyer_total_price={buyer_price:.2f}, round={info['round']})")
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
                    print(f"  Seller1 Step Reward = seller_profit({seller1_profit:.2f} * {weights['seller_profit']:.2f}) + round_cost({round_cost:.2f} * {weights['time_cost']:.2f}) = {info['step_seller1_reward']:.2f} (seller1_total_price={seller1_price:.2f}, seller1_min={env.seller1_min_price}, round={info['round']})")
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
                    print(f"  Seller2 Step Reward = seller_profit({seller2_profit:.2f} * {weights['seller_profit']:.2f}) + round_cost({round_cost:.2f} * {weights['time_cost']:.2f}) = {info['step_seller2_reward']:.2f} (seller2_total_price={seller2_price:.2f}, seller2_min={env.seller2_min_price}, round={info['round']})")
                else:
                    weighted_round_cost = round_cost * weights["time_cost"]
                    print(f"  Seller2 Step Reward = round_cost({round_cost:.2f} * {weights['time_cost']:.2f}) = {weighted_round_cost:.2f} (seller2_price not specified, round={info['round']})")
        
        if done:
            print("\n" + "=" * 60)
            print("Negotiation Ended")
            print("=" * 60)
            print(f"Status: {info['status']}")
            if info.get("selected_buyer") and info.get("selected_seller"):
                print(f"Selected Deal: Buyer {info['selected_buyer']} - Seller {info['selected_seller']}")
                print(f"Final Deal Total Price: ${info.get('final_deal_price', 0):.2f}")
                pair_key = f"b{info['selected_buyer']}s{info['selected_seller']}"
                agreed_contract = info.get(f"{pair_key}_agreed_contract")
                if agreed_contract is not None:
                    print(f"Final Contract: {agreed_contract}")
            print(
                f"Buyer1-Seller1 Total: Buyer=${info.get('b1s1_buyer_price', 0) or 0:.2f} | "
                f"Seller=${info.get('b1s1_seller_price', 0) or 0:.2f}"
            )
            print(
                f"Buyer1-Seller2 Total: Buyer=${info.get('b1s2_buyer_price', 0) or 0:.2f} | "
                f"Seller=${info.get('b1s2_seller_price', 0) or 0:.2f}"
            )
            print(
                f"Buyer2-Seller1 Total: Buyer=${info.get('b2s1_buyer_price', 0) or 0:.2f} | "
                f"Seller=${info.get('b2s1_seller_price', 0) or 0:.2f}"
            )
            print(
                f"Buyer2-Seller2 Total: Buyer=${info.get('b2s2_buyer_price', 0) or 0:.2f} | "
                f"Seller=${info.get('b2s2_seller_price', 0) or 0:.2f}"
            )
            env._print_global_score_details()
            env._print_buyer_score_details()
            env._print_seller_score_details()
            actual_rounds = info["round"]
            print(f"Total Rounds: {actual_rounds}")
            print(f"Global Reward: {reward:.3f}")
            if "buyer1_reward" in info:
                print(f"Buyer1 Reward: {info['buyer1_reward']:.3f}")
            if "buyer2_reward" in info:
                print(f"Buyer2 Reward: {info['buyer2_reward']:.3f}")
            if "seller1_reward" in info:
                print(f"Seller1 Reward: {info['seller1_reward']:.3f}")
            if "seller2_reward" in info:
                print(f"Seller2 Reward: {info['seller2_reward']:.3f}")
            if "global_score" in info:
                print(f"GlobalScore: {info['global_score']:.3f}")
            if "buyer_score" in info:
                print(f"BuyerScore: {info['buyer_score']:.3f}")
            if "seller_score" in info:
                print(f"SellerScore: {info['seller_score']:.3f}")
            if info.get("termination_reason"):
                print(f"Reason: {info['termination_reason']}")
            print("=" * 60)

            elapsed_time = time.time() - start_time
            product_info_out = info.get("product_info", {})
            results.update(
                {
                    "status": info.get("status", "unknown"),
                    "success": terminated,
                    "selected_buyer": info.get("selected_buyer"),
                    "selected_seller": info.get("selected_seller"),
                    "final_deal_price": info.get("final_deal_price"),
                    "b1s1_buyer_price": info.get("b1s1_buyer_price"),
                    "b1s1_seller_price": info.get("b1s1_seller_price"),
                    "b1s2_buyer_price": info.get("b1s2_buyer_price"),
                    "b1s2_seller_price": info.get("b1s2_seller_price"),
                    "b2s1_buyer_price": info.get("b2s1_buyer_price"),
                    "b2s1_seller_price": info.get("b2s1_seller_price"),
                    "b2s2_buyer_price": info.get("b2s2_buyer_price"),
                    "b2s2_seller_price": info.get("b2s2_seller_price"),
                    "b1s1_agreed_contract": info.get("b1s1_agreed_contract"),
                    "b1s1_buyer_utility": info.get("b1s1_buyer_utility"),
                    "b1s1_seller_utility": info.get("b1s1_seller_utility"),
                    "b1s1_z_max": info.get("b1s1_z_max"),
                    "b1s2_agreed_contract": info.get("b1s2_agreed_contract"),
                    "b1s2_buyer_utility": info.get("b1s2_buyer_utility"),
                    "b1s2_seller_utility": info.get("b1s2_seller_utility"),
                    "b1s2_z_max": info.get("b1s2_z_max"),
                    "b2s1_agreed_contract": info.get("b2s1_agreed_contract"),
                    "b2s1_buyer_utility": info.get("b2s1_buyer_utility"),
                    "b2s1_seller_utility": info.get("b2s1_seller_utility"),
                    "b2s1_z_max": info.get("b2s1_z_max"),
                    "b2s2_agreed_contract": info.get("b2s2_agreed_contract"),
                    "b2s2_buyer_utility": info.get("b2s2_buyer_utility"),
                    "b2s2_seller_utility": info.get("b2s2_seller_utility"),
                    "b2s2_z_max": info.get("b2s2_z_max"),
                    "total_rounds": info.get("round", 0),
                    "total_reward": float(reward) if reward is not None else None,
                    "buyer1_reward": info.get("buyer1_reward"),
                    "buyer2_reward": info.get("buyer2_reward"),
                    "seller1_reward": info.get("seller1_reward"),
                    "seller2_reward": info.get("seller2_reward"),
                    "global_score": info.get("global_score"),
                    "buyer_score": info.get("buyer_score"),
                    "seller_score": info.get("seller_score"),
                    "termination_reason": info.get("termination_reason"),
                    "elapsed_time": elapsed_time,
                    "buyer1_max_price": buyer1_max_price,
                    "buyer2_max_price": buyer2_max_price,
                    "seller1_min_price": seller1_min_price,
                    "seller2_min_price": seller2_min_price,
                    "buyer_seller_contract_configs": buyer_seller_contract_configs,
                    "product_info": product_info_out,
                    "model": get_model_name(model),
                }
            )
            break
    
    # Close environment
    env.close()
    print("\nFlip Flops & T-Shirt package negotiation completed!")
    
    # Ensure elapsed_time is set even if negotiation didn't complete normally
    if "elapsed_time" not in results:
        results["elapsed_time"] = time.time() - start_time
    
    # Save results to file
    try:
        # Create results directory structure
        results_dir = Path(project_root) / "agenticpay" / "results" / "multi_buyer_multi_products_multi_seller"
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
        output_file = run_dir / "Task11_s7_flipflops_tshirt_output.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("Task11 Scenario 7: Flip Flops & Marvel T-Shirt - Sequential Two-Buyer Two-Seller Two-Product Negotiation Results\n")
            f.write("Category: Clothing & Fashion\n")
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
                f.write(f"Final Deal Total Price: ${results.get('final_deal_price', 0):.2f}\n\n")
                pair_key = f"b{results['selected_buyer']}s{results['selected_seller']}"
                agreed_contract = results.get(f"{pair_key}_agreed_contract")
                if agreed_contract is not None:
                    f.write(f"Final Contract: {agreed_contract}\n\n")
            f.write("Final Total Prices (bundle):\n")
            f.write(
                f"  Buyer1-Seller1: Buyer=${results['b1s1_buyer_price']:.2f} | Seller=${results['b1s1_seller_price']:.2f}\n"
                if results.get("b1s1_buyer_price") is not None and results.get("b1s1_seller_price") is not None
                else "  Buyer1-Seller1: Not specified\n"
            )
            f.write(
                f"  Buyer1-Seller2: Buyer=${results['b1s2_buyer_price']:.2f} | Seller=${results['b1s2_seller_price']:.2f}\n"
                if results.get("b1s2_buyer_price") is not None and results.get("b1s2_seller_price") is not None
                else "  Buyer1-Seller2: Not specified\n"
            )
            f.write(
                f"  Buyer2-Seller1: Buyer=${results['b2s1_buyer_price']:.2f} | Seller=${results['b2s1_seller_price']:.2f}\n"
                if results.get("b2s1_buyer_price") is not None and results.get("b2s1_seller_price") is not None
                else "  Buyer2-Seller1: Not specified\n"
            )
            f.write(
                f"  Buyer2-Seller2: Buyer=${results['b2s2_buyer_price']:.2f} | Seller=${results['b2s2_seller_price']:.2f}\n\n"
                if results.get("b2s2_buyer_price") is not None and results.get("b2s2_seller_price") is not None
                else "  Buyer2-Seller2: Not specified\n\n"
            )
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
            pin = results.get("product_info", {}) or {}
            f.write("Products:\n")
            for i, p in enumerate(pin.get("products", []), 1):
                price_val = p.get("list_price", p.get("price", p.get("original_price", 0)))
                f.write(f"  {i}. {p.get('name', 'N/A')} by {p.get('brand', 'N/A')} — list ${float(price_val):.2f}\n")
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
    parser = argparse.ArgumentParser(description="Task11 Scenario 7: Flip Flops & Marvel T-Shirt - Sequential Two-Buyer Two-Seller Two-Product Negotiation")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="OpenVLM model name. Set OPENAI_URL/OPENVLM_BASE_URL for API endpoint, OPENVLM_MODEL for default model name."
    )
    args = parser.parse_args()
    main(model_name=args.model)
