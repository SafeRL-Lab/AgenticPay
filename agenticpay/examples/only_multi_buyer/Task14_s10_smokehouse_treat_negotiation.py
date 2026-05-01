"""Task14 Scenario 10: The Smokehouse Treat - Sequential Two-Buyer Negotiation

One seller negotiating with two potential buyers for The Smokehouse Treat by Burgers' Smokehouse (from sampled_products2.jsonl line 10).
Gift pack with smoked sausage and cheeses. Seller chooses which buyer to negotiate with each round.
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

from agenticpay.envs.only_multi_buyer.Task3_sequential_two_buyer_negotiation import Task3SequentialTwoBuyerNegotiation
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


def _run_seller_routing(
    seller,
    combined_history: list,
    observation: dict,
    routing_instruction: str,
):
    """Align with only_multi_seller: structured ``<selected_buyer>`` + retries + random fallback."""
    max_selection_retries = 2
    retry_count = 0
    inst = routing_instruction
    seller_response = None
    selected_buyer = None
    while True:
        seller_response = seller.respond(
            conversation_history=combined_history,
            current_state={
                **observation,
                "instruction": inst,
                "num_buyers": 2,
            },
        )
        selected_buyer = seller.last_selected_buyer
        if selected_buyer is not None:
            break
        if retry_count >= max_selection_retries:
            break
        retry_count += 1
        print(
            f"\n[Warning] Missing <selected_buyer>; retrying seller response "
            f"({retry_count}/{max_selection_retries})..."
        )
        inst = (
            routing_instruction
            + " IMPORTANT: You MUST include a valid <selected_buyer> block with only 1 or 2."
        )
    if selected_buyer is None:
        selected_buyer = random.choice([1, 2])
        print(
            f"\n[Warning] Failed to parse <selected_buyer> after retries; "
            f"randomly selecting Buyer {selected_buyer}."
        )
    return seller_response, selected_buyer


def main(model_name=None):
    """Main function: Demonstrates sequential multi-buyer negotiation flow
    
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
    
    print("Creating agents...")
    product_request = (
        "I want Burgers' Smokehouse Treat gift pack—smoked sausage & cheeses as listed. "
        "I also prefer the vacuum-sealed items in the photo to lie fairly snug in the tray without "
        "large obvious air pillows lifting the plastic away from the products."
    )
    v_ref, c_ref = 8.7, 4.5
    buyer1_max_price = 47.0
    buyer2_max_price = 50.0
    sb1, sb2 = buyer1_max_price / v_ref, buyer2_max_price / v_ref
    c_b1_seller, c_b2_seller = 43.0, 39.0
    ss1, ss2 = c_b1_seller / c_ref, c_b2_seller / c_ref

    def _beauty_buyer_prefs(scale: float):
        return {
            "continuous_weights": {"delivery_days": -0.30 * scale},
            "discrete_weights": {
                "return_policy": {"30_days": 1.1 * scale, "none": -1.3 * scale},
                "packaging": {"protective": 1.0 * scale, "standard": -0.4 * scale},
                "user_product_preference": {
                    "strong_match": 0.30 * scale,
                    "partial_match": 0.12 * scale,
                    "mismatch_or_uncertain": -0.25 * scale,
                },
            },
        }

    def _beauty_buyer_prefs_b2(scale: float):
        return {
            "continuous_weights": {"delivery_days": -0.20 * scale},
            "discrete_weights": {
                "return_policy": {"30_days": 0.9 * scale, "none": -1.0 * scale},
                "packaging": {"protective": 0.8 * scale, "standard": -0.2 * scale},
                "user_product_preference": {
                    "strong_match": 0.30 * scale,
                    "partial_match": 0.12 * scale,
                    "mismatch_or_uncertain": -0.25 * scale,
                },
            },
        }

    def _beauty_seller_prefs(scale: float):
        return {
            "continuous_weights": {"delivery_days": 0.20 * scale},
            "discrete_weights": {
                "return_policy": {"30_days": -1.4 * scale, "none": 1.0 * scale},
                "packaging": {"protective": -0.8 * scale, "standard": 0.3 * scale},
                "user_product_preference": {
                    "strong_match": -0.08 * scale,
                    "partial_match": -0.04 * scale,
                    "mismatch_or_uncertain": 0.01 * scale,
                },
            },
        }

    def _beauty_seller_prefs_b2(scale: float):
        return {
            "continuous_weights": {"delivery_days": 0.25 * scale},
            "discrete_weights": {
                "return_policy": {"30_days": -1.4 * scale, "none": 1.0 * scale},
                "packaging": {"protective": -0.85 * scale, "standard": 0.32 * scale},
                "user_product_preference": {
                    "strong_match": -0.08 * scale,
                    "partial_match": -0.04 * scale,
                    "mismatch_or_uncertain": 0.01 * scale,
                },
            },
        }

    buyer1_contract_config = {
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
            "price": "The total amount of money the buyer pays for the whole deal, measured in US dollars.",
            "continuous_terms.delivery_days": (
                "How many days the seller can take to deliver the Smokehouse Treat gift pack after the deal is made."
            ),
            "discrete_terms.return_policy": (
                "The return rule for the order. `30_days` means the buyer can return within 30 days; "
                "`none` means the sale is final and returns are not allowed."
            ),
            "discrete_terms.packaging": (
                "Shipment packaging / handling tier. `protective` means more careful packing for refrigerated "
                "or gift-quality presentation; `standard` means normal food shipment packaging."
            ),
            "discrete_terms.user_product_preference": (
                "How well the listing matches the buyer's preference for vacuum-sealed products sitting snug "
                "in the tray without large air pockets visibly lifting the wrap. Assign `strong_match`, "
                "`partial_match`, or `mismatch_or_uncertain`."
            ),
        },
        "continuous_bounds": {"delivery_days": {"min": 1, "max": 7}},
        "discrete_options": {
            "return_policy": ["30_days", "none"],
            "packaging": ["protective", "standard"],
            "user_product_preference": ["strong_match", "partial_match", "mismatch_or_uncertain"],
        },
        "buyer_preferences": {
            "v_base": buyer1_max_price,
            "weight_descriptions": {
                "v_base": (
                    "Your private maximum value for this gourmet gift-pack order before delivery, return, and packaging terms, measured in dollars. "
                    "A lower price is better because every dollar paid reduces your utility by 1 dollar."
                ),
                "continuous_weights.delivery_days": (
                    "How much each additional delivery day changes your utility ($/day). Negative means slower delivery is worse."
                ),
                "discrete_weights.return_policy": (
                    "How much each return-policy option changes your utility ($). Positive is good for you."
                ),
                "discrete_weights.packaging": (
                    "How much each packaging option changes your utility ($). Positive is good for you."
                ),
                "discrete_weights.user_product_preference": (
                    "How each match level on your stated product preference changes your utility ($). Positive helps you."
                ),
            },
            **_beauty_buyer_prefs(sb1),
        },
        "seller_preferences": {
            "c_base": c_b1_seller,
            "weight_descriptions": {
                "c_base": (
                    "Your private minimum cost before delivery, return, and packaging terms, measured in dollars. "
                    "A higher price is better because every dollar received increases your utility by 1 dollar."
                ),
                "continuous_weights.delivery_days": (
                    "How much each additional delivery day changes your utility ($/day). Positive means more lead time helps you."
                ),
                "discrete_weights.return_policy": (
                    "How much each return-policy option changes your utility ($)."
                ),
                "discrete_weights.packaging": (
                    "How much each packaging option changes your utility ($)."
                ),
                "discrete_weights.user_product_preference": (
                    "How each commitment level on the buyer's vacuum-pack presentation preference shifts your utility ($); "
                    "firmer commitments carry a small nonzero risk cost."
                ),
            },
            **_beauty_seller_prefs(ss1),
        },
    }
    buyer2_contract_config = {
        **{k: v for k, v in buyer1_contract_config.items() if k not in ("buyer_preferences", "seller_preferences")},
        "buyer_preferences": {
            **{k: v for k, v in buyer1_contract_config["buyer_preferences"].items() if k not in ("v_base", "continuous_weights", "discrete_weights")},
            "v_base": buyer2_max_price,
            "weight_descriptions": {
                **buyer1_contract_config["buyer_preferences"]["weight_descriptions"],
                "v_base": (
                    "Your private maximum value for this gourmet gift-pack order before delivery, return, and packaging terms, measured in dollars. "
                    "A lower price is better because every dollar paid reduces your utility by 1 dollar."
                ),
            },
            **_beauty_buyer_prefs_b2(sb2),
        },
        "seller_preferences": {
            **{k: v for k, v in buyer1_contract_config["seller_preferences"].items() if k not in ("c_base", "continuous_weights", "discrete_weights")},
            "c_base": c_b2_seller,
            **_beauty_seller_prefs_b2(ss2),
        },
    }
    buyer_contract_configs = {1: buyer1_contract_config, 2: buyer2_contract_config}
    seller_min_price = min(cfg["seller_preferences"]["c_base"] for cfg in buyer_contract_configs.values())

    buyer1 = BuyerAgent(model=model, name="Buyer1", buyer_max_price=buyer1_max_price)
    buyer2 = BuyerAgent(model=model, name="Buyer2", buyer_max_price=buyer2_max_price)
    seller = SellerAgent(model=model, name="Seller1", seller_min_price=seller_min_price)
    
    # Create environment
    print("Creating sequential multi-buyer negotiation environment...")
    env = Task3SequentialTwoBuyerNegotiation(
        buyer1_agent=buyer1,
        buyer2_agent=buyer2,
        seller_agent=seller,
        max_rounds=max_rounds,
        buyer1_max_price=buyer1_max_price,  # Buyer1 maximum acceptable price (confidential)
        buyer2_max_price=buyer2_max_price,  # Buyer2 maximum acceptable price (confidential)
        seller_min_price=seller_min_price,  # Seller minimum acceptable price (confidential)
        environment_info={
            "platform": "Amazon",
            "market_type": "B2C",
            "buyer_contract_configs": buyer_contract_configs,
        },
        price_tolerance=price_tolerance,
        reward_weights=reward_weights,  # Reward weights configuration
    )
    
    # Create user profile (text description of personal preferences)
    user_profile = None
    print(f"User Profile: {user_profile}")
    
    # Get user requirement
    # Use default requirement for automatic running
    user_requirement = product_request
    print(f"Using default requirement: {user_requirement}")
    
    # Reset environment
    print("\n" + "="*60)
    print("Starting new sequential negotiation with two buyers (The Smokehouse Treat)...")
    print("="*60)
    
    # Product: The Smokehouse Treat by Burgers' Smokehouse (from sampled_products2.jsonl line 10)
    product_image_url = "https://m.media-amazon.com/images/I/51aHD-sJ1FS.jpg"
    
    observation, info = env.reset(
        user_requirement=user_requirement,
        product_info={
            "name": "The Smokehouse Treat by Burgers' Smokehouse",
            "condition": "New",
            "brand": "Visit the Burgers' Smokehouse Store",
            "original_price": 62.00,
            "product_category": "Grocery & Gourmet Food › Food & Beverage Gifts › Meat & Seafood Gifts",
            "average_rating": 5.0,
            "total_reviews": 1,
            "seller_name": "Burgers Smokehouse",
            "asin": "B01LA37T1S",
            "full_description": "This pack offers fine smoked sausage and cheeses. It is great to serve to guests or to give as a gift for any occasion. Contains: One 12 oz. Smoked Ozark Sausage One 12 oz. Beef Sausage One 11 oz. Smoked Cheddar Cheese One 10 oz. Baby Swiss Cheese",
            "small_description": "The Best Cheese and Summer Sausages. Ready to Slice for Appetizers and Hors d'oeuvres. Makes Entertaining Easy.",
            "image_url": product_image_url,
        },
        user_profile=user_profile,  # Pass user profile
    )
    
    # Start negotiation loop
    done = False
    start_time = time.time()

    routing_instruction = (
        "You are negotiating with two buyers. Each round, choose exactly ONE buyer "
        "and output that choice in a dedicated <selected_buyer> block containing only "
        "the digit 1 or 2. Follow the required <mental_model> / <message> format and include "
        "one complete <contract>...</contract> JSON block in <message>."
    )
    
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
        # Each round: buyers respond first (if first round), then seller chooses buyer and responds
        # For sequential negotiation, we need to handle the flow:
        # 1. First round: both buyers respond first, then seller chooses one and responds
        # 2. Subsequent rounds: seller chooses buyer first, then buyer responds, then seller responds
        
        current_round = observation.get('current_round', 0)
        
        # First round: buyers respond first based on product info
        if current_round == 0:
            # Get buyer1's initial response
            buyer1_action = buyer1.respond(
                conversation_history=observation["conversation_history_buyer1"],
                current_state=observation
            )
            
            # Get buyer2's initial response
            buyer2_action = buyer2.respond(
                conversation_history=observation["conversation_history_buyer2"],
                current_state=observation
            )
            
            # Create updated conversation histories that include buyers' responses
            updated_conversation_history_buyer1 = observation["conversation_history_buyer1"].copy()
            updated_conversation_history_buyer2 = observation["conversation_history_buyer2"].copy()
            
            if buyer1_action:
                updated_conversation_history_buyer1.append({
                    "role": "buyer",
                    "content": buyer1_action,
                    "round": current_round
                })
            
            if buyer2_action:
                updated_conversation_history_buyer2.append({
                    "role": "buyer",
                    "content": buyer2_action,
                    "round": current_round
                })
            
            # Seller can see both buyers' messages and choose which one to negotiate with
            combined_history = []
            for msg in updated_conversation_history_buyer1:
                combined_history.append({**msg, "thread_label": "Talk with Buyer 1"})
            for msg in updated_conversation_history_buyer2:
                combined_history.append({**msg, "thread_label": "Talk with Buyer 2"})

            seller_response, selected_buyer = _run_seller_routing(
                seller, combined_history, observation, routing_instruction
            )
            print(f"\n[Seller chooses to negotiate with Buyer {selected_buyer} this round]")

            seller_action = seller_response
            
            # Use the buyer action for the selected buyer
            if selected_buyer == 1:
                buyer_action = buyer1_action
            else:
                buyer_action = buyer2_action
        else:
            # Subsequent rounds: seller chooses buyer first, then buyer responds, then seller responds
            combined_history = []
            for msg in observation.get("conversation_history_buyer1", []):
                combined_history.append({**msg, "thread_label": "Talk with Buyer 1"})
            for msg in observation.get("conversation_history_buyer2", []):
                combined_history.append({**msg, "thread_label": "Talk with Buyer 2"})

            seller_response, selected_buyer = _run_seller_routing(
                seller, combined_history, observation, routing_instruction
            )
            print(f"\n[Seller chooses to negotiate with Buyer {selected_buyer} this round]")

            seller_action = seller_response
            
            # Get the conversation history for the selected buyer
            if selected_buyer == 1:
                conversation_history = observation["conversation_history_buyer1"]
            else:
                conversation_history = observation["conversation_history_buyer2"]
            
            # Create updated conversation history that includes seller's message
            # So buyer can see seller's message before responding
            updated_conversation_history = conversation_history.copy()
            if seller_action:
                updated_conversation_history.append({
                    "role": "seller",
                    "content": seller_action,
                    "round": current_round
                })
            
            # Get the selected buyer's response (buyer can now see seller's message)
            if selected_buyer == 1:
                buyer_action = buyer1.respond(
                    conversation_history=updated_conversation_history,
                    current_state=observation
                )
            else:
                buyer_action = buyer2.respond(
                    conversation_history=updated_conversation_history,
                    current_state=observation
                )
        
        # Execute step with selected buyer and actions (order: buyer_action, seller_action)
        observation, reward, terminated, truncated, info = env.step(
            selected_buyer=selected_buyer,
            buyer_action=buyer_action,
            seller_action=seller_action
        )
        done = terminated or truncated
        
        # Render current state (includes all print information)
        env.render()
        
        # Flush output to ensure complete display
        sys.stdout.flush()
        
        # Display step rewards for each round with detailed calculation
        if 'step_buyer1_reward' in info or 'step_buyer2_reward' in info or 'step_seller_reward' in info:
            print(f"\n[Step Rewards] ", end="")
            if 'step_buyer1_reward' in info:
                print(f"Buyer1: {info['step_buyer1_reward']:.3f}", end="")
            if 'step_buyer2_reward' in info:
                if 'step_buyer1_reward' in info:
                    print(f" | ", end="")
                print(f"Buyer2: {info['step_buyer2_reward']:.3f}", end="")
            if 'step_seller_reward' in info:
                if 'step_buyer1_reward' in info or 'step_buyer2_reward' in info:
                    print(f" | ", end="")
                print(f"Seller: {info['step_seller_reward']:.3f}", end="")
            print()
            
            # Display detailed calculation with weights
            round_cost = -info['round']
            weights = env.reward_weights
            
            # Buyer1 step reward details
            if 'step_buyer1_reward' in info:
                buyer1_price = info.get('buyer1_price')
                if buyer1_price is not None and env.buyer1_max_price is not None:
                    buyer1_savings = env.buyer1_max_price - buyer1_price
                    weighted_savings = buyer1_savings * weights["buyer_savings"]
                    weighted_round_cost = round_cost * weights["time_cost"]
                    print(f"  Buyer1 Step Reward = buyer_savings({buyer1_savings:.2f} * {weights['buyer_savings']:.2f}) + round_cost({round_cost:.2f} * {weights['time_cost']:.2f}) = {info['step_buyer1_reward']:.2f} (buyer1_max={env.buyer1_max_price}, buyer1_price={buyer1_price:.2f}, round={info['round']})")
                else:
                    weighted_round_cost = round_cost * weights["time_cost"]
                    print(f"  Buyer1 Step Reward = round_cost({round_cost:.2f} * {weights['time_cost']:.2f}) = {weighted_round_cost:.2f} (buyer1_price not specified, round={info['round']})")
            
            # Buyer2 step reward details
            if 'step_buyer2_reward' in info:
                buyer2_price = info.get('buyer2_price')
                if buyer2_price is not None and env.buyer2_max_price is not None:
                    buyer2_savings = env.buyer2_max_price - buyer2_price
                    weighted_savings = buyer2_savings * weights["buyer_savings"]
                    weighted_round_cost = round_cost * weights["time_cost"]
                    print(f"  Buyer2 Step Reward = buyer_savings({buyer2_savings:.2f} * {weights['buyer_savings']:.2f}) + round_cost({round_cost:.2f} * {weights['time_cost']:.2f}) = {info['step_buyer2_reward']:.2f} (buyer2_max={env.buyer2_max_price}, buyer2_price={buyer2_price:.2f}, round={info['round']})")
                else:
                    weighted_round_cost = round_cost * weights["time_cost"]
                    print(f"  Buyer2 Step Reward = round_cost({round_cost:.2f} * {weights['time_cost']:.2f}) = {weighted_round_cost:.2f} (buyer2_price not specified, round={info['round']})")
            
            # Seller step reward details
            if 'step_seller_reward' in info:
                seller_price = None
                if info.get('current_selected_buyer') == 1:
                    seller_price = info.get('seller_price_buyer1')
                elif info.get('current_selected_buyer') == 2:
                    seller_price = info.get('seller_price_buyer2')
                
                if seller_price is not None and env.seller_min_price is not None:
                    seller_profit = seller_price - env.seller_min_price
                    weighted_seller_profit = seller_profit * weights["seller_profit"]
                    weighted_round_cost = round_cost * weights["time_cost"]
                    print(f"  Seller Step Reward = seller_profit({seller_profit:.2f} * {weights['seller_profit']:.2f}) + round_cost({round_cost:.2f} * {weights['time_cost']:.2f}) = {info['step_seller_reward']:.2f} (seller_price={seller_price:.2f}, seller_min={env.seller_min_price}, round={info['round']})")
                else:
                    weighted_round_cost = round_cost * weights["time_cost"]
                    print(f"  Seller Step Reward = round_cost({round_cost:.2f} * {weights['time_cost']:.2f}) = {weighted_round_cost:.2f} (seller_price not specified, round={info['round']})")
        
        # If this is the final round (agreed or timeout), display score calculations after Step Rewards
        if done:
            # Print score calculations after Step Rewards
            env._print_global_score_details()
            env._print_buyer_score_details()
            env._print_seller_score_details()
            
            print("\n" + "="*60)
            print("Negotiation Ended")
            print("="*60)
            print(f"Status: {info['status']}")
            if info.get('selected_buyer'):
                print(f"Final Selected Buyer: Buyer {info['selected_buyer']}")
                print(f"Final Deal Price: ${info.get('final_deal_price', 0):.2f}")
            if info.get('agreed_contract') is not None:
                print(f"Final Contract: {info['agreed_contract']}")
            buyer1_price = info.get('buyer1_price', 0) or 0
            seller_price_buyer1 = info.get('seller_price_buyer1', 0) or 0
            buyer2_price = info.get('buyer2_price', 0) or 0
            seller_price_buyer2 = info.get('seller_price_buyer2', 0) or 0
            print(f"Buyer1 Prices: Buyer=${buyer1_price:.2f} | Seller=${seller_price_buyer1:.2f}")
            print(f"Buyer2 Prices: Buyer=${buyer2_price:.2f} | Seller=${seller_price_buyer2:.2f}")
            # current_round has been incremented to reflect the completed round
            actual_rounds = info['round']
            print(f"Total Rounds: {actual_rounds}")
            print(f"Global Reward: {reward:.3f}")
            if 'buyer1_reward' in info:
                print(f"Buyer1 Reward: {info['buyer1_reward']:.3f}")
            if 'buyer2_reward' in info:
                print(f"Buyer2 Reward: {info['buyer2_reward']:.3f}")
            if 'seller_reward' in info:
                print(f"Seller Reward: {info['seller_reward']:.3f}")
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
            results.update({
                "status": info.get('status', 'unknown'),
                "success": terminated,
                "selected_buyer": info.get('selected_buyer'),
                "final_deal_price": info.get('final_deal_price'),
                "buyer1_price": info.get('buyer1_price'),
                "buyer2_price": info.get('buyer2_price'),
                "seller_price_buyer1": info.get('seller_price_buyer1'),
                "seller_price_buyer2": info.get('seller_price_buyer2'),
                "agreed_contract": info.get('agreed_contract'),
                "total_rounds": info.get('round', 0),
                "total_reward": float(reward) if reward is not None else None,
                "buyer1_reward": info.get('buyer1_reward'),
                "buyer2_reward": info.get('buyer2_reward'),
                "seller_reward": info.get('seller_reward'),
                "global_score": info.get('global_score'),
                "buyer_score": info.get('buyer_score'),
                "seller_score": info.get('seller_score'),
                "termination_reason": info.get('termination_reason'),
                "elapsed_time": elapsed_time,
                "buyer1_max_price": buyer1_max_price,
                "buyer2_max_price": buyer2_max_price,
                "seller_min_price": seller_min_price,
                "buyer_contract_configs": buyer_contract_configs,
                "product_info": {
                    "name": "The Smokehouse Treat by Burgers' Smokehouse",
                    "brand": "Visit the Burgers' Smokehouse Store",
                    "original_price": 62.00,
                    "product_category": "Grocery & Gourmet Food › Food & Beverage Gifts › Meat & Seafood Gifts",
                    "average_rating": 5.0,
                    "total_reviews": 1,
                    "asin": "B01LA37T1S",
                },
                "model": get_model_name(model),
            })
            break
    
    # Close environment
    env.close()
    print("\nNegotiation completed!")
    
    # Ensure elapsed_time is set even if negotiation didn't complete normally
    if "elapsed_time" not in results:
        results["elapsed_time"] = time.time() - start_time
    
    # Save results to file
    try:
        # Create results directory structure
        results_dir = Path(project_root) / "agenticpay" / "results" / "only_multi_buyer"
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
            f.write("Task14 Scenario 10: The Smokehouse Treat - Sequential Two-Buyer Negotiation Results\n")
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
            if results.get('selected_buyer'):
                f.write(f"Final Selected Buyer: Buyer {results['selected_buyer']}\n")
                f.write(f"Final Deal Price: ${results.get('final_deal_price', 0):.2f}\n\n")
            if results.get('agreed_contract') is not None:
                f.write(f"Final Contract: {results['agreed_contract']}\n\n")
            f.write("Final Prices:\n")
            f.write(f"  Buyer1 - Buyer Price: ${results['buyer1_price']:.2f}" if results.get('buyer1_price') is not None else "  Buyer1 - Buyer Price: Not specified")
            f.write("\n")
            f.write(f"  Buyer1 - Seller Price: ${results['seller_price_buyer1']:.2f}" if results.get('seller_price_buyer1') is not None else "  Buyer1 - Seller Price: Not specified")
            f.write("\n")
            f.write(f"  Buyer2 - Buyer Price: ${results['buyer2_price']:.2f}" if results.get('buyer2_price') is not None else "  Buyer2 - Buyer Price: Not specified")
            f.write("\n")
            f.write(f"  Buyer2 - Seller Price: ${results['seller_price_buyer2']:.2f}" if results.get('seller_price_buyer2') is not None else "  Buyer2 - Seller Price: Not specified")
            f.write("\n\n")
            f.write("Rewards:\n")
            if results.get('total_reward') is not None:
                f.write(f"  Total Reward: {results['total_reward']:.3f}\n")
            if results.get('buyer1_reward') is not None:
                f.write(f"  Buyer1 Reward: {results['buyer1_reward']:.3f}\n")
            if results.get('buyer2_reward') is not None:
                f.write(f"  Buyer2 Reward: {results['buyer2_reward']:.3f}\n")
            if results.get('seller_reward') is not None:
                f.write(f"  Seller Reward: {results['seller_reward']:.3f}\n")
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
    parser = argparse.ArgumentParser(description="Task14 Scenario 10: The Smokehouse Treat - Sequential Two-Buyer Negotiation")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="OpenVLM model name. Set OPENAI_URL/OPENVLM_BASE_URL for API endpoint, OPENVLM_MODEL for default model name."
    )
    args = parser.parse_args()
    main(model_name=args.model)

