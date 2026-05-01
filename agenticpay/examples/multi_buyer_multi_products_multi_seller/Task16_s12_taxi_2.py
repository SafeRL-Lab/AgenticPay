"""Task16 Scenario 12: NYC Taxi Ride - Sequential Two-Buyer Two-Seller Two-Product Negotiation

Same two fare components from two quotes: listing has no per-operator identity; two sellers each have a
different confidential floor for the **total** all-in price. Structured `<selected_seller>` routing per round
(aligned with Task5_s1). Category: Daily Life Consumption
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
            except Exception:
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
    buyer1_max_price = 16.02  # Maximum acceptable all-in bundle fare for buyer1 (confidential; below summed quote components)
    buyer2_max_price = 16.83  # Maximum acceptable all-in bundle fare for buyer2 (confidential; below summed quote components)
    seller1_min_price = 14.41  # Minimum acceptable all-in bundle fare for seller1 (confidential; higher floor than Seller 2)
    seller2_min_price = 13.27  # Minimum acceptable all-in bundle fare for seller2 (confidential)

    product_request = (
        "I want Union Sq to Lenox Hill West—base fare plus surcharges, all-in. "
        "I also prefer the pictured cab's side to read as one even yellow finish without obvious blotchy respray patches."
    )
    _m = 0.90  # MAUT weight scale vs paper baseline (~$18 v_base)

    shared_contract_fields = {
        "contrainfo": {
            "product_request": product_request,
            "initial_contract_status": (
                "No total all-in fare, curb wait time, route preference, or user product preference match "
                "has been selected or agreed before negotiation starts."
            ),
            "contract_completion_requirement": (
                "A valid offer must explicitly fill price, continuous_terms.wait_time_mins, "
                "discrete_terms.route_preference, and discrete_terms.user_product_preference. The price is the passenger-paid "
                "total for both fare components (metered base + mandatory fees)."
            ),
        },
        "field_descriptions": {
            "price": (
                "The total amount the passenger pays for the ride bundle, all-inclusive, in US dollars."
            ),
            "continuous_terms.wait_time_mins": (
                "Pickup curb delay in minutes after the driver arrives before starting the priced portion."
            ),
            "discrete_terms.route_preference": (
                "`tunnel` uses tolled crossings when faster; `local_streets` avoids tolls on congested surface roads."
            ),
            "discrete_terms.user_product_preference": (
                "How well the fare-quote imagery matches the buyer's stated preference that the cab's side read as one "
                "even yellow finish without obvious blotchy respray patches. `strong_match` when clearly satisfied; "
                "`partial_match` when ambiguous or partly satisfied; `mismatch_or_uncertain` when not satisfied or unconfirmable."
            ),
        },
        "continuous_bounds": {"wait_time_mins": {"min": 0, "max": 30}},
        "discrete_options": {
            "route_preference": ["tunnel", "local_streets"],
            "user_product_preference": ["strong_match", "partial_match", "mismatch_or_uncertain"],
        },
    }

    buyer1_preferences = {
        "v_base": buyer1_max_price,
        "weight_descriptions": {
            "v_base": (
                "Your private maximum value for the all-in bundle before wait/route terms, in dollars."
            ),
            "continuous_weights.wait_time_mins": (
                "Utility change per extra minute you keep the driver waiting at pickup (dollars per minute)."
            ),
            "discrete_weights.route_preference": (
                "Extra dollar utility for tunnel vs local routing beyond price."
            ),
            "discrete_weights.user_product_preference": (
                "Extra dollar utility for how well imagery matches your stated paint-uniformity preference."
            ),
        },
        "continuous_weights": {"wait_time_mins": 1.0 * _m},
        "discrete_weights": {
            "route_preference": {"tunnel": 4.0 * _m, "local_streets": -2.0 * _m},
            "user_product_preference": {
                "strong_match": 0.45 * _m,
                "partial_match": 0.16 * _m,
                "mismatch_or_uncertain": -0.30 * _m,
            },
        },
    }
    buyer2_preferences = json.loads(json.dumps(buyer1_preferences))
    buyer2_preferences["v_base"] = buyer2_max_price
    buyer2_preferences["continuous_weights"]["wait_time_mins"] = 0.72 * _m
    buyer2_preferences["discrete_weights"]["route_preference"] = {
        "tunnel": 3.35 * _m,
        "local_streets": -1.55 * _m,
    }

    seller1_preferences = {
        "c_base": seller1_min_price,
        "weight_descriptions": {
            "c_base": (
                "Your private minimum acceptable revenue for the bundle before wait/route terms, in dollars."
            ),
            "continuous_weights.wait_time_mins": (
                "Utility change per extra pickup wait minute for you (dollars per minute)."
            ),
            "discrete_weights.route_preference": (
                "Extra dollar utility/cost for tunnel tolls vs surface routing."
            ),
            "discrete_weights.user_product_preference": (
                "Small dollar cost or benefit when affirming alignment with the buyer's stated paint-uniformity preference."
            ),
        },
        "continuous_weights": {"wait_time_mins": -1.5 * _m},
        "discrete_weights": {
            "route_preference": {"tunnel": -3.0 * _m, "local_streets": 0.0},
            "user_product_preference": {
                "strong_match": -0.11 * _m,
                "partial_match": -0.055 * _m,
                "mismatch_or_uncertain": 0.012 * _m,
            },
        },
    }
    seller2_preferences = json.loads(json.dumps(seller1_preferences))
    seller2_preferences["c_base"] = seller2_min_price
    seller2_preferences["continuous_weights"]["wait_time_mins"] = -1.35 * _m
    seller2_preferences["discrete_weights"]["route_preference"] = {
        "tunnel": -2.72 * _m,
        "local_streets": 0.08 * _m,
    }

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

    buyer1 = BuyerAgent(model=model, name="Buyer1", buyer_max_price=buyer1_max_price)
    buyer2 = BuyerAgent(model=model, name="Buyer2", buyer_max_price=buyer2_max_price)
    seller1 = SellerAgent(model=model, name="Seller1", seller_min_price=seller1_min_price)
    seller2 = SellerAgent(model=model, name="Seller2", seller_min_price=seller2_min_price)

    # Create environment
    print("Creating sequential multi-buyer multi-seller multi-product taxi negotiation environment...")
    env = Task3SequentialTwoBuyerTwoSellerTwoProductNegotiation(
        buyer1_agent=buyer1,
        buyer2_agent=buyer2,
        seller1_agent=seller1,
        seller2_agent=seller2,
        max_rounds=max_rounds,
        initial_seller1_price=26.0,  # Initial total fare offered by seller1 for both fare components
        initial_seller2_price=28.0,  # Initial total fare offered by seller2 for both fare components
        buyer1_max_price=buyer1_max_price,  # Buyer1 total max price (confidential)
        buyer2_max_price=buyer2_max_price,  # Buyer2 total max price (confidential)
        seller1_min_price=seller1_min_price,  # Seller1 total min price (confidential)
        seller2_min_price=seller2_min_price,  # Seller2 total min price (confidential)
        environment_info={
            "platform": "NYC Street Hail",
            "market_type": "Service Negotiation (Ride Fare Bundle)",
            "traffic_context": "Cross-neighborhood Manhattan route",
            "category": "Daily Life Consumption",
            "note": "Multiple third-party quotes exist for the same two-service cart; prices are all-in totals.",
            "buyer_seller_contract_configs": buyer_seller_contract_configs,
        },
        price_tolerance=price_tolerance,
        reward_weights=reward_weights,  # Reward weights configuration
    )
    
    # Create user profile (text description of personal preferences)
    user_profile = None
    print(f"User Profile: {user_profile}")
    
    product_info = {
        "products": [
            {
                "name": "Taxi Base Fare Segment: Union Sq -> Lenox Hill West",
                "brand": "NYC Yellow Taxi",
                "price": 11.4,
                "condition": "Service",
                "service_type": "Metered Base Fare",
                "pickup_location": "Union Sq, Manhattan, New York, NY",
                "dropoff_location": "Lenox Hill West, Manhattan, New York, NY",
                "trip_distance_miles": 1.93,
                "historical_trip_time": "Around 10-15 minutes",
                "RatecodeID": 1,
                "Passenger Count": 1,
                "Historical Total Amount": 20.58,
                "full_description": "Base fare component of a Manhattan yellow taxi ride.",
                "image_url": os.path.join(
                    project_root,
                    "agenticpay",
                    "data",
                    "NYC_taxi_data",
                    "img",
                    "yellow_tripdata_2026-02_sample_10",
                    "image_1.png",
                ),
            },
            {
                "name": "Taxi Surcharges and Extras Package (same trip)",
                "brand": "NYC Taxi Fare Rules",
                "price": 9.18,
                "condition": "Service",
                "service_type": "Surcharges and Taxes",
                "components": {
                    "congestion_surcharge": 2.5,
                    "cbd_congestion_fee": 0.75,
                    "improvement_surcharge": 1.0,
                    "mta_tax": 0.5,
                    "extra": 0.0,
                    "tip_amount": 3.43,
                    "tolls_amount": 0.0,
                    "airport_fee": 0.0
                },
                "pricing_rule": "Negotiated total must include base fare and all mandatory fees as final all-in passenger payment.",
                "full_description": "Surcharge/tax/typical extras component for this trip based on the sampled NYC taxi record.",
                "image_url": os.path.join(
                    project_root,
                    "agenticpay",
                    "data",
                    "NYC_taxi_data",
                    "img",
                    "yellow_tripdata_2026-02_sample_10",
                    "image_1.png",
                ),
            },
        ]
    }
    
    # Calculate total product price
    total_product_price = sum(p["price"] for p in product_info["products"])
    print(f"\nProducts (Taxi Fare Components):")
    for i, p in enumerate(product_info["products"], 1):
        print(f"  {i}. {p['name']}: ${p['price']:.2f}")
    print(f"  Total Package Price: ${total_product_price:.2f}")
    
    user_requirement = product_request
    print(f"Using default requirement: {user_requirement}")
    
    # Reset environment
    print("\n" + "="*60)
    print("Starting new sequential negotiation for taxi fare components...")
    print("Two buyers competing with two sellers for Base Fare + Surcharges Package")
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
        "task": "Task16_s12_taxi_2_multi_buyer_multi_products_multi_seller",
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
            "one complete <contract>...</contract> JSON block with price (all-in total), "
            "continuous_terms.wait_time_mins in [0,30], discrete_terms.route_preference "
            "either tunnel or local_streets, and discrete_terms.user_product_preference "
            "one of strong_match, partial_match, mismatch_or_uncertain."
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
                current_state=observation
            )
        elif buyer1_selected_seller == 2:
            seller2_action_buyer1 = seller2.respond(
                conversation_history=conversation_history_b1s2,
                current_state=observation
            )
        
        if buyer2_selected_seller == 1:
            seller1_action_buyer2 = seller1.respond(
                conversation_history=conversation_history_b2s1,
                current_state=observation
            )
        elif buyer2_selected_seller == 2:
            seller2_action_buyer2 = seller2.respond(
                conversation_history=conversation_history_b2s2,
                current_state=observation
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
                    "product_info": product_info_out,
                    "model": get_model_name(model),
                }
            )
            break
    
    # Close environment
    env.close()
    print("\nTaxi fare bundle negotiation completed!")
    
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
        output_file = run_dir / "Task16_s12_taxi_2_output.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("Task16 Scenario 12: NYC Taxi Ride - Sequential Two-Buyer Two-Seller Two-Product Negotiation Results\n")
            f.write("Category: Daily Life Consumption\n")
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
            f.write("Final Prices (All-in Taxi Fare for Both Components):\n")
            f.write(f"  Buyer1-Seller1: Buyer=${results['b1s1_buyer_price']:.2f} | Seller=${results['b1s1_seller_price']:.2f}" if results.get('b1s1_buyer_price') is not None and results.get('b1s1_seller_price') is not None else "  Buyer1-Seller1: Not specified")
            f.write("\n")
            f.write(f"  Buyer1-Seller2: Buyer=${results['b1s2_buyer_price']:.2f} | Seller=${results['b1s2_seller_price']:.2f}" if results.get('b1s2_buyer_price') is not None and results.get('b1s2_seller_price') is not None else "  Buyer1-Seller2: Not specified")
            f.write("\n")
            f.write(f"  Buyer2-Seller1: Buyer=${results['b2s1_buyer_price']:.2f} | Seller=${results['b2s1_seller_price']:.2f}" if results.get('b2s1_buyer_price') is not None and results.get('b2s1_seller_price') is not None else "  Buyer2-Seller1: Not specified")
            f.write("\n")
            f.write(f"  Buyer2-Seller2: Buyer=${results['b2s2_buyer_price']:.2f} | Seller=${results['b2s2_seller_price']:.2f}" if results.get('b2s2_buyer_price') is not None and results.get('b2s2_seller_price') is not None else "  Buyer2-Seller2: Not specified")
            f.write("\n\n")
            product_info = results.get('product_info', {})
            f.write("Fare Components:\n")
            if 'products' in product_info:
                for i, p in enumerate(product_info['products'], 1):
                    f.write(f"  {i}. {p.get('name', 'N/A')} by {p.get('brand', 'N/A')} - ${p.get('price', 0):.2f}\n")
                total_price = sum(p.get('price', 0) for p in product_info.get('products', []))
                f.write(f"  Total Product Price: ${total_price:.2f}\n")
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
    parser = argparse.ArgumentParser(description="Task16 Scenario 12: NYC Taxi Ride - Sequential Two-Buyer Two-Seller Two-Product Negotiation")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name to use (e.g., 'gemini-3-pro-all', 'gpt-5.2', 'claude-sonnet-4-5-20250929'). If not provided, uses default model."
    )
    args = parser.parse_args()
    main(model_name=args.model)
