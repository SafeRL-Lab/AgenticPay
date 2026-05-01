"""Task15 Scenario 11: NYC Taxi Ride - Sequential Two-Seller Negotiation (Gramercy to Murray Hill)

The buyer asks to purchase one product (same route / offering) from the marketplace. Product info is a single
listing without per-seller details; two independent sellers each negotiate that same ride with different
confidential floor (minimum) prices.
Category: Daily Life Consumption
Tests agent multi-seller negotiation with route image (image + text).
"""

import os
import sys
import json
import time
import argparse
import random
from pathlib import Path
from datetime import datetime

# Add project path (4 levels up from script to reach repo root AgenticPayGym)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, project_root)

from agenticpay.envs.only_multi_seller.Task3_sequential_two_seller_negotiation import Task3SequentialTwoSellerNegotiation
from agenticpay.agents.buyer_agent import BuyerAgent
from agenticpay.agents.seller_agent import SellerAgent
from agenticpay.models.custom_llm import CustomLLM
from agenticpay.models.openai_vlm import OpenAIVLM
from agenticpay.models.qwen3_vl import Qwen3VL
from agenticpay.models.vllm_lm import VLLMLLM
from agenticpay.models.sglang_vlm import SGLangVLM
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



def main(model_name=None):
    """Main function: Demonstrates sequential multi-seller negotiation flow
    
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

    # Use OpenAIVLM (Vision Language Model) for route map + text (image + text)
    model_name = model_name or "gpt-5.4"  # gpt-4o, gpt-4o-mini, gpt-4-vision-preview, etc.
    model = OpenAIVLM(model=model_name, api_key=api_key)

    # Build absolute path to model directory
    # model_path = os.path.join(project_root, "models", "download_models", "Qwen3-8B-Instruct")
    # model_path = os.path.abspath(model_path)

    # vLLM LLM Model
    # model = VLLMLLM(
    #     model_path=model_path,
    #     trust_remote_code=True,
    #     gpu_memory_utilization=0.9,
    #     tensor_parallel_size=4, # 4 GPUs
    # )

    # SGLang VLM Model
    # model = SGLangVLM(
    #     model_path=model_path,
    # )
    
    print(f"✓ Successfully initialized: {model}")
    
    print("Creating agents...")
    # Same ride (route): v_base / c_base are confidential reservation prices; product_info reference_total_amount is a higher public anchor.
    user_requirement = (
        "I want a yellow cab from Gramercy to Murray Hill, all-in fare. "
        "I also prefer the plotted route trace to jog through at least one clean right-angle corner rather than "
        "reading as one perfectly straight ruler line across every intermediate block shown."
    )
    product_request = user_requirement
    shared_contract_fields = {
        "contrainfo": {
            "product_request": product_request,
            "initial_contract_status": (
                "No all-in fare price, passenger wait time before pickup, route preference (tunnel vs local streets), "
                "or user product preference match has been selected or agreed before negotiation starts."
            ),
            "contract_completion_requirement": (
                "A valid offer must explicitly fill price, continuous_terms.wait_time_mins, "
                "discrete_terms.route_preference, and discrete_terms.user_product_preference."
            ),
        },
        "field_descriptions": {
            "price": (
                "The total all-in amount in US dollars the passenger pays for the ride, including base fare "
                "and mandatory surcharges/taxes in the listing; no extra fees after the deal."
            ),
            "continuous_terms.wait_time_mins": (
                "Minutes after the deal before the passenger meets the driver at pickup; higher means more curb wait for the driver."
            ),
            "discrete_terms.route_preference": (
                "`tunnel` uses a tolled crossing when it saves time; `local_streets` stays on congested surface streets without added toll."
            ),
            "discrete_terms.user_product_preference": (
                "How well the routed path matches the buyer's stated preference for at least one clear right-angle corner "
                "versus a ruler-straight corridor between endpoints. Use `strong_match` when the preference is clearly "
                "satisfied, `partial_match` when it is only partly satisfied, and `mismatch_or_uncertain` when it is "
                "not satisfied or cannot be confirmed."
            ),
        },
        "continuous_bounds": {"wait_time_mins": {"min": 0, "max": 30}},
        "discrete_options": {
            "route_preference": ["tunnel", "local_streets"],
            "user_product_preference": ["strong_match", "partial_match", "mismatch_or_uncertain"],
        },
        "buyer_preferences": {
            "v_base": 10.42,
            "weight_descriptions": {
                "v_base": (
                    "Private max willingness to pay for this ride before wait/route terms (USD); each $1 fare lowers utility by $1."
                ),
                "continuous_weights.wait_time_mins": (
                    "USD per extra minute you need before curbside pickup; positive if delay before you appear helps you."
                ),
                "discrete_weights.route_preference": (
                    "USD utility per route; tunnel is better when you are time-constrained."
                ),
                "discrete_weights.user_product_preference": (
                    "How much each level of match to your stated route-shape preference changes your utility, measured in dollars."
                ),
            },
            "continuous_weights": {"wait_time_mins": 1.0},
            "discrete_weights": {
                "route_preference": {"tunnel": 4.0, "local_streets": -2.0},
                "user_product_preference": {
                    "strong_match": 0.30,
                    "partial_match": 0.12,
                    "mismatch_or_uncertain": -0.25,
                },
            },
        },
    }
    seller1_contract_config = {
        **shared_contract_fields,
        "seller_preferences": {
            "c_base": 9.06,
            "weight_descriptions": {
                "c_base": (
                    "Private minimum acceptable all-in payout before wait/route terms (USD); each $1 fare increases utility by $1."
                ),
                "continuous_weights.wait_time_mins": (
                    "USD per minute of passenger delay at pickup; negative when idling costs you."
                ),
                "discrete_weights.route_preference": (
                    "USD utility per route; tunnel often implies tolls you bear."
                ),
                "discrete_weights.user_product_preference": (
                    "How much each level of commitment to the buyer's stated route-shape preference changes your "
                    "utility (USD). Stronger commitments imply a small nonzero representation risk."
                ),
            },
            "continuous_weights": {"wait_time_mins": -1.65},
            "discrete_weights": {
                "route_preference": {"tunnel": -3.2, "local_streets": 0.0},
                "user_product_preference": {
                    "strong_match": -0.08,
                    "partial_match": -0.04,
                    "mismatch_or_uncertain": 0.01,
                },
            },
        },
    }
    seller2_contract_config = {
        **shared_contract_fields,
        "seller_preferences": {
            "c_base": 8.03,
            "weight_descriptions": seller1_contract_config["seller_preferences"]["weight_descriptions"],
            "continuous_weights": {"wait_time_mins": -1.35},
            "discrete_weights": {
                "route_preference": {"tunnel": -2.85, "local_streets": 0.0},
                "user_product_preference": {
                    "strong_match": -0.08,
                    "partial_match": -0.04,
                    "mismatch_or_uncertain": 0.01,
                },
            },
        },
    }
    seller_contract_configs = {1: seller1_contract_config, 2: seller2_contract_config}
    buyer_max_price = shared_contract_fields["buyer_preferences"]["v_base"]
    seller1_min_price = seller1_contract_config["seller_preferences"]["c_base"]
    seller2_min_price = seller2_contract_config["seller_preferences"]["c_base"]

    buyer = BuyerAgent(model=model, name="Buyer1", buyer_max_price=buyer_max_price)
    seller1 = SellerAgent(model=model, name="Seller1", seller_min_price=seller1_min_price)
    seller2 = SellerAgent(model=model, name="Seller2", seller_min_price=seller2_min_price)
    
    # Create environment
    print("Creating sequential multi-seller negotiation environment...")
    env = Task3SequentialTwoSellerNegotiation(
        buyer_agent=buyer,
        seller1_agent=seller1,
        seller2_agent=seller2,
        max_rounds=max_rounds,
        initial_seller1_price=18.00,  # Opening ask — same ride, different offer
        initial_seller2_price=20.00,  # Opening ask — same ride, different offer
        buyer_max_price=buyer_max_price,    # Buyer max willing to pay (confidential)
        seller1_min_price=seller1_min_price,  # Seller1 minimum acceptable price (confidential)
        seller2_min_price=seller2_min_price,  # Seller2 minimum acceptable price (confidential)
        environment_info={
            "platform": "NYC Street Hail / Ride Apps",
            "market_type": "B2C",
            "note": "Multiple third-party offers exist for the same route and service.",
            "seller_contract_configs": seller_contract_configs,
        },
        price_tolerance=price_tolerance,
        reward_weights=reward_weights,  # Reward weights configuration
    )
    
    # User profile (preferences only; no seller identity — sellers differ only in negotiation/pricing)
    user_profile = None
    print(f"User Profile: {user_profile}")
    
    # Get user requirement
    # print("\n" + "="*60)
    # print("Please enter the product requirement you want to purchase:")
    # user_requirement = input("> ").strip()
    # if not user_requirement:
    #     print("No requirement entered, using default requirement...")
    #     user_requirement = "I want one direct taxi ride from Gramercy to Murray Hill at an all-in flat fare."
    #     print(f"Using default requirement: {user_requirement}")
    print(f"Using default requirement: {user_requirement}")
    
    # Reset environment
    print("\n" + "="*60)
    print("Starting new sequential negotiation with two sellers...")
    print("="*60)
    
    # Route image for VLM: local screenshot path
    product_image_url = os.path.join(
        project_root,
        "agenticpay",
        "data",
        "NYC_taxi_data",
        "img",
        "yellow_tripdata_2026-02_sample_10",
        "image_0.png",
    )
    
    observation, info = env.reset(
        user_requirement=user_requirement,
        product_info={
            "name": "NYC yellow taxi: Gramercy → Murray Hill (all-in flat fare)",
            "product_category": "Transportation & Mobility › Taxi › Manhattan",
            "pickup_location": "Gramercy, Manhattan, New York, NY",
            "dropoff_location": "Murray Hill, Manhattan, New York, NY",
            "trip_distance": "0.94 miles",
            "trip_time_estimate": "Less than 5 minutes",
            "passenger_count": 1,
            "historical_fare_amount": 7.2,
            "reference_total_amount": 12.95,
            "mandatory_surcharges": [
                "$2.50 (Congestion Surcharge for driving below 96th St in Manhattan)",
                "$0.75 (CBD Congestion Fee)",
                "$1.00 (Improvement Surcharge)",
                "$0.50 (MTA State Tax)",
            ],
            "tolls": "$0.00",
            "pricing_rules": "The negotiated price (### BUYER_PRICE($X) ### or ### SELLER_PRICE($Y) ###) MUST be the TOTAL final amount the passenger pays. It MUST include the driver's base fare PLUS all mandatory surcharges and taxes listed above. No fees can be added later.",
            "route_note": "See the attached route image for distance and Manhattan traffic context.",
            "image_url": product_image_url,
        },
        user_profile=user_profile,  # Pass user profile
    )
    
    # Start negotiation loop
    done = False
    start_time = time.time()
    
    # Initialize results dictionary
    results = {
        "task": "Task15_s11_taxi_1",
        "timestamp": datetime.now().isoformat(),
        "user_requirement": user_requirement,
        "user_profile": user_profile,
        "status": "unknown",
        "success": False,
        "error": None,
    }
    
    while not done:
        # Each round, buyer chooses one seller to negotiate with
        # Buyer can see both sellers' information in the observation
        # Let buyer decide which seller to negotiate with and provide negotiation message
        # We'll use a combined conversation history that includes both sellers' conversations
        combined_history = []
        # Add seller1 messages with prefix
        for msg in observation.get("conversation_history_seller1", []):
            combined_history.append({
                **msg,
                "thread_label": "Talk with Seller 1",
            })
        # Add seller2 messages with prefix
        for msg in observation.get("conversation_history_seller2", []):
            combined_history.append({
                **msg,
                "thread_label": "Talk with Seller 2",
            })
        # Get buyer's response - buyer should choose a seller via a structured <selected_seller> block
        routing_instruction = (
            "You are negotiating with two sellers. Each round, choose exactly ONE seller "
            "and output that choice in a dedicated <selected_seller> block containing only "
            "the digit 1 or 2. Then put only your negotiation text in <message>."
        )
        buyer_response = buyer.respond(
            conversation_history=combined_history,
            current_state={
                **observation,
                "instruction": routing_instruction
            }
        )

        # Routing relies on the structured <selected_seller> block.
        # If parsing fails, retry a few times; if still missing, fallback to random seller.
        selected_seller = buyer.last_selected_seller
        max_selection_retries = 2
        retry_count = 0
        while selected_seller is None and retry_count < max_selection_retries:
            retry_count += 1
            print(f"\n[Warning] Missing <selected_seller>; retrying buyer response ({retry_count}/{max_selection_retries})...")
            buyer_response = buyer.respond(
                conversation_history=combined_history,
                current_state={
                    **observation,
                    "instruction": (
                        routing_instruction
                        + " IMPORTANT: You MUST include a valid <selected_seller> block with only 1 or 2."
                    )
                }
            )
            selected_seller = buyer.last_selected_seller
        if selected_seller is None:
            selected_seller = random.choice([1, 2])
            print(f"\n[Warning] Failed to parse <selected_seller> after retries; randomly selecting Seller {selected_seller}.")
        print(f"\n[Buyer chooses to negotiate with Seller {selected_seller} this round]")
        
        # BuyerAgent returns only the <message> block as the negotiation message.
        buyer_action = buyer_response
        
        # Get the conversation history for the selected seller
        # Create updated conversation history that includes buyer's response
        # So seller can see buyer's message before responding
        if selected_seller == 1:
            conversation_history = observation["conversation_history_seller1"].copy()
        else:
            conversation_history = observation["conversation_history_seller2"].copy()
        
        # Add buyer's message to the conversation history
        if buyer_action:
            current_round = observation.get("current_round", 0)
            conversation_history.append({
                "role": "buyer",
                "content": buyer_action,
                "round": current_round
            })
        
        # Get the selected seller's response (seller can now see buyer's message)
        if selected_seller == 1:
            seller_action = seller1.respond(
                conversation_history=conversation_history,
                current_state=observation
            )
        else:
            seller_action = seller2.respond(
                conversation_history=conversation_history,
                current_state=observation
            )
        
        # Execute step with selected seller and actions
        observation, reward, terminated, truncated, info = env.step(
            selected_seller=selected_seller,
            buyer_action=buyer_action,
            seller_action=seller_action
        )
        done = terminated or truncated
        
        # Render current state (includes all print information)
        env.render()
        
        # Flush output to ensure complete display
        sys.stdout.flush()
        
        # Display step rewards for each round with detailed calculation
        if 'step_buyer_reward' in info or 'step_seller1_reward' in info or 'step_seller2_reward' in info:
            print(f"\n[Step Rewards] ", end="")
            if 'step_buyer_reward' in info:
                print(f"Buyer: {info['step_buyer_reward']:.3f}", end="")
            if 'step_seller1_reward' in info:
                if 'step_buyer_reward' in info:
                    print(f" | ", end="")
                print(f"Seller1: {info['step_seller1_reward']:.3f}", end="")
            if 'step_seller2_reward' in info:
                if 'step_buyer_reward' in info or 'step_seller1_reward' in info:
                    print(f" | ", end="")
                print(f"Seller2: {info['step_seller2_reward']:.3f}", end="")
            print()
            
            # Display detailed calculation with weights
            round_cost = -info['round']
            weights = env.reward_weights
            
            # Buyer step reward details
            if 'step_buyer_reward' in info:
                buyer_price = None
                if info.get('current_selected_seller') == 1:
                    buyer_price = info.get('buyer_price_seller1')
                elif info.get('current_selected_seller') == 2:
                    buyer_price = info.get('buyer_price_seller2')
                
                if buyer_price is not None and env.buyer_max_price is not None:
                    buyer_savings = env.buyer_max_price - buyer_price
                    weighted_savings = buyer_savings * weights["buyer_savings"]
                    weighted_round_cost = round_cost * weights["time_cost"]
                    print(f"  Buyer Step Reward = buyer_savings({buyer_savings:.2f} * {weights['buyer_savings']:.2f}) + round_cost({round_cost:.2f} * {weights['time_cost']:.2f}) = {info['step_buyer_reward']:.2f} (buyer_max={env.buyer_max_price}, buyer_price={buyer_price:.2f}, round={info['round']})")
                else:
                    weighted_round_cost = round_cost * weights["time_cost"]
                    print(f"  Buyer Step Reward = round_cost({round_cost:.2f} * {weights['time_cost']:.2f}) = {weighted_round_cost:.2f} (buyer_price not specified, round={info['round']})")
            
            # Seller1 step reward details
            if 'step_seller1_reward' in info and info.get('seller1_price') is not None:
                seller1_price = info.get('seller1_price', 0)
                seller1_min = env.seller1_min_price
                if seller1_min is not None:
                    seller1_profit = seller1_price - seller1_min
                    weighted_seller1_profit = seller1_profit * weights["seller_profit"]
                    weighted_round_cost = round_cost * weights["time_cost"]
                    print(f"  Seller1 Step Reward = seller_profit({seller1_profit:.2f} * {weights['seller_profit']:.2f}) + round_cost({round_cost:.2f} * {weights['time_cost']:.2f}) = {info['step_seller1_reward']:.2f} (seller1_price={seller1_price:.2f}, seller1_min={seller1_min}, round={info['round']})")
                else:
                    weighted_round_cost = round_cost * weights["time_cost"]
                    print(f"  Seller1 Step Reward = round_cost({round_cost:.2f} * {weights['time_cost']:.2f}) = {weighted_round_cost:.2f} (seller1_price={seller1_price:.2f}, seller1_min not specified, round={info['round']})")
            elif 'step_seller1_reward' in info:
                weighted_round_cost = round_cost * weights["time_cost"]
                print(f"  Seller1 Step Reward = round_cost({round_cost:.2f} * {weights['time_cost']:.2f}) = {weighted_round_cost:.2f} (seller1_price not specified, round={info['round']})")
            
            # Seller2 step reward details
            if 'step_seller2_reward' in info and info.get('seller2_price') is not None:
                seller2_price = info.get('seller2_price', 0)
                seller2_min = env.seller2_min_price
                if seller2_min is not None:
                    seller2_profit = seller2_price - seller2_min
                    weighted_seller2_profit = seller2_profit * weights["seller_profit"]
                    weighted_round_cost = round_cost * weights["time_cost"]
                    print(f"  Seller2 Step Reward = seller_profit({seller2_profit:.2f} * {weights['seller_profit']:.2f}) + round_cost({round_cost:.2f} * {weights['time_cost']:.2f}) = {info['step_seller2_reward']:.2f} (seller2_price={seller2_price:.2f}, seller2_min={seller2_min}, round={info['round']})")
                else:
                    weighted_round_cost = round_cost * weights["time_cost"]
                    print(f"  Seller2 Step Reward = round_cost({round_cost:.2f} * {weights['time_cost']:.2f}) = {weighted_round_cost:.2f} (seller2_price={seller2_price:.2f}, seller2_min not specified, round={info['round']})")
            elif 'step_seller2_reward' in info:
                weighted_round_cost = round_cost * weights["time_cost"]
                print(f"  Seller2 Step Reward = round_cost({round_cost:.2f} * {weights['time_cost']:.2f}) = {weighted_round_cost:.2f} (seller2_price not specified, round={info['round']})")
        
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
            if info.get('selected_seller'):
                print(f"Final Selected Seller: Seller {info['selected_seller']}")
                print(f"Final Deal Price: ${info.get('final_deal_price', 0):.2f}")
                agreed_contract = info.get(f"agreed_contract_seller{info['selected_seller']}")
                if agreed_contract is not None:
                    print(f"Final Contract: {agreed_contract}")
            seller1_price = info.get('seller1_price', 0) or 0
            buyer_price_seller1 = info.get('buyer_price_seller1', 0) or 0
            seller2_price = info.get('seller2_price', 0) or 0
            buyer_price_seller2 = info.get('buyer_price_seller2', 0) or 0
            print(f"Seller1 Prices: Seller=${seller1_price:.2f} | Buyer=${buyer_price_seller1:.2f}")
            print(f"Seller2 Prices: Seller=${seller2_price:.2f} | Buyer=${buyer_price_seller2:.2f}")
            # current_round has been incremented to reflect the completed round
            actual_rounds = info['round']
            print(f"Total Rounds: {actual_rounds}")
            print(f"Global Reward: {reward:.3f}")
            if 'buyer_reward' in info:
                print(f"Buyer Reward: {info['buyer_reward']:.3f}")
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
            results.update({
                "status": info.get('status', 'unknown'),
                "success": terminated,
                "selected_seller": info.get('selected_seller'),
                "final_deal_price": info.get('final_deal_price'),
                "seller1_price": info.get('seller1_price'),
                "seller2_price": info.get('seller2_price'),
                "buyer_price_seller1": info.get('buyer_price_seller1'),
                "buyer_price_seller2": info.get('buyer_price_seller2'),
                "total_rounds": info.get('round', 0),
                "total_reward": float(reward) if reward is not None else None,
                "buyer_reward": info.get('buyer_reward'),
                "seller1_reward": info.get('seller1_reward'),
                "seller2_reward": info.get('seller2_reward'),
                "global_score": info.get('global_score'),
                "buyer_score": info.get('buyer_score'),
                "seller_score": info.get('seller_score'),
                "termination_reason": info.get('termination_reason'),
                "elapsed_time": elapsed_time,
                "buyer_max_price": buyer_max_price,
                "seller1_min_price": seller1_min_price,
                "seller2_min_price": seller2_min_price,
                "seller_contract_configs": seller_contract_configs,
                "agreed_contract_seller1": info.get("agreed_contract_seller1"),
                "agreed_contract_seller2": info.get("agreed_contract_seller2"),
                "buyer_utility_seller1": info.get("buyer_utility_seller1"),
                "seller_utility_seller1": info.get("seller_utility_seller1"),
                "z_max_seller1": info.get("z_max_seller1"),
                "buyer_utility_seller2": info.get("buyer_utility_seller2"),
                "seller_utility_seller2": info.get("seller_utility_seller2"),
                "z_max_seller2": info.get("z_max_seller2"),
                "product_info": {
                    "name": "NYC yellow taxi: Gramercy → Murray Hill (all-in flat fare)",
                    "product_category": "Transportation & Mobility › Taxi › Manhattan",
                    "pickup_location": "Gramercy, Manhattan, New York, NY",
                    "dropoff_location": "Murray Hill, Manhattan, New York, NY",
                    "trip_distance": "0.94 miles",
                    "trip_time_estimate": "Less than 5 minutes",
                    "passenger_count": 1,
                    "historical_fare_amount": 7.2,
                    "reference_total_amount": 12.95,
                    "image_url": product_image_url,
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
        results_dir = Path(project_root) / "agenticpay" / "results" / "only_multi_seller"
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
        
        # Save output text (we'll create a simple output file with key information)
        output_file = run_dir / "Task15_s11_taxi_1_output.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("Task15 Scenario 11: NYC Taxi Ride - Sequential Two-Seller Negotiation Results\n")
            f.write("Category: Daily Life Consumption\n")
            f.write("Scenario: Gramercy to Murray Hill short-distance taxi ride fare negotiation\n")
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
            if results.get('selected_seller'):
                f.write(f"Final Selected Seller: Seller {results['selected_seller']}\n")
                f.write(f"Final Deal Price: ${results.get('final_deal_price', 0):.2f}\n\n")
                agreed_contract = results.get(f"agreed_contract_seller{results['selected_seller']}")
                if agreed_contract is not None:
                    f.write(f"Final Contract: {agreed_contract}\n\n")
            f.write("Contract Utilities:\n")
            if results.get('z_max_seller1') is not None:
                f.write(f"  Seller1 Z_max: {results['z_max_seller1']:.3f}\n")
            if results.get('buyer_utility_seller1') is not None:
                f.write(f"  Seller1 Buyer Utility: {results['buyer_utility_seller1']:.3f}\n")
            if results.get('seller_utility_seller1') is not None:
                f.write(f"  Seller1 Seller Utility: {results['seller_utility_seller1']:.3f}\n")
            if results.get('z_max_seller2') is not None:
                f.write(f"  Seller2 Z_max: {results['z_max_seller2']:.3f}\n")
            if results.get('buyer_utility_seller2') is not None:
                f.write(f"  Seller2 Buyer Utility: {results['buyer_utility_seller2']:.3f}\n")
            if results.get('seller_utility_seller2') is not None:
                f.write(f"  Seller2 Seller Utility: {results['seller_utility_seller2']:.3f}\n")
            f.write("\n")
            f.write("Final Prices:\n")
            f.write(f"  Seller1 - Seller Price: ${results['seller1_price']:.2f}" if results.get('seller1_price') is not None else "  Seller1 - Seller Price: Not specified")
            f.write("\n")
            f.write(f"  Seller1 - Buyer Price: ${results['buyer_price_seller1']:.2f}" if results.get('buyer_price_seller1') is not None else "  Seller1 - Buyer Price: Not specified")
            f.write("\n")
            f.write(f"  Seller2 - Seller Price: ${results['seller2_price']:.2f}" if results.get('seller2_price') is not None else "  Seller2 - Seller Price: Not specified")
            f.write("\n")
            f.write(f"  Seller2 - Buyer Price: ${results['buyer_price_seller2']:.2f}" if results.get('buyer_price_seller2') is not None else "  Seller2 - Buyer Price: Not specified")
            f.write("\n\n")
            f.write("Rewards:\n")
            if results.get('total_reward') is not None:
                f.write(f"  Total Reward: {results['total_reward']:.3f}\n")
            if results.get('buyer_reward') is not None:
                f.write(f"  Buyer Reward: {results['buyer_reward']:.3f}\n")
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
    parser = argparse.ArgumentParser(description="Task15 Scenario 11: NYC Taxi Ride - Sequential Two-Seller Negotiation (Gramercy to Murray Hill)")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name to use (e.g., 'gemini-3-pro-all', 'gpt-5.2', 'claude-sonnet-4-5-20250929'). If not provided, uses default model."
    )
    args = parser.parse_args()
    main(model_name=args.model)
