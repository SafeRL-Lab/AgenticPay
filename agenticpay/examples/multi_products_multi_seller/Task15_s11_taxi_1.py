"""Task15 Scenario 11: NYC taxi — same two line items, sequential two-seller (bundle total)

Buyer wants the same two items (main trip + mandatory fees) and negotiates TOTAL bundle price.
Both sellers list identical line items; each has a different floor price and opening offer.
Category: Daily Life Consumption
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime

# Add project path (4 levels up from script to reach repo root AgenticPayGym)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, project_root)

from agenticpay.envs.multi_products_multi_seller.Task3_sequential_two_seller_per_one_product_negotiation import Task3SequentialTwoSellerPerOneProductNegotiation
from agenticpay.agents.buyer_agent import BuyerAgent
from agenticpay.agents.seller_agent import SellerAgent
from agenticpay.models.openai_vlm import OpenAIVLM

# Import configuration parameters
examples_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, examples_dir)
try:
    from config import reward_weights, max_rounds, price_tolerance, OPENAI_API_KEY
except ImportError:
    # Default values if config not available
    reward_weights = {"buyer_savings": 1.0, "seller_profit": 1.0, "time_cost": 0.1}
    max_rounds = 20
    price_tolerance = 1.0
    OPENAI_API_KEY = None


def get_model_name(model):
    """Extract model name from model object"""
    if hasattr(model, 'model'):
        return model.model
    elif hasattr(model, 'model_id'):
        return model.model_id
    elif hasattr(model, 'model_path'):
        model_path = model.model_path
        return os.path.basename(model_path) if model_path else str(model)
    else:
        model_str = str(model)
        if "model=" in model_str:
            try:
                return model_str.split("model=")[1].split(")")[0]
            except Exception:
                return model_str
        else:
            return model_str


def main(model_name=None):
    """Sequential two-seller negotiation for the same two-line-item bundle (total price)."""

    print("Initializing model...")

    # Check API key
    api_key = os.getenv("OPENAI_API_KEY") or OPENAI_API_KEY
    if not api_key:
        print("Warning: OPENAI_API_KEY not set. Please set it to use OpenAI models.")
        print("You can set it with: export OPENAI_API_KEY='your-key-here'")
        return

    # Use OpenAIVLM (Vision Language Model) for negotiation with product images (image + text)
    model_name = model_name or "gpt-4o-mini"  # gpt-4o, gpt-4o-mini, gpt-4-vision-preview, etc.
    model = OpenAIVLM(model=model_name, api_key=api_key)

    print(f"✓ Successfully initialized: {model}")

    print("Creating agents...")
    buyer_max_price = 10.43
    seller1_min_price = 9.06
    seller2_min_price = 8.36

    product_request = (
        "I want Gramercy → Murray Hill—best all-in for ride plus fees. "
        "I also prefer the vehicle's sidewalls below the doors to look clean without heavy streaked mud buildup."
    )
    user_requirement = product_request

    shared_contract_fields = {
        "contrainfo": {
            "product_request": product_request,
            "initial_contract_status": (
                "No all-in bundle price, pickup wait time, route preference, or user product preference match has been "
                "agreed before negotiation starts."
            ),
            "contract_completion_requirement": (
                "A valid offer must explicitly fill price, continuous_terms.wait_time_mins, discrete_terms.route_preference, "
                "and discrete_terms.user_product_preference. The price is the all-in total for the main trip "
                "plus mandatory fees line items together."
            ),
        },
        "field_descriptions": {
            "price": (
                "The all-in total fare the buyer pays for the bundled taxi trip (metered portion plus "
                "mandatory surcharges/fees/taxes as listed), measured in US dollars."
            ),
            "continuous_terms.wait_time_mins": (
                "How many minutes the driver waits after arrival before pickup "
                "(how long the passenger needs to get curbside)."
            ),
            "discrete_terms.route_preference": (
                "Routing: `tunnel` uses tolled crossings / faster links where applicable; "
                "`local_streets` stays on congested surface streets without tunnel toll to the driver."
            ),
            "discrete_terms.user_product_preference": (
                "How well the assigned taxi matches the buyer's stated preference for lower door-side bodywork below the waist line "
                "that looks largely clean without streaked mud buildup. "
                "Use `strong_match` when clearly satisfied, `partial_match` when only partly satisfied, "
                "and `mismatch_or_uncertain` when not satisfied or cannot be confirmed."
            ),
        },
        "continuous_bounds": {
            "wait_time_mins": {"min": 0, "max": 30},
        },
        "discrete_options": {
            "route_preference": ["tunnel", "local_streets"],
            "user_product_preference": ["strong_match", "partial_match", "mismatch_or_uncertain"],
        },
        "buyer_preferences": {
            "v_base": buyer_max_price,
            "weight_descriptions": {
                "v_base": (
                    "Your private maximum all-in value for this ride bundle before wait and route terms, in dollars. "
                    "Paying `price` reduces utility one-for-one."
                ),
                "continuous_weights.wait_time_mins": (
                    "Dollar change in utility per extra minute of driver wait before pickup; "
                    "positive means you value more time to get downstairs."
                ),
                "discrete_weights.route_preference": (
                    "Dollar utility for each route choice (time vs congestion vs toll exposure)."
                ),
                "discrete_weights.user_product_preference": (
                    "Dollar utility for each match level versus your stated vehicle cleanliness preference; "
                    "positive is better for you."
                ),
            },
            "continuous_weights": {"wait_time_mins": 1.0},
            "discrete_weights": {
                "route_preference": {"tunnel": 4.0, "local_streets": -2.0},
                "user_product_preference": {
                    "strong_match": 0.14,
                    "partial_match": 0.055,
                    "mismatch_or_uncertain": -0.11,
                },
            },
        },
    }
    seller1_contract_config = {
        **shared_contract_fields,
        "seller_preferences": {
            "c_base": seller1_min_price,
            "weight_descriptions": {
                "c_base": (
                    "Your private minimum all-in revenue for this bundle before wait and route terms, in dollars. "
                    "Receiving `price` raises utility one-for-one."
                ),
                "continuous_weights.wait_time_mins": (
                    "Dollar change per minute of idle waiting; negative means waiting is costly."
                ),
                "discrete_weights.route_preference": (
                    "Dollar utility per route; tunnel often implies tolls and different trip cost."
                ),
                "discrete_weights.user_product_preference": (
                    "Dollar utility for each commitment level on the buyer's stated vehicle preference; stronger commitments "
                    "carry a small nonzero risk or sourcing cost."
                ),
            },
            "continuous_weights": {"wait_time_mins": -1.5},
            "discrete_weights": {
                "route_preference": {"tunnel": -3.0, "local_streets": 0.0},
                "user_product_preference": {
                    "strong_match": -0.045,
                    "partial_match": -0.022,
                    "mismatch_or_uncertain": 0.006,
                },
            },
        },
    }
    seller2_contract_config = {
        **shared_contract_fields,
        "seller_preferences": {
            "c_base": seller2_min_price,
            "weight_descriptions": seller1_contract_config["seller_preferences"]["weight_descriptions"],
            "continuous_weights": {"wait_time_mins": -1.35},
            "discrete_weights": {
                "route_preference": {"tunnel": -2.55, "local_streets": -0.1},
                "user_product_preference": {
                    "strong_match": -0.045,
                    "partial_match": -0.022,
                    "mismatch_or_uncertain": 0.006,
                },
            },
        },
    }
    seller_contract_configs = {
        1: seller1_contract_config,
        2: seller2_contract_config,
    }

    buyer = BuyerAgent(model=model, name="Buyer1", buyer_max_price=buyer_max_price)
    seller1 = SellerAgent(model=model, name="Seller1", seller_min_price=seller1_min_price)
    seller2 = SellerAgent(model=model, name="Seller2", seller_min_price=seller2_min_price)

    print("Creating sequential multi-seller negotiation environment...")
    env = Task3SequentialTwoSellerPerOneProductNegotiation(
        buyer_agent=buyer,
        seller1_agent=seller1,
        seller2_agent=seller2,
        max_rounds=max_rounds,
        initial_seller1_price=18.00,
        initial_seller2_price=20.00,
        buyer_max_price=buyer_max_price,
        seller1_min_price=seller1_min_price,
        seller2_min_price=seller2_min_price,
        environment_info={
            "platform": "NYC Street Hail",
            "market_type": "B2C",
            "comparison_enabled": True,
            "traffic_context": "Dense Manhattan local roads",
            "seller_contract_configs": seller_contract_configs,
        },
        price_tolerance=0,
        reward_weights=reward_weights,
    )

    user_profile = None
    print(f"User Profile: {user_profile}")

    print(f"Using default requirement: {user_requirement}")

    product_image_url = os.path.join(
        project_root,
        "agenticpay",
        "data",
        "NYC_taxi_data",
        "img",
        "yellow_tripdata_2026-02_sample_10",
        "image_0.png",
    )

    bundle_product_info = {
        "products": [
            {
                "name": "NYC yellow taxi — Gramercy to Murray Hill (main trip / metered fare portion)",
                "condition": "Metered on-trip service",
                "brand": "NYC Yellow Taxi",
                "service_type": "Point-to-point taxi ride",
                "pickup_location": "Gramercy, Manhattan, New York, NY",
                "dropoff_location": "Murray Hill, Manhattan, New York, NY",
                "trip_distance_miles": 0.94,
                "historical_trip_time": "Less than 5 minutes",
                "VendorID": 7,
                "RatecodeID": 1,
                "Passenger Count": 1,
                "Historical Fare Amount": 7.2,
                "price": 7.2,
                "original_price": 7.2,
                "availability_status": "Available now",
                "product_category": "Transportation & Mobility > Taxi Service",
                "average_rating": 4.7,
                "total_reviews": 128,
                "full_description": "Main trip component for the Gramercy–Murray Hill leg; bundle total is trip plus mandatory fees line item.",
                "image_url": product_image_url,
            },
            {
                "name": "NYC yellow taxi — mandatory surcharges and taxes (this route)",
                "condition": "Regulatory surcharges, fees, and taxes (bundled line item)",
                "brand": "NYC Yellow Taxi",
                "price": 5.75,
                "original_price": 5.75,
                "Historical Total Amount (reference)": 12.95,
                "mandatory_surcharges": [
                    "$2.50 (Congestion Surcharge)",
                    "$0.75 (CBD Congestion Fee)",
                    "$1.00 (Improvement Surcharge)",
                    "$0.50 (MTA State Tax)",
                ],
                "tolls": 0.0,
                "availability_status": "In effect per TLC rules",
                "product_category": "Transportation & Mobility > Taxi Service",
                "full_description": "Itemized add-ons and taxes applicable to this route; the negotiated ### BUYER_PRICE($X) ### is the all-in total for line 1 + line 2 together.",
                "image_url": product_image_url,
            },
        ]
    }

    print("\n" + "="*60)
    print("Sequential negotiation: two sellers, same 2-line-item taxi bundle, different bundle offers...")
    print("="*60)

    observation, info = env.reset(
        user_requirement=user_requirement,
        seller1_product_info=bundle_product_info,
        seller2_product_info=bundle_product_info,
        user_profile=user_profile,
    )

    done = False
    start_time = time.time()

    results = {
        "task": "Task15_s11_taxi_1_multi_products_multi_seller",
        "category": "Daily Life Consumption",
        "scenario": "Gramercy to Murray Hill: same two line items, two sellers negotiate all-in total (different floors)",
        "timestamp": datetime.now().isoformat(),
        "user_requirement": user_requirement,
        "user_profile": user_profile,
        "status": "unknown",
        "success": False,
        "error": None,
    }

    while not done:
        combined_history = []
        for msg in observation.get("conversation_history_seller1", []):
            combined_history.append({**msg, "content": f"[Seller 1] {msg['content']}"})
        for msg in observation.get("conversation_history_seller2", []):
            combined_history.append({**msg, "content": f"[Seller 2] {msg['content']}"})

        buyer_response = buyer.respond(
            conversation_history=combined_history,
            current_state={
                **observation,
                "instruction": "Two sellers offer the SAME two items as one all-in trip bundle. Each round pick ONE seller (use <selected_seller>) and negotiate the TOTAL all-in price for both line items together."
            }
        )

        selected_seller = Task3SequentialTwoSellerPerOneProductNegotiation.resolve_selected_seller(
            buyer_response, observation, buyer.last_selected_seller
        )
        print(f"\n[Buyer chooses to negotiate with Seller {selected_seller} this round]")

        buyer_action = buyer_response

        if selected_seller == 1:
            conversation_history = observation["conversation_history_seller1"]
        else:
            conversation_history = observation["conversation_history_seller2"]

        updated_conversation_history = conversation_history.copy()
        if buyer_action:
            current_round = observation.get("current_round", 0)
            updated_conversation_history.append({
                "role": "buyer",
                "content": buyer_action,
                "round": current_round
            })

        if selected_seller == 1:
            seller_action = seller1.respond(
                conversation_history=updated_conversation_history,
                current_state=observation
            )
        else:
            seller_action = seller2.respond(
                conversation_history=updated_conversation_history,
                current_state=observation
            )

        observation, reward, terminated, truncated, info = env.step(
            selected_seller=selected_seller,
            buyer_action=buyer_action,
            seller_action=seller_action
        )
        done = terminated or truncated

        env.render()
        sys.stdout.flush()

        if 'step_seller1_reward' in info or 'step_seller2_reward' in info or 'step_buyer_reward' in info:
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
                    print(f"  Seller2 Step Reward = seller_profit({seller2_profit:.2f} * {weights['seller_profit']:.2f}) + round_cost({round_cost:.2f} * {weights['time_cost']:.2f}) = {info['step_seller2_reward']:.2f} (seller2_price={seller2_price:.2f}, seller2_min={seller2_min}, round={info['round']})")
                else:
                    weighted_round_cost = round_cost * weights["time_cost"]
                    print(f"  Seller2 Step Reward = round_cost({round_cost:.2f} * {weights['time_cost']:.2f}) = {weighted_round_cost:.2f} (seller2_price={seller2_price:.2f}, seller2_min not specified, round={info['round']})")
            elif 'step_seller2_reward' in info:
                weighted_round_cost = round_cost * weights["time_cost"]
                print(f"  Seller2 Step Reward = round_cost({round_cost:.2f} * {weights['time_cost']:.2f}) = {weighted_round_cost:.2f} (seller2_price not specified, round={info['round']})")

        # If this is the final round (agreed or timeout), display score calculations after Step Rewards
        if done:
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
                bundle = info.get('seller1_product_info', {}) or {}
                plist = bundle.get('products') or []
                if len(plist) >= 2:
                    print(f"Bundle: (1) {plist[0].get('name', 'N/A')} | (2) {plist[1].get('name', 'N/A')}")
                elif plist:
                    print(f"Items: {plist[0].get('name', 'N/A')}")
            seller1_price = info.get('seller1_price', 0) or 0
            buyer_price_seller1 = info.get('buyer_price_seller1', 0) or 0
            seller2_price = info.get('seller2_price', 0) or 0
            buyer_price_seller2 = info.get('buyer_price_seller2', 0) or 0
            print(f"Seller1 Prices: Seller=${seller1_price:.2f} | Buyer=${buyer_price_seller1:.2f}")
            print(f"Seller2 Prices: Seller=${seller2_price:.2f} | Buyer=${buyer_price_seller2:.2f}")
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

            elapsed_time = time.time() - start_time
            seller1_product_info = info.get('seller1_product_info', {})
            seller2_product_info = info.get('seller2_product_info', {})
            actual_rounds = info.get('round', 0)
            results.update({
                "status": info.get('status', 'unknown'),
                "success": terminated,
                "selected_seller": info.get('selected_seller'),
                "final_deal_price": info.get('final_deal_price'),
                "seller1_price": info.get('seller1_price'),
                "seller2_price": info.get('seller2_price'),
                "buyer_price_seller1": info.get('buyer_price_seller1'),
                "buyer_price_seller2": info.get('buyer_price_seller2'),
                "total_rounds": actual_rounds,
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
                "seller1_product_info": seller1_product_info,
                "seller2_product_info": seller2_product_info,
                "model": get_model_name(model),
            })
            break

    env.close()
    print("\nNegotiation completed!")

    if "elapsed_time" not in results:
        results["elapsed_time"] = time.time() - start_time

    try:
        results_dir = Path(project_root) / "agenticpay" / "results" / "multi_products_multi_seller"
        results_dir.mkdir(parents=True, exist_ok=True)

        model_name_safe = get_model_name(model).replace("/", "_").replace("\\", "_").replace(":", "_")
        model_dir = results_dir / model_name_safe
        model_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = model_dir / f"batch_evaluation_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=True)

        summary_file = run_dir / "summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        script_stem = Path(__file__).stem
        output_file = run_dir / f"{script_stem}_output.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("Task15 Scenario 11: NYC taxi — two-product bundle, sequential two-seller negotiation results\n")
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
            if results.get('selected_seller'):
                f.write(f"Final Selected Seller: Seller {results['selected_seller']}\n")
                f.write(f"Final Deal Price: ${results.get('final_deal_price', 0):.2f}\n")
                binfo = results.get('seller1_product_info', {}) or {}
                pl = binfo.get('products') or []
                if len(pl) >= 2:
                    f.write(f"Bundle: {pl[0].get('name', 'N/A')} + {pl[1].get('name', 'N/A')}\n\n")
            f.write("Final Prices:\n")
            f.write(f"  Seller1 - Seller Price: ${results['seller1_price']:.2f}" if results.get('seller1_price') is not None else "  Seller1 - Seller Price: Not specified")
            f.write("\n")
            f.write(f"  Seller1 - Buyer Price: ${results['buyer_price_seller1']:.2f}" if results.get('buyer_price_seller1') is not None else "  Seller1 - Buyer Price: Not specified")
            f.write("\n")
            f.write(f"  Seller2 - Seller Price: ${results['seller2_price']:.2f}" if results.get('seller2_price') is not None else "  Seller2 - Seller Price: Not specified")
            f.write("\n")
            f.write(f"  Seller2 - Buyer Price: ${results['buyer_price_seller2']:.2f}" if results.get('buyer_price_seller2') is not None else "  Seller2 - Buyer Price: Not specified")
            f.write("\n\n")
            f.write("Products (same bundle for both sellers):\n")
            shared = results.get('seller1_product_info', {}) or {}
            for i, p in enumerate(shared.get('products') or [], 1):
                pr = p.get('price', p.get('original_price', 0))
                f.write(f"  {i}. {p.get('name', 'N/A')} (${float(pr):.2f})\n")
            f.write("\n")
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
    parser = argparse.ArgumentParser(description="Task15 Scenario 11: NYC Taxi Ride - Sequential Two-Seller Per One Product Negotiation")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name to use (e.g., 'gemini-3-pro-all', 'gpt-5.2', 'claude-sonnet-4-5-20250929'). If not provided, uses default model."
    )
    args = parser.parse_args()
    main(model_name=args.model)
