"""Task26 Scenario 22: Rent House 2 — same two-rental bundle, sequential two-landlord negotiation (image + text)

Tenant wants the same two listed units as one package and negotiates TOTAL monthly rent for both.
Both landlords list identical line items; each has a different floor price and opening offer.
Category: Real Estate
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


def main(model_name=None):
    """Sequential two-seller negotiation for the same two-rental bundle (total monthly rent)."""

    print("Initializing model...")

    api_key = os.getenv("OPENAI_API_KEY") or OPENAI_API_KEY
    if not api_key:
        print("Warning: OPENAI_API_KEY not set. Please set it to use OpenAI models.")
        print("You can set it with: export OPENAI_API_KEY='your-key-here'")
        return

    model_name = model_name or "gpt-4o-mini"
    model = OpenAIVLM(model=model_name, api_key=api_key)

    print(f"✓ Successfully initialized: {model}")

    # Listing line-item rents sum to 1520 + 1650 = 3170 USD/mo (public bundle reference). Private floors sit below that summed anchor (Seller1_min > Seller2_min; buyer_max below reference).
    print("Creating agents...")
    _bundle_buyer_max = 2466.0
    _bundle_seller1_min = 2219.0
    _bundle_seller2_min = 1966.0
    product_request = (
        "I want Sydney CBD studio plus Bondi 2BR—best total monthly rent. "
        "I also prefer kitchen listing photos that leave a clear open stretch of counter for prep instead of clutter wall-to-wall."
    )
    shared_contract_fields = {
        "contrainfo": {
            "product_request": product_request,
            "initial_contract_status": (
                "No total bundle monthly rent, lease length, utility-inclusion term, or user product preference match has been "
                "selected or agreed before negotiation starts."
            ),
            "contract_completion_requirement": (
                "A valid offer must explicitly fill price, continuous_terms.lease_months, discrete_terms.include_utilities, "
                "and discrete_terms.user_product_preference. The price is the TOTAL monthly rent for both rental units together."
            ),
        },
        "field_descriptions": {
            "price": (
                "The total amount of money the tenant pays per month for the full two-rental bundle, measured in US dollars."
            ),
            "continuous_terms.lease_months": (
                "The number of months in the lease term for the bundle. It must be within the configured bounds."
            ),
            "discrete_terms.include_utilities": (
                "Whether standard utilities are included in the total monthly rent. true means utilities are included; "
                "false means the tenant pays utilities separately."
            ),
            "discrete_terms.user_product_preference": (
                "How well kitchen listing photos match the buyer's stated preference for a clearly open counter stretch usable for "
                "prep rather than clutter packed wall-to-wall across the worktop. "
                "Use `strong_match` when clearly satisfied, `partial_match` when only partly satisfied, "
                "and `mismatch_or_uncertain` when not satisfied or cannot be confirmed."
            ),
        },
        "continuous_bounds": {
            "lease_months": {"min": 1, "max": 24},
        },
        "discrete_options": {
            "include_utilities": [True, False],
            "user_product_preference": ["strong_match", "partial_match", "mismatch_or_uncertain"],
        },
        "buyer_preferences": {
            "v_base": _bundle_buyer_max,
            "weight_descriptions": {
                "v_base": (
                    "Your private maximum total monthly rent for the two-unit bundle before lease-length and utility terms, "
                    "measured in dollars. A lower total rent is better for you because every dollar paid reduces your utility by 1 dollar."
                ),
                "continuous_weights.lease_months": (
                    "How much each additional lease month changes your utility, measured in dollars per month. "
                    "A negative number means longer commitment is worse for you."
                ),
                "discrete_weights.include_utilities": (
                    "How much utility inclusion changes your utility, measured in dollars. "
                    "Positive numbers are good for you; negative numbers are bad for you."
                ),
                "discrete_weights.user_product_preference": (
                    "How much each alignment level between the listing depiction and your stated kitchen-photo preference "
                    "changes your utility, measured in dollars."
                ),
            },
            "continuous_weights": {"lease_months": -10.0},
            "discrete_weights": {
                "include_utilities": {True: 100.0, False: 0.0},
                "user_product_preference": {
                    "strong_match": 42.0,
                    "partial_match": 17.0,
                    "mismatch_or_uncertain": -35.0,
                },
            },
        },
    }
    seller1_contract_config = {
        **shared_contract_fields,
        "seller_preferences": {
            "c_base": _bundle_seller1_min,
            "weight_descriptions": {
                "c_base": (
                    "Your private minimum acceptable total monthly rent for the two-unit bundle before lease-length and "
                    "utility terms, measured in dollars. A higher total rent is better for you because every dollar received "
                    "increases your utility by 1 dollar."
                ),
                "continuous_weights.lease_months": (
                    "How much each additional lease month changes your utility, measured in dollars per month. "
                    "A positive number means longer occupancy reduces vacancy risk for you."
                ),
                "discrete_weights.include_utilities": (
                    "How much including utilities in the bundle rent changes your utility, measured in dollars. "
                    "Positive numbers are good for you; negative numbers are bad for you."
                ),
                "discrete_weights.user_product_preference": (
                    "How much each commitment tier on marketing depiction accuracy shifts your utility; "
                    "strong tiers carry modest extra disclosure friction."
                ),
            },
            "continuous_weights": {"lease_months": 20.0},
            "discrete_weights": {
                "include_utilities": {True: -60.0, False: 0.0},
                "user_product_preference": {
                    "strong_match": -14.0,
                    "partial_match": -7.0,
                    "mismatch_or_uncertain": 2.0,
                },
            },
        },
    }
    seller2_contract_config = {
        **shared_contract_fields,
        "seller_preferences": {
            "c_base": _bundle_seller2_min,
            "weight_descriptions": seller1_contract_config["seller_preferences"]["weight_descriptions"],
            "continuous_weights": {"lease_months": 18.0},
            "discrete_weights": {
                "include_utilities": {True: -55.0, False: 0.0},
                "user_product_preference": {
                    "strong_match": -14.0,
                    "partial_match": -7.0,
                    "mismatch_or_uncertain": 2.0,
                },
            },
        },
    }
    seller_contract_configs = {
        1: seller1_contract_config,
        2: seller2_contract_config,
    }
    buyer_max_price = shared_contract_fields["buyer_preferences"]["v_base"]
    seller1_min_price = seller1_contract_config["seller_preferences"]["c_base"]
    seller2_min_price = seller2_contract_config["seller_preferences"]["c_base"]

    buyer = BuyerAgent(model=model, name="Buyer1", buyer_max_price=buyer_max_price)
    seller1 = SellerAgent(model=model, name="Seller1", seller_min_price=seller1_min_price)
    seller2 = SellerAgent(model=model, name="Seller2", seller_min_price=seller2_min_price)

    print("Creating sequential multi-seller negotiation environment...")
    env = Task3SequentialTwoSellerPerOneProductNegotiation(
        buyer_agent=buyer,
        seller1_agent=seller1,
        seller2_agent=seller2,
        max_rounds=max_rounds,
        initial_seller1_price=3220.0,
        initial_seller2_price=3190.0,
        buyer_max_price=buyer_max_price,
        seller1_min_price=seller1_min_price,
        seller2_min_price=seller2_min_price,
        environment_info={
            "platform": "Airbnb (long-term lease framing)",
            "market_type": "Residential Rental",
            "comparison_enabled": True,
            "seller_contract_configs": seller_contract_configs,
        },
        price_tolerance=0,
        reward_weights=reward_weights,
    )

    user_profile = None
    print(f"User Profile: {user_profile}")

    user_requirement = product_request
    print(f"Using default requirement: {user_requirement}")

    img1 = "https://a0.muscache.com/im/pictures/0c59647f-273c-4510-a1f3-eb8a3f6cc650.jpg?aki_policy=large"
    img2 = "https://a0.muscache.com/im/pictures/0c59647f-273c-4510-a1f3-eb8a3f6cc650.jpg?aki_policy=large"

    bundle_product_info = {
        "products": [
            {
                "name": "Mezzos Studio in Sydney CBD (Airbnb 14096512)",
                "condition": "Move-in ready",
                "size": "0 bed · Apartment · Entire home/apt · 1.0 bath",
                "price": 1520.0,
                "original_price": 1520.0,
                "availability_status": "Available now.",
                "product_category": "Real Estate › Rentals › Apartment › Entire home/apt",
                "average_rating": 4.45,
                "total_reviews": 221,
                "asin": "AIRBNB-14096512",
                "full_description": "Furnished studio at Circular Quay, CBD. Wi-Fi included. List reference $1520/mo as a line item toward a two-unit long-term package.",
                "image_url": img1,
                "listing_age": "Listing 14096512",
            },
            {
                "name": "Sunny 2BR apartment near Bondi Beach (Airbnb 16289600)",
                "condition": "Move-in ready",
                "size": "2 bed · Apartment · Entire home/apt · 1.0 bath",
                "price": 1650.0,
                "original_price": 1650.0,
                "availability_status": "Available now.",
                "product_category": "Real Estate › Rentals › Apartment › Entire home/apt",
                "average_rating": 3.0,
                "total_reviews": 1,
                "asin": "AIRBNB-16289600",
                "full_description": "Between Bondi Junction and Bondi Beach; two double beds, equipped kitchen, park next door. List reference $1650/mo as a line item toward a two-unit long-term package.",
                "image_url": img2,
                "listing_age": "Listing 16289600",
            },
        ]
    }

    print("\n" + "="*60)
    print("Sequential negotiation: two landlords, same 2-rental bundle, different total offers...")
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
        "task": "Task26_s22_rent_house_2_negotiation",
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
                "instruction": "Two landlords offer the SAME two rentals as one bundle. Each round pick ONE seller (use <selected_seller>) and negotiate the TOTAL monthly rent for both units together."
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
                # Display product info for selected seller
                bundle = info.get('seller1_product_info', {}) or {}
                plist = bundle.get('products') or []
                if len(plist) >= 2:
                    print(f"Bundle: (1) {plist[0].get('name', 'N/A')} | (2) {plist[1].get('name', 'N/A')}")
                elif plist:
                    print(f"Selected bundle item: {plist[0].get('name', 'N/A')}")
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
            seller1_product_info = info.get('seller1_product_info', {})
            seller2_product_info = info.get('seller2_product_info', {})
            # current_round has been incremented to reflect the completed round
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

        model_name = get_model_name(model)
        model_name_safe = model_name.replace("/", "_").replace("\\", "_").replace(":", "_")
        model_dir = results_dir / model_name_safe
        model_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = model_dir / f"batch_evaluation_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=True)

        summary_file = run_dir / "summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        output_file = run_dir / "Task26_s22_rent_house_2_output.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("Task26 Scenario 22: Rent House 2 — two-rental bundle, sequential two-landlord results\n")
            f.write("Category: Real Estate\n")
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
            f.write("Listings (same bundle for both landlords):\n")
            shared = results.get('seller1_product_info', {}) or {}
            for i, p in enumerate(shared.get('products') or [], 1):
                pr = p.get('price', p.get('original_price', 0))
                f.write(f"  {i}. {p.get('name', 'N/A')} (${float(pr):.2f} line-item reference)\n")
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
    parser = argparse.ArgumentParser(description="Task26: two-rental bundle, sequential two-landlord negotiation")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name to use. If not provided, uses default model."
    )
    args = parser.parse_args()
    main(model_name=args.model)
