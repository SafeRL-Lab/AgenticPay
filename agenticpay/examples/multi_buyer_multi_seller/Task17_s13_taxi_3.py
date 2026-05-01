"""Task17 Scenario 13: NYC Taxi - Sequential Two-Buyer Two-Seller Negotiation (LaGuardia to East Chelsea)

Same ride (route) from two marketplace offers: listing has no per-seller identity; two sellers each
have a different confidential floor. Two buyers each pick one seller per round (structured routing).
Category: Daily Life Consumption
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
    """Sequential two-buyer two-seller NYC taxi negotiation (image + text)."""
    print("Initializing model...")

    api_key = os.getenv("OPENAI_API_KEY") or OPENAI_API_KEY
    if not api_key:
        print("Warning: OPENAI_API_KEY not set. Please set it to use OpenAI models.")
        print("You can set it with: export OPENAI_API_KEY='your-key-here'")
        return

    model_name = model_name or "gpt-5.4"
    model = OpenAIVLM(model=model_name, api_key=api_key)
    print(f"✓ Successfully initialized: {model}")

    print("Creating agents...")
    user_requirement = (
        "I want LaGuardia → East Chelsea yellow cab—all-in fare. "
        "I also prefer the rear trunk lines up evenly with the bumper with no obvious large offset gap."
    )
    product_request = user_requirement

    buyer1_max_price = 53.43  # Maximum acceptable all-in total for Buyer 1 (confidential; below listing reference total)
    buyer2_max_price = 55.27  # Maximum acceptable all-in total for Buyer 2 (confidential; below listing reference total)
    seller1_min_price = 47.33  # Minimum acceptable all-in total for Seller 1 (confidential; below listing reference total)
    seller2_min_price = 42.18  # Minimum acceptable all-in total for Seller 2 (confidential; below listing reference total)

    shared_contract_fields = {
        "contrainfo": {
            "product_request": product_request,
            "initial_contract_status": (
                "No all-in fare, curbside wait time, route preference, or user product preference match has been "
                "selected or agreed before negotiation starts."
            ),
            "contract_completion_requirement": (
                "A valid offer must explicitly fill price, continuous_terms.wait_time_mins, "
                "discrete_terms.route_preference, and discrete_terms.user_product_preference."
            ),
        },
        "field_descriptions": {
            "price": (
                "The total all-in amount the passenger pays for this yellow cab ride, in US dollars, "
                "including mandatory surcharges described in the listing."
            ),
            "continuous_terms.wait_time_mins": (
                "How many minutes the driver will wait at pickup after arriving before the passenger must appear "
                "(longer wait can mean less rushing for the rider but more idle time for the driver)."
            ),
            "discrete_terms.route_preference": (
                "Routing choice for Manhattan traffic: `tunnel` uses tolled crossings when they save time; "
                "`local_streets` stays on congested surface streets to avoid tunnel tolls."
            ),
            "discrete_terms.user_product_preference": (
                "Match to the buyer's stated trunk-to-bumper alignment check (even shut lines; no obvious large "
                "offset gap). `strong_match` / `partial_match` / `mismatch_or_uncertain`."
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
                "Your private maximum total fare for this ride before wait-time and route terms, in dollars. "
                "A lower fare is better because each dollar paid reduces your utility by one dollar."
            ),
            "continuous_weights.wait_time_mins": (
                "How each extra minute of driver wait at pickup changes your utility ($/minute). "
                "Positive means you value not having to rush downstairs immediately."
            ),
            "discrete_weights.route_preference": (
                "Dollar utility shift for each route option; positive is good for you, negative is bad."
            ),
            "discrete_weights.user_product_preference": (
                "Dollar utility shift per match level versus your stated vehicle appearance preference."
            ),
        },
        "continuous_weights": {"wait_time_mins": 1.0},
        "discrete_weights": {
            "route_preference": {"tunnel": 4.0, "local_streets": -2.0},
            "user_product_preference": {
                "strong_match": 0.22,
                "partial_match": 0.09,
                "mismatch_or_uncertain": -0.18,
            },
        },
    }
    buyer2_preferences = json.loads(json.dumps(buyer1_preferences))
    buyer2_preferences["v_base"] = buyer2_max_price
    buyer2_preferences["continuous_weights"]["wait_time_mins"] = 0.82
    buyer2_preferences["discrete_weights"]["route_preference"] = {"tunnel": 3.7, "local_streets": -1.75}

    seller1_preferences = {
        "c_base": seller1_min_price,
        "weight_descriptions": {
            "c_base": (
                "Your private minimum acceptable all-in payout for this ride before wait-time and route terms, "
                "in dollars. A higher fare is better because each dollar received increases your utility by one dollar."
            ),
            "continuous_weights.wait_time_mins": (
                "How each extra minute waiting curbside changes your utility ($/minute). "
                "Negative numbers mean waiting is costly."
            ),
            "discrete_weights.route_preference": (
                "Dollar utility shift for each route option; tolls and traffic trade off here."
            ),
            "discrete_weights.user_product_preference": (
                "Dollar shift per match tier; stronger appearance confirmation carries small nonzero assurance cost."
            ),
        },
        "continuous_weights": {"wait_time_mins": -1.5},
        "discrete_weights": {
            "route_preference": {"tunnel": -3.0, "local_streets": 0.0},
            "user_product_preference": {
                "strong_match": -0.06,
                "partial_match": -0.03,
                "mismatch_or_uncertain": 0.008,
            },
        },
    }
    seller2_preferences = json.loads(json.dumps(seller1_preferences))
    seller2_preferences["c_base"] = seller2_min_price
    seller2_preferences["continuous_weights"]["wait_time_mins"] = -1.38
    seller2_preferences["discrete_weights"]["route_preference"] = {"tunnel": -2.95, "local_streets": 0.11}

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

    print("Creating sequential multi-buyer multi-seller negotiation environment...")
    env = Task3SequentialTwoBuyerTwoSellerNegotiation(
        buyer1_agent=buyer1,
        buyer2_agent=buyer2,
        seller1_agent=seller1,
        seller2_agent=seller2,
        max_rounds=max_rounds,
        initial_seller1_price=75.0,
        initial_seller2_price=78.0,
        buyer1_max_price=buyer1_max_price,
        buyer2_max_price=buyer2_max_price,
        seller1_min_price=seller1_min_price,
        seller2_min_price=seller2_min_price,
        environment_info={
            "platform": "NYC Street Hail / Ride Apps",
            "market_type": "B2C",
            "buyer_seller_contract_configs": buyer_seller_contract_configs,
        },
        price_tolerance=price_tolerance,
        reward_weights=reward_weights,
    )

    user_profile = None
    print(f"User Profile: {user_profile}")

    print(f"Using default requirement: {user_requirement}")

    print("\n" + "=" * 60)
    print("Starting new sequential negotiation with two buyers and two sellers (NYC taxi)...")
    print("=" * 60)

    product_image_url = os.path.join(
        project_root,
        "agenticpay",
        "data",
        "NYC_taxi_data",
        "img",
        "yellow_tripdata_2026-02_sample_10",
        'image_2.png',
    )

    observation, info = env.reset(
        user_requirement=user_requirement,
        product_info={
            'name': 'NYC yellow taxi: LaGuardia Airport → East Chelsea (all-in flat fare)',
            'product_category': 'Transportation & Mobility › Taxi › NYC (airport)',
            'pickup_location': 'LaGuardia Airport, Queens, New York, NY',
            'dropoff_location': 'East Chelsea, Manhattan, New York, NY',
            'trip_distance': '9.99 miles',
            'trip_time_estimate': 'About 31 minutes',
            'passenger_count': 1,
            'historical_fare_amount': 44.3,
            'reference_total_amount': 67.81,
            'mandatory_surcharges': [
                '$2.50 (Congestion Surcharge for driving below 96th St in Manhattan)',
                '$0.75 (CBD Congestion Fee)',
                '$1.75 (Airport Fee)',
                '$6.00 (Night/peak extra)',
                '$1.00 (Improvement Surcharge)',
                '$0.50 (MTA State Tax)',
            ],
            'tolls': '$0.00',
            'pricing_rules': "The negotiated price (### BUYER_PRICE($X) ### or ### SELLER_PRICE($Y) ###) MUST be the TOTAL final amount the passenger pays. It MUST include the driver's base fare PLUS all mandatory surcharges and taxes listed above. No fees can be added later.",
            'route_note': 'See the attached route image for distance and airport-to-Manhattan context.',
            'image_url': product_image_url,
        },
        user_profile=user_profile,
    )

    done = False
    start_time = time.time()

    results = {
        "task": 'Task17_s13_taxi_3',
        "timestamp": datetime.now().isoformat(),
        "user_requirement": user_requirement,
        "user_profile": user_profile,
        "status": "unknown",
        "success": False,
        "error": None,
    }

    while not done:
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

        buyer1_action = buyer1_response
        buyer2_action = buyer2_response

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

        env.render()
        sys.stdout.flush()

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

            round_cost = -info['round']
            weights = env.reward_weights

            if 'step_buyer1_reward' in info:
                buyer_price = None
                if info.get('buyer1_selected_seller') == 1:
                    buyer_price = info.get('b1s1_buyer_price')
                elif info.get('buyer1_selected_seller') == 2:
                    buyer_price = info.get('b1s2_buyer_price')

                if buyer_price is not None and env.buyer1_max_price is not None:
                    buyer_savings = env.buyer1_max_price - buyer_price
                    _ = buyer_savings * weights["buyer_savings"]
                    weighted_round_cost = round_cost * weights["time_cost"]
                    print(f"  Buyer1 Step Reward = buyer_savings({buyer_savings:.2f} * {weights['buyer_savings']:.2f}) + round_cost({round_cost:.2f} * {weights['time_cost']:.2f}) = {info['step_buyer1_reward']:.2f} (buyer1_max={env.buyer1_max_price}, buyer_price={buyer_price:.2f}, round={info['round']})")
                else:
                    weighted_round_cost = round_cost * weights["time_cost"]
                    print(f"  Buyer1 Step Reward = round_cost({round_cost:.2f} * {weights['time_cost']:.2f}) = {weighted_round_cost:.2f} (buyer_price not specified, round={info['round']})")

            if 'step_buyer2_reward' in info:
                buyer_price = None
                if info.get('buyer2_selected_seller') == 1:
                    buyer_price = info.get('b2s1_buyer_price')
                elif info.get('buyer2_selected_seller') == 2:
                    buyer_price = info.get('b2s2_buyer_price')

                if buyer_price is not None and env.buyer2_max_price is not None:
                    buyer_savings = env.buyer2_max_price - buyer_price
                    _ = buyer_savings * weights["buyer_savings"]
                    weighted_round_cost = round_cost * weights["time_cost"]
                    print(f"  Buyer2 Step Reward = buyer_savings({buyer_savings:.2f} * {weights['buyer_savings']:.2f}) + round_cost({round_cost:.2f} * {weights['time_cost']:.2f}) = {info['step_buyer2_reward']:.2f} (buyer2_max={env.buyer2_max_price}, buyer_price={buyer_price:.2f}, round={info['round']})")
                else:
                    weighted_round_cost = round_cost * weights["time_cost"]
                    print(f"  Buyer2 Step Reward = round_cost({round_cost:.2f} * {weights['time_cost']:.2f}) = {weighted_round_cost:.2f} (buyer_price not specified, round={info['round']})")

            if 'step_seller1_reward' in info:
                seller1_price = None
                if info.get('buyer1_selected_seller') == 1 and info.get('b1s1_seller_price') is not None:
                    seller1_price = info.get('b1s1_seller_price')
                elif info.get('buyer2_selected_seller') == 1 and info.get('b2s1_seller_price') is not None:
                    seller1_price = info.get('b2s1_seller_price')
                if (info.get('buyer1_selected_seller') == 1 and info.get('buyer2_selected_seller') == 1 and
                    info.get('b1s1_seller_price') is not None and info.get('b2s1_seller_price') is not None):
                    seller1_price = max(info.get('b1s1_seller_price'), info.get('b2s1_seller_price'))

                if seller1_price is not None and env.seller1_min_price is not None:
                    seller1_profit = seller1_price - env.seller1_min_price
                    _ = seller1_profit * weights["seller_profit"]
                    weighted_round_cost = round_cost * weights["time_cost"]
                    print(f"  Seller1 Step Reward = seller_profit({seller1_profit:.2f} * {weights['seller_profit']:.2f}) + round_cost({round_cost:.2f} * {weights['time_cost']:.2f}) = {info['step_seller1_reward']:.2f} (seller1_price={seller1_price:.2f}, seller1_min={env.seller1_min_price}, round={info['round']})")
                else:
                    weighted_round_cost = round_cost * weights["time_cost"]
                    print(f"  Seller1 Step Reward = round_cost({round_cost:.2f} * {weights['time_cost']:.2f}) = {weighted_round_cost:.2f} (seller1_price not specified, round={info['round']})")

            if 'step_seller2_reward' in info:
                seller2_price = None
                if info.get('buyer1_selected_seller') == 2 and info.get('b1s2_seller_price') is not None:
                    seller2_price = info.get('b1s2_seller_price')
                elif info.get('buyer2_selected_seller') == 2 and info.get('b2s2_seller_price') is not None:
                    seller2_price = info.get('b2s2_seller_price')
                if (info.get('buyer1_selected_seller') == 2 and info.get('buyer2_selected_seller') == 2 and
                    info.get('b1s2_seller_price') is not None and info.get('b2s2_seller_price') is not None):
                    seller2_price = max(info.get('b1s2_seller_price'), info.get('b2s2_seller_price'))

                if seller2_price is not None and env.seller2_min_price is not None:
                    seller2_profit = seller2_price - env.seller2_min_price
                    _ = seller2_profit * weights["seller_profit"]
                    weighted_round_cost = round_cost * weights["time_cost"]
                    print(f"  Seller2 Step Reward = seller_profit({seller2_profit:.2f} * {weights['seller_profit']:.2f}) + round_cost({round_cost:.2f} * {weights['time_cost']:.2f}) = {info['step_seller2_reward']:.2f} (seller2_price={seller2_price:.2f}, seller2_min={env.seller2_min_price}, round={info['round']})")
                else:
                    weighted_round_cost = round_cost * weights["time_cost"]
                    print(f"  Seller2 Step Reward = round_cost({round_cost:.2f} * {weights['time_cost']:.2f}) = {weighted_round_cost:.2f} (seller2_price not specified, round={info['round']})")

        if done:
            print("\n" + "=" * 60)
            print("Negotiation Ended")
            print("=" * 60)
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
            env._print_global_score_details()
            env._print_buyer_score_details()
            env._print_seller_score_details()

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
            print("=" * 60)

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

    env.close()
    print("\nSequential multi-buyer multi-seller negotiation completed!")

    if "elapsed_time" not in results:
        results["elapsed_time"] = time.time() - start_time

    try:
        results_dir = Path(project_root) / "agenticpay" / "results" / "multi_buyer_multi_seller"
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

        output_file = run_dir / 'Task17_s13_taxi_3_output.txt'
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write('Task17 Scenario 13: NYC Taxi - Sequential Two-Buyer Two-Seller Negotiation Results (image + text)' + "\n")
            f.write("Category: Daily Life Consumption\n")
            f.write("=" * 80 + "\n\n")
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
                pair_key = f"b{results['selected_buyer']}s{results['selected_seller']}"
                agreed_contract = results.get(f"{pair_key}_agreed_contract")
                if agreed_contract is not None:
                    f.write(f"Final Contract: {agreed_contract}\n\n")
            f.write("Final Prices:\n")
            f.write(f"  Buyer1-Seller1: Buyer=${results['b1s1_buyer_price']:.2f} | Seller=${results['b1s1_seller_price']:.2f}" if results.get('b1s1_buyer_price') is not None and results.get('b1s1_seller_price') is not None else "  Buyer1-Seller1: Not specified")
            f.write("\n")
            f.write(f"  Buyer1-Seller2: Buyer=${results['b1s2_buyer_price']:.2f} | Seller=${results['b1s2_seller_price']:.2f}" if results.get('b1s2_buyer_price') is not None and results.get('b1s2_seller_price') is not None else "  Buyer1-Seller2: Not specified")
            f.write("\n")
            f.write(f"  Buyer2-Seller1: Buyer=${results['b2s1_buyer_price']:.2f} | Seller=${results['b2s1_seller_price']:.2f}" if results.get('b2s1_buyer_price') is not None and results.get('b2s1_seller_price') is not None else "  Buyer2-Seller1: Not specified")
            f.write("\n")
            f.write(f"  Buyer2-Seller2: Buyer=${results['b2s2_buyer_price']:.2f} | Seller=${results['b2s2_seller_price']:.2f}" if results.get('b2s2_buyer_price') is not None and results.get('b2s2_seller_price') is not None else "  Buyer2-Seller2: Not specified")
            f.write("\n\n")
            product_info = results.get('product_info', {})
            f.write("Ride (listing):\n")
            f.write(f"  Name: {product_info.get('name', 'N/A')}\n")
            ref = product_info.get('reference_total_amount')
            if ref is not None:
                f.write(f"  Reference total: ${float(ref):.2f}\n")
            f.write(f"  Pickup: {product_info.get('pickup_location', 'N/A')}\n")
            f.write(f"  Dropoff: {product_info.get('dropoff_location', 'N/A')}\n")
            f.write("\n")
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
    parser = argparse.ArgumentParser(description='Task17 Scenario 13: NYC Taxi - Sequential Two-Buyer Two-Seller Negotiation (LaGuardia to East Chelsea)')
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name to use (e.g., 'gemini-3-pro-all', 'gpt-5.2', 'claude-sonnet-4-5-20250929'). If not provided, uses default model."
    )
    args = parser.parse_args()
    main(model_name=args.model)
