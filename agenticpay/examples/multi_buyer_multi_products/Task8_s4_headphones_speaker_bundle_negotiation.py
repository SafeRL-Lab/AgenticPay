"""Task8 Scenario 4: Headphones & Bluetooth Speaker Bundle - Sequential Two-Buyer Two-Product Negotiation (image + text)

One seller negotiating with two buyers for the same two-item electronics bundle (kids headphones + portable speaker; total price).
Seller chooses which buyer to negotiate with each round (aligned with only_multi_buyer Task3 / Task5).
Category: Electronics
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

from agenticpay.envs.multi_buyer_multi_products.Task3_sequential_two_buyer_two_product_negotiation import Task3SequentialTwoBuyerTwoProductNegotiation
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


def _run_seller_routing(
    seller,
    combined_history: list,
    observation: dict,
    routing_instruction: str,
):
    """Structured ``<selected_buyer>`` + retries + random fallback (aligned with only_multi_buyer Task3 / Task5)."""
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
    """Main function: sequential multi-buyer bundle negotiation; seller routes buyers each round."""

    print("Initializing model...")

    api_key = os.getenv("OPENAI_API_KEY") or OPENAI_API_KEY
    if not api_key:
        print("Warning: OPENAI_API_KEY not set. Please set it to use OpenAI models.")
        print("You can set it with: export OPENAI_API_KEY='your-key-here'")
        return

    model_name = model_name or "gpt-5.4"
    model = OpenAIVLM(model=model_name, api_key=api_key)

    print(f"✓ Successfully initialized: {model}")

    # Public anchor: SKU total ~$123.48; confidential band is lower (floors ≈65–68% · caps ≈78–81% · window ≈ quoted×0.13).
    print("Creating agents...")
    product_request = (
        "I want kids Bluetooth headphones and a Sony SRS-XB33 speaker. "
        "I also prefer the headphones headband cushion to run as one continuous padded strip rather than two separated cushion pads."
    )
    _k1 = 92.36 / 24.8
    _k2 = 101.01 / 26.2
    buyer1_contract_config = {
        "contrainfo": {
            "product_request": product_request,
            "initial_contract_status": (
                "No total bundle price, delivery time, return policy, packaging option, or user product preference match "
                "has been selected or agreed before negotiation starts."
            ),
            "contract_completion_requirement": (
                "A valid offer must explicitly fill price, continuous_terms.delivery_days, "
                "discrete_terms.return_policy, discrete_terms.packaging, and "
                "discrete_terms.user_product_preference for the two-item bundle."
            ),
        },
        "field_descriptions": {
            "price": "The total amount of money the buyer pays for the whole two-item bundle, measured in US dollars.",
            "continuous_terms.delivery_days": (
                "How many days the seller can take to deliver the headphones and Bluetooth speaker after the deal is made."
            ),
            "discrete_terms.return_policy": (
                "The return rule for the bundle. `30_days` means the buyer can return the order within 30 days; "
                "`none` means the sale is final and returns are not allowed."
            ),
            "discrete_terms.packaging": (
                "The packaging used for shipment. `protective` means extra protection for the electronics; "
                "`standard` means normal packaging."
            ),
            "discrete_terms.user_product_preference": (
                "How well the bundle matches the buyer's stated preference for the headphones headband cushion to run as one "
                "continuous padded strip rather than two separated cushion pads. Use `strong_match` when that preference is clearly satisfied, "
                "`partial_match` when it is only partly satisfied, and `mismatch_or_uncertain` when it is not satisfied or "
                "cannot be confirmed."
            ),
        },
        "continuous_bounds": {"delivery_days": {"min": 1, "max": 7}},
        "discrete_options": {
            "return_policy": ["30_days", "none"],
            "packaging": ["protective", "standard"],
            "user_product_preference": ["strong_match", "partial_match", "mismatch_or_uncertain"],
        },
        "buyer_preferences": {
            "v_base": 92.36,
            "weight_descriptions": {
                "v_base": (
                    "Your private maximum value for the complete bundle before delivery, return, and packaging terms, "
                    "measured in dollars. A lower total price is better for you because every dollar paid reduces your utility by 1 dollar."
                ),
                "continuous_weights.delivery_days": (
                    "How much each additional delivery day changes your utility, measured in dollars per day. "
                    "A negative number means slower delivery is worse for you."
                ),
                "discrete_weights.return_policy": (
                    "How much each return-policy option changes your utility, measured in dollars. "
                    "Positive numbers are good for you; negative numbers are bad for you."
                ),
                "discrete_weights.packaging": (
                    "How much each packaging option changes your utility, measured in dollars. "
                    "Positive numbers are good for you; negative numbers are bad for you."
                ),
                "discrete_weights.user_product_preference": (
                    "How much each level of match to your stated product preference changes your utility, measured in dollars. "
                    "Positive numbers are good for you; negative numbers are bad for you."
                ),
            },
            "continuous_weights": {"delivery_days": -0.55 * _k1},
            "discrete_weights": {
                "return_policy": {"30_days": 1.8 * _k1, "none": -2.0 * _k1},
                "packaging": {"protective": 1.4 * _k1, "standard": -0.6 * _k1},
                "user_product_preference": {
                    "strong_match": 0.30,
                    "partial_match": 0.12,
                    "mismatch_or_uncertain": -0.25,
                },
            },
        },
        "seller_preferences": {
            "c_base": 87.92,
            "weight_descriptions": {
                "c_base": (
                    "Your private minimum cost for fulfilling the complete bundle before delivery, return, and packaging terms, "
                    "measured in dollars. A higher total price is better for you because every dollar received increases your utility by 1 dollar."
                ),
                "continuous_weights.delivery_days": (
                    "How much each additional delivery day changes your utility, measured in dollars per day. "
                    "A positive number means more delivery flexibility is better for you."
                ),
                "discrete_weights.return_policy": (
                    "How much each return-policy option changes your utility, measured in dollars. "
                    "Positive numbers are good for you; negative numbers are bad for you."
                ),
                "discrete_weights.packaging": (
                    "How much each packaging option changes your utility, measured in dollars. "
                    "Positive numbers are good for you; negative numbers are bad for you."
                ),
                "discrete_weights.user_product_preference": (
                    "How much each level of commitment to the buyer's stated product preference changes your utility, measured in dollars. "
                    "Stronger commitments carry a small nonzero risk or handling cost."
                ),
            },
            "continuous_weights": {"delivery_days": 0.35 * _k1},
            "discrete_weights": {
                "return_policy": {"30_days": -2.2 * _k1, "none": 1.5 * _k1},
                "packaging": {"protective": -1.3 * _k1, "standard": 0.5 * _k1},
                "user_product_preference": {
                    "strong_match": -0.08,
                    "partial_match": -0.04,
                    "mismatch_or_uncertain": 0.01,
                },
            },
        },
    }
    buyer2_contract_config = json.loads(json.dumps(buyer1_contract_config))
    buyer2_contract_config["buyer_preferences"]["v_base"] = 101.01
    buyer2_contract_config["buyer_preferences"]["continuous_weights"]["delivery_days"] = -0.45 * _k2
    buyer2_contract_config["buyer_preferences"]["discrete_weights"]["return_policy"] = {
        "30_days": 1.5 * _k2,
        "none": -1.6 * _k2,
    }
    buyer2_contract_config["buyer_preferences"]["discrete_weights"]["packaging"] = {
        "protective": 1.1 * _k2,
        "standard": -0.4 * _k2,
    }
    buyer2_contract_config["seller_preferences"]["c_base"] = 77.05
    buyer2_contract_config["seller_preferences"]["continuous_weights"]["delivery_days"] = 0.40 * _k2
    buyer2_contract_config["seller_preferences"]["discrete_weights"]["return_policy"] = {
        "30_days": -2.0 * _k2,
        "none": 1.4 * _k2,
    }
    buyer2_contract_config["seller_preferences"]["discrete_weights"]["packaging"] = {
        "protective": -1.1 * _k2,
        "standard": 0.4 * _k2,
    }
    buyer_contract_configs = {1: buyer1_contract_config, 2: buyer2_contract_config}
    buyer1_max_price = buyer1_contract_config["buyer_preferences"]["v_base"]
    buyer2_max_price = buyer2_contract_config["buyer_preferences"]["v_base"]
    seller_min_price = min(cfg["seller_preferences"]["c_base"] for cfg in buyer_contract_configs.values())

    buyer1 = BuyerAgent(model=model, name="Buyer1", buyer_max_price=buyer1_max_price)
    buyer2 = BuyerAgent(model=model, name="Buyer2", buyer_max_price=buyer2_max_price)
    seller = SellerAgent(model=model, name="Seller1", seller_min_price=seller_min_price)

    print("Creating sequential multi-buyer multi-product negotiation environment...")
    env = Task3SequentialTwoBuyerTwoProductNegotiation(
        buyer1_agent=buyer1,
        buyer2_agent=buyer2,
        seller_agent=seller,
        max_rounds=max_rounds,
        buyer1_max_price=buyer1_max_price,
        buyer2_max_price=buyer2_max_price,
        seller_min_price=seller_min_price,
        environment_info={
            "platform": "Amazon",
            "market_type": "B2C",
            "listing_age": "2 weeks",
            "bundle_context": (
                "Several sellers list this exact two-item bundle; offers are for the bundle total. "
                "Each seller has a different internal floor for the pair; buyers only see product facts, not seller identities."
            ),
            "buyer_contract_configs": buyer_contract_configs,
        },
        price_tolerance=price_tolerance,
        reward_weights=reward_weights,
    )

    user_profile = None
    print(f"User Profile: {user_profile}")

    product1_image_url = "https://m.media-amazon.com/images/I/41B+OC0qnOL.jpg"
    product2_image_url = "https://m.media-amazon.com/images/I/41+lMIUpYbL.jpg"

    product_info = {
        "products": [
            {
                "name": "Kids Wireless Headphones, Adjustable Headband, Stereo Sound, 3.5mm Jack, Kids Bluetooth Headphones, Volume Control, Foldable, Build-in Microphone, Over-Ear Headphones for Kids for School Home, Travel",
                "condition": "New",
                "price": 14.99,
                "brand": "NVRADCHUA",
                "original_price": 14.99,
                "product_category": "Electronics › Headphones › Over-Ear Headphones",
                "average_rating": 4.0,
                "total_reviews": 2,
                "asin": "B09KQNH5C6",
                "full_description": "WIRELESS & WIRED KIDS HEADPHONES: Built with 5.0 Bluetooth chip for fast and stable connection, also with 3.5mm jack. Compatible with smartphones, laptops, tablets, computers, TVs.",
                "image_url": product1_image_url,
            },
            {
                "name": "Sony Extra Bass Portable Bluetooth Speaker Black - SRS-XB33/BC (Renewed)",
                "condition": "Renewed",
                "price": 108.49,
                "brand": "Sony",
                "original_price": 108.49,
                "product_category": "Electronics › Portable Audio & Video › Portable Speakers & Docks › Portable Bluetooth Speakers",
                "average_rating": 4.5,
                "total_reviews": 962,
                "asin": "B08FZDJRQ7",
                "full_description": "This pre-owned or refurbished product has been professionally inspected and tested to work and look like new.",
                "image_url": product2_image_url,
            },
        ]
    }

    total_product_price = sum(p["price"] for p in product_info["products"])
    print(f"\nProducts (Bundle):")
    for i, p in enumerate(product_info["products"], 1):
        print(f"  {i}. {p['name']}: ${p['price']:.2f}")
    print(f"  Total Bundle Price: ${total_product_price:.2f}")

    user_requirement = product_request
    print(f"Using default requirement: {user_requirement}")

    print("\n" + "=" * 60)
    print("Starting sequential negotiation for the headphones + speaker bundle...")
    print("=" * 60)

    observation, info = env.reset(
        user_requirement=user_requirement,
        product_info=product_info,
        user_profile=user_profile,
    )

    done = False
    start_time = time.time()

    routing_instruction = (
        "You are negotiating with two buyers for the SAME two-product bundle; all prices are the TOTAL for both items. "
        "Each round, choose exactly ONE buyer and output that choice in a dedicated <selected_buyer> block containing only "
        "the digit 1 or 2. Follow the required <mental_model> / <message> format and include "
        "one complete <contract>...</contract> JSON block in <message>."
    )

    results = {
        "task": "Task8_s4_headphones_speaker_bundle_negotiation",
        "timestamp": datetime.now().isoformat(),
        "user_requirement": user_requirement,
        "user_profile": user_profile,
        "status": "unknown",
        "success": False,
        "error": None,
    }

    while not done:
        current_round = observation.get("current_round", 0)

        if current_round == 0:
            buyer1_action = buyer1.respond(
                conversation_history=observation["conversation_history_buyer1"],
                current_state=observation,
            )
            buyer2_action = buyer2.respond(
                conversation_history=observation["conversation_history_buyer2"],
                current_state=observation,
            )

            updated_conversation_history_buyer1 = observation["conversation_history_buyer1"].copy()
            updated_conversation_history_buyer2 = observation["conversation_history_buyer2"].copy()

            if buyer1_action:
                updated_conversation_history_buyer1.append({
                    "role": "buyer",
                    "content": buyer1_action,
                    "round": current_round,
                })
            if buyer2_action:
                updated_conversation_history_buyer2.append({
                    "role": "buyer",
                    "content": buyer2_action,
                    "round": current_round,
                })

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
            buyer_action = buyer1_action if selected_buyer == 1 else buyer2_action
        else:
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

            if selected_buyer == 1:
                conversation_history = observation["conversation_history_buyer1"]
            else:
                conversation_history = observation["conversation_history_buyer2"]

            updated_conversation_history = conversation_history.copy()
            if seller_action:
                updated_conversation_history.append({
                    "role": "seller",
                    "content": seller_action,
                    "round": current_round,
                })

            if selected_buyer == 1:
                buyer_action = buyer1.respond(
                    conversation_history=updated_conversation_history,
                    current_state=observation,
                )
            else:
                buyer_action = buyer2.respond(
                    conversation_history=updated_conversation_history,
                    current_state=observation,
                )

        observation, reward, terminated, truncated, info = env.step(
            selected_buyer=selected_buyer,
            buyer_action=buyer_action,
            seller_action=seller_action,
        )
        done = terminated or truncated

        env.render()
        sys.stdout.flush()

        if "step_buyer1_reward" in info or "step_buyer2_reward" in info or "step_seller_reward" in info:
            print(f"\n[Step Rewards] ", end="")
            if "step_buyer1_reward" in info:
                print(f"Buyer1: {info['step_buyer1_reward']:.3f}", end="")
            if "step_buyer2_reward" in info:
                if "step_buyer1_reward" in info:
                    print(f" | ", end="")
                print(f"Buyer2: {info['step_buyer2_reward']:.3f}", end="")
            if "step_seller_reward" in info:
                if "step_buyer1_reward" in info or "step_buyer2_reward" in info:
                    print(f" | ", end="")
                print(f"Seller: {info['step_seller_reward']:.3f}", end="")
            print()

            round_cost = -info["round"]
            weights = env.reward_weights

            if "step_buyer1_reward" in info:
                buyer1_price = info.get("buyer1_price")
                if buyer1_price is not None and env.buyer1_max_price is not None:
                    buyer1_savings = env.buyer1_max_price - buyer1_price
                    print(
                        f"  Buyer1 Step Reward = buyer_savings({buyer1_savings:.2f} * {weights['buyer_savings']:.2f}) + "
                        f"round_cost({round_cost:.2f} * {weights['time_cost']:.2f}) = {info['step_buyer1_reward']:.2f} "
                        f"(buyer1_max={env.buyer1_max_price}, buyer1_price={buyer1_price:.2f}, round={info['round']})"
                    )
                else:
                    print(
                        f"  Buyer1 Step Reward = round_cost({round_cost:.2f} * {weights['time_cost']:.2f}) = "
                        f"{round_cost * weights['time_cost']:.2f} (buyer1_price not specified, round={info['round']})"
                    )

            if "step_buyer2_reward" in info:
                buyer2_price = info.get("buyer2_price")
                if buyer2_price is not None and env.buyer2_max_price is not None:
                    buyer2_savings = env.buyer2_max_price - buyer2_price
                    print(
                        f"  Buyer2 Step Reward = buyer_savings({buyer2_savings:.2f} * {weights['buyer_savings']:.2f}) + "
                        f"round_cost({round_cost:.2f} * {weights['time_cost']:.2f}) = {info['step_buyer2_reward']:.2f} "
                        f"(buyer2_max={env.buyer2_max_price}, buyer2_price={buyer2_price:.2f}, round={info['round']})"
                    )
                else:
                    print(
                        f"  Buyer2 Step Reward = round_cost({round_cost:.2f} * {weights['time_cost']:.2f}) = "
                        f"{round_cost * weights['time_cost']:.2f} (buyer2_price not specified, round={info['round']})"
                    )

            if "step_seller_reward" in info:
                seller_price = None
                if info.get("current_selected_buyer") == 1:
                    seller_price = info.get("seller_price_buyer1")
                elif info.get("current_selected_buyer") == 2:
                    seller_price = info.get("seller_price_buyer2")

                if seller_price is not None and env.seller_min_price is not None:
                    seller_profit = seller_price - env.seller_min_price
                    print(
                        f"  Seller Step Reward = seller_profit({seller_profit:.2f} * {weights['seller_profit']:.2f}) + "
                        f"round_cost({round_cost:.2f} * {weights['time_cost']:.2f}) = {info['step_seller_reward']:.2f} "
                        f"(seller_price={seller_price:.2f}, seller_min={env.seller_min_price}, round={info['round']})"
                    )
                else:
                    print(
                        f"  Seller Step Reward = round_cost({round_cost:.2f} * {weights['time_cost']:.2f}) = "
                        f"{round_cost * weights['time_cost']:.2f} (seller_price not specified, round={info['round']})"
                    )

        if done:
            env._print_global_score_details()
            env._print_buyer_score_details()
            env._print_seller_score_details()

            print("\n" + "=" * 60)
            print("Negotiation Ended")
            print("=" * 60)
            print(f"Status: {info['status']}")
            if info.get("selected_buyer"):
                print(f"Final Selected Buyer: Buyer {info['selected_buyer']}")
                print(f"Final Deal Total Price: ${info.get('final_deal_price', 0):.2f}")
            if info.get("agreed_contract") is not None:
                print(f"Final Contract: {info['agreed_contract']}")
            buyer1_price = info.get("buyer1_price", 0) or 0
            seller_price_buyer1 = info.get("seller_price_buyer1", 0) or 0
            buyer2_price = info.get("buyer2_price", 0) or 0
            seller_price_buyer2 = info.get("seller_price_buyer2", 0) or 0
            print(f"Buyer1 Total Prices: Buyer=${buyer1_price:.2f} | Seller=${seller_price_buyer1:.2f}")
            print(f"Buyer2 Total Prices: Buyer=${buyer2_price:.2f} | Seller=${seller_price_buyer2:.2f}")
            actual_rounds = info["round"]
            print(f"Total Rounds: {actual_rounds}")
            print(f"Global Reward: {reward:.3f}")
            if "buyer1_reward" in info:
                print(f"Buyer1 Reward: {info['buyer1_reward']:.3f}")
            if "buyer2_reward" in info:
                print(f"Buyer2 Reward: {info['buyer2_reward']:.3f}")
            if "seller_reward" in info:
                print(f"Seller Reward: {info['seller_reward']:.3f}")
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
            results.update({
                "status": info.get("status", "unknown"),
                "success": terminated,
                "selected_buyer": info.get("selected_buyer"),
                "final_deal_price": info.get("final_deal_price"),
                "buyer1_price": info.get("buyer1_price"),
                "buyer2_price": info.get("buyer2_price"),
                "seller_price_buyer1": info.get("seller_price_buyer1"),
                "seller_price_buyer2": info.get("seller_price_buyer2"),
                "agreed_contract": info.get("agreed_contract"),
                "total_rounds": info.get("round", 0),
                "total_reward": float(reward) if reward is not None else None,
                "buyer1_reward": info.get("buyer1_reward"),
                "buyer2_reward": info.get("buyer2_reward"),
                "seller_reward": info.get("seller_reward"),
                "global_score": info.get("global_score"),
                "buyer_score": info.get("buyer_score"),
                "seller_score": info.get("seller_score"),
                "termination_reason": info.get("termination_reason"),
                "elapsed_time": elapsed_time,
                "buyer1_max_price": buyer1_max_price,
                "buyer2_max_price": buyer2_max_price,
                "seller_min_price": seller_min_price,
                "buyer_contract_configs": buyer_contract_configs,
                "product_info": product_info,
                "model": get_model_name(model),
            })
            break

    env.close()
    print("\nNegotiation completed!")

    if "elapsed_time" not in results:
        results["elapsed_time"] = time.time() - start_time

    try:
        results_dir = Path(project_root) / "agenticpay" / "results" / "multi_buyer_multi_products"
        results_dir.mkdir(parents=True, exist_ok=True)

        model_name_safe = get_model_name(model).replace("/", "_").replace("\\", "_").replace(":", "_")
        model_dir = results_dir / model_name_safe
        model_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = model_dir / f"batch_evaluation_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=True)

        summary_file = run_dir / "summary.json"
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        output_file = run_dir / "Task8_s4_headphones_speaker_bundle_output.txt"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write("Task8 Scenario 4: Headphones & Bluetooth Speaker Bundle - Sequential Two-Buyer Two-Product Negotiation Results\n")
            f.write("Category: Electronics\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Timestamp: {results['timestamp']}\n")
            f.write(f"Model: {results['model']}\n")
            f.write(f"User Requirement: {results['user_requirement']}\n")
            f.write(f"User Profile: {results['user_profile']}\n\n")
            f.write(f"Status: {results['status']}\n")
            f.write(f"Success: {results['success']}\n")
            f.write(f"Total Rounds: {results['total_rounds']}\n")
            elapsed_time = results.get("elapsed_time", 0)
            f.write(f"Elapsed Time: {elapsed_time:.2f}s\n\n")
            if results.get("selected_buyer"):
                f.write(f"Final Selected Buyer: Buyer {results['selected_buyer']}\n")
                f.write(f"Final Deal Total Price: ${results.get('final_deal_price', 0):.2f}\n\n")
            if results.get("agreed_contract") is not None:
                f.write(f"Final Contract: {results['agreed_contract']}\n\n")
            f.write("Final Prices (bundle total):\n")
            f.write(
                f"  Buyer1: Buyer=${results['buyer1_price']:.2f} | Seller=${results['seller_price_buyer1']:.2f}"
                if results.get("buyer1_price") is not None and results.get("seller_price_buyer1") is not None
                else "  Buyer1: Not specified"
            )
            f.write("\n")
            f.write(
                f"  Buyer2: Buyer=${results['buyer2_price']:.2f} | Seller=${results['seller_price_buyer2']:.2f}"
                if results.get("buyer2_price") is not None and results.get("seller_price_buyer2") is not None
                else "  Buyer2: Not specified"
            )
            f.write("\n\n")
            pi = results.get("product_info", {})
            f.write("Products:\n")
            if "products" in pi:
                for i, p in enumerate(pi["products"], 1):
                    f.write(f"  {i}. {p.get('name', 'N/A')} by {p.get('brand', 'N/A')} - ${p.get('price', 0):.2f}\n")
                total_price = sum(p.get("price", 0) for p in pi.get("products", []))
                f.write(f"  Total Product Price: ${total_price:.2f}\n")
            f.write("\n")
            f.write("Rewards:\n")
            if results.get("total_reward") is not None:
                f.write(f"  Total Reward: {results['total_reward']:.3f}\n")
            if results.get("buyer1_reward") is not None:
                f.write(f"  Buyer1 Reward: {results['buyer1_reward']:.3f}\n")
            if results.get("buyer2_reward") is not None:
                f.write(f"  Buyer2 Reward: {results['buyer2_reward']:.3f}\n")
            if results.get("seller_reward") is not None:
                f.write(f"  Seller Reward: {results['seller_reward']:.3f}\n")
            f.write("\n")
            f.write("Scores:\n")
            if results.get("global_score") is not None:
                f.write(f"  Global Score: {results['global_score']:.3f}\n")
            if results.get("buyer_score") is not None:
                f.write(f"  Buyer Score: {results['buyer_score']:.3f}\n")
            if results.get("seller_score") is not None:
                f.write(f"  Seller Score: {results['seller_score']:.3f}\n")
            f.write("\n")
            if results.get("termination_reason"):
                f.write(f"Termination Reason: {results['termination_reason']}\n")
            if results.get("error"):
                f.write(f"\nError: {results['error']}\n")

        print(f"\nResults saved to: {run_dir}")
        print(f"  - Summary JSON: {summary_file}")
        print(f"  - Output Text: {output_file}")
    except Exception as e:
        print(f"\nWarning: Failed to save results: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Task8 Scenario 4: Headphones & Bluetooth Speaker Bundle - Sequential Two-Buyer Two-Product Negotiation (image + text)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name to use. If not provided, uses default model.",
    )
    args = parser.parse_args()
    main(model_name=args.model)
