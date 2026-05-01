"""Task27 Scenario 23: Sydney — Sequential Two-Buyer Two-Listing Rental Bundle

Several landlords may list the same two units; negotiation is for the combined monthly total (long-term lease framing).
Aligned with Task5: seller routes via ``<selected_buyer>``; contract mode uses the larger of the two buyers' ``Z_max`` as ``z_market`` for normalized scores.
Listing copy derived from ``airbnb_embeddings_sample10.jsonl`` (``_id`` 14096512, 16289600).
Category: Real Estate — Residential Rentals
"""

import os
import sys
import json
import time
import random
import argparse
from pathlib import Path
from datetime import datetime

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, project_root)

from agenticpay.envs.multi_buyer_multi_products.Task3_sequential_two_buyer_two_product_negotiation import Task3SequentialTwoBuyerTwoProductNegotiation
from agenticpay.agents.buyer_agent import BuyerAgent
from agenticpay.agents.seller_agent import SellerAgent
from agenticpay.models.openai_vlm import OpenAIVLM
from agenticpay.examples.config import reward_weights, max_rounds, price_tolerance, OPENAI_API_KEY


def get_model_name(model):
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
    print("Initializing model...")

    api_key = os.getenv("OPENAI_API_KEY") or OPENAI_API_KEY
    if not api_key:
        print("Warning: OPENAI_API_KEY not set. Please set it to use OpenAI models.")
        print("You can set it with: export OPENAI_API_KEY='your-key-here'")
        return

    model_name = model_name or "gpt-5.4"
    model = OpenAIVLM(model=model_name, api_key=api_key)

    print(f"✓ Successfully initialized: {model}")

    # Anchor ~$6730/month from listing totals; confidential band is materially lower (~64–65% floor vs anchor · ~76–79% cap).
    print("Creating agents...")
    product_request = (
        "I want Mezzos Studio in Sydney CBD and Whole sunny apartment near Bondi Beach—one monthly rent. "
        "I also prefer interior shots where main living areas show hard surface plank-style flooring rather than wall-to-wall carpet."
    )
    buyer1_contract_config = {
        "contrainfo": {
            "product_request": product_request,
            "initial_contract_status": (
                "No combined monthly bundle rent, lease length in months, utilities-included boolean, or user product preference match "
                "has been fixed before negotiation starts."
            ),
            "contract_completion_requirement": (
                "A valid offer must set price (total USD/month for both units), "
                "continuous_terms.lease_months, discrete_terms.include_utilities, and discrete_terms.user_product_preference."
            ),
        },
        "field_descriptions": {
            "price": (
                "Total combined monthly rent for both listings in the bundle, in US dollars "
                "(not nightly)."
            ),
            "continuous_terms.lease_months": (
                "Lease commitment length in months for the bundle (one number applies to the whole deal)."
            ),
            "discrete_terms.include_utilities": (
                "If true, quoted rent includes major utilities as bundled; if false, tenant pays utilities separately."
            ),
            "discrete_terms.user_product_preference": (
                "How well the bundle matches the buyer's stated preference for interior shots where main living areas show hard surface "
                "plank-style flooring rather than wall-to-wall carpet. Use `strong_match` when that preference is clearly satisfied, "
                "`partial_match` when it is only partly satisfied, and `mismatch_or_uncertain` when it is not satisfied or "
                "cannot be confirmed."
            ),
        },
        "continuous_bounds": {"lease_months": {"min": 1, "max": 24}},
        "discrete_options": {
            "include_utilities": [True, False],
            "user_product_preference": ["strong_match", "partial_match", "mismatch_or_uncertain"],
        },
        "buyer_preferences": {
            "v_base": 5034.0,
            "weight_descriptions": {
                "v_base": (
                    "Maximum acceptable bundled rent reservation before lease/utilities splits ($/month; confidential; materially below summed listing anchors)."
                ),
                "continuous_weights.lease_months": (
                    "$/month change per extra committed month; negative means you dislike long leases."
                ),
                "discrete_weights.include_utilities": (
                    "One-time $/month-style utility bump for bundled vs separate utility bills."
                ),
                "discrete_weights.user_product_preference": (
                    "How much each level of match to your stated product preference changes your utility, measured in dollars. "
                    "Positive numbers are good for you; negative numbers are bad for you."
                ),
            },
            "continuous_weights": {"lease_months": -11.0},
            "discrete_weights": {
                "include_utilities": {True: 128.0, False: 0.0},
                "user_product_preference": {
                    "strong_match": 0.30,
                    "partial_match": 0.12,
                    "mismatch_or_uncertain": -0.25,
                },
            },
        },
        "seller_preferences": {
            "c_base": 4792.0,
            "weight_descriptions": {
                "c_base": (
                    "Minimum acceptable bundled landlord reservation revenue before lease/utilities ($/month; confidential; below summed listing anchors)."
                ),
                "continuous_weights.lease_months": (
                    "$/month per extra committed month; positive means you value reduced vacancy risk."
                ),
                "discrete_weights.include_utilities": (
                    "Cost/annoyance of bundling utilities into rent (negative for include_utilities true)."
                ),
                "discrete_weights.user_product_preference": (
                    "How much each level of commitment to the buyer's stated product preference changes your utility, measured in dollars. "
                    "Stronger commitments carry a small nonzero risk or handling cost."
                ),
            },
            "continuous_weights": {"lease_months": 19.0},
            "discrete_weights": {
                "include_utilities": {True: -80.0, False: 0.0},
                "user_product_preference": {
                    "strong_match": -0.08,
                    "partial_match": -0.04,
                    "mismatch_or_uncertain": 0.01,
                },
            },
        },
    }
    buyer2_contract_config = json.loads(json.dumps(buyer1_contract_config))
    buyer2_contract_config["buyer_preferences"]["v_base"] = 5505.0
    buyer2_contract_config["buyer_preferences"]["continuous_weights"]["lease_months"] = -8.5
    buyer2_contract_config["buyer_preferences"]["discrete_weights"]["include_utilities"] = {
        True: 158.0,
        False: 0.0,
    }
    buyer2_contract_config["seller_preferences"]["c_base"] = 4200.0
    buyer2_contract_config["seller_preferences"]["continuous_weights"]["lease_months"] = 23.0
    buyer2_contract_config["seller_preferences"]["discrete_weights"]["include_utilities"] = {
        True: -93.0,
        False: 0.0,
    }
    buyer_contract_configs = {
        1: buyer1_contract_config,
        2: buyer2_contract_config,
    }
    buyer1_max_price = buyer1_contract_config["buyer_preferences"]["v_base"]
    buyer2_max_price = buyer2_contract_config["buyer_preferences"]["v_base"]
    seller_min_price = min(
        cfg["seller_preferences"]["c_base"] for cfg in buyer_contract_configs.values()
    )

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
            "platform": "Airbnb (listing-style; combined monthly lease)",
            "market_type": "Residential Rental",
            "availability_status": "Available for lease discussion.",
            "listing_age": "2019 scrape (jsonl): 14096512; 16289600 — airbnb_embeddings_sample10.jsonl",
            "bundle_context": (
                "Several landlords list this exact two-unit bundle; offers are for the combined monthly total. "
                "Each seller has a different internal floor for the pair; tenants only see listing facts, not owner identities."
            ),
            "buyer_contract_configs": buyer_contract_configs,
        },
        price_tolerance=price_tolerance,
        reward_weights=reward_weights,
    )

    user_profile = None
    print(f"User Profile: {user_profile}")

    product_info = {
        "products": [
            {
                "name": "Mezzos Studio in Sydney CBD",
                "brand": "Sydney, Australia",
                "price": 3250.0,
                "condition": "Fully furnished · Wi‑Fi (long-term monthly lease framing)",
                "color": "N/A",
                "size": "Apartment · entire home/apt · studio (0 BR) · 1 bath · accommodates 4 · listing 14096512",
                "original_price": 3250.0,
                "availability_status": "Available for lease discussion.",
                "product_category": "Real Estate › Rentals › Apartments › Studio",
                "average_rating": 4.45,
                "total_reviews": 221,
                "asin": "AIRBNB-14096512",
                "full_description": (
                    "Entire studio at Circular Quay in the CBD; furnished with Wi‑Fi. "
                    "Monthly component for bundle-pricing scenarios."
                ),
                "image_url": "https://a0.muscache.com/im/pictures/0c59647f-273c-4510-a1f3-eb8a3f6cc650.jpg?aki_policy=large",
            },
            {
                "name": "Whole sunny apartment near Bondi Beach",
                "brand": "Sydney, Australia",
                "price": 3480.0,
                "condition": "Furnishing completing per listing note (long-term monthly lease framing)",
                "color": "N/A",
                "size": "Apartment · entire home/apt · 2 BR · 1 bath · accommodates 4 · listing 16289600",
                "original_price": 3480.0,
                "availability_status": "Available for lease discussion.",
                "product_category": "Real Estate › Rentals › Apartments",
                "average_rating": 3.0,
                "total_reviews": 1,
                "asin": "AIRBNB-16289600",
                "full_description": (
                    "Sunny apartment between Bondi Junction and Bondi Beach; two double beds and equipped kitchen; "
                    "park nearby. Monthly component for bundle-pricing scenarios."
                ),
                "image_url": "https://a0.muscache.com/im/pictures/0c59647f-273c-4510-a1f3-eb8a3f6cc650.jpg?aki_policy=large",
            },
        ]
    }

    total_product_price = sum(p["price"] for p in product_info["products"])
    print(f"\nProducts (two-listing rental bundle — Sydney):")
    for i, p in enumerate(product_info["products"], 1):
        print(f"  {i}. {p['name']}: ${p['price']:.2f}")
    print(f"  Total Bundle Reference Sum: ${total_product_price:.2f}")

    user_requirement = product_request
    print(f"Using default requirement: {user_requirement}")

    print("\n" + "=" * 60)
    print("Starting sequential negotiation for the two-listing rental bundle...")
    print("=" * 60)

    observation, info = env.reset(
        user_requirement=user_requirement,
        product_info=product_info,
        user_profile=user_profile,
    )

    done = False
    start_time = time.time()

    routing_instruction = (
        "You are negotiating with two buyers for the SAME two-listing monthly-rent bundle; all prices are the "
        "TOTAL combined monthly rent for both units. "
        "Each round, choose exactly ONE buyer and output that choice in a dedicated <selected_buyer> block containing only "
        "the digit 1 or 2. Follow the required <mental_model> / <message> format and include "
        "one complete <contract>...</contract> JSON block in <message> (price, lease_months, include_utilities)."
    )

    results = {
        "task": "Task27_s23_rent_house_3",
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
            buyer1_price = info.get("buyer1_price", 0) or 0
            seller_price_buyer1 = info.get("seller_price_buyer1", 0) or 0
            buyer2_price = info.get("buyer2_price", 0) or 0
            seller_price_buyer2 = info.get("seller_price_buyer2", 0) or 0
            print(f"Buyer1 Total Prices: Buyer=${buyer1_price:.2f} | Seller=${seller_price_buyer1:.2f}")
            print(f"Buyer2 Total Prices: Buyer=${buyer2_price:.2f} | Seller=${seller_price_buyer2:.2f}")
            print(f"Total Rounds: {info['round']}")
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

            results.update({
                "status": info.get("status", "unknown"),
                "success": terminated,
                "selected_buyer": info.get("selected_buyer"),
                "final_deal_price": info.get("final_deal_price"),
                "buyer1_price": info.get("buyer1_price"),
                "buyer2_price": info.get("buyer2_price"),
                "seller_price_buyer1": info.get("seller_price_buyer1"),
                "seller_price_buyer2": info.get("seller_price_buyer2"),
                "total_rounds": info.get("round", 0),
                "total_reward": float(reward) if reward is not None else None,
                "buyer1_reward": info.get("buyer1_reward"),
                "buyer2_reward": info.get("buyer2_reward"),
                "seller_reward": info.get("seller_reward"),
                "global_score": info.get("global_score"),
                "buyer_score": info.get("buyer_score"),
                "seller_score": info.get("seller_score"),
                "termination_reason": info.get("termination_reason"),
                "elapsed_time": time.time() - start_time,
                "buyer1_max_price": buyer1_max_price,
                "buyer2_max_price": buyer2_max_price,
                "seller_min_price": seller_min_price,
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
        output_file = run_dir / "Task27_s23_rent_house_3_output.txt"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write("Task27 Scenario 23: Sydney Two-Listing Rental Bundle Results\n")
            f.write("Category: Real Estate — Residential Rentals\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Timestamp: {results['timestamp']}\n")
            f.write(f"Model: {results['model']}\n")
            f.write(f"User Requirement: {results['user_requirement']}\n")
            f.write(f"User Profile: {results['user_profile']}\n\n")
            f.write(f"Status: {results['status']}\n")
            f.write(f"Success: {results['success']}\n")
            f.write(f"Total Rounds: {results['total_rounds']}\n")
            f.write(f"Elapsed Time: {results.get('elapsed_time', 0):.2f}s\n\n")
            if results.get("selected_buyer"):
                f.write(f"Final Selected Buyer: Buyer {results['selected_buyer']}\n")
                f.write(f"Final Deal Total Price: ${results.get('final_deal_price', 0):.2f}\n\n")
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
                    f.write(f"  {i}. {p.get('name', 'N/A')} — {p.get('brand', 'N/A')} — ${p.get('price', 0):.2f}\n")
                f.write(f"  Total Product Price: ${sum(p.get('price', 0) for p in pi.get('products', [])):.2f}\n")
            f.write("\nRewards:\n")
            if results.get("total_reward") is not None:
                f.write(f"  Total Reward: {results['total_reward']:.3f}\n")
            if results.get("buyer1_reward") is not None:
                f.write(f"  Buyer1 Reward: {results['buyer1_reward']:.3f}\n")
            if results.get("buyer2_reward") is not None:
                f.write(f"  Buyer2 Reward: {results['buyer2_reward']:.3f}\n")
            if results.get("seller_reward") is not None:
                f.write(f"  Seller Reward: {results['seller_reward']:.3f}\n")
            f.write("\nScores:\n")
            if results.get("global_score") is not None:
                f.write(f"  Global Score: {results['global_score']:.3f}\n")
            if results.get("buyer_score") is not None:
                f.write(f"  Buyer Score: {results['buyer_score']:.3f}\n")
            if results.get("seller_score") is not None:
                f.write(f"  Seller Score: {results['seller_score']:.3f}\n")
            f.write("\n")
            if results.get("termination_reason"):
                f.write(f"Termination Reason: {results['termination_reason']}\n")
        print(f"\nResults saved to: {run_dir}")
        print(f"  - Summary JSON: {summary_file}")
        print(f"  - Output Text: {output_file}")
    except Exception as e:
        print(f"\nWarning: Failed to save results: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Task27 Scenario 23: Sydney Two-Listing Rental Bundle")
    parser.add_argument("--model", type=str, default=None, help="Model name.")
    args = parser.parse_args()
    main(model_name=args.model)
