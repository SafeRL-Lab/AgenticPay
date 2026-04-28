"""Task5 Scenario 1: Beauty Product Bundle - Sequential Two-Buyer Two-Seller Negotiation (2 products, image + text)

Same two-item bundle and two third-party offers: product listing has no per-seller identity; two sellers
each have a different confidential floor for the **total** price. Two buyers each pick one seller per
round (structured `<selected_seller>` routing, aligned with multi_buyer_multi_seller Task5).
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
    if hasattr(model, "model"):
        return model.model
    elif hasattr(model, "model_id"):
        return model.model_id
    elif hasattr(model, "model_path"):
        model_path = model.model_path
        return os.path.basename(model_path) if model_path else str(model)
    else:
        model_str = str(model)
        if "model=" in model_str:
            try:
                return model_str.split("model=")[1].split(")")[0]
            except Exception:
                return model_str
        return model_str


def _run_buyer_routing(buyer, combined_history: list, observation: dict, routing_instruction: str):
    """Structured ``<selected_seller>`` + retries + random fallback (aligned with multi_buyer_multi_seller Task5)."""
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
    """Main function: sequential two-buyer two-seller negotiation for a fixed two-product bundle (total price)."""
    print("Initializing model...")

    api_key = os.getenv("OPENAI_API_KEY") or OPENAI_API_KEY
    if not api_key:
        print("Warning: OPENAI_API_KEY not set. Please set it to use OpenAI models.")
        print("You can set it with: export OPENAI_API_KEY='your-key-here'")
        return

    model_name = model_name or "gpt-5.4"
    model = OpenAIVLM(model=model_name, api_key=api_key)

    print(f"✓ Successfully initialized: {model}")

    # Same two-SKU bundle from two offers: each seller has a different confidential floor (total USD)
    print("Creating agents...")
    buyer1_max_price = 22.51  # Buyer 1 max WTP for the bundle (confidential; lower than buyer 2)
    buyer2_max_price = 24.39  # Buyer 2 max WTP (confidential)
    seller1_min_price = 20.89  # Seller 1 floor (confidential; higher than seller 2)
    seller2_min_price = 18.71  # Seller 2 floor (confidential)

    buyer1 = BuyerAgent(model=model, name="Buyer1", buyer_max_price=buyer1_max_price)
    buyer2 = BuyerAgent(model=model, name="Buyer2", buyer_max_price=buyer2_max_price)
    seller1 = SellerAgent(model=model, name="Seller1", seller_min_price=seller1_min_price)
    seller2 = SellerAgent(model=model, name="Seller2", seller_min_price=seller2_min_price)

    print("Creating sequential multi-buyer multi-seller two-product negotiation environment...")
    env = Task3SequentialTwoBuyerTwoSellerTwoProductNegotiation(
        buyer1_agent=buyer1,
        buyer2_agent=buyer2,
        seller1_agent=seller1,
        seller2_agent=seller2,
        max_rounds=max_rounds,
        initial_seller1_price=36.0,  # Opening ask (total) — same items, different listing
        initial_seller2_price=37.5,  # Opening ask (total) — same items, different listing
        buyer1_max_price=buyer1_max_price,
        buyer2_max_price=buyer2_max_price,
        seller1_min_price=seller1_min_price,
        seller2_min_price=seller2_min_price,
        environment_info={
            "platform": "Amazon",
            "market_type": "B2C",
            "note": "Multiple third-party offers exist for the same two-SKU cart; prices are bundle totals.",
        },
        price_tolerance=price_tolerance,
        reward_weights=reward_weights,
    )

    user_profile = None
    print(f"User Profile: {user_profile}")

    # Concise English user query: exactly two products (simulated search / assistant request)
    user_requirement = "I want these two, new: Turquoise Glass eyeshadow and NOU Oliban men's EDT."
    print(f"Using default requirement: {user_requirement}")

    product1_image_url = "https://m.media-amazon.com/images/I/41IiEBGouZL.jpg"
    product2_image_url = "https://m.media-amazon.com/images/I/51gDhcURgKL.jpg"

    product_info = {
        "products": [
            {
                "name": "Maybelline New York Expert Wear Eyeshadow Singles, 130s Turquoise Glass Perfect Pastels, 0.09 Ounce",
                "condition": "New",
                "brand": "Maybelline New York",
                "shade": "130s Turquoise Glass Perfect Pastels",
                "size": "0.09 Ounce",
                "price": 7.50,
                "list_price": 7.50,
                "original_price": 7.98,
                "product_category": "Beauty & Personal Care › Makeup › Eyes › Eyeshadow",
                "average_rating": 4.2,
                "total_reviews": 54,
                "full_description": "Easy to use. Lots to choose. All-day crease-proof wear. Rich, velvety textures. Glides on effortlessly with superior smoothness.",
                "asin": "B0046VILG4",
                "image_url": product1_image_url,
            },
            {
                "name": "NOU Oliban Eau de Toilette for Men, Woody Oriental, 1.7 Fl Oz",
                "condition": "New",
                "brand": "NOU",
                "price": 21.95,
                "list_price": 21.95,
                "original_price": 21.95,
                "product_category": "Beauty & Personal Care › Fragrance",
                "average_rating": 4.0,
                "total_reviews": 6,
                "full_description": "Woody oriental EDT infused with essential oils; notes include elemi, olibanum, patchouli, sandalwood, leather, and vanilla.",
                "asin": "B08XQWJX8P",
                "image_url": product2_image_url,
            },
        ]
    }

    print("\n" + "=" * 60)
    print("Starting new sequential negotiation (two products, total price; two buyers, two sellers)...")
    print("=" * 60)

    observation, info = env.reset(
        user_requirement=user_requirement,
        product_info=product_info,
        user_profile=user_profile,
    )

    done = False
    start_time = time.time()

    results = {
        "task": "Task5_s1_beauty_product_bundle_negotiation",
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
            "the digit 1 or 2. Then put only your negotiation text in <message>. "
            "The price you discuss is the **total** for both products in the cart."
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
                conversation_history_b1s1.append(
                    {"role": "buyer", "content": buyer1_action, "round": current_round}
                )
        else:
            conversation_history_b1s2 = observation["conversation_history_b1s2"].copy()
            if buyer1_action:
                current_round = observation.get("current_round", 0)
                conversation_history_b1s2.append(
                    {"role": "buyer", "content": buyer1_action, "round": current_round}
                )

        if buyer2_selected_seller == 1:
            conversation_history_b2s1 = observation["conversation_history_b2s1"].copy()
            if buyer2_action:
                current_round = observation.get("current_round", 0)
                conversation_history_b2s1.append(
                    {"role": "buyer", "content": buyer2_action, "round": current_round}
                )
        else:
            conversation_history_b2s2 = observation["conversation_history_b2s2"].copy()
            if buyer2_action:
                current_round = observation.get("current_round", 0)
                conversation_history_b2s2.append(
                    {"role": "buyer", "content": buyer2_action, "round": current_round}
                )

        seller1_action_buyer1 = None
        seller1_action_buyer2 = None
        seller2_action_buyer1 = None
        seller2_action_buyer2 = None

        if buyer1_selected_seller == 1:
            seller1_action_buyer1 = seller1.respond(
                conversation_history=conversation_history_b1s1,
                current_state=observation,
            )
        elif buyer1_selected_seller == 2:
            seller2_action_buyer1 = seller2.respond(
                conversation_history=conversation_history_b1s2,
                current_state=observation,
            )

        if buyer2_selected_seller == 1:
            seller1_action_buyer2 = seller1.respond(
                conversation_history=conversation_history_b2s1,
                current_state=observation,
            )
        elif buyer2_selected_seller == 2:
            seller2_action_buyer2 = seller2.respond(
                conversation_history=conversation_history_b2s2,
                current_state=observation,
            )

        observation, reward, terminated, truncated, info = env.step(
            buyer1_selected_seller=buyer1_selected_seller,
            buyer2_selected_seller=buyer2_selected_seller,
            buyer1_action=buyer1_action,
            buyer2_action=buyer2_action,
            seller1_action_buyer1=seller1_action_buyer1,
            seller1_action_buyer2=seller1_action_buyer2,
            seller2_action_buyer1=seller2_action_buyer1,
            seller2_action_buyer2=seller2_action_buyer2,
        )
        done = terminated or truncated

        env.render()
        sys.stdout.flush()

        if (
            "step_buyer1_reward" in info
            or "step_buyer2_reward" in info
            or "step_seller1_reward" in info
            or "step_seller2_reward" in info
        ):
            print(f"\n[Step Rewards] ", end="")
            if "step_buyer1_reward" in info:
                print(f"Buyer1: {info['step_buyer1_reward']:.3f}", end="")
            if "step_buyer2_reward" in info:
                if "step_buyer1_reward" in info:
                    print(" | ", end="")
                print(f"Buyer2: {info['step_buyer2_reward']:.3f}", end="")
            if "step_seller1_reward" in info:
                if "step_buyer1_reward" in info or "step_buyer2_reward" in info:
                    print(" | ", end="")
                print(f"Seller1: {info['step_seller1_reward']:.3f}", end="")
            if "step_seller2_reward" in info:
                if (
                    "step_buyer1_reward" in info
                    or "step_buyer2_reward" in info
                    or "step_seller1_reward" in info
                ):
                    print(" | ", end="")
                print(f"Seller2: {info['step_seller2_reward']:.3f}", end="")
            print()

            round_cost = -info["round"]
            weights = env.reward_weights

            if "step_buyer1_reward" in info:
                buyer_price = None
                if info.get("buyer1_selected_seller") == 1:
                    buyer_price = info.get("b1s1_buyer_price")
                elif info.get("buyer1_selected_seller") == 2:
                    buyer_price = info.get("b1s2_buyer_price")

                if buyer_price is not None and env.buyer1_max_price is not None:
                    buyer_savings = env.buyer1_max_price - buyer_price
                    print(
                        f"  Buyer1 Step Reward = buyer_savings({buyer_savings:.2f} * {weights['buyer_savings']:.2f}) + "
                        f"round_cost({round_cost:.2f} * {weights['time_cost']:.2f}) = {info['step_buyer1_reward']:.2f} "
                        f"(buyer1_max={env.buyer1_max_price}, buyer_total_price={buyer_price:.2f}, round={info['round']})"
                    )
                else:
                    weighted_round_cost = round_cost * weights["time_cost"]
                    print(
                        f"  Buyer1 Step Reward = round_cost({round_cost:.2f} * {weights['time_cost']:.2f}) = {weighted_round_cost:.2f} "
                        f"(buyer_price not specified, round={info['round']})"
                    )

            if "step_buyer2_reward" in info:
                buyer_price = None
                if info.get("buyer2_selected_seller") == 1:
                    buyer_price = info.get("b2s1_buyer_price")
                elif info.get("buyer2_selected_seller") == 2:
                    buyer_price = info.get("b2s2_buyer_price")

                if buyer_price is not None and env.buyer2_max_price is not None:
                    buyer_savings = env.buyer2_max_price - buyer_price
                    print(
                        f"  Buyer2 Step Reward = buyer_savings({buyer_savings:.2f} * {weights['buyer_savings']:.2f}) + "
                        f"round_cost({round_cost:.2f} * {weights['time_cost']:.2f}) = {info['step_buyer2_reward']:.2f} "
                        f"(buyer2_max={env.buyer2_max_price}, buyer_total_price={buyer_price:.2f}, round={info['round']})"
                    )
                else:
                    weighted_round_cost = round_cost * weights["time_cost"]
                    print(
                        f"  Buyer2 Step Reward = round_cost({round_cost:.2f} * {weights['time_cost']:.2f}) = {weighted_round_cost:.2f} "
                        f"(buyer_price not specified, round={info['round']})"
                    )

            if "step_seller1_reward" in info:
                seller1_price = None
                if info.get("buyer1_selected_seller") == 1 and info.get("b1s1_seller_price") is not None:
                    seller1_price = info.get("b1s1_seller_price")
                elif info.get("buyer2_selected_seller") == 1 and info.get("b2s1_seller_price") is not None:
                    seller1_price = info.get("b2s1_seller_price")
                if (
                    info.get("buyer1_selected_seller") == 1
                    and info.get("buyer2_selected_seller") == 1
                    and info.get("b1s1_seller_price") is not None
                    and info.get("b2s1_seller_price") is not None
                ):
                    seller1_price = max(info.get("b1s1_seller_price"), info.get("b2s1_seller_price"))

                if seller1_price is not None and env.seller1_min_price is not None:
                    seller1_profit = seller1_price - env.seller1_min_price
                    print(
                        f"  Seller1 Step Reward = seller_profit({seller1_profit:.2f} * {weights['seller_profit']:.2f}) + "
                        f"round_cost({round_cost:.2f} * {weights['time_cost']:.2f}) = {info['step_seller1_reward']:.2f} "
                        f"(seller1_total_price={seller1_price:.2f}, seller1_min={env.seller1_min_price}, round={info['round']})"
                    )
                else:
                    weighted_round_cost = round_cost * weights["time_cost"]
                    print(
                        f"  Seller1 Step Reward = round_cost({round_cost:.2f} * {weights['time_cost']:.2f}) = {weighted_round_cost:.2f} "
                        f"(seller1_price not specified, round={info['round']})"
                    )

            if "step_seller2_reward" in info:
                seller2_price = None
                if info.get("buyer1_selected_seller") == 2 and info.get("b1s2_seller_price") is not None:
                    seller2_price = info.get("b1s2_seller_price")
                elif info.get("buyer2_selected_seller") == 2 and info.get("b2s2_seller_price") is not None:
                    seller2_price = info.get("b2s2_seller_price")
                if (
                    info.get("buyer1_selected_seller") == 2
                    and info.get("buyer2_selected_seller") == 2
                    and info.get("b1s2_seller_price") is not None
                    and info.get("b2s2_seller_price") is not None
                ):
                    seller2_price = max(info.get("b1s2_seller_price"), info.get("b2s2_seller_price"))

                if seller2_price is not None and env.seller2_min_price is not None:
                    seller2_profit = seller2_price - env.seller2_min_price
                    print(
                        f"  Seller2 Step Reward = seller_profit({seller2_profit:.2f} * {weights['seller_profit']:.2f}) + "
                        f"round_cost({round_cost:.2f} * {weights['time_cost']:.2f}) = {info['step_seller2_reward']:.2f} "
                        f"(seller2_total_price={seller2_price:.2f}, seller2_min={env.seller2_min_price}, round={info['round']})"
                    )
                else:
                    weighted_round_cost = round_cost * weights["time_cost"]
                    print(
                        f"  Seller2 Step Reward = round_cost({round_cost:.2f} * {weights['time_cost']:.2f}) = {weighted_round_cost:.2f} "
                        f"(seller2_price not specified, round={info['round']})"
                    )

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

    env.close()
    print("\nSequential two-buyer two-seller two-product bundle negotiation completed!")

    if "elapsed_time" not in results:
        results["elapsed_time"] = time.time() - start_time

    try:
        results_dir = Path(project_root) / "agenticpay" / "results" / "multi_buyer_multi_products_multi_seller"
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
        output_file = run_dir / "Task5_s1_beauty_product_bundle_output.txt"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write(
                "Task5 Scenario 1: Beauty Product Bundle - Sequential Two-Buyer Two-Seller Two-Product "
                "Negotiation Results (image + text)\n"
            )
            f.write("Category: Daily Life Consumption\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Timestamp: {results['timestamp']}\n")
            f.write(f"Model: {results.get('model', '')}\n")
            f.write(f"User Requirement: {results['user_requirement']}\n")
            f.write(f"User Profile: {results['user_profile']}\n\n")
            f.write(f"Status: {results['status']}\n")
            f.write(f"Success: {results['success']}\n")
            f.write(f"Total Rounds: {results.get('total_rounds', 0)}\n")
            f.write(f"Elapsed Time: {results.get('elapsed_time', 0):.2f}s\n\n")
            if results.get("selected_buyer") and results.get("selected_seller"):
                f.write(f"Selected Deal: Buyer {results['selected_buyer']} - Seller {results['selected_seller']}\n")
                f.write(f"Final Deal Total Price: ${results.get('final_deal_price', 0):.2f}\n\n")
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
            pin = results.get("product_info", {}) or {}
            f.write("Products:\n")
            for i, p in enumerate(pin.get("products", []), 1):
                price_val = p.get("list_price", p.get("price", p.get("original_price", 0)))
                f.write(f"  {i}. {p.get('name', 'N/A')} by {p.get('brand', 'N/A')} — list ${float(price_val):.2f}\n")
            f.write("\n")
            f.write("Rewards:\n")
            if results.get("total_reward") is not None:
                f.write(f"  Total Reward: {results['total_reward']:.3f}\n")
            if results.get("buyer1_reward") is not None:
                f.write(f"  Buyer1 Reward: {results['buyer1_reward']:.3f}\n")
            if results.get("buyer2_reward") is not None:
                f.write(f"  Buyer2 Reward: {results['buyer2_reward']:.3f}\n")
            if results.get("seller1_reward") is not None:
                f.write(f"  Seller1 Reward: {results['seller1_reward']:.3f}\n")
            if results.get("seller2_reward") is not None:
                f.write(f"  Seller2 Reward: {results['seller2_reward']:.3f}\n")
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
        description="Task5 Scenario 1: Beauty product bundle - sequential 2x2 negotiation (2 products, image + text)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name. If not provided, uses default.",
    )
    args = parser.parse_args()
    main(model_name=args.model)
