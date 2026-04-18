"""Task16 Scenario 12: NYC Taxi Ride - Sequential Two-Seller Per One Product Negotiation

Buyer negotiates with two sellers offering the same Union Sq -> Lenox Hill West route under different service levels.
Buyer chooses one seller per round to negotiate with.
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
import re

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
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


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


def extract_seller_choice(buyer_response: str, observation: dict) -> int:
    """Extract seller choice from buyer's response."""
    response_lower = buyer_response.lower()

    if re.search(r'seller\s*2|second\s+seller|seller\s*two', response_lower):
        return 2
    elif re.search(r'seller\s*1|first\s+seller|seller\s*one', response_lower):
        return 1

    seller1_price = observation.get("seller1_price")
    seller2_price = observation.get("seller2_price")

    price_match = re.search(r'\$?(\d+\.?\d*)', buyer_response)
    if price_match:
        mentioned_price = float(price_match.group(1))
        if seller1_price is not None and abs(mentioned_price - seller1_price) < 5:
            return 1
        elif seller2_price is not None and abs(mentioned_price - seller2_price) < 5:
            return 2

    if seller1_price is not None and seller2_price is not None:
        return 1 if seller1_price <= seller2_price else 2
    elif seller1_price is not None:
        return 1
    elif seller2_price is not None:
        return 2

    return 1


def main(model_name=None):
    """Main function: Demonstrates sequential multi-seller negotiation flow with different products."""

    print("Initializing model...")

    api_key = os.getenv("OPENAI_API_KEY") or OPENAI_API_KEY or "token-abc123"
    openvlm_base_url = os.getenv("OPENAI_URL") or os.getenv("OPENVLM_BASE_URL", "http://localhost:8000/v1")
    openvlm_model = os.getenv("OPENVLM_MODEL", "openvlm")

    model = OpenAIVLM(
        model=model_name or openvlm_model,
        api_key=api_key,
        base_url=openvlm_base_url,
    )

    print(f"✓ Successfully initialized: {model}")

    print("Creating agents...")
    buyer_max_price = 20.58
    seller1_min_price = 17.38
    seller2_min_price = 18.88

    buyer = BuyerAgent(model=model, buyer_max_price=buyer_max_price)
    seller1 = SellerAgent(model=model, seller_min_price=seller1_min_price)
    seller2 = SellerAgent(model=model, seller_min_price=seller2_min_price)

    print("Creating sequential multi-seller negotiation environment...")
    env = Task3SequentialTwoSellerPerOneProductNegotiation(
        buyer_agent=buyer,
        seller1_agent=seller1,
        seller2_agent=seller2,
        max_rounds=max_rounds,
        initial_seller1_price=26.00,
        initial_seller2_price=28.00,
        buyer_max_price=buyer_max_price,
        seller1_min_price=seller1_min_price,
        seller2_min_price=seller2_min_price,
        environment_info={
            "platform": "NYC Street Hail",
            "market_type": "Service Negotiation (Two Drivers, One Route)",
            "traffic_context": "Dense Manhattan local roads",
        },
        price_tolerance=price_tolerance,
        reward_weights=reward_weights,
    )

    user_profile = "Price-sensitive rider comparing two driver options for the same NYC route. Prioritizes a transparent all-in final fare with no hidden fees."
    print(f"User Profile: {user_profile}")

    user_requirement = "I need a direct ride from Union Sq to Lenox Hill West. The trip is under two miles, so I expect a reasonable all-in flat fare with no surprise charges."
    print(f"Using default requirement: {user_requirement}")

    print("\n" + "="*60)
    print("Starting new sequential negotiation (Seller1: standard taxi, Seller2: premium taxi)...")
    print("="*60)

    product_image_url = os.path.join(
        project_root,
        "agenticpay",
        "data",
        "NYC_taxi_data",
        "img",
        "yellow_tripdata_2026-02_sample_10",
        "image_1.png",
    )

    observation, info = env.reset(
        user_requirement=user_requirement,
        seller1_product_info={
            "name": "Standard Yellow Taxi Ride: Union Sq -> Lenox Hill West",
            "condition": "Metered standard yellow taxi service",
            "brand": "NYC Yellow Taxi (Standard Driver)",
            "original_price": 26.00,
            "price": 26.00,
            "availability_status": "Available now",
            "product_category": "Transportation & Mobility > Taxi Service",
            "average_rating": 4.7,
            "total_reviews": 128,
            "seller_name": "Standard Driver",
            "service_type": "Point-to-point taxi ride",
            "pickup_location": "Union Sq, Manhattan, New York, NY",
            "dropoff_location": "Lenox Hill West, Manhattan, New York, NY",
            "trip_distance_miles": 1.93,
            "historical_trip_time": "Around 10-15 minutes (traffic dependent)",
            "VendorID": 7,
            "RatecodeID": 1,
            "Passenger Count": 1,
            "Historical Fare Amount": 11.4,
            "Historical Total Amount": 20.58,
            "mandatory_surcharges": ['$2.50 (Congestion Surcharge)', '$0.75 (CBD Congestion Fee)', '$1.00 (Improvement Surcharge)', '$0.50 (MTA State Tax)'],
            "tolls": 0.0,
            "full_description": "Standard taxi option for the route from Union Sq to Lenox Hill West. Final negotiated price must be all-in and include required NYC surcharges.",
            "image_url": product_image_url,
        },
        seller2_product_info={
            "name": "Premium Yellow Taxi Ride: Union Sq -> Lenox Hill West",
            "condition": "Premium ride with cleaner vehicle and faster pick-up",
            "brand": "NYC Yellow Taxi (Premium Driver)",
            "original_price": 28.00,
            "price": 28.00,
            "availability_status": "Available now",
            "product_category": "Transportation & Mobility > Taxi Service",
            "average_rating": 4.9,
            "total_reviews": 92,
            "seller_name": "Premium Driver",
            "service_type": "Point-to-point taxi ride",
            "pickup_location": "Union Sq, Manhattan, New York, NY",
            "dropoff_location": "Lenox Hill West, Manhattan, New York, NY",
            "trip_distance_miles": 1.93,
            "historical_trip_time": "Around 10-15 minutes (traffic dependent)",
            "VendorID": 7,
            "RatecodeID": 1,
            "Passenger Count": 1,
            "Historical Fare Amount": 11.4,
            "Historical Total Amount": 20.58,
            "mandatory_surcharges": ['$2.50 (Congestion Surcharge)', '$0.75 (CBD Congestion Fee)', '$1.00 (Improvement Surcharge)', '$0.50 (MTA State Tax)'],
            "tolls": 0.0,
            "full_description": "Premium taxi option for the same route with better service quality. Final negotiated price must be all-in and include required NYC surcharges.",
            "image_url": product_image_url,
        },
        user_profile=user_profile,
    )

    done = False
    start_time = time.time()

    results = {
        "task": "Task16_s12_taxi_2_multi_products_multi_seller",
        "category": "Daily Life Consumption",
        "scenario": "Union Sq to Lenox Hill West taxi ride fare negotiation (two service levels)",
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
                "instruction": "You are negotiating with two sellers offering the same route under different service levels. Each round, choose ONE seller (1 or 2) and provide your negotiation message clearly."
            }
        )

        selected_seller = extract_seller_choice(buyer_response, observation)
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
                if info['selected_seller'] == 1:
                    product_info = info.get('seller1_product_info', {})
                    print(f"Selected Product: {product_info.get('name', 'N/A')} by {product_info.get('brand', 'N/A')}")
                elif info['selected_seller'] == 2:
                    product_info = info.get('seller2_product_info', {})
                    print(f"Selected Product: {product_info.get('name', 'N/A')} by {product_info.get('brand', 'N/A')}")
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
            f.write("Task16 Scenario 12: NYC Taxi Ride - Sequential Two-Seller Per One Product Negotiation Results\n")
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
                selected_product = results.get('seller1_product_info', {}) if results['selected_seller'] == 1 else results.get('seller2_product_info', {})
                f.write(f"Selected Product: {selected_product.get('name', 'N/A')} by {selected_product.get('brand', 'N/A')}\n\n")
            f.write("Final Prices:\n")
            f.write(f"  Seller1 - Seller Price: ${results['seller1_price']:.2f}" if results.get('seller1_price') is not None else "  Seller1 - Seller Price: Not specified")
            f.write("\n")
            f.write(f"  Seller1 - Buyer Price: ${results['buyer_price_seller1']:.2f}" if results.get('buyer_price_seller1') is not None else "  Seller1 - Buyer Price: Not specified")
            f.write("\n")
            f.write(f"  Seller2 - Seller Price: ${results['seller2_price']:.2f}" if results.get('seller2_price') is not None else "  Seller2 - Seller Price: Not specified")
            f.write("\n")
            f.write(f"  Seller2 - Buyer Price: ${results['buyer_price_seller2']:.2f}" if results.get('buyer_price_seller2') is not None else "  Seller2 - Buyer Price: Not specified")
            f.write("\n\n")
            f.write("Services:\n")
            seller1_product = results.get('seller1_product_info', {})
            p1 = seller1_product.get('price') or seller1_product.get('original_price', 0)
            f.write(f"  Seller1 Service: {seller1_product.get('name', 'N/A')} by {seller1_product.get('brand', 'N/A')} (${p1:.2f})\n")
            seller2_product = results.get('seller2_product_info', {})
            p2 = seller2_product.get('price') or seller2_product.get('original_price', 0)
            f.write(f"  Seller2 Service: {seller2_product.get('name', 'N/A')} by {seller2_product.get('brand', 'N/A')} (${p2:.2f})\n")
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
    parser = argparse.ArgumentParser(description="Task16 Scenario 12: NYC Taxi Ride - Sequential Two-Seller Per One Product Negotiation")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="OpenVLM model name. Set OPENAI_URL/OPENVLM_BASE_URL for API endpoint, OPENVLM_MODEL for default model name."
    )
    args = parser.parse_args()
    main(model_name=args.model)
