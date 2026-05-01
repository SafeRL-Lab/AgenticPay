"""Task10 Scenario 6: Bookshelf & Wall Sconce Bundle - Two-Product Negotiation

Buyer negotiates for 4-Tier Ladder Bookshelf and Fanyate Wall Sconce bundle.
Bundle purchase with total price negotiation.
Category: Home & Kitchen / Tools & Home Improvement
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

from agenticpay import make  # Use registration system
from agenticpay.agents.buyer_agent import BuyerAgent
from agenticpay.agents.seller_agent import SellerAgent
from agenticpay.models.custom_llm import CustomLLM
from agenticpay.models.openai_vlm import OpenAIVLM
from agenticpay.models.qwen3_vl import Qwen3VL
from agenticpay.models.vllm_lm import VLLMLLM
from agenticpay.models.sglang_vlm import SGLangVLM

# Import configuration parameters
examples_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, examples_dir)
from config import reward_weights, buyer_reward_aggregation, seller_reward_aggregation, max_rounds, price_tolerance, OPENAI_API_KEY


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
    """Main function: Demonstrates two-product negotiation flow
    
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
    
    # Use OpenAIVLM (Vision Language Model) for bundle negotiation with product images (image + text)
    # Supports OpenAI API or OpenAI-compatible API (set OPENAI_URL for local OpenVLM etc.)
    model_name = model_name or "gpt-4o-mini"  # gpt-4o, gpt-4o-mini, gpt-4-vision-preview, etc.
    base_url = os.getenv("OPENAI_URL") or os.getenv("OPENVLM_BASE_URL")  # None = official OpenAI API
    model = OpenAIVLM(
        model=model_name,
        api_key=api_key,
        base_url=base_url,
    )

    print(f"✓ Successfully initialized: {model}")
    
    # Create Agents (set their respective bottom prices, this information is confidential, unknown to each other)
    # buyer_max_price and seller_min_price are confidential bundle reservation bounds below the public listing reference total.
    print("Creating agents...")
    product_request = (
        "I want a 4-tier black ladder bookshelf and the Fanyate 2-light oil-rubbed bronze sconce pack. "
        "I also prefer the bookshelf shelves to line up level along the front edge without visible sag."
    )
    # Multi-dimensional contract setup for reusable MAUT scoring in env.
    contract_config = {
        "contrainfo": {
            "product_request": product_request,
            "initial_contract_status": (
                "No total bundle price, delivery time, return policy, packaging option, or user product preference match "
                "has been selected or agreed before negotiation starts."
            ),
            "contract_completion_requirement": (
                "A valid offer must explicitly fill price, continuous_terms.delivery_days, "
                "discrete_terms.return_policy, discrete_terms.packaging, and "
                "discrete_terms.user_product_preference."
            ),
        },
        "field_descriptions": {
            "price": "The total amount of money the buyer pays for the whole two-product home goods bundle, measured in US dollars.",
            "continuous_terms.delivery_days": (
                "How many days the seller can take to deliver both home goods after the deal is made."
            ),
            "discrete_terms.return_policy": (
                "The return rule for the bundle order. `30_days` means the buyer can return the items within 30 days; "
                "`none` means the sale is final and returns are not allowed."
            ),
            "discrete_terms.packaging": (
                "The packaging used for shipment. `protective` means extra protection for the bookshelf parts and sconce glass shades; "
                "`standard` means normal packaging."
            ),
            "discrete_terms.user_product_preference": (
                "How well the bundle matches the buyer's stated preference for bookshelf shelves that line up level "
                "along the front edge without visible sag. Use `strong_match` when the preference is clearly satisfied, "
                "`partial_match` when it is only partly satisfied, and `mismatch_or_uncertain` when it is not "
                "satisfied or cannot be confirmed."
            ),
        },
        "continuous_bounds": {
            "delivery_days": {"min": 1, "max": 7}
        },
        "discrete_options": {
            "return_policy": ["30_days", "none"],
            "packaging": ["protective", "standard"],
            "user_product_preference": ["strong_match", "partial_match", "mismatch_or_uncertain"],
        },
        "buyer_preferences": {
            "v_base": 117.73,
            "weight_descriptions": {
                "v_base": (
                    "Your private maximum value for this two-product home goods bundle before delivery, return, and packaging terms, measured in dollars. "
                    "A lower price is better for you because every dollar paid reduces your utility by 1 dollar."
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
                    "How much each level of match to your stated product preference changes your utility, "
                    "measured in dollars. Positive numbers are good for you; negative numbers are bad for you."
                ),
            },
            "continuous_weights": {"delivery_days": -0.25},
            "discrete_weights": {
                "return_policy": {"30_days": 1.0, "none": -1.2},
                "packaging": {"protective": 0.9, "standard": -0.3},
                "user_product_preference": {
                    "strong_match": 0.30,
                    "partial_match": 0.12,
                    "mismatch_or_uncertain": -0.25,
                },
            },
        },
        "seller_preferences": {
            "c_base": 96.60,
            "weight_descriptions": {
                "c_base": (
                    "Your private minimum cost for fulfilling this two-product home goods bundle before delivery, return, and packaging terms, measured in dollars. "
                    "A higher price is better for you because every dollar received increases your utility by 1 dollar."
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
                    "How much each level of commitment to the buyer's stated product preference changes your "
                    "utility, measured in dollars. Stronger commitments carry a small nonzero risk or handling cost."
                ),
            },
            "continuous_weights": {"delivery_days": 0.20},
            "discrete_weights": {
                "return_policy": {"30_days": -1.4, "none": 1.0},
                "packaging": {"protective": -0.8, "standard": 0.3},
                "user_product_preference": {
                    "strong_match": -0.08,
                    "partial_match": -0.04,
                    "mismatch_or_uncertain": 0.01,
                },
            },
        },
    }
    buyer_max_price = contract_config["buyer_preferences"]["v_base"]  # Keep for backward-compatible price displays
    seller_min_price = contract_config["seller_preferences"]["c_base"]  # Keep for backward-compatible price displays
    buyer = BuyerAgent(model=model, name="Buyer1", buyer_max_price=buyer_max_price)
    seller = SellerAgent(model=model, name="Seller1", seller_min_price=seller_min_price)
    
    # Create environment using registration system
    print("Creating two-product negotiation environment...")
    env = make(
        "Task2_two_product_negotiation-v0",
        buyer_agent=buyer,
        seller_agent=seller,
        max_rounds=max_rounds,
        buyer_max_price=buyer_max_price,  # Buyer total max price (confidential, for both products)
        seller_min_price=seller_min_price,  # Seller total min price (confidential, for both products)
        environment_info={
            "platform": "Amazon",
            "market_type": "B2C",
            "contract_config": contract_config,
        },
        price_tolerance=price_tolerance,
    )
    
    # Create user profile (text description of personal preferences)
    user_profile = None
    print(f"User Profile: {user_profile}")
    
    # Define two products with their individual prices
    # Product 1: 4-Tier Bookshelf (from Task9_s6_bookshelf_negotiation example)
    # Product 2: Fanyate Wall Sconce (from sampled_products2.jsonl line 6)
    # The product_info should contain a list of two products
    product_info = {
        "products": [
            {
                "name": "4-Tier Ladder Bookshelf Organizer, Iron Open Bookcase Organizer (Black)",
                "price": 36.94,
                "condition": "New",
                "brand": "Brand: Kcelarec",
                "product_category": "Home & Kitchen › Furniture › Home Office Furniture › Bookcases",
                "average_rating": 5.0,
                "total_reviews": 1,
                "seller_name": "Kcelarec",
                "asin": "B088WSDHTW",
                "full_description": "If you are looking for a practical bookshelf, you can't miss this Widen 4 Tiers Bookshelf. This bookshelf is made of high quality material, which is stable, sturdy and durable. Its design of 4 tiers can hold a lot of books, and its strong bearing capacity can bear 44-88 lbs. You can put books in this bookshelf, and also place many other items like potting, decoration, etc. Made of high quality iron. Stable, sturdy and durable. Practical, design of 4 tiers can hold a lot of items. 44-88 lbs strong bearing capacity. Easy to install. Dimensions: (23.62 x 13.78 x 57.87) inches.",
                "image_url": "https://m.media-amazon.com/images/I/41Tbj+f2soL.jpg",
            },
            {
                "name": "Fanyate Antique Industrial Wall Sconce, 2-Light Bathroom Light Fixture Oil Rubbed Bronze Vanity Light with Clear Glass Shade Suitable for Bathroom Living Room Hallway ORB, 2 Pack",
                "price": 113.99,
                "condition": "New",
                "brand": "Visit the Fanyate Store",
                "product_category": "Tools & Home Improvement › Lighting & Ceiling Fans › Wall Lights › Wall Lamps & Sconces",
                "average_rating": 4.7,
                "total_reviews": 55,
                "seller_name": "Fanyate",
                "asin": "B0928LGTVF",
                "full_description": "【ANTIQUE INDUSTRIAL STYLE】Unique Oil Rubbed Bronze painting finished metal lamp body mated with clear glass shade, adding more antique and industrial atmosphere and bringing a quiet and comfortable feeling to your life. 【PRODUCT INSPECTION】The width of this light is 13.8'', the depth is 6.6,'' and the height is 9.8''. Compatible with E26 base bulb. The max wattage of the bulb is 60W. (Bulb is not included.) 【EASY INSTALLATION】Easy installation to save your time. The installation instruction and mounting screws are included in the package for your quick installation. 【APPLICABLE SPACE】These wall lights are suitable for any space you want to decorate. Not only suitable for bathroom, also living room, study, porch, kitchen, dining room, cafe, bar, bedroom, shop, lounge decoration. 【GORGEOUS SHOPPING EXPERIENCE】You can get not only good value from this lamp but also our services and a 1-year warranty that will guarantee your complete satisfaction with your purchase.",
                "image_url": "https://m.media-amazon.com/images/I/41icQciKVIS.jpg",
            },
        ]
    }
    
    # Calculate total product price
    total_product_price = sum(p["price"] for p in product_info["products"])
    print(f"\nProducts:")
    for i, p in enumerate(product_info["products"], 1):
        print(f"  {i}. {p['name']}: ${p['price']:.2f}")
    print(f"  Total Product Price: ${total_product_price:.2f}")
    
    # Get user requirement (should describe purchasing two products)
    # Use default requirement for automatic running
    user_requirement = product_request
    print(f"Using default requirement: {user_requirement}")
    
    # Reset environment
    print("\n" + "="*60)
    print("Starting two-product negotiation...")
    print("="*60)
    
    observation, info = env.reset(
        user_requirement=user_requirement,
        product_info=product_info,
        user_profile=user_profile,
    )
    
    # Start negotiation loop
    done = False
    start_time = time.time()
    
    # Initialize results dictionary
    results = {
        "task": "Task10_s6_bookshelf_sconce_bundle_negotiation",
        "timestamp": datetime.now().isoformat(),
        "user_requirement": user_requirement,
        "user_profile": user_profile,
        "status": "unknown",
        "success": False,
        "error": None,
    }
    
    while not done:
        # Each round: buyer responds first, then seller responds (seeing buyer's message)
        # Get buyer's response
        buyer_action = buyer.respond(
            conversation_history=observation["conversation_history"],
            current_state=observation
        )
        
        # Create updated conversation history that includes buyer's response
        # So seller can see buyer's message before responding
        updated_conversation_history = observation["conversation_history"].copy()
        if buyer_action:
            current_round = observation.get("current_round", 0)
            updated_conversation_history.append({
                "role": "buyer",
                "content": buyer_action,
                "round": current_round
            })
        
        # Get seller's response (seller can now see buyer's message)
        seller_action = seller.respond(
            conversation_history=updated_conversation_history,
            current_state=observation
        )
        
        # Execute step with both actions
        observation, reward, terminated, truncated, info = env.step(
            buyer_action=buyer_action,
            seller_action=seller_action
        )
        done = terminated or truncated
        
        # Render current state (includes all print information)
        env.render()
        
        # Flush output to ensure complete display
        sys.stdout.flush()
        
        # If this is the final round (agreed or timeout), display score calculations after Round Summary
        if done:
            # Print score calculations after Round Summary
            env._print_global_score_details()
            env._print_buyer_score_details()
            env._print_seller_score_details()
            
            print("\n" + "="*60)
            print("Negotiation Ended")
            print("="*60)
            print(f"Status: {info['status']}")
            seller_price = info.get('seller_price')
            buyer_price = info.get('buyer_price')
            seller_price_str = f"${seller_price:.2f}" if seller_price is not None else "Not specified"
            buyer_price_str = f"${buyer_price:.2f}" if buyer_price is not None else "Not specified"
            print(f"Final Total Prices: Seller={seller_price_str} | Buyer={buyer_price_str}")
            if info.get('agreed_price'):
                print(f"Agreed Total Price: ${info.get('agreed_price', 0):.2f}")
            if info.get('agreed_contract') is not None:
                print(f"Final Contract: {info['agreed_contract']}")
            # current_round has been incremented to reflect the completed round
            actual_rounds = info['round']
            print(f"Total Rounds: {actual_rounds}")
            print(f"Reward: {reward:.3f}")
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
                "seller_price": info.get('seller_price'),
                "buyer_price": info.get('buyer_price'),
                "agreed_price": info.get('agreed_price'),
                "agreed_contract": info.get('agreed_contract'),
                "total_rounds": info.get('round', 0),
                "total_reward": float(reward) if reward is not None else None,
                "global_score": info.get('global_score'),
                "buyer_score": info.get('buyer_score'),
                "seller_score": info.get('seller_score'),
                "buyer_utility": info.get('buyer_utility'),
                "seller_utility": info.get('seller_utility'),
                "z_max": info.get('z_max'),
                "termination_reason": info.get('termination_reason'),
                "elapsed_time": elapsed_time,
                "buyer_max_price": buyer_max_price,
                "seller_min_price": seller_min_price,
                "contract_config": contract_config,
                "product_info": product_info,
                "model": get_model_name(model),
            })
            break
    
    # Close environment
    env.close()
    print("\nTwo-product negotiation completed!")
    
    # Ensure elapsed_time is set even if negotiation didn't complete normally
    if "elapsed_time" not in results:
        results["elapsed_time"] = time.time() - start_time
    
    # Save results to file
    try:
        # Create results directory structure
        results_dir = Path(project_root) / "agenticpay" / "results" / "only_multi_products"
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
        output_file = run_dir / "Task10_s6_bookshelf_sconce_bundle_output.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("Task10 Scenario 6: Bookshelf & Wall Sconce Bundle - Two-Product Negotiation Results\n")
            f.write("Category: Home & Kitchen\n")
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
            f.write("Final Prices:\n")
            f.write(f"  Seller Total Price: ${results['seller_price']:.2f}" if results.get('seller_price') is not None else "  Seller Total Price: Not specified")
            f.write("\n")
            f.write(f"  Buyer Total Price: ${results['buyer_price']:.2f}" if results.get('buyer_price') is not None else "  Buyer Total Price: Not specified")
            f.write("\n")
            if results.get('agreed_price'):
                f.write(f"  Agreed Total Price: ${results['agreed_price']:.2f}\n")
            if results.get('agreed_contract') is not None:
                f.write(f"  Agreed Contract: {results['agreed_contract']}\n")
            f.write("\n")
            f.write("Products:\n")
            for i, p in enumerate(results.get('product_info', {}).get('products', []), 1):
                f.write(f"  {i}. {p.get('name', 'Unknown')}: ${p.get('price', 0):.2f}\n")
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
    parser = argparse.ArgumentParser(description="Task10 Scenario 6: Bookshelf & Wall Sconce Bundle - Two-Product Negotiation")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name to use (e.g., gpt-4o-mini, gpt-4o). Set OPENAI_URL for OpenAI-compatible API (e.g., local OpenVLM)."
    )
    args = parser.parse_args()
    main(model_name=args.model)

