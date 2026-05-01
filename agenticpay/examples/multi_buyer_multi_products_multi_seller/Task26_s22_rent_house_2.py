"""Task26 Scenario 22: Barcelona — Sequential Two-Buyer Two-Seller Two-Listing Rental Bundle

Long-term monthly rent: two tenants negotiate combined monthly rent for the same pair of listings with two landlords (Airbnb 2019 crawl sample; not same-day short-term rates).
Listings from ``airbnb_embeddings_sample10.jsonl`` entries 2 and 6:
``5*Super host Apartment+terrace  8 min from Ramblas`` (``_id`` 1332929) +
``Double room in Barcelona Center with balcony!!!!`` (``_id`` 23160633).
Confidential floors/ceilings sit below the listing component-sum anchor; ``seller*_min_price < buyer*_max_price < quoted_price``.
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
    
    # Create Agents (set their respective bottom prices, this information is confidential, unknown to each other)
    # Combined monthly rent for both Barcelona listings (confidential)
    print("Creating agents...")
    buyer1_max_price = 2668.0  # Maximum acceptable combined monthly rent for buyer1 (confidential; below summed listing components)
    buyer2_max_price = 2819.0  # Maximum acceptable combined monthly rent for buyer2 (confidential; below summed listing components)
    seller1_min_price = 2398.0  # Minimum acceptable combined monthly rent for seller1 bundle (confidential)
    seller2_min_price = 2139.0  # Minimum acceptable combined monthly rent for seller2 bundle (confidential)
    
    buyer1 = BuyerAgent(model=model, name="Buyer1", buyer_max_price=buyer1_max_price)
    buyer2 = BuyerAgent(model=model, name="Buyer2", buyer_max_price=buyer2_max_price)
    seller1 = SellerAgent(model=model, name="Seller1", seller_min_price=seller1_min_price)
    seller2 = SellerAgent(model=model, name="Seller2", seller_min_price=seller2_min_price)

    user_requirement = (
        "I want the Ramblas terrace apartment and Double room in Barcelona Center—one monthly rent. "
        "I also prefer the terrace listing shots to capture a readable outdoor railing or balustrade edge, not only interior furniture cropped tight."
    )
    product_request = user_requirement

    shared_contract_fields = {
        "contrainfo": {
            "product_request": product_request,
            "initial_contract_status": (
                "No total combined monthly rent, bundled lease length, utilities-included policy, "
                "or user product preference match has been selected before negotiation starts."
            ),
            "contract_completion_requirement": (
                "A valid offer must set price (total USD monthly rent for both listings together), "
                "continuous_terms.lease_months (integer months within bounds), "
                "discrete_terms.include_utilities (boolean true/false), and discrete_terms.user_product_preference "
                "(strong_match|partial_match|mismatch_or_uncertain)."
            ),
        },
        "field_descriptions": {
            "price": (
                "Total combined monthly rent in US dollars for the two-listing bundle (both units together)."
            ),
            "continuous_terms.lease_months": (
                "Lease term in whole months applying to the bundled package (same commitment length for both listings)."
            ),
            "discrete_terms.include_utilities": (
                "If true, quoted rent includes bundled utilities where applicable; if false, tenant pays utilities separately."
            ),
            "discrete_terms.user_product_preference": (
                "How well listing photos align with the buyer's stated preference that terrace shots show a readable "
                "outdoor railing or balustrade edge, not only tight interior furniture crops. "
                "`strong_match` when clearly satisfied for both bundle listings; `partial_match` when mixed or ambiguous; "
                "`mismatch_or_uncertain` when not satisfied or unconfirmable."
            ),
        },
        "continuous_bounds": {"lease_months": {"min": 1, "max": 24}},
        "discrete_options": {
            "include_utilities": [True, False],
            "user_product_preference": ["strong_match", "partial_match", "mismatch_or_uncertain"],
        },
    }
    buyer1_preferences = {
        "v_base": buyer1_max_price,
        "weight_descriptions": {
            "v_base": (
                "Your private reservation value for the bundle’s monthly rent before lease length and utilities terms, in USD."
            ),
            "continuous_weights.lease_months": (
                "Utility change per additional month of lease (USD/month); negative means you prefer shorter leases."
            ),
            "discrete_weights.include_utilities": (
                "Utility change for each utilities-included option (USD); higher is better for you when included."
            ),
            "discrete_weights.user_product_preference": (
                "USD change for photo alignment with your stated terrace-railing visibility preference."
            ),
        },
        "continuous_weights": {"lease_months": -10.0},
        "discrete_weights": {
            "include_utilities": {True: 195.0, False: 0.0},
            "user_product_preference": {
                "strong_match": 19.5,
                "partial_match": 7.9,
                "mismatch_or_uncertain": -15.2,
            },
        },
    }
    buyer2_preferences = json.loads(json.dumps(buyer1_preferences))
    buyer2_preferences["v_base"] = buyer2_max_price
    buyer2_preferences["continuous_weights"]["lease_months"] = -8.0
    buyer2_preferences["discrete_weights"]["include_utilities"] = {True: 170.0, False: 0.0}
    buyer2_preferences["discrete_weights"]["user_product_preference"] = {
        "strong_match": 17.0,
        "partial_match": 6.9,
        "mismatch_or_uncertain": -13.2,
    }
    seller1_preferences = {
        "c_base": seller1_min_price,
        "weight_descriptions": {
            "c_base": (
                "Your private floor for acceptable total monthly rent before lease/utilities terms, in USD."
            ),
            "continuous_weights.lease_months": (
                "Utility change per additional month (USD/month); positive means longer leases reduce vacancy risk for you."
            ),
            "discrete_weights.include_utilities": (
                "Utility/cost impact when utilities are bundled into the rent (USD); negative values are costs to you."
            ),
            "discrete_weights.user_product_preference": (
                "Small USD carrying cost when you commit to strong photo alignment with the buyer's terrace-railing preference."
            ),
        },
        "continuous_weights": {"lease_months": 19.0},
        "discrete_weights": {
            "include_utilities": {True: -88.0, False: 0.0},
            "user_product_preference": {
                "strong_match": -4.9,
                "partial_match": -2.4,
                "mismatch_or_uncertain": 0.62,
            },
        },
    }
    seller2_preferences = json.loads(json.dumps(seller1_preferences))
    seller2_preferences["c_base"] = seller2_min_price
    seller2_preferences["continuous_weights"]["lease_months"] = 20.0
    seller2_preferences["discrete_weights"]["include_utilities"] = {True: -80.0, False: 0.0}
    seller2_preferences["discrete_weights"]["user_product_preference"] = {
        "strong_match": -4.45,
        "partial_match": -2.18,
        "mismatch_or_uncertain": 0.56,
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

    print("Creating sequential multi-buyer multi-seller two-product negotiation environment...")
    env = Task3SequentialTwoBuyerTwoSellerTwoProductNegotiation(
        buyer1_agent=buyer1,
        buyer2_agent=buyer2,
        seller1_agent=seller1,
        seller2_agent=seller2,
        max_rounds=max_rounds,
        initial_seller1_price=3540.0,  # Opening combined monthly rent — seller1 bundle offer
        initial_seller2_price=3510.0,  # Opening combined monthly rent — seller2 bundle offer
        buyer1_max_price=buyer1_max_price,  # Buyer1 total max price (confidential)
        buyer2_max_price=buyer2_max_price,  # Buyer2 total max price (confidential)
        seller1_min_price=seller1_min_price,  # Seller1 total min price (confidential)
        seller2_min_price=seller2_min_price,  # Seller2 total min price (confidential)
        environment_info={
            "platform": "Airbnb (listing-style; combined monthly lease)",
            "market_type": "Residential Rental",
            "note": "Two third-party bundle offers for the same two-listing cart; prices are total monthly rent.",
            "availability_status": "Available for lease discussion.",
            "listing_age": "2019 scrape (jsonl): 1332929 last_scraped 2019-03-08; 23160633 last_scraped 2019-03-08 — airbnb_embeddings_sample10.jsonl",
            "category": "Real Estate",
            "buyer_seller_contract_configs": buyer_seller_contract_configs,
        },
        price_tolerance=price_tolerance,
        reward_weights=reward_weights,  # Reward weights configuration
    )

    user_profile = None
    print(f"User Profile: {user_profile}")
    print(f"Using default requirement: {user_requirement}")

    # Two rental listings per bundle; images from airbnb_embeddings_sample10.jsonl (1332929, 23160633)
    product_info = {
        "products": [
            {
                "name": "5*Super host Apartment+terrace  8 min from Ramblas",
                "price": 1890.0,
                "condition": "Bright penthouse-style · terrace (long-term monthly lease framing)",
                "color": "N/A",
                "size": "Apartment · entire home/apt · 2 BR · 1 bath · accommodates 4 · listing 1332929",
                "original_price": 1890.0,
                "availability_status": "Available for lease discussion.",
                "product_category": "Real Estate › Rentals › Apartments",
                "average_rating": 4.7,
                "total_reviews": 320,
                "asin": "AIRBNB-1332929",
                "full_description": "Space text (jsonl excerpt): bright apartment with sun and terrace with table for four; penthouse 5th floor with lift; two rooms with natural light. Nightly sample price in data: $63 — monthly bundle component. https://www.airbnb.com/rooms/1332929",
                "image_url": "https://a0.muscache.com/im/users/7218242/profile_pic/1372707884/original.jpg?aki_policy=large",
            },
            {
                "name": "Double room in Barcelona Center with balcony!!!!",
                "price": 1560.0,
                "condition": "Private room in shared apartment (long-term monthly lease framing)",
                "color": "N/A",
                "size": "Apartment · private room · 1 BR · 1 bath · accommodates 2 · listing 23160633",
                "original_price": 1560.0,
                "availability_status": "Available for lease discussion.",
                "product_category": "Real Estate › Rentals › Private Room",
                "average_rating": 5.0,
                "total_reviews": 4,
                "asin": "AIRBNB-23160633",
                "full_description": "Summary (jsonl): Perfectly locaded! 2min walking to Pl Universitat, 10 min walking to Pl. Catalunya, 2 min walking to Sant Antoni Market. Nightly sample price in data: $50 — second component of combined monthly rent. https://www.airbnb.com/rooms/23160633",
                "image_url": "https://a0.muscache.com/im/pictures/e051cbb3-0418-4200-8254-95b0bb8db441.jpg?aki_policy=large",
            },
        ]
    }

    total_product_price = sum(p["price"] for p in product_info["products"])
    print(f"\nProducts (two-listing rental bundle — Barcelona):")
    for i, p in enumerate(product_info["products"], 1):
        print(f"  {i}. {p['name']}: ${p['price']:.2f}")
    print(f"  Component sum (two listings): ${total_product_price:.2f}")
    
    # Reset environment
    print("\n" + "="*60)
    print("Starting new sequential negotiation — Barcelona two-listing bundle...")
    print("Two buyers competing for bundled monthly rent (terrace apt + center room)")
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
        "task": "Task26_s22_rent_house_2",
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
            "one complete <contract>...</contract> JSON block for the selected seller. "
            "The contract `price` is the **total** combined monthly rent for both listings."
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
        
        # Execute step with selected sellers and actions
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
        
        # Render current state (includes all print information)
        env.render()
        
        # Flush output to ensure complete display
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
                pair_key = f"b{info['selected_buyer']}s{info['selected_seller']}"
                agreed_contract = info.get(f"{pair_key}_agreed_contract")
                if agreed_contract is not None:
                    print(f"Final Contract: {agreed_contract}")
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
    
    # Close environment
    env.close()
    print("\nBarcelona two-listing rental bundle negotiation completed!")
    
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
        output_file = run_dir / "Task26_s22_rent_house_2_output.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("Task26 Scenario 22: Barcelona Two-Listing Rental Bundle — Negotiation Results\n")
            f.write("Category: Real Estate — Residential Rentals\n")
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
            f.write("Final Prices (Combined monthly rent for both listings):\n")
            f.write(f"  Buyer1-Seller1: Buyer=${results['b1s1_buyer_price']:.2f} | Seller=${results['b1s1_seller_price']:.2f}" if results.get('b1s1_buyer_price') is not None and results.get('b1s1_seller_price') is not None else "  Buyer1-Seller1: Not specified")
            f.write("\n")
            f.write(f"  Buyer1-Seller2: Buyer=${results['b1s2_buyer_price']:.2f} | Seller=${results['b1s2_seller_price']:.2f}" if results.get('b1s2_buyer_price') is not None and results.get('b1s2_seller_price') is not None else "  Buyer1-Seller2: Not specified")
            f.write("\n")
            f.write(f"  Buyer2-Seller1: Buyer=${results['b2s1_buyer_price']:.2f} | Seller=${results['b2s1_seller_price']:.2f}" if results.get('b2s1_buyer_price') is not None and results.get('b2s1_seller_price') is not None else "  Buyer2-Seller1: Not specified")
            f.write("\n")
            f.write(f"  Buyer2-Seller2: Buyer=${results['b2s2_buyer_price']:.2f} | Seller=${results['b2s2_seller_price']:.2f}" if results.get('b2s2_buyer_price') is not None and results.get('b2s2_seller_price') is not None else "  Buyer2-Seller2: Not specified")
            f.write("\n\n")
            product_info = results.get('product_info', {})
            f.write("Products:\n")
            if 'products' in product_info:
                for i, p in enumerate(product_info['products'], 1):
                    f.write(f"  {i}. {p.get('name', 'N/A')} — ${p.get('price', 0):.2f}\n")
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
    parser = argparse.ArgumentParser(description="Task26 Scenario 22: Barcelona two-listing rental bundle — Sequential Two-Buyer Two-Seller")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name to use (e.g., 'gemini-3-pro-all', 'gpt-5.2', 'claude-sonnet-4-5-20250929'). If not provided, uses default model."
    )
    args = parser.parse_args()
    main(model_name=args.model)
