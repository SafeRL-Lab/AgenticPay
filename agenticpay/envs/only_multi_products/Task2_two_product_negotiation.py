"""Task2 Two-Product Negotiation Environment Implementation

Supports negotiation for two products with total price negotiation.
"""

from __future__ import annotations

import json
import re
from copy import deepcopy
from typing import Any, Dict, Optional, Tuple, List

from agenticpay.core import BaseEnv, NegotiationStatus, NegotiationInfo
from agenticpay.agents.base_agent import BaseAgent
from agenticpay.memory.conversation_memory import ConversationMemory
from agenticpay.utils.negotiation_state import NegotiationState


class Task2TwoProductNegotiation(BaseEnv):
    """Task2 Two-Product Negotiation Environment
    
    Manages negotiation process for two products with total price negotiation.
    buyer_max_price and seller_min_price represent the total expected cost for both products.
    """
    
    def __init__(
        self,
        buyer_agent: BaseAgent,
        seller_agent: BaseAgent,
        max_rounds: int = 20,
        initial_seller_price: float = 100.0,
        buyer_max_price: Optional[float] = None,
        seller_min_price: Optional[float] = None,
        environment_info: Optional[Dict[str, Any]] = None,
        price_tolerance: float = 1.0,
        gamma: float = 0.99,
        deal_score_weight: float = 10.0,
        quality_score_weight: float = 80.0,
        efficiency_score_weight: float = 10.0,
        failure_penalty_weight: float = 15.0,
        buyer_deal_weight: float = 10.0,
        buyer_utility_weight: float = 80.0,
        buyer_efficiency_weight: float = 10.0,
        buyer_failure_penalty_weight: float = 15.0,
        seller_deal_weight: float = 10.0,
        seller_utility_weight: float = 80.0,
        seller_efficiency_weight: float = 10.0,
        seller_failure_penalty_weight: float = 15.0,
    ):
        """Initialize two-product negotiation environment
        
        Args:
            buyer_agent: Buyer Agent
            seller_agent: Seller Agent
            max_rounds: Maximum number of negotiation rounds
            initial_seller_price: Initial total price offered by seller for both products
            buyer_max_price: Maximum acceptable total price for buyer (confidential, for both products)
            seller_min_price: Minimum acceptable total price for seller (confidential, for both products)
            environment_info: Environment information (e.g., season, weather, etc.)
            price_tolerance: Price tolerance for determining agreement
            gamma: Discount factor for GlobalScore calculation, controls penalty for longer negotiations (default: 0.99, range: 0.97-0.995)
            deal_score_weight: Weight D for DealScore component (default: 10.0)
            quality_score_weight: Weight W for QualityScore component (default: 80.0)
            efficiency_score_weight: Weight E for EfficiencyScore component (default: 10.0)
            failure_penalty_weight: Weight F for FailurePenalty component (default: 15.0)
            buyer_deal_weight: Weight Db for Buyer Deal Bonus (default: 10.0)
            buyer_utility_weight: Weight Wb for Buyer utility component (default: 80.0)
            buyer_efficiency_weight: Weight Eb for Buyer Efficiency Bonus (default: 10.0)
            buyer_failure_penalty_weight: Weight Fb for Buyer Failure Penalty (default: 15.0)
            seller_deal_weight: Weight Ds for Seller Deal Bonus (default: 10.0)
            seller_utility_weight: Weight Ws for Seller utility component (default: 80.0)
            seller_efficiency_weight: Weight Es for Seller Efficiency Bonus (default: 10.0)
            seller_failure_penalty_weight: Weight Fs for Seller Failure Penalty (default: 15.0)
        """
        self.buyer_agent = buyer_agent
        self.seller_agent = seller_agent
        self.max_rounds = max_rounds
        self.initial_seller_price = initial_seller_price
        self.buyer_max_price = buyer_max_price
        self.seller_min_price = seller_min_price
        self.environment_info = environment_info or {}
        self.contract_config = self.environment_info.get("contract_config") or {}
        self.use_contract_mode = bool(self.contract_config)
        self.z_max = self._calculate_z_max() if self.use_contract_mode else None
        self.price_tolerance = price_tolerance
        
        # Score calculation parameters
        self.gamma = gamma
        self.deal_score_weight = deal_score_weight  # D
        self.quality_score_weight = quality_score_weight  # W
        self.efficiency_score_weight = efficiency_score_weight  # E
        self.failure_penalty_weight = failure_penalty_weight  # F
        # Buyer score weights
        self.buyer_deal_weight = buyer_deal_weight  # Db
        self.buyer_utility_weight = buyer_utility_weight  # Wb
        self.buyer_efficiency_weight = buyer_efficiency_weight  # Eb
        self.buyer_failure_penalty_weight = buyer_failure_penalty_weight  # Fb
        # Seller score weights
        self.seller_deal_weight = seller_deal_weight  # Ds
        self.seller_utility_weight = seller_utility_weight  # Ws
        self.seller_efficiency_weight = seller_efficiency_weight  # Es
        self.seller_failure_penalty_weight = seller_failure_penalty_weight  # Fs
        
        # Call parent class initialization
        super().__init__()
        
        # State management
        self.memory = ConversationMemory()
        self.state = NegotiationState()
        self.current_round = 0
        self.negotiation_info = NegotiationInfo()
        self.product_info: Optional[Dict[str, Any]] = None
        self.product_images: Optional[List[str]] = None  # For VLM img input
        self.state.metadata["buyer_contract"] = None
        self.state.metadata["seller_contract"] = None
        self.state.metadata["agreed_contract"] = None
        self.state.metadata["buyer_utility"] = None
        self.state.metadata["seller_utility"] = None
        self.state.metadata["z_max"] = self.z_max

    def _build_public_contract_config(self) -> Dict[str, Any]:
        """Build contract config fields that both parties are allowed to see."""
        if not self.contract_config:
            return {}
        public_config: Dict[str, Any] = {}
        for key in ("continuous_bounds", "discrete_options", "field_descriptions", "contrainfo"):
            if key in self.contract_config:
                public_config[key] = deepcopy(self.contract_config[key])
        return public_config

    def _build_role_contract_config(self, role: str) -> Dict[str, Any]:
        """Build role-specific contract config with only own private preferences."""
        role_config = self._build_public_contract_config()
        if role == "buyer" and "buyer_preferences" in self.contract_config:
            role_config["buyer_preferences"] = deepcopy(self.contract_config["buyer_preferences"])
        if role == "seller" and "seller_preferences" in self.contract_config:
            role_config["seller_preferences"] = deepcopy(self.contract_config["seller_preferences"])
        return role_config

    def _build_role_environment_info(self, role: str) -> Dict[str, Any]:
        """Build role-specific environment info without leaking counterparty preferences."""
        role_env_info = deepcopy(self.environment_info)
        if self.use_contract_mode:
            role_env_info["contract_config"] = self._build_role_contract_config(role)
        return role_env_info

    def _normalize_contract(self, contract: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Normalize contract schema to a stable internal representation."""
        if not isinstance(contract, dict):
            return None
        price = contract.get("price")
        try:
            price = float(price)
        except (TypeError, ValueError):
            return None
        if price <= 0:
            return None

        continuous_terms = contract.get("continuous_terms", {})
        discrete_terms = contract.get("discrete_terms", {})
        if not isinstance(continuous_terms, dict) or not isinstance(discrete_terms, dict):
            return None

        return {
            "price": price,
            "continuous_terms": continuous_terms,
            "discrete_terms": discrete_terms,
        }

    def _extract_contract(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract contract JSON from <contract>...</contract> block."""
        if not text:
            return None
        contract_match = re.search(r"<contract>\s*(.*?)\s*</contract>", text, re.DOTALL | re.IGNORECASE)
        if not contract_match:
            return None
        try:
            contract_obj = json.loads(contract_match.group(1).strip())
        except json.JSONDecodeError:
            return None
        return self._normalize_contract(contract_obj)

    def _validate_contract(self, contract: Optional[Dict[str, Any]]) -> bool:
        """Validate contract against configured bounds/options in contract mode."""
        if not contract:
            return False
        if not self.use_contract_mode:
            return True

        continuous_bounds = self.contract_config.get("continuous_bounds", {})
        discrete_options = self.contract_config.get("discrete_options", {})
        continuous_terms = contract.get("continuous_terms", {})
        discrete_terms = contract.get("discrete_terms", {})

        for term in continuous_terms.keys():
            if term not in continuous_bounds:
                return False
        for term in discrete_terms.keys():
            if term not in discrete_options:
                return False

        for term, bounds in continuous_bounds.items():
            if term not in continuous_terms:
                return False
            try:
                numeric_value = float(continuous_terms.get(term))
            except (TypeError, ValueError):
                return False
            min_v = bounds.get("min")
            max_v = bounds.get("max")
            if min_v is not None and numeric_value < min_v:
                return False
            if max_v is not None and numeric_value > max_v:
                return False

        for term, options in discrete_options.items():
            if term not in discrete_terms:
                return False
            if discrete_terms.get(term) not in options:
                return False

        return True

    def _calculate_contract_utilities(self, contract: Dict[str, Any]) -> Tuple[float, float]:
        """Compute raw MAUT utilities (U_b, U_s) from contract config."""
        buyer_prefs = self.contract_config.get("buyer_preferences", {})
        seller_prefs = self.contract_config.get("seller_preferences", {})

        price = float(contract["price"])
        buyer_utility = float(buyer_prefs.get("v_base", 0.0)) - price
        seller_utility = price - float(seller_prefs.get("c_base", 0.0))

        buyer_cw = buyer_prefs.get("continuous_weights", {})
        seller_cw = seller_prefs.get("continuous_weights", {})
        for term, value in contract.get("continuous_terms", {}).items():
            try:
                numeric_value = float(value)
            except (TypeError, ValueError):
                continue
            buyer_utility += float(buyer_cw.get(term, 0.0)) * numeric_value
            seller_utility += float(seller_cw.get(term, 0.0)) * numeric_value

        buyer_dw = buyer_prefs.get("discrete_weights", {})
        seller_dw = seller_prefs.get("discrete_weights", {})
        for term, value in contract.get("discrete_terms", {}).items():
            buyer_utility += float(buyer_dw.get(term, {}).get(value, 0.0))
            seller_utility += float(seller_dw.get(term, {}).get(value, 0.0))

        return buyer_utility, seller_utility

    def _calculate_z_max(self) -> Optional[float]:
        """Compute theoretical max surplus Z_max for contract mode."""
        if not self.contract_config:
            return None

        buyer_prefs = self.contract_config.get("buyer_preferences", {})
        seller_prefs = self.contract_config.get("seller_preferences", {})
        z_max = float(buyer_prefs.get("v_base", 0.0)) - float(seller_prefs.get("c_base", 0.0))

        buyer_cw = buyer_prefs.get("continuous_weights", {})
        seller_cw = seller_prefs.get("continuous_weights", {})
        for term, bounds in self.contract_config.get("continuous_bounds", {}).items():
            total_weight = float(buyer_cw.get(term, 0.0)) + float(seller_cw.get(term, 0.0))
            min_v = float(bounds.get("min", 0.0))
            max_v = float(bounds.get("max", 0.0))
            z_max += total_weight * (max_v if total_weight >= 0 else min_v)

        buyer_dw = buyer_prefs.get("discrete_weights", {})
        seller_dw = seller_prefs.get("discrete_weights", {})
        for term, options in self.contract_config.get("discrete_options", {}).items():
            best = None
            for opt in options:
                candidate = float(buyer_dw.get(term, {}).get(opt, 0.0)) + float(seller_dw.get(term, {}).get(opt, 0.0))
                if best is None or candidate > best:
                    best = candidate
            if best is not None:
                z_max += best

        return z_max

    def _contracts_compatible(self, buyer_contract: Dict[str, Any], seller_contract: Dict[str, Any]) -> bool:
        """Check whether two contracts are compatible enough to settle."""
        if buyer_contract.get("continuous_terms", {}) != seller_contract.get("continuous_terms", {}):
            return False
        if buyer_contract.get("discrete_terms", {}) != seller_contract.get("discrete_terms", {}):
            return False
        buyer_price = float(buyer_contract["price"])
        seller_price = float(seller_contract["price"])
        if abs(buyer_price - seller_price) <= self.price_tolerance:
            return True
        return seller_price <= buyer_price
    
    def reset(
        self,
        user_requirement: str = "",
        product_info: Optional[Dict[str, Any]] = None,
        user_profile: Optional[Any] = None,
        **kwargs: Any,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Reset environment, start new negotiation
        
        Args:
            user_requirement: User requirement description (should describe purchasing two products)
            product_info: Product information containing two products and their prices
                Expected format: {
                    "products": [
                        {"name": "Product1", "price": 100.0, ...},
                        {"name": "Product2", "price": 80.0, ...}
                    ]
                }
            user_profile: User profile
            **kwargs: Other parameters
            
        Returns:
            (observation, info) Initial observation and info
        """
        # Reset state
        self.memory.clear()
        self.state = NegotiationState()
        self.current_round = 0
        self.negotiation_info = NegotiationInfo()
        self.product_info = product_info or {}
        self.state.metadata["buyer_contract"] = None
        self.state.metadata["seller_contract"] = None
        self.state.metadata["agreed_contract"] = None
        self.state.metadata["buyer_utility"] = None
        self.state.metadata["seller_utility"] = None
        self.state.metadata["z_max"] = self.z_max
        
        # Extract product information
        products = self.product_info.get("products", [])
        if len(products) < 2:
            raise ValueError("product_info must contain at least 2 products in 'products' list")
        
        # Extract product_images for VLM (from image_url in each product)
        product_images = []
        for p in products:
            img_url = p.get("image_path") or p.get("image_url")
            if img_url:
                product_images.append(img_url)
        if not product_images:
            product_images = None
        self.product_images = product_images
        
        # Calculate total price of both products
        total_product_price = sum(p.get("price", 0.0) for p in products)
        
        # Initialize Agents (include product_images for VLM to use img input)
        buyer_context = {
            "user_requirement": user_requirement,
            "max_price": self.buyer_max_price,  # Total max price for both products
            "user_profile": user_profile,
            "environment_info": self._build_role_environment_info("buyer"),
            "product_info": self.product_info,  # Buyer can see both products
            "product_images": product_images,  # For VLM: product images (URL/path)
            "contract_config": self._build_role_contract_config("buyer"),
        }
        self.buyer_agent.initialize(buyer_context)
        
        seller_context = {
            "product_info": self.product_info,  # Seller can see both products
            "product_images": product_images,  # For VLM: product images (URL/path)
            "initial_price": self.initial_seller_price,  # Initial total price
            "min_price": self.seller_min_price,  # Total min price for both products
            "environment_info": self._build_role_environment_info("seller"),
            "contract_config": self._build_role_contract_config("seller"),
        }
        self.seller_agent.initialize(seller_context)
        
        # No initial seller offer - negotiation starts with buyer's first message
        # Build observation
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(
        self, 
        buyer_action: Optional[str] = None, 
        seller_action: Optional[str] = None
    ) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """Execute one negotiation step
        
        Each round, both buyer and seller make their responses, then check if agreement is reached.
        Prices represent total price for both products.
        
        Args:
            buyer_action: Buyer's response text (optional, can be None if buyer doesn't respond)
            seller_action: Seller's response text (optional, can be None if seller doesn't respond)
            
        Returns:
            (observation, reward, terminated, truncated, info)
        """
        # Add messages to memory (both agents respond in the same round)
        # Order: buyer -> seller
        if buyer_action is not None:
            self.memory.add_message("buyer", buyer_action, self.current_round)
            buyer_contract = self._extract_contract(buyer_action) if self.use_contract_mode else None
            if buyer_contract and self._validate_contract(buyer_contract):
                self.state.metadata["buyer_contract"] = buyer_contract
                buyer_price = float(buyer_contract["price"])
            else:
                buyer_price = self._extract_price(buyer_action)
            if buyer_price is not None:
                self.state.update(buyer_price=buyer_price)
                self.negotiation_info.buyer_price = buyer_price
                self.negotiation_info.current_price = buyer_price
        
        if seller_action is not None:
            self.memory.add_message("seller", seller_action, self.current_round)
            seller_contract = self._extract_contract(seller_action) if self.use_contract_mode else None
            if seller_contract and self._validate_contract(seller_contract):
                self.state.metadata["seller_contract"] = seller_contract
                seller_price = float(seller_contract["price"])
            else:
                seller_price = self._extract_price(seller_action)
            if seller_price is not None:
                self.state.update(seller_price=seller_price)
                self.negotiation_info.seller_price = seller_price
                self.negotiation_info.current_price = seller_price
        
        # Check if agreement is reached (after both agents have responded)
        terminated = False
        truncated = False
        reward = 0.0
        
        if self._check_agreement():
            terminated = True
            self.negotiation_info.status = NegotiationStatus.AGREED
            if self.use_contract_mode:
                buyer_contract = self.state.metadata.get("buyer_contract")
                seller_contract = self.state.metadata.get("seller_contract")
                if buyer_contract and seller_contract and self._contracts_compatible(buyer_contract, seller_contract):
                    if float(seller_contract["price"]) <= float(buyer_contract["price"]):
                        agreed_price = float(seller_contract["price"])
                    else:
                        agreed_price = (float(buyer_contract["price"]) + float(seller_contract["price"])) / 2
                    self.state.metadata["agreed_contract"] = {
                        "price": agreed_price,
                        "continuous_terms": buyer_contract.get("continuous_terms", {}),
                        "discrete_terms": buyer_contract.get("discrete_terms", {}),
                    }
                else:
                    agreed_price = (self.state.buyer_price + self.state.seller_price) / 2
            else:
                # When seller's offer <= buyer's offer: deal at seller's price; otherwise use midpoint
                if self.state.seller_price <= self.state.buyer_price:
                    agreed_price = self.state.seller_price
                else:
                    agreed_price = (self.state.buyer_price + self.state.seller_price) / 2
            self.state.update(agreed_price=agreed_price)
            self.negotiation_info.current_price = agreed_price
            # Increment current_round to reflect that this round is completed
            # This ensures round count is accurate when calculating final scores
            self.current_round += 1
            self.negotiation_info.round_count = self.current_round
            reward = self._calculate_reward()
        elif self.current_round >= self.max_rounds:
            truncated = True
            self.negotiation_info.status = NegotiationStatus.TIMEOUT
            # Increment current_round to reflect that this round is completed
            # This ensures round count is accurate when calculating final scores
            self.current_round += 1
            self.negotiation_info.round_count = self.current_round
            reward = self._calculate_reward()
        else:
            # Move to next round
            self.current_round += 1
            self.negotiation_info.round_count = self.current_round
        
        # Build observation and info
        observation = self._get_observation()
        info = self._get_info()
        if terminated or truncated:
            info["termination_reason"] = "agreed" if terminated else "timeout"
            # Calculate GlobalScore, BuyerScore, and SellerScore for final result
            # Note: current_round has been incremented to reflect the completed round
            # Don't print here - will be printed in example code after Step Rewards
            global_score = self._calculate_global_score(print_details=False)
            info["global_score"] = global_score
            buyer_score = self._calculate_buyer_score(print_details=False)
            info["buyer_score"] = buyer_score
            seller_score = self._calculate_seller_score(print_details=False)
            info["seller_score"] = seller_score
        
        return observation, reward, terminated, truncated, info
    
    def render(self, mode: str = "human") -> Optional[str]:
        """Render current state
        
        Displays buyer and seller outputs for each round, followed by a round summary
        including prices, agreement status, and reason.
        Prices shown are total prices for both products.
        
        Args:
            mode: Render mode, "human" prints to console, "text" returns text
            
        Returns:
            Returns string if mode="text", otherwise returns None
        """
        output_lines = []
        
        # Display product info
        if self.product_info:
            products = self.product_info.get("products", [])
            if products:
                output_lines.append(f"\n{'='*60}")
                output_lines.append("Products:")
                for i, p in enumerate(products, 1):
                    name = p.get("name", f"Product {i}")
                    price = p.get("price", 0.0)
                    output_lines.append(f"  {i}. {name}: ${price:.2f}")
                total_price = sum(p.get("price", 0.0) for p in products)
                output_lines.append(f"  Total Product Price: ${total_price:.2f}")
                output_lines.append(f"{'='*60}")
        
        # Get messages from the round that just completed
        # Note: In step(), messages are added to current_round
        # - If agreement reached: current_round stays the same, messages are in current_round
        # - If no agreement: current_round is incremented, messages are in current_round - 1
        history = self.memory.get_history()
        if history:
            # Determine which round's messages to display
            # Messages are stored with the round value at the time of storage (before current_round is incremented)
            # In step(), messages are added first, then current_round is incremented
            # So for any completed round, messages are stored at current_round - 1
            round_to_display = self.current_round - 1 if self.current_round > 0 else 0
            
            round_messages = [
                msg for msg in history if msg["round"] == round_to_display
            ]
            
            if round_messages:
                # Display round number: current_round is already incremented, so it represents the completed round number
                display_round = self.current_round
                output_lines.append(f"\n{'='*60}")
                output_lines.append(f"Round {display_round} - Negotiation Output")
                output_lines.append(f"{'='*60}")
                
                # Display buyer message first (if exists)
                buyer_msg = next(
                    (msg for msg in round_messages if msg["role"] == "buyer"), 
                    None
                )
                if buyer_msg:
                    output_lines.append(f"\n[BUYER Output]:")
                    output_lines.append(f"  {buyer_msg['content']}")
                
                # Display seller message (if exists)
                seller_msg = next(
                    (msg for msg in round_messages if msg["role"] == "seller"), 
                    None
                )
                if seller_msg:
                    output_lines.append(f"\n[SELLER Output]:")
                    output_lines.append(f"  {seller_msg['content']}")
        
        # Round summary section
        output_lines.append(f"\n{'-'*60}")
        output_lines.append(f"Round {self.current_round} Summary:")
        output_lines.append(f"{'-'*60}")
        
        if self.use_contract_mode:
            buyer_contract = self.state.metadata.get("buyer_contract")
            seller_contract = self.state.metadata.get("seller_contract")
            agreed_contract = self.state.metadata.get("agreed_contract")
            output_lines.append(f"  Buyer Contract: {buyer_contract if buyer_contract else 'Not specified'}")
            output_lines.append(f"  Seller Contract: {seller_contract if seller_contract else 'Not specified'}")
            if agreed_contract:
                output_lines.append(f"  Agreed Contract: {agreed_contract}")
        else:
            # Display buyer price (total for both products)
            if self.state.buyer_price is not None:
                output_lines.append(f"  Buyer Total Price: ${self.state.buyer_price:.2f}")
            else:
                output_lines.append(f"  Buyer Total Price: Not specified")
            
            # Display seller price (total for both products)
            if self.state.seller_price is not None:
                output_lines.append(f"  Seller Total Price: ${self.state.seller_price:.2f}")
            else:
                output_lines.append(f"  Seller Total Price: Not specified")
        
        # Check agreement status and provide reason
        is_agreed = self._check_agreement()
        agreement_reason = self._get_agreement_reason()
        
        output_lines.append(f"  Agreement Status: {'✓ AGREED' if is_agreed else '✗ NOT AGREED'}")
        output_lines.append(f"  Reason: {agreement_reason}")
        
        # Display agreed price if agreement is reached
        if self.state.agreed_price is not None:
            output_lines.append(f"  Agreed Total Price: ${self.state.agreed_price:.2f}")
        
        # Display negotiation status
        status_display = {
            NegotiationStatus.ONGOING: "Ongoing",
            NegotiationStatus.AGREED: "Agreed",
            NegotiationStatus.FAILED: "Failed",
            NegotiationStatus.TIMEOUT: "Timeout"
        }
        output_lines.append(f"  Negotiation Status: {status_display.get(self.negotiation_info.status, 'Unknown')}")
        
        output_lines.append(f"{'='*60}\n")
        
        output = "\n".join(output_lines)
        
        if mode == "human":
            print(output)
            return None
        else:
            return output
    
    def _get_agreement_reason(self) -> str:
        """Get reason for agreement or non-agreement
        
        Returns:
            String describing why agreement was or was not reached
        """
        if self.use_contract_mode:
            buyer_contract = self.state.metadata.get("buyer_contract")
            seller_contract = self.state.metadata.get("seller_contract")
            if not buyer_contract and not seller_contract:
                return "Both buyer and seller contract JSON are missing"
            if not buyer_contract:
                return "Buyer contract JSON is missing or invalid"
            if not seller_contract:
                return "Seller contract JSON is missing or invalid"
            if not self._contracts_compatible(buyer_contract, seller_contract):
                return "Contract terms or price are not compatible yet"
            return "Both sides submitted compatible contract JSON"

        if self.state.buyer_price is None or self.state.seller_price is None:
            if self.state.buyer_price is None and self.state.seller_price is None:
                return "Both buyer and seller prices are not specified yet"
            elif self.state.buyer_price is None:
                return "Buyer price is not specified yet"
            else:
                return "Seller price is not specified yet"
        
        price_diff = abs(self.state.buyer_price - self.state.seller_price)
        
        if self.state.seller_price <= self.state.buyer_price:
            return f"Seller's offer (${self.state.seller_price:.2f}) <= buyer's offer (${self.state.buyer_price:.2f}), deal at seller's price"
        if price_diff <= self.price_tolerance:
            return f"Price difference (${price_diff:.2f}) is within tolerance (${self.price_tolerance:.2f})"
        else:
            return f"Price difference (${price_diff:.2f}) exceeds tolerance (${self.price_tolerance:.2f})"
    
    def close(self):
        """Close environment, cleanup resources"""
        self.memory.clear()
        self.state = NegotiationState()
    
    def _get_observation(self) -> Dict[str, Any]:
        """Get current observation"""
        obs = {
            "conversation_history": self.memory.get_history(),
            "current_round": self.current_round,
            "seller_price": self.state.seller_price,  # Total price for both products
            "buyer_price": self.state.buyer_price,  # Total price for both products
            "status": self.negotiation_info.status.value,
            "product_info": self.product_info,
            "contract_config": self._build_public_contract_config() if self.use_contract_mode else None,
            "buyer_contract": self.state.metadata.get("buyer_contract"),
            "seller_contract": self.state.metadata.get("seller_contract"),
            "agreed_contract": self.state.metadata.get("agreed_contract"),
        }
        # Include product_images for VLM (agent passes img to model when is_vlm)
        if getattr(self, "product_images", None) is not None:
            obs["product_images"] = self.product_images
        return obs
    
    def _get_info(self) -> Dict[str, Any]:
        """Get current info"""
        return {
            "round": self.current_round,
            "status": self.negotiation_info.status.value,
            "seller_price": self.state.seller_price,  # Total price for both products
            "buyer_price": self.state.buyer_price,  # Total price for both products
            "agreed_price": self.state.agreed_price,  # Total agreed price for both products
            "buyer_contract": self.state.metadata.get("buyer_contract"),
            "seller_contract": self.state.metadata.get("seller_contract"),
            "agreed_contract": self.state.metadata.get("agreed_contract"),
            "buyer_utility": self.state.metadata.get("buyer_utility"),
            "seller_utility": self.state.metadata.get("seller_utility"),
            "z_max": self.z_max,
            "negotiation_info": self.negotiation_info,
            "product_info": self.product_info,
        }
    
    def _extract_price(self, text: str) -> Optional[float]:
        """Extract price from text
        
        Priority: 
        1. Extract from ### BUYER_PRICE($X) ### or ### SELLER_PRICE($X) ### format (preferred)
        2. Fall back to ### $X ### format
        3. Fall back to other price patterns
        
        Args:
            text: Text containing price
            
        Returns:
            Extracted price, returns None if not found
        """
        def parse_price(price_str: str) -> Optional[float]:
            """Parse price string, removing commas and converting to float"""
            try:
                # Remove commas from price string (e.g., "8,750" -> "8750")
                cleaned = price_str.replace(',', '')
                price = float(cleaned)
                if price > 0:
                    return price
            except ValueError:
                pass
            return None
        
        # Priority 1: Extract price from ### BUYER_PRICE($X) ### or ### SELLER_PRICE($X) ### format
        # Matches: ### BUYER_PRICE($100.50) ###, ### SELLER_PRICE($150) ###, ### BUYER_PRICE($8,750) ###, etc.
        labeled_price_pattern = r'###\s*(?:BUYER_PRICE|SELLER_PRICE)\s*\(\$([\d,]+\.?\d*)\)\s*###'
        matches = re.findall(labeled_price_pattern, text, re.IGNORECASE)
        if matches:
            price = parse_price(matches[-1])  # Take the last match
            if price is not None:
                return price
        
        # Priority 2: Extract price from ### $X ### format (backward compatibility)
        # Matches: ### $100.50 ###, ### $100 ###, ###$120###, ### $8,750 ###, etc.
        triple_hash_pattern = r'###\s*\$([\d,]+\.?\d*)\s*###'
        matches = re.findall(triple_hash_pattern, text, re.IGNORECASE)
        if matches:
            price = parse_price(matches[-1])  # Take the last match
            if price is not None:
                return price
        
        # Priority 3: Fall back to other price patterns
        fallback_patterns = [
            r'\$([\d,]+\.?\d*)',  # $100.50 or $100 or $8,750
            r'([\d,]+\.?\d*)\s*dollars?',  # 100.50 dollars or 8,750 dollars
            r'([\d,]+\.?\d*)\s*USD',  # 100.50 USD or 8,750 USD
            r'price.*?([\d,]+\.?\d*)',  # price 100.50 or price 8,750
            r'offer.*?([\d,]+\.?\d*)',  # offer 100.50 or offer 8,750
            r'total.*?([\d,]+\.?\d*)',  # total 100.50 or total 8,750
        ]
        
        for pattern in fallback_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                price = parse_price(matches[-1])  # Take the last match
                if price is not None:
                    return price
        
        return None
    
    def _check_agreement(self) -> bool:
        """Check if agreement is reached
        
        Agreement is reached when:
        1. The total prices of both buyer and seller are within the tolerance range, or
        2. Seller's offer is less than or equal to buyer's offer (seller_price <= buyer_price).
        
        Returns:
            Whether agreement is reached
        """
        if self.use_contract_mode:
            buyer_contract = self.state.metadata.get("buyer_contract")
            seller_contract = self.state.metadata.get("seller_contract")
            if not buyer_contract or not seller_contract:
                return False
            if not self._validate_contract(buyer_contract) or not self._validate_contract(seller_contract):
                return False
            return self._contracts_compatible(buyer_contract, seller_contract)

        if self.state.buyer_price is None or self.state.seller_price is None:
            return False
        
        price_diff = abs(self.state.buyer_price - self.state.seller_price)
        if price_diff <= self.price_tolerance:
            return True
        if self.state.seller_price <= self.state.buyer_price:
            return True
        return False
    
    def _calculate_reward(self) -> float:
        """Calculate reward
        
        Calculate reward value based on negotiation result.
        
        If deal is reached:
            reward = buyer savings + seller profit + time cost (negative, based on rounds)
            - buyer savings = buyer_max_price - deal_price (money saved by buyer for both products)
            - seller profit = deal_price - seller_min_price (extra profit for seller for both products)
            - time cost = -current_round (penalty for number of rounds taken)
        
        If deal is not reached:
            reward = time cost (negative, based on rounds)
            - time cost = -current_round (penalty for number of rounds taken)
        
        Returns:
            Reward value
        """
        # Time cost: negative value based on number of rounds
        time_cost = -self.current_round
        
        if self.negotiation_info.status == NegotiationStatus.AGREED:
            # Deal reached: buyer savings + seller profit + time cost
            if self.state.agreed_price is None:
                print(f"Reward = time_cost = {time_cost:.2f} (round={self.current_round})")
                return time_cost
            
            deal_price = self.state.agreed_price
            reward = 0.0
            buyer_savings = 0.0
            seller_profit = 0.0
            
            # Calculate buyer savings: buyer_max_price - deal_price (for both products)
            if self.buyer_max_price is not None:
                buyer_savings = self.buyer_max_price - deal_price
                reward += buyer_savings
            
            # Calculate seller profit: deal_price - seller_min_price (for both products)
            if self.seller_min_price is not None:
                seller_profit = deal_price - self.seller_min_price
                reward += seller_profit
            
            # Add time cost (negative penalty)
            reward += time_cost
            
            print(f"Reward = buyer_savings({buyer_savings:.2f}) + seller_profit({seller_profit:.2f}) + time_cost({time_cost:.2f}) = {reward:.2f} (buyer_max={self.buyer_max_price}, deal_price={deal_price:.2f}, seller_min={self.seller_min_price}, round={self.current_round})")
            
            return reward
        
        else:
            # Deal not reached: only time cost (negative penalty)
            print(f"Reward = time_cost = {time_cost:.2f} (round={self.current_round}, deal not reached)")
            return time_cost

    def _get_contract_score_terms(self) -> Optional[Dict[str, float]]:
        """Return normalized utility terms for contract-mode scoring."""
        if not self.use_contract_mode:
            return None
        if not self.z_max or self.z_max <= 0:
            return None

        final_contract = self.state.metadata.get("agreed_contract")
        if not final_contract and self.state.metadata.get("buyer_contract") and self.state.metadata.get("seller_contract"):
            buyer_contract = self.state.metadata["buyer_contract"]
            seller_contract = self.state.metadata["seller_contract"]
            if self._contracts_compatible(buyer_contract, seller_contract):
                resolved_price = float(seller_contract["price"]) if float(seller_contract["price"]) <= float(buyer_contract["price"]) else (
                    float(seller_contract["price"]) + float(buyer_contract["price"])
                ) / 2
                final_contract = {
                    "price": resolved_price,
                    "continuous_terms": buyer_contract.get("continuous_terms", {}),
                    "discrete_terms": buyer_contract.get("discrete_terms", {}),
                }

        if not final_contract or not self._validate_contract(final_contract):
            return None

        u_b, u_s = self._calculate_contract_utilities(final_contract)
        self.state.metadata["buyer_utility"] = u_b
        self.state.metadata["seller_utility"] = u_s
        if u_b < 0 or u_s < 0:
            return None

        r_b = u_b / self.z_max
        r_s = u_s / self.z_max
        return {
            "u_b": u_b,
            "u_s": u_s,
            "r_b": r_b,
            "r_s": r_s,
            "q": 4.0 * r_b * r_s,
        }
    
    def _calculate_global_score(self, print_details: bool = True) -> float:
        """Calculate GlobalScore based on the optimized formula
        
        Returns:
            GlobalScore value (only calculated at final result)
        """
        round_index = max(0, self.current_round - 1)
        discount = self.gamma ** round_index

        if self.use_contract_mode:
            score_terms = self._get_contract_score_terms()
            feasible_deal = (self.negotiation_info.status == NegotiationStatus.AGREED) or (self.state.agreed_price is not None)
            if feasible_deal and score_terms is not None:
                deal_score = self.deal_score_weight * discount
                quality_score = self.quality_score_weight * score_terms["q"] * discount
                efficiency_score = self.efficiency_score_weight * discount
                global_score = deal_score + quality_score + efficiency_score
                if print_details:
                    print(f"\n[GlobalScore Calculation - Contract Mode]")
                    print(f"  z_max = {self.z_max:.4f}")
                    print(f"  u_b = {score_terms['u_b']:.4f}, u_s = {score_terms['u_s']:.4f}")
                    print(f"  r_b = {score_terms['r_b']:.4f}, r_s = {score_terms['r_s']:.4f}")
                    print(f"  Q = 4 * r_b * r_s = {score_terms['q']:.4f}")
                    print(f"  DealScore = {deal_score:.3f}, QualityScore = {quality_score:.3f}, EfficiencyScore = {efficiency_score:.3f}")
                    print(f"  GlobalScore = {global_score:.3f}")
                return global_score

            failure_penalty = -self.failure_penalty_weight * (1.0 - discount)
            if print_details:
                print(f"\n[GlobalScore Calculation - Contract Mode]")
                print(f"  Failed to produce a feasible valid contract")
                print(f"  round_index = {round_index}, discount = γ^{round_index} = {discount:.6f}")
                print(f"  FailurePenalty = {failure_penalty:.3f}")
            return failure_penalty

        # Check if we have required prices
        if self.buyer_max_price is None or self.seller_min_price is None:
            # Calculate discount for failure penalty
            failure_penalty = -self.failure_penalty_weight * (1.0 - discount)
            
            if print_details:
                print(f"\n[GlobalScore Calculation]")
                print(f"  buyer_max_price or seller_min_price is None")
                print(f"  round_index = {round_index}, gamma = {self.gamma}, discount = γ^{round_index} = {discount:.6f}")
                print(f"  FailurePenalty = -F({self.failure_penalty_weight:.1f}) * (1 - discount({discount:.6f})) = {failure_penalty:.3f}")
                print(f"  GlobalScore = {failure_penalty:.3f}")
            return failure_penalty
        
        # Calculate Z
        Z = self.buyer_max_price - self.seller_min_price
        
        # Calculate discount = γ^(t-1)
        round_index = max(0, self.current_round)
        discount = self.gamma ** round_index
        
        # Check feasible_deal: whether negotiation reached agreement
        feasible_deal = (self.negotiation_info.status == NegotiationStatus.AGREED) or (self.state.agreed_price is not None)
        
        # Get the final price (agreed_price if available, otherwise use current prices)
        if self.state.agreed_price is not None:
            final_price = self.state.agreed_price
        elif self.state.buyer_price is not None and self.state.seller_price is not None:
            # Use average if both prices are available but not agreed
            final_price = (self.state.buyer_price + self.state.seller_price) / 2
        elif self.state.buyer_price is not None:
            final_price = self.state.buyer_price
        elif self.state.seller_price is not None:
            final_price = self.state.seller_price
        else:
            # No price available - calculate failure penalty
            failure_penalty = -self.failure_penalty_weight * (1.0 - discount)
            
            if print_details:
                print(f"\n[GlobalScore Calculation]")
                print(f"  Z = buyer_max_price({self.buyer_max_price:.2f}) - seller_min_price({self.seller_min_price:.2f}) = {Z:.2f}")
                print(f"  No final price available")
                print(f"  feasible_deal = {feasible_deal}")
                print(f"  valid_range = (Z > 0) = {Z > 0}")
                print(f"  round_index = {round_index}, gamma = {self.gamma}, discount = γ^{round_index} = {discount:.6f}")
                print(f"  FailurePenalty = -F({self.failure_penalty_weight:.1f}) * (1 - discount({discount:.6f})) = {failure_penalty:.3f}")
                print(f"  GlobalScore = {failure_penalty:.3f}")
            return failure_penalty
        
        # Check valid_range: (Z > 0) and (seller_min_price <= p <= buyer_max_price)
        valid_range = (Z > 0) and (self.seller_min_price <= final_price <= self.buyer_max_price)
        
        # If feasible_deal and valid_range, calculate success scores
        if feasible_deal and valid_range:
            # Calculate utilities
            u_b = (self.buyer_max_price - final_price) / Z
            u_s = (final_price - self.seller_min_price) / Z
            
            # Calculate Q = 4 * u_b * u_s (in [0,1])
            Q = 4.0 * u_b * u_s
            
            # Calculate component scores
            deal_score = self.deal_score_weight * discount  # D * discount
            quality_score = self.quality_score_weight * Q * discount  # W * Q * discount
            efficiency_score = self.efficiency_score_weight * discount  # E * discount
            
            # Calculate GlobalScore
            global_score = deal_score + quality_score + efficiency_score
            
            if print_details:
                # Debug output header
                print(f"\n[GlobalScore Calculation]")
                print(f"  Z = buyer_max_price({self.buyer_max_price:.2f}) - seller_min_price({self.seller_min_price:.2f}) = {Z:.2f}")
                print(f"  final_price = {final_price:.2f}")
                print(f"  feasible_deal = {feasible_deal} (negotiation status: {self.negotiation_info.status.value})")
                print(f"  valid_range = (Z > 0) and (seller_min_price({self.seller_min_price:.2f}) <= final_price({final_price:.2f}) <= buyer_max_price({self.buyer_max_price:.2f})) = {valid_range}")
                print(f"  round_index = {round_index}, gamma = {self.gamma}, discount = γ^{round_index} = {discount:.6f}")
                # Debug output for success case
                print(f"  u_b = (buyer_max_price({self.buyer_max_price:.2f}) - final_price({final_price:.2f})) / Z({Z:.2f}) = {u_b:.4f}")
                print(f"  u_s = (final_price({final_price:.2f}) - seller_min_price({self.seller_min_price:.2f})) / Z({Z:.2f}) = {u_s:.4f}")
                print(f"  Q = 4 * u_b({u_b:.4f}) * u_s({u_s:.4f}) = {Q:.4f}")
                print(f"  DealScore = D({self.deal_score_weight:.1f}) * discount({discount:.6f}) = {deal_score:.3f}")
                print(f"  QualityScore = W({self.quality_score_weight:.1f}) * Q({Q:.4f}) * discount({discount:.6f}) = {quality_score:.3f}")
                print(f"  EfficiencyScore = E({self.efficiency_score_weight:.1f}) * discount({discount:.6f}) = {efficiency_score:.3f}")
                print(f"  GlobalScore = DealScore({deal_score:.3f}) + QualityScore({quality_score:.3f}) + EfficiencyScore({efficiency_score:.3f}) = {global_score:.3f}")
            
            return global_score
        else:
            # Calculate failure penalty
            failure_penalty = -self.failure_penalty_weight * (1.0 - discount)
            
            if print_details:
                # Debug output header
                print(f"\n[GlobalScore Calculation]")
                print(f"  Z = buyer_max_price({self.buyer_max_price:.2f}) - seller_min_price({self.seller_min_price:.2f}) = {Z:.2f}")
                print(f"  final_price = {final_price:.2f}")
                print(f"  feasible_deal = {feasible_deal} (negotiation status: {self.negotiation_info.status.value})")
                print(f"  valid_range = (Z > 0) and (seller_min_price({self.seller_min_price:.2f}) <= final_price({final_price:.2f}) <= buyer_max_price({self.buyer_max_price:.2f})) = {valid_range}")
                print(f"  round_index = {round_index}, gamma = {self.gamma}, discount = γ^{round_index} = {discount:.6f}")
                # Debug output for failure case
                print(f"  FailurePenalty = -F({self.failure_penalty_weight:.1f}) * (1 - discount({discount:.6f})) = {failure_penalty:.3f}")
                print(f"  GlobalScore = {failure_penalty:.3f}")
            
            return failure_penalty
    
    def _calculate_buyer_score(self, print_details: bool = True) -> float:
        """Calculate BuyerScore based on the formula
        
        Returns:
            BuyerScore value (only calculated at final result)
        """
        round_index = max(0, self.current_round - 1)
        discount = self.gamma ** round_index

        if self.use_contract_mode:
            score_terms = self._get_contract_score_terms()
            feasible_deal = (self.negotiation_info.status == NegotiationStatus.AGREED) or (self.state.agreed_price is not None)
            if feasible_deal and score_terms is not None:
                buyer_score = discount * (
                    self.buyer_deal_weight
                    + self.buyer_utility_weight * score_terms["r_b"]
                    + self.buyer_efficiency_weight
                )
                if print_details:
                    print(f"\n[BuyerScore Calculation - Contract Mode]")
                    print(f"  z_max = {self.z_max:.4f}, r_b = {score_terms['r_b']:.4f}")
                    print(f"  BuyerScore = {buyer_score:.3f}")
                return buyer_score

            buyer_score = -self.buyer_failure_penalty_weight * (1.0 - discount)
            if print_details:
                print(f"\n[BuyerScore Calculation - Contract Mode]")
                print(f"  Failed to produce a feasible valid contract")
                print(f"  BuyerScore = {buyer_score:.3f}")
            return buyer_score

        # Check if we have required prices
        if self.buyer_max_price is None or self.seller_min_price is None:
            # Calculate discount for failure penalty
            buyer_score = -self.buyer_failure_penalty_weight * (1.0 - discount)
            
            if print_details:
                print(f"\n[BuyerScore Calculation]")
                print(f"  buyer_max_price or seller_min_price is None")
                print(f"  round_index = {round_index}, gamma = {self.gamma}, discount = γ^{round_index} = {discount:.6f}")
                print(f"  BuyerScore = -Fb({self.buyer_failure_penalty_weight:.1f}) * (1 - discount({discount:.6f})) = {buyer_score:.3f}")
            return buyer_score
        
        # Calculate Z
        Z = self.buyer_max_price - self.seller_min_price
        
        # Calculate discount = γ^(t-1)
        round_index = max(0, self.current_round)
        discount = self.gamma ** round_index
        
        # Check feasible_deal: whether negotiation reached agreement
        feasible_deal = (self.negotiation_info.status == NegotiationStatus.AGREED) or (self.state.agreed_price is not None)
        
        # Get the final price
        if self.state.agreed_price is not None:
            final_price = self.state.agreed_price
        elif self.state.buyer_price is not None and self.state.seller_price is not None:
            final_price = (self.state.buyer_price + self.state.seller_price) / 2
        elif self.state.buyer_price is not None:
            final_price = self.state.buyer_price
        elif self.state.seller_price is not None:
            final_price = self.state.seller_price
        else:
            # No price available - calculate failure penalty
            buyer_score = -self.buyer_failure_penalty_weight * (1.0 - discount)
            
            if print_details:
                print(f"\n[BuyerScore Calculation]")
                print(f"  Z = buyer_max_price({self.buyer_max_price:.2f}) - seller_min_price({self.seller_min_price:.2f}) = {Z:.2f}")
                print(f"  No final price available")
                print(f"  round_index = {round_index}, gamma = {self.gamma}, discount = γ^{round_index} = {discount:.6f}")
                print(f"  BuyerScore = -Fb({self.buyer_failure_penalty_weight:.1f}) * (1 - discount({discount:.6f})) = {buyer_score:.3f}")
            return buyer_score
        
        # Check valid_range: (Z > 0) and (seller_min_price <= p <= buyer_max_price)
        valid_range = (Z > 0) and (self.seller_min_price <= final_price <= self.buyer_max_price)
        
        # If feasible_deal and valid_range, calculate success score
        if feasible_deal and valid_range:
            # Calculate utility
            u_b = (self.buyer_max_price - final_price) / Z
            
            # Calculate BuyerScore = discount * (Db + Wb * u_b + Eb)
            buyer_score = discount * (self.buyer_deal_weight + self.buyer_utility_weight * u_b + self.buyer_efficiency_weight)
            
            if print_details:
                # Debug output header
                print(f"\n[BuyerScore Calculation]")
                print(f"  Z = buyer_max_price({self.buyer_max_price:.2f}) - seller_min_price({self.seller_min_price:.2f}) = {Z:.2f}")
                print(f"  final_price = {final_price:.2f}")
                print(f"  feasible_deal = {feasible_deal} (negotiation status: {self.negotiation_info.status.value})")
                print(f"  valid_range = (Z > 0) and (seller_min_price({self.seller_min_price:.2f}) <= final_price({final_price:.2f}) <= buyer_max_price({self.buyer_max_price:.2f})) = {valid_range}")
                print(f"  round_index = {round_index}, gamma = {self.gamma}, discount = γ^{round_index} = {discount:.6f}")
                # Debug output for success case
                print(f"  u_b = (buyer_max_price({self.buyer_max_price:.2f}) - final_price({final_price:.2f})) / Z({Z:.2f}) = {u_b:.4f}")
                print(f"  BuyerScore = discount({discount:.6f}) * (Db({self.buyer_deal_weight:.1f}) + Wb({self.buyer_utility_weight:.1f}) * u_b({u_b:.4f}) + Eb({self.buyer_efficiency_weight:.1f}))")
                print(f"  BuyerScore = {discount:.6f} * ({self.buyer_deal_weight:.1f} + {self.buyer_utility_weight * u_b:.4f} + {self.buyer_efficiency_weight:.1f}) = {buyer_score:.3f}")
            
            return buyer_score
        else:
            # Calculate failure penalty (out-of-range deals treated as failures)
            buyer_score = -self.buyer_failure_penalty_weight * (1.0 - discount)
            
            if print_details:
                # Debug output header
                print(f"\n[BuyerScore Calculation]")
                print(f"  Z = buyer_max_price({self.buyer_max_price:.2f}) - seller_min_price({self.seller_min_price:.2f}) = {Z:.2f}")
                print(f"  final_price = {final_price:.2f}")
                print(f"  feasible_deal = {feasible_deal} (negotiation status: {self.negotiation_info.status.value})")
                print(f"  valid_range = (Z > 0) and (seller_min_price({self.seller_min_price:.2f}) <= final_price({final_price:.2f}) <= buyer_max_price({self.buyer_max_price:.2f})) = {valid_range}")
                print(f"  round_index = {round_index}, gamma = {self.gamma}, discount = γ^{round_index} = {discount:.6f}")
                # Debug output for failure case
                print(f"  BuyerScore = -Fb({self.buyer_failure_penalty_weight:.1f}) * (1 - discount({discount:.6f})) = {buyer_score:.3f}")
            
            return buyer_score
    
    def _calculate_seller_score(self, print_details: bool = True) -> float:
        """Calculate SellerScore based on the formula
        
        Returns:
            SellerScore value (only calculated at final result)
        """
        round_index = max(0, self.current_round - 1)
        discount = self.gamma ** round_index

        if self.use_contract_mode:
            score_terms = self._get_contract_score_terms()
            feasible_deal = (self.negotiation_info.status == NegotiationStatus.AGREED) or (self.state.agreed_price is not None)
            if feasible_deal and score_terms is not None:
                seller_score = discount * (
                    self.seller_deal_weight
                    + self.seller_utility_weight * score_terms["r_s"]
                    + self.seller_efficiency_weight
                )
                if print_details:
                    print(f"\n[SellerScore Calculation - Contract Mode]")
                    print(f"  z_max = {self.z_max:.4f}, r_s = {score_terms['r_s']:.4f}")
                    print(f"  SellerScore = {seller_score:.3f}")
                return seller_score

            seller_score = -self.seller_failure_penalty_weight * (1.0 - discount)
            if print_details:
                print(f"\n[SellerScore Calculation - Contract Mode]")
                print(f"  Failed to produce a feasible valid contract")
                print(f"  SellerScore = {seller_score:.3f}")
            return seller_score

        # Check if we have required prices
        if self.buyer_max_price is None or self.seller_min_price is None:
            # Calculate discount for failure penalty
            seller_score = -self.seller_failure_penalty_weight * (1.0 - discount)
            
            if print_details:
                print(f"\n[SellerScore Calculation]")
                print(f"  buyer_max_price or seller_min_price is None")
                print(f"  round_index = {round_index}, gamma = {self.gamma}, discount = γ^{round_index} = {discount:.6f}")
                print(f"  SellerScore = -Fs({self.seller_failure_penalty_weight:.1f}) * (1 - discount({discount:.6f})) = {seller_score:.3f}")
            return seller_score
        
        # Calculate Z
        Z = self.buyer_max_price - self.seller_min_price
        
        # Calculate discount = γ^(t-1)
        round_index = max(0, self.current_round)
        discount = self.gamma ** round_index
        
        # Check feasible_deal: whether negotiation reached agreement
        feasible_deal = (self.negotiation_info.status == NegotiationStatus.AGREED) or (self.state.agreed_price is not None)
        
        # Get the final price
        if self.state.agreed_price is not None:
            final_price = self.state.agreed_price
        elif self.state.buyer_price is not None and self.state.seller_price is not None:
            final_price = (self.state.buyer_price + self.state.seller_price) / 2
        elif self.state.buyer_price is not None:
            final_price = self.state.buyer_price
        elif self.state.seller_price is not None:
            final_price = self.state.seller_price
        else:
            # No price available - calculate failure penalty
            seller_score = -self.seller_failure_penalty_weight * (1.0 - discount)
            
            if print_details:
                print(f"\n[SellerScore Calculation]")
                print(f"  Z = buyer_max_price({self.buyer_max_price:.2f}) - seller_min_price({self.seller_min_price:.2f}) = {Z:.2f}")
                print(f"  No final price available")
                print(f"  round_index = {round_index}, gamma = {self.gamma}, discount = γ^{round_index} = {discount:.6f}")
                print(f"  SellerScore = -Fs({self.seller_failure_penalty_weight:.1f}) * (1 - discount({discount:.6f})) = {seller_score:.3f}")
            return seller_score
        
        # Check valid_range: (Z > 0) and (seller_min_price <= p <= buyer_max_price)
        valid_range = (Z > 0) and (self.seller_min_price <= final_price <= self.buyer_max_price)
        
        # If feasible_deal and valid_range, calculate success score
        if feasible_deal and valid_range:
            # Calculate utility
            u_s = (final_price - self.seller_min_price) / Z
            
            # Calculate SellerScore = discount * (Ds + Ws * u_s + Es)
            seller_score = discount * (self.seller_deal_weight + self.seller_utility_weight * u_s + self.seller_efficiency_weight)
            
            if print_details:
                # Debug output header
                print(f"\n[SellerScore Calculation]")
                print(f"  Z = buyer_max_price({self.buyer_max_price:.2f}) - seller_min_price({self.seller_min_price:.2f}) = {Z:.2f}")
                print(f"  final_price = {final_price:.2f}")
                print(f"  feasible_deal = {feasible_deal} (negotiation status: {self.negotiation_info.status.value})")
                print(f"  valid_range = (Z > 0) and (seller_min_price({self.seller_min_price:.2f}) <= final_price({final_price:.2f}) <= buyer_max_price({self.buyer_max_price:.2f})) = {valid_range}")
                print(f"  round_index = {round_index}, gamma = {self.gamma}, discount = γ^{round_index} = {discount:.6f}")
                # Debug output for success case
                print(f"  u_s = (final_price({final_price:.2f}) - seller_min_price({self.seller_min_price:.2f})) / Z({Z:.2f}) = {u_s:.4f}")
                print(f"  SellerScore = discount({discount:.6f}) * (Ds({self.seller_deal_weight:.1f}) + Ws({self.seller_utility_weight:.1f}) * u_s({u_s:.4f}) + Es({self.seller_efficiency_weight:.1f}))")
                print(f"  SellerScore = {discount:.6f} * ({self.seller_deal_weight:.1f} + {self.seller_utility_weight * u_s:.4f} + {self.seller_efficiency_weight:.1f}) = {seller_score:.3f}")
            
            return seller_score
        else:
            # Calculate failure penalty (out-of-range deals treated as failures)
            seller_score = -self.seller_failure_penalty_weight * (1.0 - discount)
            
            if print_details:
                # Debug output header
                print(f"\n[SellerScore Calculation]")
                print(f"  Z = buyer_max_price({self.buyer_max_price:.2f}) - seller_min_price({self.seller_min_price:.2f}) = {Z:.2f}")
                print(f"  final_price = {final_price:.2f}")
                print(f"  feasible_deal = {feasible_deal} (negotiation status: {self.negotiation_info.status.value})")
                print(f"  valid_range = (Z > 0) and (seller_min_price({self.seller_min_price:.2f}) <= final_price({final_price:.2f}) <= buyer_max_price({self.buyer_max_price:.2f})) = {valid_range}")
                print(f"  round_index = {round_index}, gamma = {self.gamma}, discount = γ^{round_index} = {discount:.6f}")
                # Debug output for failure case
                print(f"  SellerScore = -Fs({self.seller_failure_penalty_weight:.1f}) * (1 - discount({discount:.6f})) = {seller_score:.3f}")
            
            return seller_score
    
    def _print_global_score_details(self):
        """Print GlobalScore calculation details (called from example code after Step Rewards)"""
        self._calculate_global_score(print_details=True)
    
    def _print_buyer_score_details(self):
        """Print BuyerScore calculation details (called from example code after Step Rewards)"""
        self._calculate_buyer_score(print_details=True)
    
    def _print_seller_score_details(self):
        """Print SellerScore calculation details (called from example code after Step Rewards)"""
        self._calculate_seller_score(print_details=True)

