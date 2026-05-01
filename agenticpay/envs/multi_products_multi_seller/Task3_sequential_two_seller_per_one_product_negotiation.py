"""Task3 Sequential Two-Seller Per One Product Negotiation Environment Implementation

Supports sequential negotiation where buyer chooses one seller per round to negotiate with.
Each seller has their own unique product. Buyer can switch between two sellers and make a deal with either seller.
"""

from __future__ import annotations

import json
import re
from copy import deepcopy
from typing import Any, Dict, Optional, Tuple

from agenticpay.core import BaseEnv, NegotiationStatus, NegotiationInfo
from agenticpay.agents.base_agent import BaseAgent
from agenticpay.memory.conversation_memory import ConversationMemory
from agenticpay.utils.negotiation_state import NegotiationState


def _image_urls_from_product_info(product_info: Dict[str, Any]) -> list:
    """Collect image URLs from flat product_info or from product_info['products']."""
    urls: list = []
    if not product_info:
        return urls
    top = product_info.get("image_path") or product_info.get("image_url")
    if top:
        urls.append(top)
    for p in product_info.get("products") or []:
        if not isinstance(p, dict):
            continue
        u = p.get("image_path") or p.get("image_url")
        if u:
            urls.append(u)
    seen: set = set()
    out: list = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


class Task3SequentialTwoSellerPerOneProductNegotiation(BaseEnv):
    """Task3 Sequential Two-Seller Per One Product Negotiation Environment
    
    Manages sequential negotiation process where buyer chooses one seller per round to negotiate with.
    Each seller has their own unique product. Buyer can switch between two sellers and make a deal with either seller.
    """
    
    def __init__(
        self,
        buyer_agent: BaseAgent,
        seller1_agent: BaseAgent,
        seller2_agent: BaseAgent,
        max_rounds: int = 20,
        initial_seller1_price: float = 100.0,
        initial_seller2_price: float = 110.0,
        buyer_max_price: Optional[float] = None,
        seller1_min_price: Optional[float] = None,
        seller2_min_price: Optional[float] = None,
        environment_info: Optional[Dict[str, Any]] = None,
        price_tolerance: float = 1.0,
        reward_weights: Optional[Dict[str, float]] = None,
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
        """Initialize sequential multi-seller negotiation environment
        
        Args:
            buyer_agent: Buyer Agent
            seller1_agent: First Seller Agent
            seller2_agent: Second Seller Agent
            max_rounds: Maximum number of negotiation rounds
            initial_seller1_price: Initial price offered by seller1
            initial_seller2_price: Initial price offered by seller2
            buyer_max_price: Maximum acceptable price for buyer (confidential)
            seller1_min_price: Minimum acceptable price for seller1 (confidential)
            seller2_min_price: Minimum acceptable price for seller2 (confidential)
            environment_info: Environment information (e.g., season, weather, etc.)
            price_tolerance: Price tolerance for determining agreement
            reward_weights: Reward weights configuration dict with keys:
                - buyer_savings: weight for buyer savings (default: 1.0)
                - seller_profit: weight for seller profit (default: 1.0)
                - time_cost: weight for time cost (default: 0.1)
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
        self.seller1_agent = seller1_agent
        self.seller2_agent = seller2_agent
        self.max_rounds = max_rounds
        self.initial_seller1_price = initial_seller1_price
        self.initial_seller2_price = initial_seller2_price
        self.buyer_max_price = buyer_max_price
        self.seller1_min_price = seller1_min_price
        self.seller2_min_price = seller2_min_price
        self.environment_info = environment_info or {}
        self.contract_configs = self._normalize_contract_configs(self.environment_info)
        self.use_contract_mode = bool(self.contract_configs)
        self.z_max_by_seller = {
            seller_id: self._calculate_z_max(config)
            for seller_id, config in self.contract_configs.items()
        } if self.use_contract_mode else {}
        self.price_tolerance = price_tolerance
        
        # Set default reward weights
        default_weights = {
            "buyer_savings": 1.0,      # Buyer savings weight
            "seller_profit": 1.0,      # Seller profit weight
            "time_cost": 0.1,          # Time cost weight (reduced impact)
        }
        if reward_weights is not None:
            default_weights.update(reward_weights)
        self.reward_weights = default_weights
        
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
        
        # State management - separate for each seller
        self.memory_seller1 = ConversationMemory()
        self.memory_seller2 = ConversationMemory()
        self.state_seller1 = NegotiationState()
        self.state_seller2 = NegotiationState()
        self.current_round = 0
        self.negotiation_info = NegotiationInfo()
        
        # Track which seller is currently selected and which seller was chosen for the deal
        self.current_selected_seller: Optional[int] = None  # 1 or 2, selected for current round
        self.final_selected_seller: Optional[int] = None  # 1 or 2, chosen for final deal
        self.final_deal_price: Optional[float] = None
        
        # Store product info for each seller
        self.seller1_product_info: Optional[Dict[str, Any]] = None
        self.seller2_product_info: Optional[Dict[str, Any]] = None

        self._reset_contract_metadata()

    def _normalize_contract_configs(self, environment_info: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
        """Normalize seller-specific or shared contract config into {1: cfg, 2: cfg}."""
        raw_configs = environment_info.get("seller_contract_configs")
        if isinstance(raw_configs, dict):
            configs: Dict[int, Dict[str, Any]] = {}
            for seller_id in (1, 2):
                value = raw_configs.get(seller_id) or raw_configs.get(str(seller_id)) or raw_configs.get(f"seller{seller_id}")
                if isinstance(value, dict):
                    configs[seller_id] = deepcopy(value)
            if configs:
                return configs

        shared_config = environment_info.get("contract_config")
        if isinstance(shared_config, dict) and shared_config:
            return {1: deepcopy(shared_config), 2: deepcopy(shared_config)}
        return {}

    def _reset_contract_metadata(self) -> None:
        """Initialize per-seller contract state used by contract-mode scoring."""
        for state in (self.state_seller1, self.state_seller2):
            state.metadata["buyer_contract"] = None
            state.metadata["seller_contract"] = None
            state.metadata["agreed_contract"] = None
            state.metadata["buyer_utility"] = None
            state.metadata["seller_utility"] = None
            state.metadata["z_max"] = None
        for seller_id, z_max in self.z_max_by_seller.items():
            self._get_seller_state(seller_id).metadata["z_max"] = z_max

    def _get_seller_state(self, seller_id: int) -> NegotiationState:
        if seller_id == 1:
            return self.state_seller1
        if seller_id == 2:
            return self.state_seller2
        raise ValueError(f"seller_id must be 1 or 2, got {seller_id}")

    def _get_seller_contract_config(self, seller_id: int) -> Dict[str, Any]:
        return self.contract_configs.get(seller_id, {})

    def _build_public_contract_config(self, seller_id: int) -> Dict[str, Any]:
        """Build contract config fields both sides may see for one seller."""
        config = self._get_seller_contract_config(seller_id)
        if not config:
            return {}
        public_config: Dict[str, Any] = {"seller_id": seller_id}
        for key in ("continuous_bounds", "discrete_options", "field_descriptions", "contrainfo"):
            if key in config:
                public_config[key] = deepcopy(config[key])
        return public_config

    def _build_role_contract_config(self, role: str, seller_id: int) -> Dict[str, Any]:
        """Expose only the role's private preferences for one seller."""
        config = self._get_seller_contract_config(seller_id)
        role_config = self._build_public_contract_config(seller_id)
        if role == "buyer" and "buyer_preferences" in config:
            role_config["buyer_preferences"] = deepcopy(config["buyer_preferences"])
        if role == "seller" and "seller_preferences" in config:
            role_config["seller_preferences"] = deepcopy(config["seller_preferences"])
        return role_config

    def _build_buyer_contract_configs(self) -> Dict[int, Dict[str, Any]]:
        return {
            seller_id: self._build_role_contract_config("buyer", seller_id)
            for seller_id in (1, 2)
            if seller_id in self.contract_configs
        }

    def _build_role_environment_info(self, role: str, seller_id: Optional[int] = None) -> Dict[str, Any]:
        """Build role-specific environment info without leaking counterparty preferences."""
        role_env_info = deepcopy(self.environment_info)
        role_env_info.pop("contract_config", None)
        role_env_info.pop("seller_contract_configs", None)
        if not self.use_contract_mode:
            return role_env_info

        if role == "buyer":
            role_env_info["seller_contract_configs"] = self._build_buyer_contract_configs()
            if seller_id is not None:
                role_env_info["contract_config"] = self._build_role_contract_config("buyer", seller_id)
        elif role == "seller" and seller_id is not None:
            role_env_info["contract_config"] = self._build_role_contract_config("seller", seller_id)
        return role_env_info

    def _normalize_contract(self, contract: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Normalize contract schema to a stable internal representation."""
        if not isinstance(contract, dict):
            return None
        try:
            price = float(contract.get("price"))
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

    def _validate_contract(self, contract: Optional[Dict[str, Any]], seller_id: int) -> bool:
        """Validate contract against the selected seller's bounds/options."""
        if not contract:
            return False
        if not self.use_contract_mode:
            return True

        config = self._get_seller_contract_config(seller_id)
        continuous_bounds = config.get("continuous_bounds", {})
        discrete_options = config.get("discrete_options", {})
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

    def _calculate_contract_utilities(self, contract: Dict[str, Any], seller_id: int) -> Tuple[float, float]:
        """Compute raw MAUT utilities (U_b, U_s) for one seller's contract config."""
        config = self._get_seller_contract_config(seller_id)
        buyer_prefs = config.get("buyer_preferences", {})
        seller_prefs = config.get("seller_preferences", {})

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

    def _calculate_z_max(self, config: Dict[str, Any]) -> Optional[float]:
        """Compute theoretical max surplus Z_max for one seller."""
        if not config:
            return None

        buyer_prefs = config.get("buyer_preferences", {})
        seller_prefs = config.get("seller_preferences", {})
        z_max = float(buyer_prefs.get("v_base", 0.0)) - float(seller_prefs.get("c_base", 0.0))

        buyer_cw = buyer_prefs.get("continuous_weights", {})
        seller_cw = seller_prefs.get("continuous_weights", {})
        for term, bounds in config.get("continuous_bounds", {}).items():
            total_weight = float(buyer_cw.get(term, 0.0)) + float(seller_cw.get(term, 0.0))
            min_v = float(bounds.get("min", 0.0))
            max_v = float(bounds.get("max", 0.0))
            z_max += total_weight * (max_v if total_weight >= 0 else min_v)

        buyer_dw = buyer_prefs.get("discrete_weights", {})
        seller_dw = seller_prefs.get("discrete_weights", {})
        for term, options in config.get("discrete_options", {}).items():
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

    def _resolve_agreed_contract(self, seller_id: int) -> Optional[Dict[str, Any]]:
        """Build the final contract for a compatible buyer/seller pair."""
        state = self._get_seller_state(seller_id)
        buyer_contract = state.metadata.get("buyer_contract")
        seller_contract = state.metadata.get("seller_contract")
        if not buyer_contract or not seller_contract:
            return None
        if not self._contracts_compatible(buyer_contract, seller_contract):
            return None

        buyer_price = float(buyer_contract["price"])
        seller_price = float(seller_contract["price"])
        agreed_price = seller_price if seller_price <= buyer_price else (buyer_price + seller_price) / 2
        return {
            "price": agreed_price,
            "continuous_terms": buyer_contract.get("continuous_terms", {}),
            "discrete_terms": buyer_contract.get("discrete_terms", {}),
        }

    def _get_market_best_contract_seller(self) -> Optional[int]:
        """Choose the seller with the highest theoretical total utility."""
        candidates = [
            (seller_id, z_max)
            for seller_id, z_max in self.z_max_by_seller.items()
            if z_max is not None
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda item: item[1])[0]
    
    def reset(
        self,
        user_requirement: str = "",
        seller1_product_info: Optional[Dict[str, Any]] = None,
        seller2_product_info: Optional[Dict[str, Any]] = None,
        user_profile: Optional[Any] = None,
        **kwargs: Any,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Reset environment, start new negotiation
        
        Args:
            user_requirement: User requirement description
            seller1_product_info: Product information for seller1
            seller2_product_info: Product information for seller2
            user_profile: User profile
            **kwargs: Other parameters
            
        Returns:
            (observation, info) Initial observation and info
        """
        # Reset state
        self.memory_seller1.clear()
        self.memory_seller2.clear()
        self.state_seller1 = NegotiationState()
        self.state_seller2 = NegotiationState()
        self.current_round = 0
        self.negotiation_info = NegotiationInfo()
        self.current_selected_seller = None
        self.final_selected_seller = None
        self.final_deal_price = None
        self._reset_contract_metadata()
        
        # Store product info for each seller
        self.seller1_product_info = seller1_product_info or {}
        self.seller2_product_info = seller2_product_info or {}
        
        # Extract product_images for VLM (top-level image_url and/or products[].image_url)
        imgs_s1 = _image_urls_from_product_info(self.seller1_product_info)
        imgs_s2 = _image_urls_from_product_info(self.seller2_product_info)
        product_images: list = []
        seen_pi: set = set()
        for u in imgs_s1 + imgs_s2:
            if u not in seen_pi:
                seen_pi.add(u)
                product_images.append(u)
        if not product_images:
            product_images = None
        self.product_images = product_images
        
        # Initialize Buyer Agent (buyer knows about both sellers and their products)
        buyer_context = {
            "user_requirement": user_requirement,
            "max_price": self.buyer_max_price,
            "user_profile": user_profile,
            "environment_info": self._build_role_environment_info("buyer"),
            "seller1_product_info": self.seller1_product_info,
            "seller2_product_info": self.seller2_product_info,
            "num_sellers": 2,  # Inform buyer there are 2 sellers
            "negotiation_mode": "sequential",  # Inform buyer this is sequential negotiation
            "product_images": product_images,  # For VLM: product images (URL/path) for img input
            "seller_contract_configs": self._build_buyer_contract_configs() if self.use_contract_mode else {},
        }
        prods = self.seller1_product_info.get("products")
        if (
            prods
            and isinstance(prods, list)
            and len(prods) >= 2
            and self.seller1_product_info is self.seller2_product_info
        ):
            buyer_context["negotiation_scope"] = (
                "Both sellers list the SAME two products as one bundle. "
                "Compare their TOTAL bundle prices only; ### BUYER_PRICE($X) ### must be the combined price for both items."
            )
        self.buyer_agent.initialize(buyer_context)
        
        # Initialize Seller1 Agent with its own product
        seller1_images = imgs_s1 if imgs_s1 else None
        seller1_context = {
            "product_info": self.seller1_product_info,
            "initial_price": self.initial_seller1_price,
            "min_price": self.seller1_min_price,
            "environment_info": self._build_role_environment_info("seller", 1),
            "seller_id": 1,  # Identify as seller 1
            "product_images": seller1_images,  # For VLM: product image (URL/path) for img input
            "contract_config": self._build_role_contract_config("seller", 1) if self.use_contract_mode else {},
        }
        self.seller1_agent.initialize(seller1_context)
        
        # Initialize Seller2 Agent with its own product
        seller2_images = imgs_s2 if imgs_s2 else None
        seller2_context = {
            "product_info": self.seller2_product_info,
            "initial_price": self.initial_seller2_price,
            "min_price": self.seller2_min_price,
            "environment_info": self._build_role_environment_info("seller", 2),
            "seller_id": 2,  # Identify as seller 2
            "product_images": seller2_images,  # For VLM: product image (URL/path) for img input
            "contract_config": self._build_role_contract_config("seller", 2) if self.use_contract_mode else {},
        }
        self.seller2_agent.initialize(seller2_context)
        
        # No initial seller offers - negotiation starts with buyer's first message
        # Build observation
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(
        self, 
        selected_seller: int,  # 1 or 2, which seller buyer chooses to negotiate with this round
        buyer_action: Optional[str] = None,
        seller_action: Optional[str] = None,
    ) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """Execute one negotiation step
        
        Each round, buyer chooses one seller to negotiate with, then buyer and that seller exchange messages.
        Order: buyer -> seller
        
        Args:
            selected_seller: Which seller (1 or 2) buyer chooses to negotiate with this round
            buyer_action: Buyer's response (optional)
            seller_action: Selected seller's response (optional)
            
        Returns:
            (observation, reward, terminated, truncated, info)
        """
        if selected_seller not in [1, 2]:
            raise ValueError(f"selected_seller must be 1 or 2, got {selected_seller}")
        
        self.current_selected_seller = selected_seller
        
        # Add messages to memory in order: buyer -> seller
        # Process buyer action first
        if buyer_action is not None:
            if selected_seller == 1:
                self.memory_seller1.add_message("buyer", buyer_action, self.current_round)
                buyer_contract = self._extract_contract(buyer_action) if self.use_contract_mode else None
                if buyer_contract and self._validate_contract(buyer_contract, 1):
                    self.state_seller1.metadata["buyer_contract"] = buyer_contract
                    buyer_price = float(buyer_contract["price"])
                else:
                    buyer_price = self._extract_price(buyer_action)
                if buyer_price is not None:
                    self.state_seller1.update(buyer_price=buyer_price)
            else:  # selected_seller == 2
                self.memory_seller2.add_message("buyer", buyer_action, self.current_round)
                buyer_contract = self._extract_contract(buyer_action) if self.use_contract_mode else None
                if buyer_contract and self._validate_contract(buyer_contract, 2):
                    self.state_seller2.metadata["buyer_contract"] = buyer_contract
                    buyer_price = float(buyer_contract["price"])
                else:
                    buyer_price = self._extract_price(buyer_action)
                if buyer_price is not None:
                    self.state_seller2.update(buyer_price=buyer_price)
        
        # Process seller action after buyer
        if seller_action is not None:
            if selected_seller == 1:
                self.memory_seller1.add_message("seller", seller_action, self.current_round)
                seller_contract = self._extract_contract(seller_action) if self.use_contract_mode else None
                if seller_contract and self._validate_contract(seller_contract, 1):
                    self.state_seller1.metadata["seller_contract"] = seller_contract
                    seller_price = float(seller_contract["price"])
                else:
                    seller_price = self._extract_price(seller_action)
                if seller_price is not None:
                    self.state_seller1.update(seller_price=seller_price)
            else:  # selected_seller == 2
                self.memory_seller2.add_message("seller", seller_action, self.current_round)
                seller_contract = self._extract_contract(seller_action) if self.use_contract_mode else None
                if seller_contract and self._validate_contract(seller_contract, 2):
                    self.state_seller2.metadata["seller_contract"] = seller_contract
                    seller_price = float(seller_contract["price"])
                else:
                    seller_price = self._extract_price(seller_action)
                if seller_price is not None:
                    self.state_seller2.update(seller_price=seller_price)
        
        # Check if deal can be made with the selected seller
        # Deal is made when: (1) price difference <= tolerance, or (2) seller's offer <= buyer's offer
        if selected_seller == 1:
            if self.use_contract_mode:
                agreed_contract = self._resolve_agreed_contract(1)
                if agreed_contract:
                    self.final_selected_seller = 1
                    self.final_deal_price = float(agreed_contract["price"])
                    self.state_seller1.metadata["agreed_contract"] = agreed_contract
            elif (buyer_action is not None and 
                self.state_seller1.buyer_price is not None and 
                self.state_seller1.seller_price is not None):
                price_diff = abs(self.state_seller1.buyer_price - self.state_seller1.seller_price)
                if price_diff <= self.price_tolerance:
                    self.final_selected_seller = 1
                    self.final_deal_price = (self.state_seller1.buyer_price + self.state_seller1.seller_price) / 2
                elif self.state_seller1.seller_price <= self.state_seller1.buyer_price:
                    self.final_selected_seller = 1
                    self.final_deal_price = self.state_seller1.seller_price
        else:  # selected_seller == 2
            if self.use_contract_mode:
                agreed_contract = self._resolve_agreed_contract(2)
                if agreed_contract:
                    self.final_selected_seller = 2
                    self.final_deal_price = float(agreed_contract["price"])
                    self.state_seller2.metadata["agreed_contract"] = agreed_contract
            elif (buyer_action is not None and 
                self.state_seller2.buyer_price is not None and 
                self.state_seller2.seller_price is not None):
                price_diff = abs(self.state_seller2.buyer_price - self.state_seller2.seller_price)
                if price_diff <= self.price_tolerance:
                    self.final_selected_seller = 2
                    self.final_deal_price = (self.state_seller2.buyer_price + self.state_seller2.seller_price) / 2
                elif self.state_seller2.seller_price <= self.state_seller2.buyer_price:
                    self.final_selected_seller = 2
                    self.final_deal_price = self.state_seller2.seller_price
        
        # Check if deal is made
        terminated = False
        truncated = False
        reward = 0.0
        buyer_reward = 0.0
        seller1_reward = 0.0
        seller2_reward = 0.0
        
        if self.final_selected_seller is not None and self.final_deal_price is not None:
            terminated = True
            self.negotiation_info.status = NegotiationStatus.AGREED
            # Increment current_round to reflect that this round is completed
            # This ensures round count is accurate when calculating final scores
            self.current_round += 1
            self.negotiation_info.round_count = self.current_round
            reward = self._calculate_reward()
            buyer_reward = self._calculate_buyer_reward()
            seller1_reward = self._calculate_seller_reward(1)
            seller2_reward = self._calculate_seller_reward(2)
        elif self.current_round >= self.max_rounds:
            truncated = True
            self.negotiation_info.status = NegotiationStatus.TIMEOUT
            # Increment current_round to reflect that this round is completed
            # This ensures round count is accurate when calculating final scores
            self.current_round += 1
            self.negotiation_info.round_count = self.current_round
            reward = self._calculate_reward()
            buyer_reward = self._calculate_buyer_reward()
            seller1_reward = self._calculate_seller_reward(1)
            seller2_reward = self._calculate_seller_reward(2)
        else:
            # Move to next round
            self.current_round += 1
            self.negotiation_info.round_count = self.current_round
        
        # Calculate step rewards for every round
        # Only calculate for the selected seller in this round (sequential negotiation)
        step_buyer_reward = self._calculate_step_buyer_reward()
        step_seller1_reward = self._calculate_step_seller_reward(1) if selected_seller == 1 else None
        step_seller2_reward = self._calculate_step_seller_reward(2) if selected_seller == 2 else None
        
        # Build observation and info
        observation = self._get_observation()
        info = self._get_info()
        
        # Add step rewards to info for every step
        info["step_buyer_reward"] = step_buyer_reward
        if step_seller1_reward is not None:
            info["step_seller1_reward"] = step_seller1_reward
        if step_seller2_reward is not None:
            info["step_seller2_reward"] = step_seller2_reward
        
        if terminated or truncated:
            info["termination_reason"] = "agreed" if terminated else "timeout"
            if terminated:
                info["selected_seller"] = self.final_selected_seller
                info["final_deal_price"] = self.final_deal_price
            info["buyer_reward"] = buyer_reward
            info["seller1_reward"] = seller1_reward
            info["seller2_reward"] = seller2_reward
            # Calculate GlobalScore, BuyerScore, and SellerScore for final result
            # Note: current_round has been incremented to reflect the completed round
            # Don't print here - will be printed in example file after Step Rewards
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
        
        Args:
            mode: Render mode, "human" prints to console, "text" returns text
            
        Returns:
            Returns string if mode="text", otherwise returns None
        """
        output_lines = []
        
        # Get messages from the round that just completed
        # Note: In step(), messages are added to current_round
        # - If agreement reached: current_round stays the same, messages are in current_round
        # - If no agreement: current_round is incremented, messages are in current_round - 1
        history_seller1 = self.memory_seller1.get_history()
        history_seller2 = self.memory_seller2.get_history()
        
        # Determine which round's messages to display
        # Messages are stored with the round value at the time of storage (before current_round is incremented)
        # In step(), messages are added first, then current_round is incremented
        # So for any completed round, messages are stored at current_round - 1
        round_to_display = self.current_round - 1 if self.current_round > 0 else 0
        
        # Display round number: current_round is already incremented, so it represents the completed round number
        display_round = self.current_round
        
        output_lines.append(f"\n{'='*60}")
        output_lines.append(f"Round {display_round} - Sequential Negotiation Output")
        output_lines.append(f"{'='*60}")
        
        # Display which seller was selected this round
        if self.current_selected_seller is not None:
            output_lines.append(f"\n[Selected Seller: Seller {self.current_selected_seller}]")
        
        # Display Seller1 conversation (if this round negotiated with seller1)
        if self.current_selected_seller == 1:
            output_lines.append(f"\n[SELLER 1 Conversation]:")
            if history_seller1:
                round_messages_s1 = [
                    msg for msg in history_seller1 if msg["round"] == round_to_display
                ]
                if round_messages_s1:
                    # Display buyer message first (if exists)
                    buyer_msg_s1 = next(
                        (msg for msg in round_messages_s1 if msg["role"] == "buyer"), 
                        None
                    )
                    if buyer_msg_s1:
                        output_lines.append(f"  [BUYER]: {buyer_msg_s1['content']}")
                    
                    # Display seller message (if exists)
                    seller_msg_s1 = next(
                        (msg for msg in round_messages_s1 if msg["role"] == "seller"), 
                        None
                    )
                    if seller_msg_s1:
                        output_lines.append(f"  [SELLER]: {seller_msg_s1['content']}")
        
        # Display Seller2 conversation (if this round negotiated with seller2)
        if self.current_selected_seller == 2:
            output_lines.append(f"\n[SELLER 2 Conversation]:")
            if history_seller2:
                round_messages_s2 = [
                    msg for msg in history_seller2 if msg["round"] == round_to_display
                ]
                if round_messages_s2:
                    # Display buyer message first (if exists)
                    buyer_msg_s2 = next(
                        (msg for msg in round_messages_s2 if msg["role"] == "buyer"), 
                        None
                    )
                    if buyer_msg_s2:
                        output_lines.append(f"  [BUYER]: {buyer_msg_s2['content']}")
                    
                    # Display seller message (if exists)
                    seller_msg_s2 = next(
                        (msg for msg in round_messages_s2 if msg["role"] == "seller"), 
                        None
                    )
                    if seller_msg_s2:
                        output_lines.append(f"  [SELLER]: {seller_msg_s2['content']}")
        
        # Round summary section
        output_lines.append(f"\n{'-'*60}")
        output_lines.append(f"Round {self.current_round} Summary:")
        output_lines.append(f"{'-'*60}")
        
        # Display Seller1 prices
        output_lines.append(f"\nSeller 1:")
        if self.use_contract_mode:
            buyer_contract = self.state_seller1.metadata.get("buyer_contract")
            seller_contract = self.state_seller1.metadata.get("seller_contract")
            agreed_contract = self.state_seller1.metadata.get("agreed_contract")
            output_lines.append(f"  Buyer Contract: {buyer_contract if buyer_contract else 'Not specified'}")
            output_lines.append(f"  Seller Contract: {seller_contract if seller_contract else 'Not specified'}")
            if agreed_contract:
                output_lines.append(f"  Agreed Contract: {agreed_contract}")
        else:
            if self.state_seller1.buyer_price is not None:
                output_lines.append(f"  Buyer Price: ${self.state_seller1.buyer_price:.2f}")
            else:
                output_lines.append(f"  Buyer Price: Not specified")
            if self.state_seller1.seller_price is not None:
                output_lines.append(f"  Seller Price: ${self.state_seller1.seller_price:.2f}")
            else:
                output_lines.append(f"  Seller Price: Not specified")
        
        # Display Seller2 prices
        output_lines.append(f"\nSeller 2:")
        if self.use_contract_mode:
            buyer_contract = self.state_seller2.metadata.get("buyer_contract")
            seller_contract = self.state_seller2.metadata.get("seller_contract")
            agreed_contract = self.state_seller2.metadata.get("agreed_contract")
            output_lines.append(f"  Buyer Contract: {buyer_contract if buyer_contract else 'Not specified'}")
            output_lines.append(f"  Seller Contract: {seller_contract if seller_contract else 'Not specified'}")
            if agreed_contract:
                output_lines.append(f"  Agreed Contract: {agreed_contract}")
        else:
            if self.state_seller2.buyer_price is not None:
                output_lines.append(f"  Buyer Price: ${self.state_seller2.buyer_price:.2f}")
            else:
                output_lines.append(f"  Buyer Price: Not specified")
            if self.state_seller2.seller_price is not None:
                output_lines.append(f"  Seller Price: ${self.state_seller2.seller_price:.2f}")
            else:
                output_lines.append(f"  Seller Price: Not specified")
        
        # Display deal status
        if self.final_selected_seller is not None:
            output_lines.append(f"\n  ✓ DEAL MADE with Seller {self.final_selected_seller}")
            if self.final_deal_price is not None:
                output_lines.append(f"  Final Deal Price: ${self.final_deal_price:.2f}")
        else:
            output_lines.append(f"\n  ✗ NO DEAL YET")
        
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
    
    def close(self):
        """Close environment, cleanup resources"""
        self.memory_seller1.clear()
        self.memory_seller2.clear()
        self.state_seller1 = NegotiationState()
        self.state_seller2 = NegotiationState()
    
    @staticmethod
    def resolve_selected_seller(
        buyer_message: str,
        observation: Dict[str, Any],
        last_selected_seller: Optional[int] = None,
    ) -> int:
        """Resolve which seller (1 or 2) the buyer is addressing this turn.
        
        Priority: explicit ``last_selected_seller`` from BuyerAgent (``<selected_seller>`` in full model
        output), then ``<selected_seller>`` inside ``buyer_message``, then natural-language cues,
        then price match against current/initial seller offers, then lower listed seller price.
        """
        if last_selected_seller in (1, 2):
            return int(last_selected_seller)
        text = buyer_message or ""
        m = re.search(r"<selected_seller>\s*(\d+)\s*</selected_seller>", text, re.IGNORECASE | re.DOTALL)
        if m:
            v = int(m.group(1))
            if v in (1, 2):
                return v
        tl = text.lower()
        if re.search(r"seller\s*2|second\s+seller|seller\s*two", tl):
            return 2
        if re.search(r"seller\s*1|first\s+seller|seller\s*one", tl):
            return 1
        s1 = observation.get("seller1_price")
        s2 = observation.get("seller2_price")
        init1 = observation.get("initial_seller1_price")
        init2 = observation.get("initial_seller2_price")
        price_match = re.search(r"\$?([\d,]+\.?\d*)", text)
        if price_match:
            try:
                mentioned = float(price_match.group(1).replace(",", ""))
            except ValueError:
                mentioned = None
            if mentioned is not None and mentioned > 0:
                tol = max(3.0, 0.12 * mentioned)
                best_id: Optional[int] = None
                best_d = float("inf")
                for sid, ref in (
                    (1, s1),
                    (2, s2),
                    (1, init1),
                    (2, init2),
                ):
                    if ref is None:
                        continue
                    d = abs(mentioned - float(ref))
                    if d < best_d:
                        best_d, best_id = d, sid
                if best_id is not None and best_d <= tol:
                    return best_id
        if s1 is not None and s2 is not None:
            return 1 if float(s1) <= float(s2) else 2
        if s1 is not None:
            return 1
        if s2 is not None:
            return 2
        return 1
    
    def _get_observation(self) -> Dict[str, Any]:
        """Get current observation"""
        obs = {
            "conversation_history_seller1": self.memory_seller1.get_history(),
            "conversation_history_seller2": self.memory_seller2.get_history(),
            "current_round": self.current_round,
            "current_selected_seller": self.current_selected_seller,
            "seller1_price": self.state_seller1.seller_price,
            "buyer_price_seller1": self.state_seller1.buyer_price,
            "seller2_price": self.state_seller2.seller_price,
            "buyer_price_seller2": self.state_seller2.buyer_price,
            "status": self.negotiation_info.status.value,
            "final_selected_seller": self.final_selected_seller,
            "final_deal_price": self.final_deal_price,
            "seller1_product_info": self.seller1_product_info,
            "seller2_product_info": self.seller2_product_info,
            "initial_seller1_price": self.initial_seller1_price,
            "initial_seller2_price": self.initial_seller2_price,
            "contract_configs": {
                seller_id: self._build_public_contract_config(seller_id)
                for seller_id in (1, 2)
            } if self.use_contract_mode else None,
            "contract_config": (
                self._build_public_contract_config(self.current_selected_seller)
                if self.use_contract_mode and self.current_selected_seller in (1, 2)
                else None
            ),
            "buyer_contract_seller1": self.state_seller1.metadata.get("buyer_contract"),
            "seller_contract_seller1": self.state_seller1.metadata.get("seller_contract"),
            "agreed_contract_seller1": self.state_seller1.metadata.get("agreed_contract"),
            "buyer_contract_seller2": self.state_seller2.metadata.get("buyer_contract"),
            "seller_contract_seller2": self.state_seller2.metadata.get("seller_contract"),
            "agreed_contract_seller2": self.state_seller2.metadata.get("agreed_contract"),
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
            "current_selected_seller": self.current_selected_seller,
            "seller1_price": self.state_seller1.seller_price,
            "buyer_price_seller1": self.state_seller1.buyer_price,
            "seller2_price": self.state_seller2.seller_price,
            "buyer_price_seller2": self.state_seller2.buyer_price,
            "final_selected_seller": self.final_selected_seller,
            "final_deal_price": self.final_deal_price,
            "negotiation_info": self.negotiation_info,
            "seller1_product_info": self.seller1_product_info,
            "seller2_product_info": self.seller2_product_info,
            "initial_seller1_price": self.initial_seller1_price,
            "initial_seller2_price": self.initial_seller2_price,
            "buyer_contract_seller1": self.state_seller1.metadata.get("buyer_contract"),
            "seller_contract_seller1": self.state_seller1.metadata.get("seller_contract"),
            "agreed_contract_seller1": self.state_seller1.metadata.get("agreed_contract"),
            "buyer_utility_seller1": self.state_seller1.metadata.get("buyer_utility"),
            "seller_utility_seller1": self.state_seller1.metadata.get("seller_utility"),
            "z_max_seller1": self.z_max_by_seller.get(1),
            "buyer_contract_seller2": self.state_seller2.metadata.get("buyer_contract"),
            "seller_contract_seller2": self.state_seller2.metadata.get("seller_contract"),
            "agreed_contract_seller2": self.state_seller2.metadata.get("agreed_contract"),
            "buyer_utility_seller2": self.state_seller2.metadata.get("buyer_utility"),
            "seller_utility_seller2": self.state_seller2.metadata.get("seller_utility"),
            "z_max_seller2": self.z_max_by_seller.get(2),
            "contract_configs": self.contract_configs if self.use_contract_mode else None,
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
        ]
        
        for pattern in fallback_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                price = parse_price(matches[-1])  # Take the last match
                if price is not None:
                    return price
        
        return None
    
    def _check_make_deal(self, text: str) -> bool:
        """Check if buyer wants to make a deal
        
        Args:
            text: Buyer's response text
            
        Returns:
            Whether buyer wants to make a deal
        """
        # First check for the fixed format "MAKE_DEAL"
        if 'MAKE_DEAL' in text.upper():
            return True
        
        # More specific patterns to avoid false positives
        # Exclude phrases like "I hope we can reach an agreement" which are just expressions of hope
        make_deal_patterns = [
            r'\bi\s+accept\b',  # "I accept"
            r'\bi\s+agree\b',  # "I agree"
            r'\baccept\s+your\s+offer\b',  # "accept your offer"
            r'\bagree\s+to\s+the\s+deal\b',  # "agree to the deal"
            r'\bmake\s+a\s+deal\b',  # "make a deal"
            r'\bwe\s+have\s+a\s+deal\b',  # "we have a deal"
            r'\blet\'?s\s+do\s+it\b',  # "let's do it"
            r'\bi\'?ll\s+take\s+it\b',  # "I'll take it"
            r'\bi\'?m\s+accepting\b',  # "I'm accepting"
            r'\bi\'?m\s+agreeing\b',  # "I'm agreeing"
            r'\bdeal\s*!',  # "deal!"
            r'\bi\s+accept\s+the\s+price\b',  # "I accept the price"
        ]
        
        text_lower = text.lower()
        
        # Exclude common false positive patterns
        false_positive_patterns = [
            r'hope\s+.*\s+agreement',  # "hope ... agreement"
            r'hoping\s+.*\s+agreement',  # "hoping ... agreement"
            r'would\s+like\s+.*\s+agreement',  # "would like ... agreement"
            r'looking\s+forward\s+.*\s+agreement',  # "looking forward ... agreement"
        ]
        
        # Check for false positives first
        for pattern in false_positive_patterns:
            if re.search(pattern, text_lower):
                return False
        
        # Check for actual deal patterns
        for pattern in make_deal_patterns:
            if re.search(pattern, text_lower):
                return True
        
        return False
    
    def _calculate_reward(self) -> float:
        """Calculate global reward
        
        Calculate reward value based on negotiation result.
        If deal is reached with a seller, use that seller's min_price for calculation.
        
        If deal is reached:
            reward = buyer savings + seller profit + time cost (negative, based on rounds)
            - buyer savings = buyer_max_price - deal_price (money saved by buyer)
            - seller profit = deal_price - seller_min_price (extra profit for seller)
            - time cost = -current_round (penalty for number of rounds taken)
        
        If deal is not reached:
            reward = time cost (negative, based on rounds)
            - time cost = -current_round (penalty for number of rounds taken)
        
        Returns:
            Reward value
        """
        # Time cost: negative value based on number of rounds
        time_cost = -self.current_round
        
        if self.negotiation_info.status == NegotiationStatus.AGREED and self.final_selected_seller is not None and self.final_deal_price is not None:
            # Deal reached: buyer savings + seller profit + time cost
            deal_price = self.final_deal_price
            reward = 0.0
            buyer_savings = 0.0
            seller_profit = 0.0
            
            # Get the selected seller's min_price
            selected_seller_min_price = None
            if self.final_selected_seller == 1:
                selected_seller_min_price = self.seller1_min_price
            elif self.final_selected_seller == 2:
                selected_seller_min_price = self.seller2_min_price
            
            # Calculate buyer savings: buyer_max_price - deal_price
            if self.buyer_max_price is not None:
                buyer_savings = self.buyer_max_price - deal_price
                reward += buyer_savings * self.reward_weights["buyer_savings"]
            
            # Calculate seller profit: deal_price - seller_min_price
            if selected_seller_min_price is not None:
                seller_profit = deal_price - selected_seller_min_price
                reward += seller_profit * self.reward_weights["seller_profit"]
            
            # Add time cost (negative penalty)
            reward += time_cost * self.reward_weights["time_cost"]
            
            weighted_buyer_savings = buyer_savings * self.reward_weights["buyer_savings"] if self.buyer_max_price is not None else 0.0
            weighted_seller_profit = seller_profit * self.reward_weights["seller_profit"] if selected_seller_min_price is not None else 0.0
            weighted_time_cost = time_cost * self.reward_weights["time_cost"]
            print(f"Global Reward = buyer_savings({buyer_savings:.2f} * {self.reward_weights['buyer_savings']:.2f}) + seller{self.final_selected_seller}_profit({seller_profit:.2f} * {self.reward_weights['seller_profit']:.2f}) + time_cost({time_cost:.2f} * {self.reward_weights['time_cost']:.2f}) = {reward:.2f} (buyer_max={self.buyer_max_price}, deal_price={deal_price:.2f}, seller{self.final_selected_seller}_min={selected_seller_min_price}, round={self.current_round})")
            
            return reward
        
        else:
            # Deal not reached: only time cost (negative penalty)
            weighted_time_cost = time_cost * self.reward_weights["time_cost"]
            print(f"Global Reward = time_cost({time_cost:.2f} * {self.reward_weights['time_cost']:.2f}) = {weighted_time_cost:.2f} (round={self.current_round}, deal not reached)")
            return weighted_time_cost
    
    def _calculate_buyer_reward(self) -> float:
        """Calculate reward from buyer's perspective
        
        Calculate reward value based on negotiation result from buyer's perspective.
        This reward does not include seller profit.
        
        If deal is reached:
            reward = buyer savings + time cost (negative, based on rounds)
            - buyer savings = buyer_max_price - deal_price (money saved by buyer)
            - time cost = -current_round (penalty for number of rounds taken)
        
        If deal is not reached:
            reward = time cost (negative, based on rounds)
            - time cost = -current_round (penalty for number of rounds taken)
        
        Returns:
            Reward value from buyer's perspective
        """
        # Time cost: negative value based on number of rounds
        time_cost = -self.current_round
        
        if self.negotiation_info.status == NegotiationStatus.AGREED and self.final_selected_seller is not None and self.final_deal_price is not None:
            # Deal reached: buyer savings + time cost
            deal_price = self.final_deal_price
            reward = 0.0
            buyer_savings = 0.0
            
            # Calculate buyer savings: buyer_max_price - deal_price
            if self.buyer_max_price is not None:
                buyer_savings = self.buyer_max_price - deal_price
                reward += buyer_savings * self.reward_weights["buyer_savings"]
            
            # Add time cost (negative penalty)
            reward += time_cost * self.reward_weights["time_cost"]
            
            weighted_buyer_savings = buyer_savings * self.reward_weights["buyer_savings"] if self.buyer_max_price is not None else 0.0
            weighted_time_cost = time_cost * self.reward_weights["time_cost"]
            print(f"Buyer Reward = buyer_savings({buyer_savings:.2f} * {self.reward_weights['buyer_savings']:.2f}) + time_cost({time_cost:.2f} * {self.reward_weights['time_cost']:.2f}) = {reward:.2f} (buyer_max={self.buyer_max_price}, deal_price={deal_price:.2f}, round={self.current_round})")
            
            return reward
        
        else:
            # Deal not reached: only time cost (negative penalty)
            weighted_time_cost = time_cost * self.reward_weights["time_cost"]
            print(f"Buyer Reward = time_cost({time_cost:.2f} * {self.reward_weights['time_cost']:.2f}) = {weighted_time_cost:.2f} (round={self.current_round}, deal not reached)")
            return weighted_time_cost
    
    def _calculate_seller_reward(self, seller_id: int) -> float:
        """Calculate reward from seller's perspective
        
        Calculate reward value based on negotiation result from seller's perspective.
        This reward does not include buyer savings.
        
        If deal is reached with this seller:
            reward = seller profit + time cost (negative, based on rounds)
            - seller profit = deal_price - seller_min_price (extra profit for seller)
            - time cost = -current_round (penalty for number of rounds taken)
        
        If deal is not reached or reached with another seller:
            reward = time cost (negative, based on rounds)
            - time cost = -current_round (penalty for number of rounds taken)
        
        Args:
            seller_id: Seller ID (1 or 2)
        
        Returns:
            Reward value from seller's perspective
        """
        # Time cost: negative value based on number of rounds
        time_cost = -self.current_round
        
        # Check if deal was reached with this seller
        deal_reached_with_this_seller = (
            self.negotiation_info.status == NegotiationStatus.AGREED and
            self.final_selected_seller == seller_id and
            self.final_deal_price is not None
        )
        
        if deal_reached_with_this_seller:
            # Deal reached with this seller: seller profit + time cost
            deal_price = self.final_deal_price
            reward = 0.0
            seller_profit = 0.0
            
            # Get this seller's min_price
            seller_min_price = None
            if seller_id == 1:
                seller_min_price = self.seller1_min_price
            elif seller_id == 2:
                seller_min_price = self.seller2_min_price
            
            # Calculate seller profit: deal_price - seller_min_price
            if seller_min_price is not None:
                seller_profit = deal_price - seller_min_price
                reward += seller_profit * self.reward_weights["seller_profit"]
            
            # Add time cost (negative penalty)
            reward += time_cost * self.reward_weights["time_cost"]
            
            weighted_seller_profit = seller_profit * self.reward_weights["seller_profit"] if seller_min_price is not None else 0.0
            weighted_time_cost = time_cost * self.reward_weights["time_cost"]
            print(f"Seller{seller_id} Reward = seller_profit({seller_profit:.2f} * {self.reward_weights['seller_profit']:.2f}) + time_cost({time_cost:.2f} * {self.reward_weights['time_cost']:.2f}) = {reward:.2f} (deal_price={deal_price:.2f}, seller{seller_id}_min={seller_min_price}, round={self.current_round})")
            
            return reward
        
        else:
            # Deal not reached or reached with another seller: only time cost (negative penalty)
            weighted_time_cost = time_cost * self.reward_weights["time_cost"]
            print(f"Seller{seller_id} Reward = time_cost({time_cost:.2f} * {self.reward_weights['time_cost']:.2f}) = {weighted_time_cost:.2f} (round={self.current_round}, deal not reached with this seller)")
            return weighted_time_cost
    
    def _calculate_step_buyer_reward(self) -> float:
        """Calculate step reward from buyer's perspective for current round
        
        Calculate reward value based on buyer's current offer in this round with the selected seller.
        This is calculated every round, not just at the end.
        
        reward = buyer savings (from current offer) + round cost
        - buyer savings = buyer_max_price - buyer_price (money saved by current offer)
        - round cost = -current_round (penalty for number of rounds taken)
        
        Returns:
            Step reward value from buyer's perspective for current round
        """
        # Round cost: negative value based on number of rounds
        round_cost = -self.current_round
        
        # Calculate buyer reward with the selected seller
        reward = 0.0
        buyer_savings = 0.0
        
        # Get buyer price from the selected seller
        if self.current_selected_seller == 1:
            buyer_price = self.state_seller1.buyer_price
        elif self.current_selected_seller == 2:
            buyer_price = self.state_seller2.buyer_price
        else:
            buyer_price = None
        
        # Calculate buyer savings: buyer_max_price - buyer_price
        if buyer_price is not None and self.buyer_max_price is not None:
            buyer_savings = self.buyer_max_price - buyer_price
            reward += buyer_savings * self.reward_weights["buyer_savings"]
        
        # Add round cost (negative penalty)
        reward += round_cost * self.reward_weights["time_cost"]
        
        return reward
    
    def _calculate_step_seller_reward(self, seller_id: int) -> float:
        """Calculate step reward from seller's perspective for current round
        
        Calculate reward value based on seller's current offer in this round.
        This is calculated every round, not just at the end.
        
        reward = seller profit (from current offer) + round cost
        - seller profit = seller_price - seller_min_price (profit from current offer)
        - round cost = -current_round (penalty for number of rounds taken)
        
        If seller_price is not specified yet, only round cost is returned.
        
        Args:
            seller_id: Seller ID (1 or 2)
        
        Returns:
            Step reward value from seller's perspective for current round
        """
        # Round cost: negative value based on number of rounds
        round_cost = -self.current_round
        reward = 0.0
        seller_profit = 0.0
        
        # Get seller state
        seller_state = None
        seller_min_price = None
        if seller_id == 1:
            seller_state = self.state_seller1
            seller_min_price = self.seller1_min_price
        elif seller_id == 2:
            seller_state = self.state_seller2
            seller_min_price = self.seller2_min_price
        
        # Calculate seller profit from current offer: seller_price - seller_min_price
        if seller_state is not None and seller_state.seller_price is not None and seller_min_price is not None:
            seller_profit = seller_state.seller_price - seller_min_price
            reward += seller_profit * self.reward_weights["seller_profit"]
        
        # Add round cost (negative penalty)
        reward += round_cost * self.reward_weights["time_cost"]
        
        return reward
    
    def _get_selected_seller_min_price(self) -> Optional[float]:
        """Get the final selected seller's min_price
        
        Returns:
            Final selected seller's min_price, or None if no seller is selected
        """
        if self.final_selected_seller == 1:
            return self.seller1_min_price
        elif self.final_selected_seller == 2:
            return self.seller2_min_price
        return None
    
    def _score_Z_and_winner_min(self) -> Tuple[Optional[float], Optional[float]]:
        """Bargaining range width Z for GlobalScore / BuyerScore / SellerScore.
        
        When both sellers have a minimum, Z = buyer_max - min(s1_min, s2_min) so utilities reflect
        the competitive surplus band; u_b and u_s still use the **closing** seller's min in the
        numerator for u_s and the same Z in the denominator (buyer surplus vs best outside seller).
        """
        winner_min = self._get_selected_seller_min_price()
        if self.buyer_max_price is None or winner_min is None:
            return None, winner_min
        if self.seller1_min_price is not None and self.seller2_min_price is not None:
            competitive_floor = min(self.seller1_min_price, self.seller2_min_price)
            Z = self.buyer_max_price - competitive_floor
        else:
            Z = self.buyer_max_price - winner_min
        return Z, winner_min

    def _get_contract_score_terms(self) -> Optional[Dict[str, float]]:
        """Return normalized utility terms using the market-best seller utility space."""
        if not self.use_contract_mode or self.final_selected_seller not in (1, 2):
            return None

        market_best_seller = self._get_market_best_contract_seller()
        if market_best_seller is None:
            return None
        z_market = self.z_max_by_seller.get(market_best_seller)
        if not z_market or z_market <= 0:
            return None

        selected_state = self._get_seller_state(self.final_selected_seller)
        final_contract = selected_state.metadata.get("agreed_contract")
        if not final_contract:
            final_contract = self._resolve_agreed_contract(self.final_selected_seller)
            if final_contract:
                selected_state.metadata["agreed_contract"] = final_contract

        if not final_contract or not self._validate_contract(final_contract, self.final_selected_seller):
            return None

        u_b, u_s = self._calculate_contract_utilities(final_contract, self.final_selected_seller)
        selected_state.metadata["buyer_utility"] = u_b
        selected_state.metadata["seller_utility"] = u_s
        if u_b < 0 or u_s < 0:
            return None

        r_b = u_b / z_market
        r_s = u_s / z_market
        return {
            "u_b": u_b,
            "u_s": u_s,
            "r_b": r_b,
            "r_s": r_s,
            "q": 4.0 * r_b * r_s,
            "z_market": z_market,
            "market_best_seller": float(market_best_seller),
            "selected_seller": float(self.final_selected_seller),
            "selected_z_max": self.z_max_by_seller.get(self.final_selected_seller) or 0.0,
        }
    
    def _calculate_global_score(self, print_details: bool = True) -> float:
        """Calculate GlobalScore based on the optimized formula
        
        Uses competitive Z = buyer_max - min(seller mins) when both sellers exist; otherwise
        Z = buyer_max - closing seller min. Closing seller's min anchors u_s.
        If no seller is selected, calculates failure penalty.
        
        Let:
        - buyer_max_price = maximum price the buyer is willing to pay
        - seller_min_price = minimum price the final selected seller is willing to accept
        - competitive_floor = min(seller1_min_price, seller2_min_price) when both set, else seller_min_price
        - Z = buyer_max_price - competitive_floor (normalizes u_b and u_s vs best competing seller)
        - γ (gamma) controls how strongly longer negotiations are penalized (default: 0.99)
        
        valid_range = (Z > 0) and (seller_min_price <= p <= buyer_max_price)
        feasible_deal = negotiation reached agreement
        
        discount = γ^(t-1)  # where t is the round number (1-based)
        
        If feasible_deal and valid_range:
            u_b = (buyer_max_price - p) / Z          # in [0, 1]
            u_s = (p - seller_min_price) / Z         # in [0, 1]
            Q = 4 * u_b * u_s                        # in [0, 1]
            
            DealScore       = D * discount
            QualityScore    = W * Q * discount
            EfficiencyScore = E * discount
            
            GlobalScore = DealScore + QualityScore + EfficiencyScore
        Else:
            FailurePenalty = -F * (1 - discount)
            GlobalScore = FailurePenalty
        
        Settings:
            D = deal_score_weight (default: 10)
            W = quality_score_weight (default: 80)
            E = efficiency_score_weight (default: 10)
            F = failure_penalty_weight (default: 15)
            γ = gamma (default: 0.99)
            T = max_rounds
        
        Returns:
            GlobalScore value (only calculated at final result)
        """
        round_index = max(0, self.current_round - 1)
        discount = self.gamma ** round_index

        if self.use_contract_mode:
            score_terms = self._get_contract_score_terms()
            feasible_deal = (self.negotiation_info.status == NegotiationStatus.AGREED) or (self.final_deal_price is not None)
            if feasible_deal and score_terms is not None:
                deal_score = self.deal_score_weight * discount
                quality_score = self.quality_score_weight * score_terms["q"] * discount
                efficiency_score = self.efficiency_score_weight * discount
                global_score = deal_score + quality_score + efficiency_score
                if print_details:
                    print(f"\n[GlobalScore Calculation - Contract Mode]")
                    print(f"  market_best_seller = Seller {int(score_terms['market_best_seller'])}")
                    print(f"  z_market = {score_terms['z_market']:.4f}, selected_seller_z_max = {score_terms['selected_z_max']:.4f}")
                    print(f"  final_selected_seller = Seller {int(score_terms['selected_seller'])}")
                    print(f"  u_b = {score_terms['u_b']:.4f}, u_s = {score_terms['u_s']:.4f}")
                    print(f"  r_b = {score_terms['r_b']:.4f}, r_s = {score_terms['r_s']:.4f}")
                    print(f"  Q = 4 * r_b * r_s = {score_terms['q']:.4f}")
                    print(f"  round_index = {round_index}, discount = γ^{round_index} = {discount:.6f}")
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

        # Get final selected seller's min_price
        selected_seller_min_price = self._get_selected_seller_min_price()
        
        # Check if we have required prices
        if self.buyer_max_price is None or selected_seller_min_price is None:
            # Calculate discount for failure penalty
            round_index = max(0, self.current_round)
            discount = self.gamma ** round_index
            failure_penalty = -self.failure_penalty_weight * (1.0 - discount)
            
            if print_details:
                print(f"\n[GlobalScore Calculation]")
                print(f"  buyer_max_price or selected_seller_min_price is None")
                print(f"  round_index = {round_index}, gamma = {self.gamma}, discount = γ^{round_index} = {discount:.6f}")
                print(f"  FailurePenalty = -F({self.failure_penalty_weight:.1f}) * (1 - discount({discount:.6f})) = {failure_penalty:.3f}")
                print(f"  GlobalScore = {failure_penalty:.3f}")
            return failure_penalty
        
        Z, _ = self._score_Z_and_winner_min()
        competitive_floor: Optional[float] = None
        if self.seller1_min_price is not None and self.seller2_min_price is not None:
            competitive_floor = min(self.seller1_min_price, self.seller2_min_price)
        
        # Calculate discount = γ^(t-1)
        round_index = max(0, self.current_round)
        discount = self.gamma ** round_index
        
        # Check feasible_deal: whether negotiation reached agreement
        feasible_deal = (self.negotiation_info.status == NegotiationStatus.AGREED) or (self.final_deal_price is not None)
        
        # Get the final price
        if self.final_deal_price is not None:
            final_price = self.final_deal_price
        else:
            # No price available - calculate failure penalty
            failure_penalty = -self.failure_penalty_weight * (1.0 - discount)
            
            if print_details:
                print(f"\n[GlobalScore Calculation]")
                if competitive_floor is not None:
                    print(f"  Z = buyer_max({self.buyer_max_price:.2f}) - min(seller mins)({competitive_floor:.2f}) = {Z:.2f} (closing seller min = {selected_seller_min_price:.2f})")
                else:
                    print(f"  Z = buyer_max({self.buyer_max_price:.2f}) - closing seller min({selected_seller_min_price:.2f}) = {Z:.2f}")
                print(f"  No final price available")
                print(f"  feasible_deal = {feasible_deal}")
                print(f"  valid_range = (Z > 0) = {Z > 0}")
                print(f"  round_index = {round_index}, gamma = {self.gamma}, discount = γ^{round_index} = {discount:.6f}")
                print(f"  FailurePenalty = -F({self.failure_penalty_weight:.1f}) * (1 - discount({discount:.6f})) = {failure_penalty:.3f}")
                print(f"  GlobalScore = {failure_penalty:.3f}")
            return failure_penalty
        
        # Check valid_range: (Z > 0) and (seller_min_price <= p <= buyer_max_price)
        valid_range = (Z > 0) and (selected_seller_min_price <= final_price <= self.buyer_max_price)
        
        # If feasible_deal and valid_range, calculate success scores
        if feasible_deal and valid_range:
            # Calculate utilities
            u_b = (self.buyer_max_price - final_price) / Z
            u_s = (final_price - selected_seller_min_price) / Z
            
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
                if competitive_floor is not None:
                    print(f"  Z = buyer_max({self.buyer_max_price:.2f}) - min(seller mins)({competitive_floor:.2f}) = {Z:.2f} (closing seller min = {selected_seller_min_price:.2f})")
                else:
                    print(f"  Z = buyer_max({self.buyer_max_price:.2f}) - closing seller min({selected_seller_min_price:.2f}) = {Z:.2f}")
                print(f"  final_price = {final_price:.2f}")
                print(f"  feasible_deal = {feasible_deal} (negotiation status: {self.negotiation_info.status.value})")
                print(f"  valid_range = (Z > 0) and (selected_seller_min_price({selected_seller_min_price:.2f}) <= final_price({final_price:.2f}) <= buyer_max_price({self.buyer_max_price:.2f})) = {valid_range}")
                print(f"  round_index = {round_index}, gamma = {self.gamma}, discount = γ^{round_index} = {discount:.6f}")
                # Debug output for success case
                print(f"  u_b = (buyer_max_price({self.buyer_max_price:.2f}) - final_price({final_price:.2f})) / Z({Z:.2f}) = {u_b:.4f}")
                print(f"  u_s = (final_price({final_price:.2f}) - closing seller min({selected_seller_min_price:.2f})) / Z({Z:.2f}) = {u_s:.4f}")
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
                if competitive_floor is not None:
                    print(f"  Z = buyer_max({self.buyer_max_price:.2f}) - min(seller mins)({competitive_floor:.2f}) = {Z:.2f} (closing seller min = {selected_seller_min_price:.2f})")
                else:
                    print(f"  Z = buyer_max({self.buyer_max_price:.2f}) - closing seller min({selected_seller_min_price:.2f}) = {Z:.2f}")
                print(f"  final_price = {final_price:.2f}")
                print(f"  feasible_deal = {feasible_deal} (negotiation status: {self.negotiation_info.status.value})")
                print(f"  valid_range = (Z > 0) and (selected_seller_min_price({selected_seller_min_price:.2f}) <= final_price({final_price:.2f}) <= buyer_max_price({self.buyer_max_price:.2f})) = {valid_range}")
                print(f"  round_index = {round_index}, gamma = {self.gamma}, discount = γ^{round_index} = {discount:.6f}")
                # Debug output for failure case
                print(f"  FailurePenalty = -F({self.failure_penalty_weight:.1f}) * (1 - discount({discount:.6f})) = {failure_penalty:.3f}")
                print(f"  GlobalScore = {failure_penalty:.3f}")
            
            return failure_penalty
    
    def _calculate_buyer_score(self, print_details: bool = True) -> float:
        """Calculate BuyerScore based on the formula
        
        u_b = (buyer_max_price - p) / Z with Z from _score_Z_and_winner_min() (competitive when two sellers).
        
        discount = γ^(t-1)  # where t is the round number (1-based)
        
        If feasible_deal and valid_range:
            BuyerScore = discount * (Db + Wb * u_b + Eb)
        Else:
            BuyerScore = -Fb * (1 - discount)
        
        Settings:
            Db = buyer_deal_weight (default: 10)
            Wb = buyer_utility_weight (default: 80)
            Eb = buyer_efficiency_weight (default: 10)
            Fb = buyer_failure_penalty_weight (default: 15)
            γ = gamma (default: 0.99)
        
        Note: Out-of-range deals are treated as failures (same logic as failure)
        
        Returns:
            BuyerScore value (only calculated at final result)
        """
        round_index = max(0, self.current_round - 1)
        discount = self.gamma ** round_index

        if self.use_contract_mode:
            score_terms = self._get_contract_score_terms()
            feasible_deal = (self.negotiation_info.status == NegotiationStatus.AGREED) or (self.final_deal_price is not None)
            if feasible_deal and score_terms is not None:
                buyer_score = discount * (
                    self.buyer_deal_weight
                    + self.buyer_utility_weight * score_terms["r_b"]
                    + self.buyer_efficiency_weight
                )
                if print_details:
                    print(f"\n[BuyerScore Calculation - Contract Mode]")
                    print(f"  market_best_seller = Seller {int(score_terms['market_best_seller'])}")
                    print(f"  z_market = {score_terms['z_market']:.4f}, r_b = {score_terms['r_b']:.4f}")
                    print(f"  round_index = {round_index}, discount = γ^{round_index} = {discount:.6f}")
                    print(f"  BuyerScore = {buyer_score:.3f}")
                return buyer_score

            buyer_score = -self.buyer_failure_penalty_weight * (1.0 - discount)
            if print_details:
                print(f"\n[BuyerScore Calculation - Contract Mode]")
                print(f"  Failed to produce a feasible valid contract")
                print(f"  BuyerScore = {buyer_score:.3f}")
            return buyer_score

        # Get final selected seller's min_price
        selected_seller_min_price = self._get_selected_seller_min_price()
        
        # Check if we have required prices
        if self.buyer_max_price is None or selected_seller_min_price is None:
            # Calculate discount for failure penalty
            round_index = max(0, self.current_round)
            discount = self.gamma ** round_index
            buyer_score = -self.buyer_failure_penalty_weight * (1.0 - discount)
            
            if print_details:
                print(f"\n[BuyerScore Calculation]")
                print(f"  buyer_max_price or selected_seller_min_price is None")
                print(f"  round_index = {round_index}, gamma = {self.gamma}, discount = γ^{round_index} = {discount:.6f}")
                print(f"  BuyerScore = -Fb({self.buyer_failure_penalty_weight:.1f}) * (1 - discount({discount:.6f})) = {buyer_score:.3f}")
            return buyer_score
        
        Z, _ = self._score_Z_and_winner_min()
        competitive_floor: Optional[float] = None
        if self.seller1_min_price is not None and self.seller2_min_price is not None:
            competitive_floor = min(self.seller1_min_price, self.seller2_min_price)
        
        # Calculate discount = γ^(t-1)
        round_index = max(0, self.current_round)
        discount = self.gamma ** round_index
        
        # Check feasible_deal: whether negotiation reached agreement
        feasible_deal = (self.negotiation_info.status == NegotiationStatus.AGREED) or (self.final_deal_price is not None)
        
        # Get the final price
        if self.final_deal_price is not None:
            final_price = self.final_deal_price
        else:
            # No price available - calculate failure penalty
            buyer_score = -self.buyer_failure_penalty_weight * (1.0 - discount)
            
            if print_details:
                print(f"\n[BuyerScore Calculation]")
                if competitive_floor is not None:
                    print(f"  Z = buyer_max({self.buyer_max_price:.2f}) - min(seller mins)({competitive_floor:.2f}) = {Z:.2f} (closing seller min = {selected_seller_min_price:.2f})")
                else:
                    print(f"  Z = buyer_max({self.buyer_max_price:.2f}) - closing seller min({selected_seller_min_price:.2f}) = {Z:.2f}")
                print(f"  No final price available")
                print(f"  round_index = {round_index}, gamma = {self.gamma}, discount = γ^{round_index} = {discount:.6f}")
                print(f"  BuyerScore = -Fb({self.buyer_failure_penalty_weight:.1f}) * (1 - discount({discount:.6f})) = {buyer_score:.3f}")
            return buyer_score
        
        # Check valid_range: (Z > 0) and (seller_min_price <= p <= buyer_max_price)
        valid_range = (Z > 0) and (selected_seller_min_price <= final_price <= self.buyer_max_price)
        
        # If feasible_deal and valid_range, calculate success score
        if feasible_deal and valid_range:
            # Calculate utility
            u_b = (self.buyer_max_price - final_price) / Z
            
            # Calculate BuyerScore = discount * (Db + Wb * u_b + Eb)
            buyer_score = discount * (self.buyer_deal_weight + self.buyer_utility_weight * u_b + self.buyer_efficiency_weight)
            
            if print_details:
                # Debug output header
                print(f"\n[BuyerScore Calculation]")
                if competitive_floor is not None:
                    print(f"  Z = buyer_max({self.buyer_max_price:.2f}) - min(seller mins)({competitive_floor:.2f}) = {Z:.2f} (closing seller min = {selected_seller_min_price:.2f})")
                else:
                    print(f"  Z = buyer_max({self.buyer_max_price:.2f}) - closing seller min({selected_seller_min_price:.2f}) = {Z:.2f}")
                print(f"  final_price = {final_price:.2f}")
                print(f"  feasible_deal = {feasible_deal} (negotiation status: {self.negotiation_info.status.value})")
                print(f"  valid_range = (Z > 0) and (selected_seller_min_price({selected_seller_min_price:.2f}) <= final_price({final_price:.2f}) <= buyer_max_price({self.buyer_max_price:.2f})) = {valid_range}")
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
                if competitive_floor is not None:
                    print(f"  Z = buyer_max({self.buyer_max_price:.2f}) - min(seller mins)({competitive_floor:.2f}) = {Z:.2f} (closing seller min = {selected_seller_min_price:.2f})")
                else:
                    print(f"  Z = buyer_max({self.buyer_max_price:.2f}) - closing seller min({selected_seller_min_price:.2f}) = {Z:.2f}")
                print(f"  final_price = {final_price:.2f}")
                print(f"  feasible_deal = {feasible_deal} (negotiation status: {self.negotiation_info.status.value})")
                print(f"  valid_range = (Z > 0) and (selected_seller_min_price({selected_seller_min_price:.2f}) <= final_price({final_price:.2f}) <= buyer_max_price({self.buyer_max_price:.2f})) = {valid_range}")
                print(f"  round_index = {round_index}, gamma = {self.gamma}, discount = γ^{round_index} = {discount:.6f}")
                # Debug output for failure case
                print(f"  BuyerScore = -Fb({self.buyer_failure_penalty_weight:.1f}) * (1 - discount({discount:.6f})) = {buyer_score:.3f}")
            
            return buyer_score
    
    def _calculate_seller_score(self, print_details: bool = True) -> float:
        """Calculate SellerScore based on the formula
        
        Uses the final selected seller's min_price for calculation.
        If no seller is selected or deal not reached with this seller, calculates failure penalty.
        
        u_s = (p - closing seller min) / Z; Z matches GlobalScore (competitive band when two sellers).
        
        discount = γ^(t-1)  # where t is the round number (1-based)
        
        If feasible_deal and valid_range:
            SellerScore = discount * (Ds + Ws * u_s + Es)
        Else:
            SellerScore = -Fs * (1 - discount)
        
        Settings:
            Ds = seller_deal_weight (default: 10)
            Ws = seller_utility_weight (default: 80)
            Es = seller_efficiency_weight (default: 10)
            Fs = seller_failure_penalty_weight (default: 15)
            γ = gamma (default: 0.99)
        
        Note: Out-of-range deals are treated as failures (same logic as failure)
        
        Returns:
            SellerScore value (only calculated at final result)
        """
        round_index = max(0, self.current_round - 1)
        discount = self.gamma ** round_index

        if self.use_contract_mode:
            score_terms = self._get_contract_score_terms()
            feasible_deal = (self.negotiation_info.status == NegotiationStatus.AGREED) or (self.final_deal_price is not None)
            if feasible_deal and score_terms is not None:
                seller_score = discount * (
                    self.seller_deal_weight
                    + self.seller_utility_weight * score_terms["r_s"]
                    + self.seller_efficiency_weight
                )
                if print_details:
                    print(f"\n[SellerScore Calculation - Contract Mode]")
                    print(f"  market_best_seller = Seller {int(score_terms['market_best_seller'])}")
                    print(f"  z_market = {score_terms['z_market']:.4f}, r_s = {score_terms['r_s']:.4f}")
                    print(f"  round_index = {round_index}, discount = γ^{round_index} = {discount:.6f}")
                    print(f"  SellerScore = {seller_score:.3f}")
                return seller_score

            seller_score = -self.seller_failure_penalty_weight * (1.0 - discount)
            if print_details:
                print(f"\n[SellerScore Calculation - Contract Mode]")
                print(f"  Failed to produce a feasible valid contract")
                print(f"  SellerScore = {seller_score:.3f}")
            return seller_score

        # Get final selected seller's min_price
        selected_seller_min_price = self._get_selected_seller_min_price()
        
        # Check if we have required prices
        if self.buyer_max_price is None or selected_seller_min_price is None:
            # Calculate discount for failure penalty
            round_index = max(0, self.current_round)
            discount = self.gamma ** round_index
            seller_score = -self.seller_failure_penalty_weight * (1.0 - discount)
            
            if print_details:
                print(f"\n[SellerScore Calculation]")
                print(f"  buyer_max_price or selected_seller_min_price is None")
                print(f"  round_index = {round_index}, gamma = {self.gamma}, discount = γ^{round_index} = {discount:.6f}")
                print(f"  SellerScore = -Fs({self.seller_failure_penalty_weight:.1f}) * (1 - discount({discount:.6f})) = {seller_score:.3f}")
            return seller_score
        
        Z, _ = self._score_Z_and_winner_min()
        competitive_floor: Optional[float] = None
        if self.seller1_min_price is not None and self.seller2_min_price is not None:
            competitive_floor = min(self.seller1_min_price, self.seller2_min_price)
        
        # Calculate discount = γ^(t-1)
        round_index = max(0, self.current_round)
        discount = self.gamma ** round_index
        
        # Check feasible_deal: whether negotiation reached agreement
        feasible_deal = (self.negotiation_info.status == NegotiationStatus.AGREED) or (self.final_deal_price is not None)
        
        # Get the final price
        if self.final_deal_price is not None:
            final_price = self.final_deal_price
        else:
            # No price available - calculate failure penalty
            seller_score = -self.seller_failure_penalty_weight * (1.0 - discount)
            
            if print_details:
                print(f"\n[SellerScore Calculation]")
                if competitive_floor is not None:
                    print(f"  Z = buyer_max({self.buyer_max_price:.2f}) - min(seller mins)({competitive_floor:.2f}) = {Z:.2f} (closing seller min = {selected_seller_min_price:.2f})")
                else:
                    print(f"  Z = buyer_max({self.buyer_max_price:.2f}) - closing seller min({selected_seller_min_price:.2f}) = {Z:.2f}")
                print(f"  No final price available")
                print(f"  round_index = {round_index}, gamma = {self.gamma}, discount = γ^{round_index} = {discount:.6f}")
                print(f"  SellerScore = -Fs({self.seller_failure_penalty_weight:.1f}) * (1 - discount({discount:.6f})) = {seller_score:.3f}")
            return seller_score
        
        # Check valid_range: (Z > 0) and (seller_min_price <= p <= buyer_max_price)
        valid_range = (Z > 0) and (selected_seller_min_price <= final_price <= self.buyer_max_price)
        
        # If feasible_deal and valid_range, calculate success score
        if feasible_deal and valid_range:
            # Calculate utility
            u_s = (final_price - selected_seller_min_price) / Z
            
            # Calculate SellerScore = discount * (Ds + Ws * u_s + Es)
            seller_score = discount * (self.seller_deal_weight + self.seller_utility_weight * u_s + self.seller_efficiency_weight)
            
            if print_details:
                # Debug output header
                print(f"\n[SellerScore Calculation]")
                if competitive_floor is not None:
                    print(f"  Z = buyer_max({self.buyer_max_price:.2f}) - min(seller mins)({competitive_floor:.2f}) = {Z:.2f} (closing seller min = {selected_seller_min_price:.2f})")
                else:
                    print(f"  Z = buyer_max({self.buyer_max_price:.2f}) - closing seller min({selected_seller_min_price:.2f}) = {Z:.2f}")
                print(f"  final_price = {final_price:.2f}")
                print(f"  feasible_deal = {feasible_deal} (negotiation status: {self.negotiation_info.status.value})")
                print(f"  valid_range = (Z > 0) and (selected_seller_min_price({selected_seller_min_price:.2f}) <= final_price({final_price:.2f}) <= buyer_max_price({self.buyer_max_price:.2f})) = {valid_range}")
                print(f"  round_index = {round_index}, gamma = {self.gamma}, discount = γ^{round_index} = {discount:.6f}")
                # Debug output for success case
                print(f"  u_s = (final_price({final_price:.2f}) - closing seller min({selected_seller_min_price:.2f})) / Z({Z:.2f}) = {u_s:.4f}")
                print(f"  SellerScore = discount({discount:.6f}) * (Ds({self.seller_deal_weight:.1f}) + Ws({self.seller_utility_weight:.1f}) * u_s({u_s:.4f}) + Es({self.seller_efficiency_weight:.1f}))")
                print(f"  SellerScore = {discount:.6f} * ({self.seller_deal_weight:.1f} + {self.seller_utility_weight * u_s:.4f} + {self.seller_efficiency_weight:.1f}) = {seller_score:.3f}")
            
            return seller_score
        else:
            # Calculate failure penalty (out-of-range deals treated as failures)
            seller_score = -self.seller_failure_penalty_weight * (1.0 - discount)
            
            if print_details:
                # Debug output header
                print(f"\n[SellerScore Calculation]")
                if competitive_floor is not None:
                    print(f"  Z = buyer_max({self.buyer_max_price:.2f}) - min(seller mins)({competitive_floor:.2f}) = {Z:.2f} (closing seller min = {selected_seller_min_price:.2f})")
                else:
                    print(f"  Z = buyer_max({self.buyer_max_price:.2f}) - closing seller min({selected_seller_min_price:.2f}) = {Z:.2f}")
                print(f"  final_price = {final_price:.2f}")
                print(f"  feasible_deal = {feasible_deal} (negotiation status: {self.negotiation_info.status.value})")
                print(f"  valid_range = (Z > 0) and (selected_seller_min_price({selected_seller_min_price:.2f}) <= final_price({final_price:.2f}) <= buyer_max_price({self.buyer_max_price:.2f})) = {valid_range}")
                print(f"  round_index = {round_index}, gamma = {self.gamma}, discount = γ^{round_index} = {discount:.6f}")
                # Debug output for failure case
                print(f"  SellerScore = -Fs({self.seller_failure_penalty_weight:.1f}) * (1 - discount({discount:.6f})) = {seller_score:.3f}")
            
            return seller_score
    
    def _print_global_score_details(self):
        """Print GlobalScore calculation details (called from example file after Step Rewards)"""
        self._calculate_global_score(print_details=True)
    
    def _print_buyer_score_details(self):
        """Print BuyerScore calculation details (called from example file after Step Rewards)"""
        self._calculate_buyer_score(print_details=True)
    
    def _print_seller_score_details(self):
        """Print SellerScore calculation details (called from example file after Step Rewards)"""
        self._calculate_seller_score(print_details=True)

