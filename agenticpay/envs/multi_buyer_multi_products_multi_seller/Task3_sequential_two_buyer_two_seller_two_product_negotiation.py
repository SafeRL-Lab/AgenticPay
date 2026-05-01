"""Task3 Sequential Two-Buyer Two-Seller Two-Product Negotiation Environment Implementation

Supports sequential negotiation where two buyers each choose one seller per round to negotiate with
for two products. Each buyer can switch between two sellers and make a deal with either seller.
Prices represent **total** price for both products. Global / Buyer / Seller scores use the same
**market** definitions as ``multi_buyer_multi_seller.Task3_sequential_two_buyer_two_seller``:
``buyers_market_ceiling = max(buyer1_max, buyer2_max)`` and ``market_best_floor = min(seller1_min, seller2_min)``.
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


class Task3SequentialTwoBuyerTwoSellerTwoProductNegotiation(BaseEnv):
    """Task3 Sequential Two-Buyer Two-Seller Two-Product Negotiation Environment
    
    Manages sequential negotiation process where each buyer chooses one seller per round to negotiate with
    for two products. Each buyer can switch between two sellers and make a deal with either seller.
    Prices represent total price for both products.
    """
    
    def __init__(
        self,
        buyer1_agent: BaseAgent,
        buyer2_agent: BaseAgent,
        seller1_agent: BaseAgent,
        seller2_agent: BaseAgent,
        max_rounds: int = 20,
        initial_seller1_price: float = 200.0,
        initial_seller2_price: float = 220.0,
        buyer1_max_price: Optional[float] = None,
        buyer2_max_price: Optional[float] = None,
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
        """Initialize sequential multi-buyer multi-seller multi-product negotiation environment
        
        Args:
            buyer1_agent: First Buyer Agent
            buyer2_agent: Second Buyer Agent
            seller1_agent: First Seller Agent
            seller2_agent: Second Seller Agent
            max_rounds: Maximum number of negotiation rounds
            initial_seller1_price: Initial total price offered by seller1 for both products
            initial_seller2_price: Initial total price offered by seller2 for both products
            buyer1_max_price: Maximum acceptable total price for buyer1 (confidential, for both products)
            buyer2_max_price: Maximum acceptable total price for buyer2 (confidential, for both products)
            seller1_min_price: Minimum acceptable total price for seller1 (confidential, for both products)
            seller2_min_price: Minimum acceptable total price for seller2 (confidential, for both products)
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
        self.buyer1_agent = buyer1_agent
        self.buyer2_agent = buyer2_agent
        self.seller1_agent = seller1_agent
        self.seller2_agent = seller2_agent
        self.max_rounds = max_rounds
        self.initial_seller1_price = initial_seller1_price
        self.initial_seller2_price = initial_seller2_price
        self.buyer1_max_price = buyer1_max_price
        self.buyer2_max_price = buyer2_max_price
        self.seller1_min_price = seller1_min_price
        self.seller2_min_price = seller2_min_price
        self.environment_info = environment_info or {}
        self.contract_configs = self._normalize_contract_configs(self.environment_info)
        self.use_contract_mode = bool(self.contract_configs)
        self.z_max_by_pair = {
            pair: self._calculate_z_max(config)
            for pair, config in self.contract_configs.items()
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
        
        # State management - separate for each buyer-seller pair
        # buyer1-seller1
        self.memory_b1s1 = ConversationMemory()
        self.state_b1s1 = NegotiationState()
        # buyer1-seller2
        self.memory_b1s2 = ConversationMemory()
        self.state_b1s2 = NegotiationState()
        # buyer2-seller1
        self.memory_b2s1 = ConversationMemory()
        self.state_b2s1 = NegotiationState()
        # buyer2-seller2
        self.memory_b2s2 = ConversationMemory()
        self.state_b2s2 = NegotiationState()
        
        self.current_round = 0
        self.negotiation_info = NegotiationInfo()
        self.product_info: Optional[Dict[str, Any]] = None
        
        # Track which seller each buyer selected for current round and final deal
        self.buyer1_selected_seller: Optional[int] = None  # 1 or 2, selected for current round
        self.buyer2_selected_seller: Optional[int] = None  # 1 or 2, selected for current round
        self.final_selected_buyer: Optional[int] = None  # 1 or 2, chosen for final deal
        self.final_selected_seller: Optional[int] = None  # 1 or 2, chosen for final deal
        self.final_deal_price: Optional[float] = None

        self._reset_contract_metadata()

    def _pair_key(self, buyer_id: int, seller_id: int) -> str:
        return f"b{buyer_id}s{seller_id}"

    def _normalize_contract_configs(self, environment_info: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Normalize pair-specific configs, with shared/single-sided fallbacks."""
        raw_configs = environment_info.get("buyer_seller_contract_configs")
        if isinstance(raw_configs, dict):
            configs: Dict[str, Dict[str, Any]] = {}
            for buyer_id in (1, 2):
                for seller_id in (1, 2):
                    pair = self._pair_key(buyer_id, seller_id)
                    value = (
                        raw_configs.get(pair)
                        or raw_configs.get((buyer_id, seller_id))
                        or raw_configs.get(f"buyer{buyer_id}_seller{seller_id}")
                    )
                    if isinstance(value, dict):
                        configs[pair] = deepcopy(value)
            if configs:
                return configs

        shared_config = environment_info.get("contract_config")
        if isinstance(shared_config, dict) and shared_config:
            return {
                self._pair_key(buyer_id, seller_id): deepcopy(shared_config)
                for buyer_id in (1, 2)
                for seller_id in (1, 2)
            }

        buyer_configs = environment_info.get("buyer_contract_configs")
        seller_configs = environment_info.get("seller_contract_configs")
        if isinstance(buyer_configs, dict) and isinstance(seller_configs, dict):
            configs = {}
            for buyer_id in (1, 2):
                buyer_cfg = buyer_configs.get(buyer_id) or buyer_configs.get(str(buyer_id))
                for seller_id in (1, 2):
                    seller_cfg = seller_configs.get(seller_id) or seller_configs.get(str(seller_id))
                    if isinstance(buyer_cfg, dict) and isinstance(seller_cfg, dict):
                        merged = deepcopy(seller_cfg)
                        merged["buyer_preferences"] = deepcopy(buyer_cfg.get("buyer_preferences", {}))
                        configs[self._pair_key(buyer_id, seller_id)] = merged
            if configs:
                return configs

        return {}

    def _reset_contract_metadata(self) -> None:
        """Initialize per-pair contract state used by contract-mode scoring."""
        for buyer_id in (1, 2):
            for seller_id in (1, 2):
                state = self._get_pair_state(buyer_id, seller_id)
                state.metadata["buyer_contract"] = None
                state.metadata["seller_contract"] = None
                state.metadata["agreed_contract"] = None
                state.metadata["buyer_utility"] = None
                state.metadata["seller_utility"] = None
                state.metadata["z_max"] = self.z_max_by_pair.get(self._pair_key(buyer_id, seller_id))

    def _get_pair_state(self, buyer_id: int, seller_id: int) -> NegotiationState:
        if buyer_id == 1 and seller_id == 1:
            return self.state_b1s1
        if buyer_id == 1 and seller_id == 2:
            return self.state_b1s2
        if buyer_id == 2 and seller_id == 1:
            return self.state_b2s1
        if buyer_id == 2 and seller_id == 2:
            return self.state_b2s2
        raise ValueError(f"buyer_id and seller_id must be 1 or 2, got {buyer_id}, {seller_id}")

    def _get_pair_contract_config(self, buyer_id: int, seller_id: int) -> Dict[str, Any]:
        return self.contract_configs.get(self._pair_key(buyer_id, seller_id), {})

    def _build_public_contract_config(self, buyer_id: int, seller_id: int) -> Dict[str, Any]:
        """Build public contract fields for one buyer-seller pair."""
        config = self._get_pair_contract_config(buyer_id, seller_id)
        if not config:
            return {}
        public_config: Dict[str, Any] = {"buyer_id": buyer_id, "seller_id": seller_id}
        for key in ("continuous_bounds", "discrete_options", "field_descriptions", "contrainfo"):
            if key in config:
                public_config[key] = deepcopy(config[key])
        return public_config

    def _build_role_contract_config(self, role: str, buyer_id: int, seller_id: int) -> Dict[str, Any]:
        """Expose only the role's private preferences for one pair."""
        config = self._get_pair_contract_config(buyer_id, seller_id)
        role_config = self._build_public_contract_config(buyer_id, seller_id)
        if role == "buyer" and "buyer_preferences" in config:
            role_config["buyer_preferences"] = deepcopy(config["buyer_preferences"])
        if role == "seller" and "seller_preferences" in config:
            role_config["seller_preferences"] = deepcopy(config["seller_preferences"])
        return role_config

    def _build_buyer_visible_contract_configs(self, buyer_id: int) -> Dict[int, Dict[str, Any]]:
        return {
            seller_id: self._build_role_contract_config("buyer", buyer_id, seller_id)
            for seller_id in (1, 2)
            if self._pair_key(buyer_id, seller_id) in self.contract_configs
        }

    def _build_seller_visible_contract_configs(self, seller_id: int) -> Dict[int, Dict[str, Any]]:
        return {
            buyer_id: self._build_role_contract_config("seller", buyer_id, seller_id)
            for buyer_id in (1, 2)
            if self._pair_key(buyer_id, seller_id) in self.contract_configs
        }

    def _build_role_environment_info(
        self,
        role: str,
        buyer_id: Optional[int] = None,
        seller_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Build role-specific environment info without leaking counterparty preferences."""
        role_env_info = deepcopy(self.environment_info)
        for key in ("contract_config", "buyer_contract_configs", "seller_contract_configs", "buyer_seller_contract_configs"):
            role_env_info.pop(key, None)
        if not self.use_contract_mode:
            return role_env_info

        if role == "buyer" and buyer_id is not None:
            role_env_info["seller_contract_configs"] = self._build_buyer_visible_contract_configs(buyer_id)
        elif role == "seller" and seller_id is not None:
            role_env_info["buyer_contract_configs"] = self._build_seller_visible_contract_configs(seller_id)
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

    def _validate_contract(self, contract: Optional[Dict[str, Any]], buyer_id: int, seller_id: int) -> bool:
        """Validate contract against the selected pair's bounds/options."""
        if not contract:
            return False
        if not self.use_contract_mode:
            return True

        config = self._get_pair_contract_config(buyer_id, seller_id)
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

    def _calculate_contract_utilities(self, contract: Dict[str, Any], buyer_id: int, seller_id: int) -> Tuple[float, float]:
        """Compute raw MAUT utilities (U_b, U_s) for one pair's contract config."""
        config = self._get_pair_contract_config(buyer_id, seller_id)
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
        """Compute theoretical max surplus Z_max for one buyer-seller pair."""
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

    def _resolve_agreed_contract(self, buyer_id: int, seller_id: int) -> Optional[Dict[str, Any]]:
        """Build the final contract for a compatible buyer/seller pair."""
        state = self._get_pair_state(buyer_id, seller_id)
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

    def _get_market_best_contract_pair(self) -> Optional[str]:
        """Choose the buyer-seller pair with the highest theoretical total utility."""
        candidates = [
            (pair, z_max)
            for pair, z_max in self.z_max_by_pair.items()
            if z_max is not None
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda item: item[1])[0]
    
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
            product_info: Product information for the fixed two-item cart (no per-seller fields).
                Expected format: {"products": [<product1 dict>, <product2 dict>]} — exactly two entries.
            user_profile: User profile
            **kwargs: Other parameters
            
        Returns:
            (observation, info) Initial observation and info
        """
        # Reset state
        self.memory_b1s1.clear()
        self.memory_b1s2.clear()
        self.memory_b2s1.clear()
        self.memory_b2s2.clear()
        self.state_b1s1 = NegotiationState()
        self.state_b1s2 = NegotiationState()
        self.state_b2s1 = NegotiationState()
        self.state_b2s2 = NegotiationState()
        self.current_round = 0
        self.negotiation_info = NegotiationInfo()
        self.buyer1_selected_seller = None
        self.buyer2_selected_seller = None
        self.final_selected_buyer = None
        self.final_selected_seller = None
        self.final_deal_price = None
        self._reset_contract_metadata()
        self.product_info = product_info or {}
        self._product_info = self.product_info  # For info/results (aligned with Task3 multi_buyer_multi_seller)
        
        # Extract product information: exactly two products (same cart for all parties; total-price negotiation)
        products = self.product_info.get("products", [])
        if len(products) != 2:
            raise ValueError("product_info['products'] must contain exactly 2 items (bundle total price).")

        # Extract product_images for VLM (from each product; optional kwargs override, aligned with Task3)
        product_images = kwargs.get("product_images")
        if product_images is None:
            product_images = self.product_info.get("product_images") or self.product_info.get("images")
        if product_images is None:
            imgs: list = []
            for p in products:
                img_url = p.get("image_path") or p.get("image_url")
                if img_url:
                    imgs.append(img_url)
            product_images = imgs if imgs else None
        if product_images is not None and not isinstance(product_images, list):
            product_images = [product_images]
        self.product_images = product_images
        
        # Initialize Buyer1 Agent (buyer1 knows about both sellers)
        buyer1_context = {
            "user_requirement": user_requirement,
            "max_price": self.buyer1_max_price,  # Total max price for both products
            "user_profile": user_profile,
            "environment_info": self._build_role_environment_info("buyer", buyer_id=1),
            "product_info": self.product_info,  # Buyer can see both products
            "product_images": product_images,  # For VLM: product images (URL/path) for img input
            "buyer_id": 1,
            "num_sellers": 2,  # Inform buyer there are 2 sellers
            "negotiation_mode": "sequential",  # Inform buyer this is sequential negotiation
            "seller_contract_configs": self._build_buyer_visible_contract_configs(1) if self.use_contract_mode else {},
        }
        self.buyer1_agent.initialize(buyer1_context)
        
        # Initialize Buyer2 Agent (buyer2 knows about both sellers)
        buyer2_context = {
            "user_requirement": user_requirement,
            "max_price": self.buyer2_max_price,  # Total max price for both products
            "user_profile": user_profile,
            "environment_info": self._build_role_environment_info("buyer", buyer_id=2),
            "product_info": self.product_info,  # Buyer can see both products
            "product_images": product_images,  # For VLM: product images (URL/path) for img input
            "buyer_id": 2,
            "num_sellers": 2,  # Inform buyer there are 2 sellers
            "negotiation_mode": "sequential",  # Inform buyer this is sequential negotiation
            "seller_contract_configs": self._build_buyer_visible_contract_configs(2) if self.use_contract_mode else {},
        }
        self.buyer2_agent.initialize(buyer2_context)
        
        # Initialize Seller1 Agent
        seller1_context = {
            "product_info": self.product_info,  # Seller can see both products
            "product_images": product_images,  # For VLM: product images (URL/path) for img input
            "initial_price": self.initial_seller1_price,  # Initial total price for both products
            "min_price": self.seller1_min_price,  # Total min price for both products
            "environment_info": self._build_role_environment_info("seller", seller_id=1),
            "seller_id": 1,  # Identify as seller 1
            "num_buyers": 2,  # Inform seller there are 2 buyers
        }
        self.seller1_agent.initialize(seller1_context)
        
        # Initialize Seller2 Agent
        seller2_context = {
            "product_info": self.product_info,  # Seller can see both products
            "product_images": product_images,  # For VLM: product images (URL/path) for img input
            "initial_price": self.initial_seller2_price,  # Initial total price for both products
            "min_price": self.seller2_min_price,  # Total min price for both products
            "environment_info": self._build_role_environment_info("seller", seller_id=2),
            "seller_id": 2,  # Identify as seller 2
            "num_buyers": 2,  # Inform seller there are 2 buyers
        }
        self.seller2_agent.initialize(seller2_context)
        
        # No initial seller offer - negotiation starts with buyer's first message
        # Build observation
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(
        self,
        buyer1_selected_seller: int,  # 1 or 2, which seller buyer1 chooses to negotiate with this round
        buyer2_selected_seller: int,  # 1 or 2, which seller buyer2 chooses to negotiate with this round
        buyer1_action: Optional[str] = None,
        buyer2_action: Optional[str] = None,
        seller1_action_buyer1: Optional[str] = None,
        seller1_action_buyer2: Optional[str] = None,
        seller2_action_buyer1: Optional[str] = None,
        seller2_action_buyer2: Optional[str] = None,
    ) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """Execute one negotiation step
        
        Each round, each buyer chooses one seller to negotiate with, then buyers and sellers exchange messages.
        Order: buyer -> seller
        Prices represent total price for both products.
        
        Args:
            buyer1_selected_seller: Which seller (1 or 2) buyer1 chooses to negotiate with this round
            buyer2_selected_seller: Which seller (1 or 2) buyer2 chooses to negotiate with this round
            buyer1_action: Buyer1's response (optional)
            buyer2_action: Buyer2's response (optional)
            seller1_action_buyer1: Seller1's response to buyer1 (optional, only if buyer1 selected seller1)
            seller1_action_buyer2: Seller1's response to buyer2 (optional, only if buyer2 selected seller1)
            seller2_action_buyer1: Seller2's response to buyer1 (optional, only if buyer1 selected seller2)
            seller2_action_buyer2: Seller2's response to buyer2 (optional, only if buyer2 selected seller2)
            
        Returns:
            (observation, reward, terminated, truncated, info)
        """
        if buyer1_selected_seller not in [1, 2]:
            raise ValueError(f"buyer1_selected_seller must be 1 or 2, got {buyer1_selected_seller}")
        if buyer2_selected_seller not in [1, 2]:
            raise ValueError(f"buyer2_selected_seller must be 1 or 2, got {buyer2_selected_seller}")
        
        self.buyer1_selected_seller = buyer1_selected_seller
        self.buyer2_selected_seller = buyer2_selected_seller
        
        # Process buyer actions first
        # buyer1 action
        if buyer1_action is not None:
            if buyer1_selected_seller == 1:
                self.memory_b1s1.add_message("buyer", buyer1_action, self.current_round)
                buyer_contract = self._extract_contract(buyer1_action) if self.use_contract_mode else None
                if buyer_contract and self._validate_contract(buyer_contract, 1, 1):
                    self.state_b1s1.metadata["buyer_contract"] = buyer_contract
                    buyer_price = float(buyer_contract["price"])
                else:
                    buyer_price = self._extract_price(buyer1_action)
                if buyer_price is not None:
                    self.state_b1s1.update(buyer_price=buyer_price)
            else:  # buyer1_selected_seller == 2
                self.memory_b1s2.add_message("buyer", buyer1_action, self.current_round)
                buyer_contract = self._extract_contract(buyer1_action) if self.use_contract_mode else None
                if buyer_contract and self._validate_contract(buyer_contract, 1, 2):
                    self.state_b1s2.metadata["buyer_contract"] = buyer_contract
                    buyer_price = float(buyer_contract["price"])
                else:
                    buyer_price = self._extract_price(buyer1_action)
                if buyer_price is not None:
                    self.state_b1s2.update(buyer_price=buyer_price)
        
        # buyer2 action
        if buyer2_action is not None:
            if buyer2_selected_seller == 1:
                self.memory_b2s1.add_message("buyer", buyer2_action, self.current_round)
                buyer_contract = self._extract_contract(buyer2_action) if self.use_contract_mode else None
                if buyer_contract and self._validate_contract(buyer_contract, 2, 1):
                    self.state_b2s1.metadata["buyer_contract"] = buyer_contract
                    buyer_price = float(buyer_contract["price"])
                else:
                    buyer_price = self._extract_price(buyer2_action)
                if buyer_price is not None:
                    self.state_b2s1.update(buyer_price=buyer_price)
            else:  # buyer2_selected_seller == 2
                self.memory_b2s2.add_message("buyer", buyer2_action, self.current_round)
                buyer_contract = self._extract_contract(buyer2_action) if self.use_contract_mode else None
                if buyer_contract and self._validate_contract(buyer_contract, 2, 2):
                    self.state_b2s2.metadata["buyer_contract"] = buyer_contract
                    buyer_price = float(buyer_contract["price"])
                else:
                    buyer_price = self._extract_price(buyer2_action)
                if buyer_price is not None:
                    self.state_b2s2.update(buyer_price=buyer_price)
        
        # Process seller actions after buyers
        # seller1-buyer1
        if buyer1_selected_seller == 1 and seller1_action_buyer1 is not None:
            self.memory_b1s1.add_message("seller", seller1_action_buyer1, self.current_round)
            seller_contract = self._extract_contract(seller1_action_buyer1) if self.use_contract_mode else None
            if seller_contract and self._validate_contract(seller_contract, 1, 1):
                self.state_b1s1.metadata["seller_contract"] = seller_contract
                seller_price = float(seller_contract["price"])
            else:
                seller_price = self._extract_price(seller1_action_buyer1)
            if seller_price is not None:
                self.state_b1s1.update(seller_price=seller_price)
        
        # seller1-buyer2
        if buyer2_selected_seller == 1 and seller1_action_buyer2 is not None:
            self.memory_b2s1.add_message("seller", seller1_action_buyer2, self.current_round)
            seller_contract = self._extract_contract(seller1_action_buyer2) if self.use_contract_mode else None
            if seller_contract and self._validate_contract(seller_contract, 2, 1):
                self.state_b2s1.metadata["seller_contract"] = seller_contract
                seller_price = float(seller_contract["price"])
            else:
                seller_price = self._extract_price(seller1_action_buyer2)
            if seller_price is not None:
                self.state_b2s1.update(seller_price=seller_price)
        
        # seller2-buyer1
        if buyer1_selected_seller == 2 and seller2_action_buyer1 is not None:
            self.memory_b1s2.add_message("seller", seller2_action_buyer1, self.current_round)
            seller_contract = self._extract_contract(seller2_action_buyer1) if self.use_contract_mode else None
            if seller_contract and self._validate_contract(seller_contract, 1, 2):
                self.state_b1s2.metadata["seller_contract"] = seller_contract
                seller_price = float(seller_contract["price"])
            else:
                seller_price = self._extract_price(seller2_action_buyer1)
            if seller_price is not None:
                self.state_b1s2.update(seller_price=seller_price)
        
        # seller2-buyer2
        if buyer2_selected_seller == 2 and seller2_action_buyer2 is not None:
            self.memory_b2s2.add_message("seller", seller2_action_buyer2, self.current_round)
            seller_contract = self._extract_contract(seller2_action_buyer2) if self.use_contract_mode else None
            if seller_contract and self._validate_contract(seller_contract, 2, 2):
                self.state_b2s2.metadata["seller_contract"] = seller_contract
                seller_price = float(seller_contract["price"])
            else:
                seller_price = self._extract_price(seller2_action_buyer2)
            if seller_price is not None:
                self.state_b2s2.update(seller_price=seller_price)
        
        # Check if deal can be made with the selected sellers.
        deals = []  # Price mode: (buyer_id, seller_id, price); contract mode: (..., agreed_contract)

        def maybe_add_contract_deal(buyer_id: int, seller_id: int) -> None:
            agreed_contract = self._resolve_agreed_contract(buyer_id, seller_id)
            if not agreed_contract:
                return
            buyer_utility, seller_utility = self._calculate_contract_utilities(agreed_contract, buyer_id, seller_id)
            if buyer_utility < 0 or seller_utility < 0:
                return
            state = self._get_pair_state(buyer_id, seller_id)
            state.metadata["agreed_contract"] = agreed_contract
            state.metadata["buyer_utility"] = buyer_utility
            state.metadata["seller_utility"] = seller_utility
            deals.append((buyer_id, seller_id, agreed_contract, buyer_utility + seller_utility))

        def maybe_add_price_deal(buyer_id: int, seller_id: int, state: NegotiationState) -> None:
            if state.buyer_price is None or state.seller_price is None:
                return
            price_diff = abs(state.buyer_price - state.seller_price)
            if price_diff <= self.price_tolerance:
                deal_price = (state.buyer_price + state.seller_price) / 2
                deals.append((buyer_id, seller_id, deal_price))
            elif state.seller_price <= state.buyer_price:
                deals.append((buyer_id, seller_id, state.seller_price))

        selected_pairs = []
        if buyer1_selected_seller == 1 and buyer1_action is not None:
            selected_pairs.append((1, 1, self.state_b1s1))
        if buyer1_selected_seller == 2 and buyer1_action is not None:
            selected_pairs.append((1, 2, self.state_b1s2))
        if buyer2_selected_seller == 1 and buyer2_action is not None:
            selected_pairs.append((2, 1, self.state_b2s1))
        if buyer2_selected_seller == 2 and buyer2_action is not None:
            selected_pairs.append((2, 2, self.state_b2s2))

        for buyer_id, seller_id, state in selected_pairs:
            if self.use_contract_mode:
                maybe_add_contract_deal(buyer_id, seller_id)
            else:
                maybe_add_price_deal(buyer_id, seller_id, state)

        # Select the best deal by comparing buyer/seller utility across all current pair deals.
        if deals:
            best_deal = None
            best_utility = float('-inf')

            for deal in deals:
                if self.use_contract_mode:
                    buyer_id, seller_id, agreed_contract, utility = deal
                    price = float(agreed_contract["price"])
                else:
                    buyer_id, seller_id, price = deal
                    buyer_max = self.buyer1_max_price if buyer_id == 1 else self.buyer2_max_price
                    seller_min = self.seller1_min_price if seller_id == 1 else self.seller2_min_price
                    buyer_savings = (buyer_max - price) if buyer_max is not None else 0
                    seller_profit = (price - seller_min) if seller_min is not None else 0
                    utility = buyer_savings + seller_profit
                
                if utility > best_utility:
                    best_utility = utility
                    best_deal = (buyer_id, seller_id, price)
            
            if best_deal:
                self.final_selected_buyer, self.final_selected_seller, self.final_deal_price = best_deal
        
        # Check if deal is made
        terminated = False
        truncated = False
        reward = 0.0
        buyer1_reward = 0.0
        buyer2_reward = 0.0
        seller1_reward = 0.0
        seller2_reward = 0.0
        
        if self.final_selected_buyer is not None and self.final_selected_seller is not None and self.final_deal_price is not None:
            terminated = True
            self.negotiation_info.status = NegotiationStatus.AGREED
            # Increment current_round to reflect that this round is completed
            # This ensures round count is accurate when calculating final scores
            self.current_round += 1
            self.negotiation_info.round_count = self.current_round
            reward = self._calculate_reward()
            buyer1_reward = self._calculate_buyer_reward(1)
            buyer2_reward = self._calculate_buyer_reward(2)
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
            buyer1_reward = self._calculate_buyer_reward(1)
            buyer2_reward = self._calculate_buyer_reward(2)
            seller1_reward = self._calculate_seller_reward(1)
            seller2_reward = self._calculate_seller_reward(2)
        else:
            # Move to next round
            self.current_round += 1
            self.negotiation_info.round_count = self.current_round
        
        # Calculate step rewards for every round
        # Only calculate for the selected sellers in this round (sequential negotiation)
        step_buyer1_reward = self._calculate_step_buyer_reward(1)
        step_buyer2_reward = self._calculate_step_buyer_reward(2)
        step_seller1_reward = self._calculate_step_seller_reward(1) if (buyer1_selected_seller == 1 or buyer2_selected_seller == 1) else None
        step_seller2_reward = self._calculate_step_seller_reward(2) if (buyer1_selected_seller == 2 or buyer2_selected_seller == 2) else None
        
        # Build observation and info
        observation = self._get_observation()
        info = self._get_info()
        
        # Add step rewards to info for every step
        info["step_buyer1_reward"] = step_buyer1_reward
        info["step_buyer2_reward"] = step_buyer2_reward
        if step_seller1_reward is not None:
            info["step_seller1_reward"] = step_seller1_reward
        if step_seller2_reward is not None:
            info["step_seller2_reward"] = step_seller2_reward
        
        if terminated or truncated:
            info["termination_reason"] = "agreed" if terminated else "timeout"
            if terminated:
                info["selected_buyer"] = self.final_selected_buyer
                info["selected_seller"] = self.final_selected_seller
                info["final_deal_price"] = self.final_deal_price
            info["buyer1_reward"] = buyer1_reward
            info["buyer2_reward"] = buyer2_reward
            info["seller1_reward"] = seller1_reward
            info["seller2_reward"] = seller2_reward
            # Calculate GlobalScore, BuyerScore, and SellerScore for final result
            # Note: current_round has been incremented to reflect the completed round
            # Don't print here - will be printed in render() after Round Summary
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
        
        # Get messages from the round that just completed
        # Note: In step(), messages are added to current_round
        # - If agreement reached: current_round stays the same, messages are in current_round
        # - If no agreement: current_round is incremented, messages are in current_round - 1
        history_b1s1 = self.memory_b1s1.get_history()
        history_b1s2 = self.memory_b1s2.get_history()
        history_b2s1 = self.memory_b2s1.get_history()
        history_b2s2 = self.memory_b2s2.get_history()
        
        # Determine which round's messages to display
        # Messages are stored with the round value at the time of storage (before current_round is incremented)
        # In step(), messages are added first, then current_round is incremented
        # So for any completed round, messages are stored at current_round - 1
        round_to_display = self.current_round - 1 if self.current_round > 0 else 0
        
        # Determine display round number
        if self.negotiation_info.status in [NegotiationStatus.AGREED, NegotiationStatus.TIMEOUT]:
            display_round = self.current_round
        else:
            display_round = self.current_round if self.current_round > 0 else 0
        
        output_lines.append(f"\n{'='*60}")
        output_lines.append(f"Round {display_round} - Sequential Negotiation Output")
        output_lines.append(f"{'='*60}")
        
        # Display which seller each buyer selected this round
        if self.buyer1_selected_seller is not None:
            output_lines.append(f"\n[Buyer 1 Selected Seller: Seller {self.buyer1_selected_seller}]")
        if self.buyer2_selected_seller is not None:
            output_lines.append(f"[Buyer 2 Selected Seller: Seller {self.buyer2_selected_seller}]")
        
        # Display Buyer1-Seller1 conversation (if this round buyer1 negotiated with seller1)
        if self.buyer1_selected_seller == 1:
            output_lines.append(f"\n[BUYER 1 - SELLER 1 Conversation]:")
            if history_b1s1:
                round_messages = [
                    msg for msg in history_b1s1 if msg["round"] == round_to_display
                ]
                if round_messages:
                    # Display buyer message first (if exists)
                    buyer_msg = next(
                        (msg for msg in round_messages if msg["role"] == "buyer"), 
                        None
                    )
                    if buyer_msg:
                        output_lines.append(f"  [BUYER]: {buyer_msg['content']}")
                    
                    # Display seller message (if exists)
                    seller_msg = next(
                        (msg for msg in round_messages if msg["role"] == "seller"), 
                        None
                    )
                    if seller_msg:
                        output_lines.append(f"  [SELLER]: {seller_msg['content']}")
        
        # Display Buyer1-Seller2 conversation (if this round buyer1 negotiated with seller2)
        if self.buyer1_selected_seller == 2:
            output_lines.append(f"\n[BUYER 1 - SELLER 2 Conversation]:")
            if history_b1s2:
                round_messages = [
                    msg for msg in history_b1s2 if msg["round"] == round_to_display
                ]
                if round_messages:
                    # Display buyer message first (if exists)
                    buyer_msg = next(
                        (msg for msg in round_messages if msg["role"] == "buyer"), 
                        None
                    )
                    if buyer_msg:
                        output_lines.append(f"  [BUYER]: {buyer_msg['content']}")
                    
                    # Display seller message (if exists)
                    seller_msg = next(
                        (msg for msg in round_messages if msg["role"] == "seller"), 
                        None
                    )
                    if seller_msg:
                        output_lines.append(f"  [SELLER]: {seller_msg['content']}")
        
        # Display Buyer2-Seller1 conversation (if this round buyer2 negotiated with seller1)
        if self.buyer2_selected_seller == 1:
            output_lines.append(f"\n[BUYER 2 - SELLER 1 Conversation]:")
            if history_b2s1:
                round_messages = [
                    msg for msg in history_b2s1 if msg["round"] == round_to_display
                ]
                if round_messages:
                    # Display buyer message first (if exists)
                    buyer_msg = next(
                        (msg for msg in round_messages if msg["role"] == "buyer"), 
                        None
                    )
                    if buyer_msg:
                        output_lines.append(f"  [BUYER]: {buyer_msg['content']}")
                    
                    # Display seller message (if exists)
                    seller_msg = next(
                        (msg for msg in round_messages if msg["role"] == "seller"), 
                        None
                    )
                    if seller_msg:
                        output_lines.append(f"  [SELLER]: {seller_msg['content']}")
        
        # Display Buyer2-Seller2 conversation (if this round buyer2 negotiated with seller2)
        if self.buyer2_selected_seller == 2:
            output_lines.append(f"\n[BUYER 2 - SELLER 2 Conversation]:")
            if history_b2s2:
                round_messages = [
                    msg for msg in history_b2s2 if msg["round"] == round_to_display
                ]
                if round_messages:
                    # Display buyer message first (if exists)
                    buyer_msg = next(
                        (msg for msg in round_messages if msg["role"] == "buyer"), 
                        None
                    )
                    if buyer_msg:
                        output_lines.append(f"  [BUYER]: {buyer_msg['content']}")
                    
                    # Display seller message (if exists)
                    seller_msg = next(
                        (msg for msg in round_messages if msg["role"] == "seller"), 
                        None
                    )
                    if seller_msg:
                        output_lines.append(f"  [SELLER]: {seller_msg['content']}")
        
        # Round summary section
        output_lines.append(f"\n{'-'*60}")
        output_lines.append(f"Round {self.current_round} Summary:")
        output_lines.append(f"{'-'*60}")
        
        # Display Buyer1-Seller1 prices (total for both products)
        output_lines.append(f"\nBuyer 1 - Seller 1:")
        if self.state_b1s1.buyer_price is not None:
            output_lines.append(f"  Buyer Total Price: ${self.state_b1s1.buyer_price:.2f}")
        else:
            output_lines.append(f"  Buyer Total Price: Not specified")
        if self.state_b1s1.seller_price is not None:
            output_lines.append(f"  Seller Total Price: ${self.state_b1s1.seller_price:.2f}")
        else:
            output_lines.append(f"  Seller Total Price: Not specified")
        
        # Display Buyer1-Seller2 prices (total for both products)
        output_lines.append(f"\nBuyer 1 - Seller 2:")
        if self.state_b1s2.buyer_price is not None:
            output_lines.append(f"  Buyer Total Price: ${self.state_b1s2.buyer_price:.2f}")
        else:
            output_lines.append(f"  Buyer Total Price: Not specified")
        if self.state_b1s2.seller_price is not None:
            output_lines.append(f"  Seller Total Price: ${self.state_b1s2.seller_price:.2f}")
        else:
            output_lines.append(f"  Seller Total Price: Not specified")
        
        # Display Buyer2-Seller1 prices (total for both products)
        output_lines.append(f"\nBuyer 2 - Seller 1:")
        if self.state_b2s1.buyer_price is not None:
            output_lines.append(f"  Buyer Total Price: ${self.state_b2s1.buyer_price:.2f}")
        else:
            output_lines.append(f"  Buyer Total Price: Not specified")
        if self.state_b2s1.seller_price is not None:
            output_lines.append(f"  Seller Total Price: ${self.state_b2s1.seller_price:.2f}")
        else:
            output_lines.append(f"  Seller Total Price: Not specified")
        
        # Display Buyer2-Seller2 prices (total for both products)
        output_lines.append(f"\nBuyer 2 - Seller 2:")
        if self.state_b2s2.buyer_price is not None:
            output_lines.append(f"  Buyer Total Price: ${self.state_b2s2.buyer_price:.2f}")
        else:
            output_lines.append(f"  Buyer Total Price: Not specified")
        if self.state_b2s2.seller_price is not None:
            output_lines.append(f"  Seller Total Price: ${self.state_b2s2.seller_price:.2f}")
        else:
            output_lines.append(f"  Seller Total Price: Not specified")
        
        # Display deal status
        if self.final_selected_buyer is not None and self.final_selected_seller is not None:
            output_lines.append(f"\n  ✓ DEAL MADE: Buyer {self.final_selected_buyer} with Seller {self.final_selected_seller}")
            if self.final_deal_price is not None:
                output_lines.append(f"  Final Deal Total Price: ${self.final_deal_price:.2f}")
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
        self.memory_b1s1.clear()
        self.memory_b1s2.clear()
        self.memory_b2s1.clear()
        self.memory_b2s2.clear()
        self.state_b1s1 = NegotiationState()
        self.state_b1s2 = NegotiationState()
        self.state_b2s1 = NegotiationState()
        self.state_b2s2 = NegotiationState()
    
    def _get_observation(self) -> Dict[str, Any]:
        """Get current observation"""
        obs = {
            "conversation_history_b1s1": self.memory_b1s1.get_history(),
            "conversation_history_b1s2": self.memory_b1s2.get_history(),
            "conversation_history_b2s1": self.memory_b2s1.get_history(),
            "conversation_history_b2s2": self.memory_b2s2.get_history(),
            "current_round": self.current_round,
            "buyer1_selected_seller": self.buyer1_selected_seller,
            "buyer2_selected_seller": self.buyer2_selected_seller,
            "b1s1_buyer_price": self.state_b1s1.buyer_price,  # Total price for both products
            "b1s1_seller_price": self.state_b1s1.seller_price,  # Total price for both products
            "b1s2_buyer_price": self.state_b1s2.buyer_price,  # Total price for both products
            "b1s2_seller_price": self.state_b1s2.seller_price,  # Total price for both products
            "b2s1_buyer_price": self.state_b2s1.buyer_price,  # Total price for both products
            "b2s1_seller_price": self.state_b2s1.seller_price,  # Total price for both products
            "b2s2_buyer_price": self.state_b2s2.buyer_price,  # Total price for both products
            "b2s2_seller_price": self.state_b2s2.seller_price,  # Total price for both products
            "status": self.negotiation_info.status.value,
            "final_selected_buyer": self.final_selected_buyer,
            "final_selected_seller": self.final_selected_seller,
            "final_deal_price": self.final_deal_price,
            "contract_configs": {
                pair: self._build_public_contract_config(int(pair[1]), int(pair[3]))
                for pair in self.contract_configs
            } if self.use_contract_mode else None,
        }
        if self.use_contract_mode:
            for buyer_id in (1, 2):
                for seller_id in (1, 2):
                    pair = self._pair_key(buyer_id, seller_id)
                    state = self._get_pair_state(buyer_id, seller_id)
                    obs[f"{pair}_buyer_contract"] = state.metadata.get("buyer_contract")
                    obs[f"{pair}_seller_contract"] = state.metadata.get("seller_contract")
                    obs[f"{pair}_agreed_contract"] = state.metadata.get("agreed_contract")
        # Include product_images for VLM (aligned with multi_buyer_multi_seller Task3: no product_info in obs)
        if getattr(self, "product_images", None) is not None:
            obs["product_images"] = self.product_images
        return obs
    
    def _get_info(self) -> Dict[str, Any]:
        """Get current info"""
        info = {
            "round": self.current_round,
            "status": self.negotiation_info.status.value,
            "buyer1_selected_seller": self.buyer1_selected_seller,
            "buyer2_selected_seller": self.buyer2_selected_seller,
            "b1s1_buyer_price": self.state_b1s1.buyer_price,  # Total price for both products
            "b1s1_seller_price": self.state_b1s1.seller_price,  # Total price for both products
            "b1s2_buyer_price": self.state_b1s2.buyer_price,  # Total price for both products
            "b1s2_seller_price": self.state_b1s2.seller_price,  # Total price for both products
            "b2s1_buyer_price": self.state_b2s1.buyer_price,  # Total price for both products
            "b2s1_seller_price": self.state_b2s1.seller_price,  # Total price for both products
            "b2s2_buyer_price": self.state_b2s2.buyer_price,  # Total price for both products
            "b2s2_seller_price": self.state_b2s2.seller_price,  # Total price for both products
            "final_selected_buyer": self.final_selected_buyer,
            "final_selected_seller": self.final_selected_seller,
            "final_deal_price": self.final_deal_price,
            "contract_configs": self.contract_configs if self.use_contract_mode else None,
            "negotiation_info": self.negotiation_info,
        }
        if self.use_contract_mode:
            for buyer_id in (1, 2):
                for seller_id in (1, 2):
                    pair = self._pair_key(buyer_id, seller_id)
                    state = self._get_pair_state(buyer_id, seller_id)
                    info[f"{pair}_buyer_contract"] = state.metadata.get("buyer_contract")
                    info[f"{pair}_seller_contract"] = state.metadata.get("seller_contract")
                    info[f"{pair}_agreed_contract"] = state.metadata.get("agreed_contract")
                    info[f"{pair}_buyer_utility"] = state.metadata.get("buyer_utility")
                    info[f"{pair}_seller_utility"] = state.metadata.get("seller_utility")
                    info[f"{pair}_z_max"] = self.z_max_by_pair.get(pair)
        if hasattr(self, "_product_info") and self._product_info:
            info["product_info"] = self._product_info
        return info
    
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
            r'total.*?(\d+\.?\d*)',  # total 100.50
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
        
        if self.negotiation_info.status == NegotiationStatus.AGREED and self.final_selected_buyer is not None and self.final_selected_seller is not None and self.final_deal_price is not None:
            # Deal reached: buyer savings + seller profit + time cost
            deal_price = self.final_deal_price
            reward = 0.0
            buyer_savings = 0.0
            seller_profit = 0.0
            
            # Get the selected buyer's max_price and seller's min_price
            selected_buyer_max_price = None
            if self.final_selected_buyer == 1:
                selected_buyer_max_price = self.buyer1_max_price
            elif self.final_selected_buyer == 2:
                selected_buyer_max_price = self.buyer2_max_price
            
            selected_seller_min_price = None
            if self.final_selected_seller == 1:
                selected_seller_min_price = self.seller1_min_price
            elif self.final_selected_seller == 2:
                selected_seller_min_price = self.seller2_min_price
            
            # Calculate buyer savings: buyer_max_price - deal_price (for both products)
            if selected_buyer_max_price is not None:
                buyer_savings = selected_buyer_max_price - deal_price
                reward += buyer_savings * self.reward_weights["buyer_savings"]
            
            # Calculate seller profit: deal_price - seller_min_price (for both products)
            if selected_seller_min_price is not None:
                seller_profit = deal_price - selected_seller_min_price
                reward += seller_profit * self.reward_weights["seller_profit"]
            
            # Add time cost (negative penalty)
            reward += time_cost * self.reward_weights["time_cost"]
            
            weighted_buyer_savings = buyer_savings * self.reward_weights["buyer_savings"] if selected_buyer_max_price is not None else 0.0
            weighted_seller_profit = seller_profit * self.reward_weights["seller_profit"] if selected_seller_min_price is not None else 0.0
            weighted_time_cost = time_cost * self.reward_weights["time_cost"]
            print(f"Global Reward = buyer{self.final_selected_buyer}_savings({buyer_savings:.2f} * {self.reward_weights['buyer_savings']:.2f}) + seller{self.final_selected_seller}_profit({seller_profit:.2f} * {self.reward_weights['seller_profit']:.2f}) + time_cost({time_cost:.2f} * {self.reward_weights['time_cost']:.2f}) = {reward:.2f} (buyer{self.final_selected_buyer}_max={selected_buyer_max_price}, deal_price={deal_price:.2f}, seller{self.final_selected_seller}_min={selected_seller_min_price}, round={self.current_round})")
            
            return reward
        
        else:
            # Deal not reached: only time cost (negative penalty)
            weighted_time_cost = time_cost * self.reward_weights["time_cost"]
            print(f"Global Reward = time_cost({time_cost:.2f} * {self.reward_weights['time_cost']:.2f}) = {weighted_time_cost:.2f} (round={self.current_round}, deal not reached)")
            return weighted_time_cost
    
    def _calculate_buyer_reward(self, buyer_id: int) -> float:
        """Calculate reward from buyer's perspective
        
        Calculate reward value based on negotiation result from buyer's perspective.
        This reward does not include seller profit.
        
        If deal is reached with this buyer:
            reward = buyer savings + time cost (negative, based on rounds)
            - buyer savings = buyer_max_price - deal_price (money saved by buyer for both products)
            - time cost = -current_round (penalty for number of rounds taken)
        
        If deal is not reached or reached with another buyer:
            reward = time cost (negative, based on rounds)
            - time cost = -current_round (penalty for number of rounds taken)
        
        Args:
            buyer_id: Buyer ID (1 or 2)
        
        Returns:
            Reward value from buyer's perspective
        """
        # Time cost: negative value based on number of rounds
        time_cost = -self.current_round
        
        # Check if deal was reached with this buyer
        deal_reached_with_this_buyer = (
            self.negotiation_info.status == NegotiationStatus.AGREED and
            self.final_selected_buyer == buyer_id and
            self.final_deal_price is not None
        )
        
        if deal_reached_with_this_buyer:
            # Deal reached with this buyer: buyer savings + time cost
            deal_price = self.final_deal_price
            reward = 0.0
            buyer_savings = 0.0
            
            # Get this buyer's max_price
            buyer_max_price = None
            if buyer_id == 1:
                buyer_max_price = self.buyer1_max_price
            elif buyer_id == 2:
                buyer_max_price = self.buyer2_max_price
            
            # Calculate buyer savings: buyer_max_price - deal_price (for both products)
            if buyer_max_price is not None:
                buyer_savings = buyer_max_price - deal_price
                reward += buyer_savings * self.reward_weights["buyer_savings"]
            
            # Add time cost (negative penalty)
            reward += time_cost * self.reward_weights["time_cost"]
            
            weighted_buyer_savings = buyer_savings * self.reward_weights["buyer_savings"] if buyer_max_price is not None else 0.0
            weighted_time_cost = time_cost * self.reward_weights["time_cost"]
            print(f"Buyer{buyer_id} Reward = buyer_savings({buyer_savings:.2f} * {self.reward_weights['buyer_savings']:.2f}) + time_cost({time_cost:.2f} * {self.reward_weights['time_cost']:.2f}) = {reward:.2f} (buyer{buyer_id}_max={buyer_max_price}, deal_price={deal_price:.2f}, round={self.current_round})")
            
            return reward
        
        else:
            # Deal not reached or reached with another buyer: only time cost (negative penalty)
            weighted_time_cost = time_cost * self.reward_weights["time_cost"]
            print(f"Buyer{buyer_id} Reward = time_cost({time_cost:.2f} * {self.reward_weights['time_cost']:.2f}) = {weighted_time_cost:.2f} (round={self.current_round}, deal not reached with this buyer)")
            return weighted_time_cost
    
    def _calculate_seller_reward(self, seller_id: int) -> float:
        """Calculate reward from seller's perspective
        
        Calculate reward value based on negotiation result from seller's perspective.
        This reward does not include buyer savings.
        
        If deal is reached with this seller:
            reward = seller profit + time cost (negative, based on rounds)
            - seller profit = deal_price - seller_min_price (extra profit for seller for both products)
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
            
            # Calculate seller profit: deal_price - seller_min_price (for both products)
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
    
    def _calculate_step_buyer_reward(self, buyer_id: int) -> float:
        """Calculate step reward from buyer's perspective for current round
        
        Calculate reward value based on buyer's current offer in this round with the selected seller.
        This is calculated every round, not just at the end.
        
        reward = buyer savings (from current offer) + round cost
        - buyer savings = buyer_max_price - buyer_price (money saved by current offer for both products)
        - round cost = -current_round (penalty for number of rounds taken)
        
        Args:
            buyer_id: Buyer ID (1 or 2)
        
        Returns:
            Step reward value from buyer's perspective for current round
        """
        # Round cost: negative value based on number of rounds
        round_cost = -self.current_round
        
        # Calculate buyer reward with the selected seller
        reward = 0.0
        buyer_savings = 0.0
        
        # Get buyer price from the selected seller
        buyer_price = None
        buyer_max_price = None
        if buyer_id == 1:
            buyer_max_price = self.buyer1_max_price
            if self.buyer1_selected_seller == 1:
                buyer_price = self.state_b1s1.buyer_price
            elif self.buyer1_selected_seller == 2:
                buyer_price = self.state_b1s2.buyer_price
        elif buyer_id == 2:
            buyer_max_price = self.buyer2_max_price
            if self.buyer2_selected_seller == 1:
                buyer_price = self.state_b2s1.buyer_price
            elif self.buyer2_selected_seller == 2:
                buyer_price = self.state_b2s2.buyer_price
        
        # Calculate buyer savings: buyer_max_price - buyer_price (for both products)
        if buyer_price is not None and buyer_max_price is not None:
            buyer_savings = buyer_max_price - buyer_price
            reward += buyer_savings * self.reward_weights["buyer_savings"]
        
        # Add round cost (negative penalty)
        reward += round_cost * self.reward_weights["time_cost"]
        
        return reward
    
    def _calculate_step_seller_reward(self, seller_id: int) -> float:
        """Calculate step reward from seller's perspective for current round
        
        Calculate reward value based on seller's current offer in this round.
        This is calculated every round, not just at the end.
        
        reward = seller profit (from current offer) + round cost
        - seller profit = seller_price - seller_min_price (profit from current offer for both products)
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
        
        # Get seller state and min_price
        seller_state = None
        seller_min_price = None
        if seller_id == 1:
            seller_min_price = self.seller1_min_price
            # Get the most recent price from either buyer1 or buyer2
            if self.buyer1_selected_seller == 1 and self.state_b1s1.seller_price is not None:
                seller_state = self.state_b1s1
            elif self.buyer2_selected_seller == 1 and self.state_b2s1.seller_price is not None:
                seller_state = self.state_b2s1
            # If both buyers selected seller1, prefer the one with higher price
            if self.buyer1_selected_seller == 1 and self.buyer2_selected_seller == 1:
                if (self.state_b1s1.seller_price is not None and self.state_b2s1.seller_price is not None):
                    seller_state = self.state_b1s1 if self.state_b1s1.seller_price >= self.state_b2s1.seller_price else self.state_b2s1
                elif self.state_b1s1.seller_price is not None:
                    seller_state = self.state_b1s1
                elif self.state_b2s1.seller_price is not None:
                    seller_state = self.state_b2s1
        elif seller_id == 2:
            seller_min_price = self.seller2_min_price
            # Get the most recent price from either buyer1 or buyer2
            if self.buyer1_selected_seller == 2 and self.state_b1s2.seller_price is not None:
                seller_state = self.state_b1s2
            elif self.buyer2_selected_seller == 2 and self.state_b2s2.seller_price is not None:
                seller_state = self.state_b2s2
            # If both buyers selected seller2, prefer the one with higher price
            if self.buyer1_selected_seller == 2 and self.buyer2_selected_seller == 2:
                if (self.state_b1s2.seller_price is not None and self.state_b2s2.seller_price is not None):
                    seller_state = self.state_b1s2 if self.state_b1s2.seller_price >= self.state_b2s2.seller_price else self.state_b2s2
                elif self.state_b1s2.seller_price is not None:
                    seller_state = self.state_b1s2
                elif self.state_b2s2.seller_price is not None:
                    seller_state = self.state_b2s2
        
        # Calculate seller profit from current offer: seller_price - seller_min_price (for both products)
        if seller_state is not None and seller_state.seller_price is not None and seller_min_price is not None:
            seller_profit = seller_state.seller_price - seller_min_price
            reward += seller_profit * self.reward_weights["seller_profit"]
        
        # Add round cost (negative penalty)
        reward += round_cost * self.reward_weights["time_cost"]
        
        return reward

    def _get_contract_score_terms(self) -> Optional[Dict[str, float]]:
        """Return normalized utility terms using the best buyer-seller utility space."""
        if not (
            self.use_contract_mode
            and self.final_selected_buyer in (1, 2)
            and self.final_selected_seller in (1, 2)
        ):
            return None

        market_best_pair = self._get_market_best_contract_pair()
        if market_best_pair is None:
            return None
        z_market = self.z_max_by_pair.get(market_best_pair)
        if not z_market or z_market <= 0:
            return None

        selected_pair = self._pair_key(self.final_selected_buyer, self.final_selected_seller)
        selected_state = self._get_pair_state(self.final_selected_buyer, self.final_selected_seller)
        final_contract = selected_state.metadata.get("agreed_contract")
        if not final_contract:
            final_contract = self._resolve_agreed_contract(self.final_selected_buyer, self.final_selected_seller)
            if final_contract:
                selected_state.metadata["agreed_contract"] = final_contract

        if not final_contract or not self._validate_contract(final_contract, self.final_selected_buyer, self.final_selected_seller):
            return None

        u_b, u_s = self._calculate_contract_utilities(final_contract, self.final_selected_buyer, self.final_selected_seller)
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
            "market_best_pair": market_best_pair,
            "selected_pair": selected_pair,
            "selected_z_max": self.z_max_by_pair.get(selected_pair) or 0.0,
        }
    
    def _get_buyers_market_ceiling(self) -> Optional[float]:
        """Highest max acceptable total price across buyers (market buyer ceiling for scoring)."""
        caps = [p for p in (self.buyer1_max_price, self.buyer2_max_price) if p is not None]
        if not caps:
            return None
        return max(caps)

    def _get_market_best_floor(self) -> Optional[float]:
        """Lowest seller floor across sellers (best market cost for the bundle; scoring)."""
        floors = [p for p in (self.seller1_min_price, self.seller2_min_price) if p is not None]
        if not floors:
            return None
        return min(floors)

    def _print_global_score_details(self):
        """Print GlobalScore details (call from example after step rewards; aligned with Task3)."""
        self._calculate_global_score(print_details=True)

    def _print_buyer_score_details(self):
        self._calculate_buyer_score(print_details=True)

    def _print_seller_score_details(self):
        self._calculate_seller_score(print_details=True)

    def _calculate_global_score(self, print_details: bool = True) -> float:
        """GlobalScore (aligned with multi_buyer_multi_seller Task3; total bundle price).

        Uses buyers_market_ceiling = max(buyer1_max, buyer2_max) and
        market_best_floor = min(seller1_min, seller2_min) for Z and utilities.
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
                    print(f"  market_best_pair = {score_terms['market_best_pair']}")
                    print(f"  selected_pair = {score_terms['selected_pair']}")
                    print(f"  z_market = {score_terms['z_market']:.4f}, selected_z_max = {score_terms['selected_z_max']:.4f}")
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

        buyers_market_ceiling = self._get_buyers_market_ceiling()
        market_best_floor = self._get_market_best_floor()

        if buyers_market_ceiling is None or market_best_floor is None:
            round_index = max(0, self.current_round)
            discount = self.gamma ** round_index
            failure_penalty = -self.failure_penalty_weight * (1.0 - discount)

            if print_details:
                print(f"\n[GlobalScore Calculation]")
                print(f"  buyers_market_ceiling or market_best_floor is None")
                print(f"  round_index = {round_index}, gamma = {self.gamma}, discount = γ^{round_index} = {discount:.6f}")
                print(f"  FailurePenalty = -F({self.failure_penalty_weight:.1f}) * (1 - discount({discount:.6f})) = {failure_penalty:.3f}")
                print(f"  GlobalScore = {failure_penalty:.3f}")
            return failure_penalty

        Z_market = buyers_market_ceiling - market_best_floor
        round_index = max(0, self.current_round)
        discount = self.gamma ** round_index
        feasible_deal = (self.negotiation_info.status == NegotiationStatus.AGREED) or (self.final_deal_price is not None)

        if self.final_deal_price is not None:
            final_price = self.final_deal_price
        else:
            failure_penalty = -self.failure_penalty_weight * (1.0 - discount)

            if print_details:
                print(f"\n[GlobalScore Calculation]")
                print(
                    f"  buyers_market_ceiling = max(buyer1_max({self.buyer1_max_price}), buyer2_max({self.buyer2_max_price})) = {buyers_market_ceiling:.2f}"
                )
                print(
                    f"  market_best_floor = min(seller1_min({self.seller1_min_price}), seller2_min({self.seller2_min_price})) = {market_best_floor:.2f}"
                )
                print(f"  Z_market = {Z_market:.2f}")
                print(f"  No final price available")
                print(f"  feasible_deal = {feasible_deal}")
                print(f"  valid_range = (Z_market > 0) = {Z_market > 0}")
                print(f"  round_index = {round_index}, gamma = {self.gamma}, discount = γ^{round_index} = {discount:.6f}")
                print(f"  FailurePenalty = -F({self.failure_penalty_weight:.1f}) * (1 - discount({discount:.6f})) = {failure_penalty:.3f}")
                print(f"  GlobalScore = {failure_penalty:.3f}")
            return failure_penalty

        valid_range = (Z_market > 0) and (market_best_floor <= final_price <= buyers_market_ceiling)

        if feasible_deal and valid_range:
            u_b = (buyers_market_ceiling - final_price) / Z_market
            u_s = (final_price - market_best_floor) / Z_market
            Q = 4.0 * u_b * u_s
            deal_score = self.deal_score_weight * discount
            quality_score = self.quality_score_weight * Q * discount
            efficiency_score = self.efficiency_score_weight * discount
            global_score = deal_score + quality_score + efficiency_score

            if print_details:
                print(f"\n[GlobalScore Calculation]")
                print(
                    f"  buyers_market_ceiling = max(buyer1_max({self.buyer1_max_price}), buyer2_max({self.buyer2_max_price})) = {buyers_market_ceiling:.2f}"
                )
                print(
                    f"  market_best_floor = min(seller1_min({self.seller1_min_price}), seller2_min({self.seller2_min_price})) = {market_best_floor:.2f}"
                )
                print(f"  Z_market = {Z_market:.2f}")
                print(f"  final_price = {final_price:.2f}")
                print(f"  feasible_deal = {feasible_deal} (negotiation status: {self.negotiation_info.status.value})")
                print(
                    f"  valid_range = (Z_market > 0) and (market_best_floor({market_best_floor:.2f}) <= final_price({final_price:.2f}) <= buyers_market_ceiling({buyers_market_ceiling:.2f})) = {valid_range}"
                )
                print(f"  round_index = {round_index}, gamma = {self.gamma}, discount = γ^{round_index} = {discount:.6f}")
                print(f"  u_b_market = (buyers_market_ceiling - final_price) / Z_market = {u_b:.4f}")
                print(f"  u_s_market = (final_price - market_best_floor) / Z_market = {u_s:.4f}")
                print(f"  Q = 4 * u_b({u_b:.4f}) * u_s({u_s:.4f}) = {Q:.4f}")
                print(f"  DealScore = D({self.deal_score_weight:.1f}) * discount({discount:.6f}) = {deal_score:.3f}")
                print(f"  QualityScore = W({self.quality_score_weight:.1f}) * Q({Q:.4f}) * discount({discount:.6f}) = {quality_score:.3f}")
                print(f"  EfficiencyScore = E({self.efficiency_score_weight:.1f}) * discount({discount:.6f}) = {efficiency_score:.3f}")
                print(f"  GlobalScore = DealScore({deal_score:.3f}) + QualityScore({quality_score:.3f}) + EfficiencyScore({efficiency_score:.3f}) = {global_score:.3f}")

            return global_score

        failure_penalty = -self.failure_penalty_weight * (1.0 - discount)

        if print_details:
            print(f"\n[GlobalScore Calculation]")
            print(
                f"  buyers_market_ceiling = max(buyer1_max({self.buyer1_max_price}), buyer2_max({self.buyer2_max_price})) = {buyers_market_ceiling:.2f}"
            )
            print(
                f"  market_best_floor = min(seller1_min({self.seller1_min_price}), seller2_min({self.seller2_min_price})) = {market_best_floor:.2f}"
            )
            print(f"  Z_market = {Z_market:.2f}")
            print(f"  final_price = {final_price:.2f}")
            print(f"  feasible_deal = {feasible_deal} (negotiation status: {self.negotiation_info.status.value})")
            print(
                f"  valid_range = (Z_market > 0) and (market_best_floor({market_best_floor:.2f}) <= final_price({final_price:.2f}) <= buyers_market_ceiling({buyers_market_ceiling:.2f})) = {valid_range}"
            )
            print(f"  round_index = {round_index}, gamma = {self.gamma}, discount = γ^{round_index} = {discount:.6f}")
            print(f"  FailurePenalty = -F({self.failure_penalty_weight:.1f}) * (1 - discount({discount:.6f})) = {failure_penalty:.3f}")
            print(f"  GlobalScore = {failure_penalty:.3f}")

        return failure_penalty

    def _calculate_buyer_score(self, print_details: bool = True) -> float:
        """BuyerScore using buyers_market_ceiling and market_best_floor (aligned with Task3)."""
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
                    print(f"  market_best_pair = {score_terms['market_best_pair']}, selected_pair = {score_terms['selected_pair']}")
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

        buyers_market_ceiling = self._get_buyers_market_ceiling()
        market_best_floor = self._get_market_best_floor()

        if buyers_market_ceiling is None or market_best_floor is None:
            round_index = max(0, self.current_round)
            discount = self.gamma ** round_index
            buyer_score = -self.buyer_failure_penalty_weight * (1.0 - discount)

            if print_details:
                print(f"\n[BuyerScore Calculation]")
                print(f"  buyers_market_ceiling or market_best_floor is None")
                print(f"  round_index = {round_index}, gamma = {self.gamma}, discount = γ^{round_index} = {discount:.6f}")
                print(f"  BuyerScore = -Fb({self.buyer_failure_penalty_weight:.1f}) * (1 - discount({discount:.6f})) = {buyer_score:.3f}")
            return buyer_score

        Z_market = buyers_market_ceiling - market_best_floor
        round_index = max(0, self.current_round)
        discount = self.gamma ** round_index
        feasible_deal = (self.negotiation_info.status == NegotiationStatus.AGREED) or (self.final_deal_price is not None)

        if self.final_deal_price is not None:
            final_price = self.final_deal_price
        else:
            buyer_score = -self.buyer_failure_penalty_weight * (1.0 - discount)

            if print_details:
                print(f"\n[BuyerScore Calculation]")
                print(
                    f"  buyers_market_ceiling = max(buyer1_max({self.buyer1_max_price}), buyer2_max({self.buyer2_max_price})) = {buyers_market_ceiling:.2f}"
                )
                print(
                    f"  market_best_floor = min(seller1_min({self.seller1_min_price}), seller2_min({self.seller2_min_price})) = {market_best_floor:.2f}"
                )
                print(f"  Z_market = {Z_market:.2f}")
                print(f"  No final price available")
                print(f"  round_index = {round_index}, gamma = {self.gamma}, discount = γ^{round_index} = {discount:.6f}")
                print(f"  BuyerScore = -Fb({self.buyer_failure_penalty_weight:.1f}) * (1 - discount({discount:.6f})) = {buyer_score:.3f}")
            return buyer_score

        valid_range = (Z_market > 0) and (market_best_floor <= final_price <= buyers_market_ceiling)

        if feasible_deal and valid_range:
            u_b = (buyers_market_ceiling - final_price) / Z_market
            buyer_score = discount * (self.buyer_deal_weight + self.buyer_utility_weight * u_b + self.buyer_efficiency_weight)

            if print_details:
                print(f"\n[BuyerScore Calculation]")
                print(
                    f"  buyers_market_ceiling = max(buyer1_max({self.buyer1_max_price}), buyer2_max({self.buyer2_max_price})) = {buyers_market_ceiling:.2f}"
                )
                print(
                    f"  market_best_floor = min(seller1_min({self.seller1_min_price}), seller2_min({self.seller2_min_price})) = {market_best_floor:.2f}"
                )
                print(f"  Z_market = {Z_market:.2f}")
                print(f"  final_price = {final_price:.2f}")
                print(f"  feasible_deal = {feasible_deal} (negotiation status: {self.negotiation_info.status.value})")
                print(
                    f"  valid_range = (Z_market > 0) and (market_best_floor({market_best_floor:.2f}) <= final_price({final_price:.2f}) <= buyers_market_ceiling({buyers_market_ceiling:.2f})) = {valid_range}"
                )
                print(f"  round_index = {round_index}, gamma = {self.gamma}, discount = γ^{round_index} = {discount:.6f}")
                print(f"  u_b_market = (buyers_market_ceiling - final_price) / Z_market = {u_b:.4f}")
                print(f"  BuyerScore = discount({discount:.6f}) * (Db + Wb * u_b + Eb) = {buyer_score:.3f}")

            return buyer_score

        buyer_score = -self.buyer_failure_penalty_weight * (1.0 - discount)

        if print_details:
            print(f"\n[BuyerScore Calculation]")
            print(
                f"  buyers_market_ceiling = max(buyer1_max({self.buyer1_max_price}), buyer2_max({self.buyer2_max_price})) = {buyers_market_ceiling:.2f}"
            )
            print(
                f"  market_best_floor = min(seller1_min({self.seller1_min_price}), seller2_min({self.seller2_min_price})) = {market_best_floor:.2f}"
            )
            print(f"  Z_market = {Z_market:.2f}")
            print(f"  final_price = {final_price:.2f}")
            print(f"  feasible_deal = {feasible_deal} (negotiation status: {self.negotiation_info.status.value})")
            print(
                f"  valid_range = (Z_market > 0) and (market_best_floor({market_best_floor:.2f}) <= final_price({final_price:.2f}) <= buyers_market_ceiling({buyers_market_ceiling:.2f})) = {valid_range}"
            )
            print(f"  round_index = {round_index}, gamma = {self.gamma}, discount = γ^{round_index} = {discount:.6f}")
            print(f"  BuyerScore = -Fb({self.buyer_failure_penalty_weight:.1f}) * (1 - discount({discount:.6f})) = {buyer_score:.3f}")

        return buyer_score

    def _calculate_seller_score(self, print_details: bool = True) -> float:
        """SellerScore using buyers_market_ceiling and market_best_floor (aligned with Task3)."""
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
                    print(f"  market_best_pair = {score_terms['market_best_pair']}, selected_pair = {score_terms['selected_pair']}")
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

        buyers_market_ceiling = self._get_buyers_market_ceiling()
        market_best_floor = self._get_market_best_floor()

        if buyers_market_ceiling is None or market_best_floor is None:
            round_index = max(0, self.current_round)
            discount = self.gamma ** round_index
            seller_score = -self.seller_failure_penalty_weight * (1.0 - discount)

            if print_details:
                print(f"\n[SellerScore Calculation]")
                print(f"  buyers_market_ceiling or market_best_floor is None")
                print(f"  round_index = {round_index}, gamma = {self.gamma}, discount = γ^{round_index} = {discount:.6f}")
                print(f"  SellerScore = -Fs({self.seller_failure_penalty_weight:.1f}) * (1 - discount({discount:.6f})) = {seller_score:.3f}")
            return seller_score

        Z_market = buyers_market_ceiling - market_best_floor
        round_index = max(0, self.current_round)
        discount = self.gamma ** round_index
        feasible_deal = (self.negotiation_info.status == NegotiationStatus.AGREED) or (self.final_deal_price is not None)

        if self.final_deal_price is not None:
            final_price = self.final_deal_price
        else:
            seller_score = -self.seller_failure_penalty_weight * (1.0 - discount)

            if print_details:
                print(f"\n[SellerScore Calculation]")
                print(
                    f"  buyers_market_ceiling = max(buyer1_max({self.buyer1_max_price}), buyer2_max({self.buyer2_max_price})) = {buyers_market_ceiling:.2f}"
                )
                print(
                    f"  market_best_floor = min(seller1_min({self.seller1_min_price}), seller2_min({self.seller2_min_price})) = {market_best_floor:.2f}"
                )
                print(f"  Z_market = {Z_market:.2f}")
                print(f"  No final price available")
                print(f"  round_index = {round_index}, gamma = {self.gamma}, discount = γ^{round_index} = {discount:.6f}")
                print(f"  SellerScore = -Fs({self.seller_failure_penalty_weight:.1f}) * (1 - discount({discount:.6f})) = {seller_score:.3f}")
            return seller_score

        valid_range = (Z_market > 0) and (market_best_floor <= final_price <= buyers_market_ceiling)

        if feasible_deal and valid_range:
            u_s = (final_price - market_best_floor) / Z_market
            seller_score = discount * (self.seller_deal_weight + self.seller_utility_weight * u_s + self.seller_efficiency_weight)

            if print_details:
                print(f"\n[SellerScore Calculation]")
                print(
                    f"  buyers_market_ceiling = max(buyer1_max({self.buyer1_max_price}), buyer2_max({self.buyer2_max_price})) = {buyers_market_ceiling:.2f}"
                )
                print(
                    f"  market_best_floor = min(seller1_min({self.seller1_min_price}), seller2_min({self.seller2_min_price})) = {market_best_floor:.2f}"
                )
                print(f"  Z_market = {Z_market:.2f}")
                print(f"  final_price = {final_price:.2f}")
                print(f"  feasible_deal = {feasible_deal} (negotiation status: {self.negotiation_info.status.value})")
                print(
                    f"  valid_range = (Z_market > 0) and (market_best_floor({market_best_floor:.2f}) <= final_price({final_price:.2f}) <= buyers_market_ceiling({buyers_market_ceiling:.2f})) = {valid_range}"
                )
                print(f"  round_index = {round_index}, gamma = {self.gamma}, discount = γ^{round_index} = {discount:.6f}")
                print(f"  u_s_market = (final_price - market_best_floor) / Z_market = {u_s:.4f}")
                print(f"  SellerScore = discount({discount:.6f}) * (Ds + Ws * u_s + Es) = {seller_score:.3f}")

            return seller_score

        seller_score = -self.seller_failure_penalty_weight * (1.0 - discount)

        if print_details:
            print(f"\n[SellerScore Calculation]")
            print(
                f"  buyers_market_ceiling = max(buyer1_max({self.buyer1_max_price}), buyer2_max({self.buyer2_max_price})) = {buyers_market_ceiling:.2f}"
            )
            print(
                f"  market_best_floor = min(seller1_min({self.seller1_min_price}), seller2_min({self.seller2_min_price})) = {market_best_floor:.2f}"
            )
            print(f"  Z_market = {Z_market:.2f}")
            print(f"  final_price = {final_price:.2f}")
            print(f"  feasible_deal = {feasible_deal} (negotiation status: {self.negotiation_info.status.value})")
            print(
                f"  valid_range = (Z_market > 0) and (market_best_floor({market_best_floor:.2f}) <= final_price({final_price:.2f}) <= buyers_market_ceiling({buyers_market_ceiling:.2f})) = {valid_range}"
            )
            print(f"  round_index = {round_index}, gamma = {self.gamma}, discount = γ^{round_index} = {discount:.6f}")
            print(f"  SellerScore = -Fs({self.seller_failure_penalty_weight:.1f}) * (1 - discount({discount:.6f})) = {seller_score:.3f}")

        return seller_score
