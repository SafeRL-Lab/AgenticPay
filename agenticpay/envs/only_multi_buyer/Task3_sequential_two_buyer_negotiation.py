"""Task3 Sequential Two-Buyer Negotiation Environment Implementation

Supports sequential negotiation where seller chooses one buyer per round to negotiate with.
Seller can switch between two buyers and make a deal with either buyer.
"""

from __future__ import annotations

import json
import re
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

from agenticpay.core import BaseEnv, NegotiationStatus, NegotiationInfo
from agenticpay.agents.base_agent import BaseAgent
from agenticpay.memory.conversation_memory import ConversationMemory
from agenticpay.utils.negotiation_state import NegotiationState


class Task3SequentialTwoBuyerNegotiation(BaseEnv):
    """Task3 Sequential Two-Buyer Negotiation Environment
    
    Manages sequential negotiation process where seller chooses one buyer per round to negotiate with.
    Seller can switch between two buyers and make a deal with either buyer.
    """
    
    def __init__(
        self,
        buyer1_agent: BaseAgent,
        buyer2_agent: BaseAgent,
        seller_agent: BaseAgent,
        max_rounds: int = 20,
        initial_seller_price: float = 100.0,
        buyer1_max_price: Optional[float] = None,
        buyer2_max_price: Optional[float] = None,
        seller_min_price: Optional[float] = None,
        environment_info: Optional[Dict[str, Any]] = None,
        price_tolerance: float = 0.0,
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
        """Initialize sequential multi-buyer negotiation environment
        
        Args:
            buyer1_agent: First Buyer Agent
            buyer2_agent: Second Buyer Agent
            seller_agent: Seller Agent
            max_rounds: Maximum number of negotiation rounds
            initial_seller_price: Initial price offered by seller
            buyer1_max_price: Maximum acceptable price for buyer1 (confidential)
            buyer2_max_price: Maximum acceptable price for buyer2 (confidential)
            seller_min_price: Minimum acceptable price for seller (confidential)
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
        self.seller_agent = seller_agent
        self.max_rounds = max_rounds
        self.initial_seller_price = initial_seller_price
        self.buyer1_max_price = buyer1_max_price
        self.buyer2_max_price = buyer2_max_price
        self.seller_min_price = seller_min_price
        self.environment_info = environment_info or {}
        self.contract_configs = self._normalize_contract_configs(self.environment_info)
        self.use_contract_mode = bool(self.contract_configs)
        self.z_max_by_buyer = {
            buyer_id: self._calculate_z_max(config)
            for buyer_id, config in self.contract_configs.items()
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
        
        # State management - separate for each buyer
        self.memory_buyer1 = ConversationMemory()
        self.memory_buyer2 = ConversationMemory()
        self.state_buyer1 = NegotiationState()
        self.state_buyer2 = NegotiationState()
        self.current_round = 0
        self.negotiation_info = NegotiationInfo()
        
        # Track which buyer is currently selected and which buyer was chosen for the deal
        self.current_selected_buyer: Optional[int] = None  # 1 or 2, selected for current round
        self.final_selected_buyer: Optional[int] = None  # 1 or 2, chosen for final deal
        self.final_deal_price: Optional[float] = None
        self.product_images: Optional[List[str]] = None  # For VLM img input
        self._reset_contract_metadata()

    def _normalize_contract_configs(self, environment_info: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
        """Normalize buyer-specific or shared contract config into {1: cfg, 2: cfg}."""
        raw_configs = environment_info.get("buyer_contract_configs")
        if isinstance(raw_configs, dict):
            configs: Dict[int, Dict[str, Any]] = {}
            for buyer_id in (1, 2):
                value = raw_configs.get(buyer_id) or raw_configs.get(str(buyer_id)) or raw_configs.get(f"buyer{buyer_id}")
                if isinstance(value, dict):
                    configs[buyer_id] = deepcopy(value)
            if configs:
                return configs

        shared_config = environment_info.get("contract_config")
        if isinstance(shared_config, dict) and shared_config:
            return {1: deepcopy(shared_config), 2: deepcopy(shared_config)}
        return {}

    def _reset_contract_metadata(self) -> None:
        """Initialize per-buyer contract state used by contract-mode scoring."""
        for state in (self.state_buyer1, self.state_buyer2):
            state.metadata["buyer_contract"] = None
            state.metadata["seller_contract"] = None
            state.metadata["agreed_contract"] = None
            state.metadata["buyer_utility"] = None
            state.metadata["seller_utility"] = None
            state.metadata["z_max"] = None
        for buyer_id, z_max in self.z_max_by_buyer.items():
            self._get_buyer_state(buyer_id).metadata["z_max"] = z_max

    def _get_buyer_state(self, buyer_id: int) -> NegotiationState:
        if buyer_id == 1:
            return self.state_buyer1
        if buyer_id == 2:
            return self.state_buyer2
        raise ValueError(f"buyer_id must be 1 or 2, got {buyer_id}")

    def _get_buyer_contract_config(self, buyer_id: int) -> Dict[str, Any]:
        return self.contract_configs.get(buyer_id, {})

    def _build_public_contract_config(self, buyer_id: int) -> Dict[str, Any]:
        """Build contract config fields both sides may see for one buyer."""
        config = self._get_buyer_contract_config(buyer_id)
        if not config:
            return {}
        public_config: Dict[str, Any] = {"buyer_id": buyer_id}
        for key in ("continuous_bounds", "discrete_options", "field_descriptions", "contrainfo"):
            if key in config:
                public_config[key] = deepcopy(config[key])
        return public_config

    def _build_role_contract_config(self, role: str, buyer_id: int) -> Dict[str, Any]:
        """Expose only the role's private preferences for one buyer."""
        config = self._get_buyer_contract_config(buyer_id)
        role_config = self._build_public_contract_config(buyer_id)
        if role == "buyer" and "buyer_preferences" in config:
            role_config["buyer_preferences"] = deepcopy(config["buyer_preferences"])
        if role == "seller" and "seller_preferences" in config:
            role_config["seller_preferences"] = deepcopy(config["seller_preferences"])
        return role_config

    def _build_seller_contract_configs(self) -> Dict[int, Dict[str, Any]]:
        return {
            buyer_id: self._build_role_contract_config("seller", buyer_id)
            for buyer_id in (1, 2)
            if buyer_id in self.contract_configs
        }

    def _build_role_environment_info(self, role: str, buyer_id: Optional[int] = None) -> Dict[str, Any]:
        """Build role-specific environment info without leaking counterparty preferences."""
        role_env_info = deepcopy(self.environment_info)
        role_env_info.pop("contract_config", None)
        role_env_info.pop("buyer_contract_configs", None)
        if not self.use_contract_mode:
            return role_env_info

        if role == "buyer" and buyer_id is not None:
            role_env_info["contract_config"] = self._build_role_contract_config("buyer", buyer_id)
        elif role == "seller":
            role_env_info["buyer_contract_configs"] = self._build_seller_contract_configs()
            if buyer_id is not None:
                role_env_info["contract_config"] = self._build_role_contract_config("seller", buyer_id)
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

    def _validate_contract(self, contract: Optional[Dict[str, Any]], buyer_id: int) -> bool:
        """Validate contract against the selected buyer's bounds/options."""
        if not contract:
            return False
        if not self.use_contract_mode:
            return True

        config = self._get_buyer_contract_config(buyer_id)
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

    def _calculate_contract_utilities(self, contract: Dict[str, Any], buyer_id: int) -> Tuple[float, float]:
        """Compute raw MAUT utilities (U_b, U_s) for one buyer's contract config."""
        config = self._get_buyer_contract_config(buyer_id)
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
        """Compute theoretical max surplus Z_max for one buyer."""
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

    def _resolve_agreed_contract(self, buyer_id: int) -> Optional[Dict[str, Any]]:
        """Build the final contract for a compatible buyer/seller pair."""
        state = self._get_buyer_state(buyer_id)
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

    def _get_market_best_contract_buyer(self) -> Optional[int]:
        """Choose the buyer with the highest theoretical total utility."""
        candidates = [
            (buyer_id, z_max)
            for buyer_id, z_max in self.z_max_by_buyer.items()
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
            user_requirement: User requirement description
            product_info: Product information
            user_profile: User profile
            **kwargs: Other parameters
            
        Returns:
            (observation, info) Initial observation and info
        """
        # Reset state
        self.memory_buyer1.clear()
        self.memory_buyer2.clear()
        self.state_buyer1 = NegotiationState()
        self.state_buyer2 = NegotiationState()
        self.current_round = 0
        self.negotiation_info = NegotiationInfo()
        self.current_selected_buyer = None
        self.final_selected_buyer = None
        self.final_deal_price = None
        self._reset_contract_metadata()
        
        # Extract product_images for VLM (single product: image_url in product_info)
        product_info = product_info or {}
        product_images = None
        img_url = product_info.get("image_path") or product_info.get("image_url")
        if img_url:
            product_images = [img_url]
        self.product_images = product_images
        
        # Initialize Buyer1 Agent (include product_images for VLM img input)
        buyer1_context = {
            "user_requirement": user_requirement,
            "max_price": self.buyer1_max_price,
            "user_profile": user_profile,
            "environment_info": self._build_role_environment_info("buyer", 1),
            "product_info": product_info,
            "product_images": product_images,  # For VLM: product images (URL/path)
            "buyer_id": 1,  # Identify as buyer 1
        }
        self.buyer1_agent.initialize(buyer1_context)
        
        # Initialize Buyer2 Agent
        buyer2_context = {
            "user_requirement": user_requirement,
            "max_price": self.buyer2_max_price,
            "user_profile": user_profile,
            "environment_info": self._build_role_environment_info("buyer", 2),
            "product_info": product_info,
            "product_images": product_images,  # For VLM: product images (URL/path)
            "buyer_id": 2,  # Identify as buyer 2
        }
        self.buyer2_agent.initialize(buyer2_context)
        
        # Initialize Seller Agent (seller knows about both buyers)
        seller_context = {
            "product_info": product_info,
            "product_images": product_images,  # For VLM: product images (URL/path)
            "initial_price": self.initial_seller_price,
            "min_price": self.seller_min_price,
            "environment_info": self._build_role_environment_info("seller"),
            "num_buyers": 2,  # Inform seller there are 2 buyers
            "negotiation_mode": "sequential",  # Inform seller this is sequential negotiation
        }
        self.seller_agent.initialize(seller_context)
        
        # No initial seller offer - negotiation starts with buyers' first messages
        # Build observation
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(
        self, 
        selected_buyer: int,  # 1 or 2, which buyer seller chooses to negotiate with this round
        buyer_action: Optional[str] = None,
        seller_action: Optional[str] = None,
    ) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """Execute one negotiation step
        
        Each round, seller chooses one buyer to negotiate with, then buyer and seller exchange messages.
        Order: buyer -> seller (seller can see buyer's message before responding)
        
        Args:
            selected_buyer: Which buyer (1 or 2) seller chooses to negotiate with this round
            buyer_action: Selected buyer's response (optional)
            seller_action: Seller's response (optional, can see buyer's message)
            
        Returns:
            (observation, reward, terminated, truncated, info)
        """
        if selected_buyer not in [1, 2]:
            raise ValueError(f"selected_buyer must be 1 or 2, got {selected_buyer}")
        
        self.current_selected_buyer = selected_buyer
        
        # Add messages to memory in order: buyer -> seller
        # Process buyer action first
        if buyer_action is not None:
            if selected_buyer == 1:
                self.memory_buyer1.add_message("buyer", buyer_action, self.current_round)
                buyer_contract = self._extract_contract(buyer_action) if self.use_contract_mode else None
                if buyer_contract and self._validate_contract(buyer_contract, 1):
                    self.state_buyer1.metadata["buyer_contract"] = buyer_contract
                    buyer_price = float(buyer_contract["price"])
                else:
                    buyer_price = self._extract_price(buyer_action)
                if buyer_price is not None:
                    self.state_buyer1.update(buyer_price=buyer_price)
            else:  # selected_buyer == 2
                self.memory_buyer2.add_message("buyer", buyer_action, self.current_round)
                buyer_contract = self._extract_contract(buyer_action) if self.use_contract_mode else None
                if buyer_contract and self._validate_contract(buyer_contract, 2):
                    self.state_buyer2.metadata["buyer_contract"] = buyer_contract
                    buyer_price = float(buyer_contract["price"])
                else:
                    buyer_price = self._extract_price(buyer_action)
                if buyer_price is not None:
                    self.state_buyer2.update(buyer_price=buyer_price)
        
        # Process seller action after buyer (seller can see buyer's message)
        if seller_action is not None:
            if selected_buyer == 1:
                self.memory_buyer1.add_message("seller", seller_action, self.current_round)
                seller_contract = self._extract_contract(seller_action) if self.use_contract_mode else None
                if seller_contract and self._validate_contract(seller_contract, 1):
                    self.state_buyer1.metadata["seller_contract"] = seller_contract
                    seller_price = float(seller_contract["price"])
                else:
                    seller_price = self._extract_price(seller_action)
                if seller_price is not None:
                    self.state_buyer1.update(seller_price=seller_price)
            else:  # selected_buyer == 2
                self.memory_buyer2.add_message("seller", seller_action, self.current_round)
                seller_contract = self._extract_contract(seller_action) if self.use_contract_mode else None
                if seller_contract and self._validate_contract(seller_contract, 2):
                    self.state_buyer2.metadata["seller_contract"] = seller_contract
                    seller_price = float(seller_contract["price"])
                else:
                    seller_price = self._extract_price(seller_action)
                if seller_price is not None:
                    self.state_buyer2.update(seller_price=seller_price)
        
        # Check if deal can be made with the selected buyer
        # Deal is made when: (1) price difference <= tolerance, or (2) seller's offer <= buyer's offer
        if selected_buyer == 1:
            if self.use_contract_mode:
                agreed_contract = self._resolve_agreed_contract(1)
                if agreed_contract:
                    self.final_selected_buyer = 1
                    self.final_deal_price = float(agreed_contract["price"])
                    self.state_buyer1.metadata["agreed_contract"] = agreed_contract
            elif (buyer_action is not None and 
                self.state_buyer1.buyer_price is not None and 
                self.state_buyer1.seller_price is not None):
                price_diff = abs(self.state_buyer1.buyer_price - self.state_buyer1.seller_price)
                if price_diff <= self.price_tolerance:
                    self.final_selected_buyer = 1
                    self.final_deal_price = (self.state_buyer1.buyer_price + self.state_buyer1.seller_price) / 2
                elif self.state_buyer1.seller_price <= self.state_buyer1.buyer_price:
                    self.final_selected_buyer = 1
                    self.final_deal_price = self.state_buyer1.seller_price
        else:  # selected_buyer == 2
            if self.use_contract_mode:
                agreed_contract = self._resolve_agreed_contract(2)
                if agreed_contract:
                    self.final_selected_buyer = 2
                    self.final_deal_price = float(agreed_contract["price"])
                    self.state_buyer2.metadata["agreed_contract"] = agreed_contract
            elif (buyer_action is not None and 
                self.state_buyer2.buyer_price is not None and 
                self.state_buyer2.seller_price is not None):
                price_diff = abs(self.state_buyer2.buyer_price - self.state_buyer2.seller_price)
                if price_diff <= self.price_tolerance:
                    self.final_selected_buyer = 2
                    self.final_deal_price = (self.state_buyer2.buyer_price + self.state_buyer2.seller_price) / 2
                elif self.state_buyer2.seller_price <= self.state_buyer2.buyer_price:
                    self.final_selected_buyer = 2
                    self.final_deal_price = self.state_buyer2.seller_price
        
        # Check if deal is made
        terminated = False
        truncated = False
        reward = 0.0
        buyer1_reward = 0.0
        buyer2_reward = 0.0
        seller_reward = 0.0
        
        if self.final_selected_buyer is not None and self.final_deal_price is not None:
            terminated = True
            self.negotiation_info.status = NegotiationStatus.AGREED
            # Increment current_round to reflect that this round is completed
            # This ensures round count is accurate when calculating final scores
            self.current_round += 1
            self.negotiation_info.round_count = self.current_round
            reward = self._calculate_reward()
            buyer1_reward = self._calculate_buyer_reward(1)
            buyer2_reward = self._calculate_buyer_reward(2)
            seller_reward = self._calculate_seller_reward()
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
            seller_reward = self._calculate_seller_reward()
        else:
            # Move to next round
            self.current_round += 1
            self.negotiation_info.round_count = self.current_round
        
        # Calculate step rewards for every round
        # Only calculate for the selected buyer in this round (sequential negotiation)
        step_buyer1_reward = self._calculate_step_buyer_reward(1) if selected_buyer == 1 else None
        step_buyer2_reward = self._calculate_step_buyer_reward(2) if selected_buyer == 2 else None
        step_seller_reward = self._calculate_step_seller_reward()
        
        # Build observation and info
        observation = self._get_observation()
        info = self._get_info()
        
        # Add step rewards to info for every step
        if step_buyer1_reward is not None:
            info["step_buyer1_reward"] = step_buyer1_reward
        if step_buyer2_reward is not None:
            info["step_buyer2_reward"] = step_buyer2_reward
        info["step_seller_reward"] = step_seller_reward
        
        if terminated or truncated:
            info["termination_reason"] = "agreed" if terminated else "timeout"
            if terminated:
                info["selected_buyer"] = self.final_selected_buyer
                info["final_deal_price"] = self.final_deal_price
            info["buyer1_reward"] = buyer1_reward
            info["buyer2_reward"] = buyer2_reward
            info["seller_reward"] = seller_reward
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
        history_buyer1 = self.memory_buyer1.get_history()
        history_buyer2 = self.memory_buyer2.get_history()
        
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
        
        # Display which buyer was selected this round
        if self.current_selected_buyer is not None:
            output_lines.append(f"\n[Selected Buyer: Buyer {self.current_selected_buyer}]")
        
        # Display Buyer1 conversation (if this round negotiated with buyer1)
        if self.current_selected_buyer == 1 and history_buyer1:
            round_messages_b1 = [
                msg for msg in history_buyer1 if msg["round"] == round_to_display
            ]
            if round_messages_b1:
                output_lines.append(f"\n[BUYER 1 Conversation]:")
                # Display buyer message first (if exists)
                buyer1_msg = next(
                    (msg for msg in round_messages_b1 if msg["role"] == "buyer"), 
                    None
                )
                if buyer1_msg:
                    output_lines.append(f"  [BUYER]: {buyer1_msg['content']}")
                
                # Display seller message (if exists)
                seller1_msg = next(
                    (msg for msg in round_messages_b1 if msg["role"] == "seller"), 
                    None
                )
                if seller1_msg:
                    output_lines.append(f"  [SELLER]: {seller1_msg['content']}")
        
        # Display Buyer2 conversation (if this round negotiated with buyer2)
        if self.current_selected_buyer == 2 and history_buyer2:
            round_messages_b2 = [
                msg for msg in history_buyer2 if msg["round"] == round_to_display
            ]
            if round_messages_b2:
                output_lines.append(f"\n[BUYER 2 Conversation]:")
                # Display buyer message first (if exists)
                buyer2_msg = next(
                    (msg for msg in round_messages_b2 if msg["role"] == "buyer"), 
                    None
                )
                if buyer2_msg:
                    output_lines.append(f"  [BUYER]: {buyer2_msg['content']}")
                
                # Display seller message (if exists)
                seller2_msg = next(
                    (msg for msg in round_messages_b2 if msg["role"] == "seller"), 
                    None
                )
                if seller2_msg:
                    output_lines.append(f"  [SELLER]: {seller2_msg['content']}")
        
        # Round summary section
        output_lines.append(f"\n{'-'*60}")
        output_lines.append(f"Round {self.current_round} Summary:")
        output_lines.append(f"{'-'*60}")
        
        # Display Buyer1 prices
        output_lines.append(f"\nBuyer 1:")
        if self.state_buyer1.buyer_price is not None:
            output_lines.append(f"  Buyer Price: ${self.state_buyer1.buyer_price:.2f}")
        else:
            output_lines.append(f"  Buyer Price: Not specified")
        if self.state_buyer1.seller_price is not None:
            output_lines.append(f"  Seller Price: ${self.state_buyer1.seller_price:.2f}")
        else:
            output_lines.append(f"  Seller Price: Not specified")
        
        # Display Buyer2 prices
        output_lines.append(f"\nBuyer 2:")
        if self.state_buyer2.buyer_price is not None:
            output_lines.append(f"  Buyer Price: ${self.state_buyer2.buyer_price:.2f}")
        else:
            output_lines.append(f"  Buyer Price: Not specified")
        if self.state_buyer2.seller_price is not None:
            output_lines.append(f"  Seller Price: ${self.state_buyer2.seller_price:.2f}")
        else:
            output_lines.append(f"  Seller Price: Not specified")
        
        # Display deal status
        if self.final_selected_buyer is not None:
            output_lines.append(f"\n  ✓ DEAL MADE with Buyer {self.final_selected_buyer}")
            if self.final_deal_price is not None:
                output_lines.append(f"  Final Deal Price: ${self.final_deal_price:.2f}")
            if self.use_contract_mode:
                agreed_contract = self._get_buyer_state(self.final_selected_buyer).metadata.get("agreed_contract")
                if agreed_contract is not None:
                    output_lines.append(f"  Final Contract: {agreed_contract}")
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
        self.memory_buyer1.clear()
        self.memory_buyer2.clear()
        self.state_buyer1 = NegotiationState()
        self.state_buyer2 = NegotiationState()
        self._reset_contract_metadata()
    
    def _get_observation(self) -> Dict[str, Any]:
        """Get current observation"""
        obs = {
            "conversation_history_buyer1": self.memory_buyer1.get_history(),
            "conversation_history_buyer2": self.memory_buyer2.get_history(),
            "current_round": self.current_round,
            "current_selected_buyer": self.current_selected_buyer,
            "buyer1_price": self.state_buyer1.buyer_price,
            "seller_price_buyer1": self.state_buyer1.seller_price,
            "buyer2_price": self.state_buyer2.buyer_price,
            "seller_price_buyer2": self.state_buyer2.seller_price,
            "status": self.negotiation_info.status.value,
            "final_selected_buyer": self.final_selected_buyer,
            "final_deal_price": self.final_deal_price,
        }
        # Include product_images for VLM (agent passes img to model when is_vlm)
        if self.product_images is not None:
            obs["product_images"] = self.product_images
        if self.use_contract_mode:
            obs["contract_configs"] = self._build_seller_contract_configs()
            if self.current_selected_buyer in (1, 2):
                obs["contract_config"] = self._build_role_contract_config("seller", self.current_selected_buyer)
        return obs
    
    def _get_info(self) -> Dict[str, Any]:
        """Get current info"""
        info = {
            "round": self.current_round,
            "status": self.negotiation_info.status.value,
            "current_selected_buyer": self.current_selected_buyer,
            "buyer1_price": self.state_buyer1.buyer_price,
            "seller_price_buyer1": self.state_buyer1.seller_price,
            "buyer2_price": self.state_buyer2.buyer_price,
            "seller_price_buyer2": self.state_buyer2.seller_price,
            "final_selected_buyer": self.final_selected_buyer,
            "final_deal_price": self.final_deal_price,
            "negotiation_info": self.negotiation_info,
        }
        if self.use_contract_mode:
            agreed_contract = None
            if self.final_selected_buyer in (1, 2):
                agreed_contract = self._get_buyer_state(self.final_selected_buyer).metadata.get("agreed_contract")
            info.update({
                "contract_configs": self.contract_configs,
                "agreed_contract": agreed_contract,
                "buyer1_contract": self.state_buyer1.metadata.get("buyer_contract"),
                "seller_contract_buyer1": self.state_buyer1.metadata.get("seller_contract"),
                "buyer2_contract": self.state_buyer2.metadata.get("buyer_contract"),
                "seller_contract_buyer2": self.state_buyer2.metadata.get("seller_contract"),
            })
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
        If deal is reached with a buyer, use that buyer's max_price for calculation.
        
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
        
        if self.negotiation_info.status == NegotiationStatus.AGREED and self.final_selected_buyer is not None and self.final_deal_price is not None:
            # Deal reached: buyer savings + seller profit + time cost
            deal_price = self.final_deal_price
            reward = 0.0
            buyer_savings = 0.0
            seller_profit = 0.0
            
            # Get the selected buyer's max_price
            selected_buyer_max_price = None
            if self.final_selected_buyer == 1:
                selected_buyer_max_price = self.buyer1_max_price
            elif self.final_selected_buyer == 2:
                selected_buyer_max_price = self.buyer2_max_price
            
            # Calculate buyer savings: buyer_max_price - deal_price
            if selected_buyer_max_price is not None:
                buyer_savings = selected_buyer_max_price - deal_price
                reward += buyer_savings * self.reward_weights["buyer_savings"]
            
            # Calculate seller profit: deal_price - seller_min_price
            if self.seller_min_price is not None:
                seller_profit = deal_price - self.seller_min_price
                reward += seller_profit * self.reward_weights["seller_profit"]
            
            # Add time cost (negative penalty)
            reward += time_cost * self.reward_weights["time_cost"]
            
            weighted_buyer_savings = buyer_savings * self.reward_weights["buyer_savings"] if selected_buyer_max_price is not None else 0.0
            weighted_seller_profit = seller_profit * self.reward_weights["seller_profit"] if self.seller_min_price is not None else 0.0
            weighted_time_cost = time_cost * self.reward_weights["time_cost"]
            print(f"Global Reward = buyer{self.final_selected_buyer}_savings({buyer_savings:.2f} * {self.reward_weights['buyer_savings']:.2f}) + seller_profit({seller_profit:.2f} * {self.reward_weights['seller_profit']:.2f}) + time_cost({time_cost:.2f} * {self.reward_weights['time_cost']:.2f}) = {reward:.2f} (buyer{self.final_selected_buyer}_max={selected_buyer_max_price}, deal_price={deal_price:.2f}, seller_min={self.seller_min_price}, round={self.current_round})")
            
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
            - buyer savings = buyer_max_price - deal_price (money saved by buyer)
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
            
            # Calculate buyer savings: buyer_max_price - deal_price
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
    
    def _calculate_seller_reward(self) -> float:
        """Calculate reward from seller's perspective
        
        Calculate reward value based on negotiation result from seller's perspective.
        This reward does not include buyer savings.
        
        If deal is reached:
            reward = seller profit + time cost (negative, based on rounds)
            - seller profit = deal_price - seller_min_price (extra profit for seller)
            - time cost = -current_round (penalty for number of rounds taken)
        
        If deal is not reached:
            reward = time cost (negative, based on rounds)
            - time cost = -current_round (penalty for number of rounds taken)
        
        Returns:
            Reward value from seller's perspective
        """
        # Time cost: negative value based on number of rounds
        time_cost = -self.current_round
        
        if self.negotiation_info.status == NegotiationStatus.AGREED and self.final_selected_buyer is not None and self.final_deal_price is not None:
            # Deal reached: seller profit + time cost
            deal_price = self.final_deal_price
            reward = 0.0
            seller_profit = 0.0
            
            # Calculate seller profit: deal_price - seller_min_price
            if self.seller_min_price is not None:
                seller_profit = deal_price - self.seller_min_price
                reward += seller_profit * self.reward_weights["seller_profit"]
            
            # Add time cost (negative penalty)
            reward += time_cost * self.reward_weights["time_cost"]
            
            weighted_seller_profit = seller_profit * self.reward_weights["seller_profit"] if self.seller_min_price is not None else 0.0
            weighted_time_cost = time_cost * self.reward_weights["time_cost"]
            print(f"Seller Reward = seller_profit({seller_profit:.2f} * {self.reward_weights['seller_profit']:.2f}) + time_cost({time_cost:.2f} * {self.reward_weights['time_cost']:.2f}) = {reward:.2f} (deal_price={deal_price:.2f}, seller_min={self.seller_min_price}, round={self.current_round})")
            
            return reward
        
        else:
            # Deal not reached: only time cost (negative penalty)
            weighted_time_cost = time_cost * self.reward_weights["time_cost"]
            print(f"Seller Reward = time_cost({time_cost:.2f} * {self.reward_weights['time_cost']:.2f}) = {weighted_time_cost:.2f} (round={self.current_round}, deal not reached)")
            return weighted_time_cost
    
    def _calculate_step_buyer_reward(self, buyer_id: int) -> float:
        """Calculate step reward from buyer's perspective for current round
        
        Calculate reward value based on buyer's current offer in this round.
        This is calculated every round, not just at the end.
        
        reward = buyer savings (from current offer) + round cost
        - buyer savings = buyer_max_price - buyer_price (money saved by current offer)
        - round cost = -current_round (penalty for number of rounds taken)
        
        Args:
            buyer_id: Buyer ID (1 or 2)
        
        Returns:
            Step reward value from buyer's perspective for current round
        """
        # Round cost: negative value based on number of rounds
        round_cost = -self.current_round
        
        # Calculate buyer reward
        reward = 0.0
        buyer_savings = 0.0
        
        # Get buyer state
        buyer_state = None
        buyer_max_price = None
        if buyer_id == 1:
            buyer_state = self.state_buyer1
            buyer_max_price = self.buyer1_max_price
        elif buyer_id == 2:
            buyer_state = self.state_buyer2
            buyer_max_price = self.buyer2_max_price
        
        # Calculate buyer savings from current offer: buyer_max_price - buyer_price
        if buyer_state is not None and buyer_state.buyer_price is not None and buyer_max_price is not None:
            buyer_savings = buyer_max_price - buyer_state.buyer_price
            reward += buyer_savings * self.reward_weights["buyer_savings"]
        
        # Add round cost (negative penalty)
        reward += round_cost * self.reward_weights["time_cost"]
        
        return reward
    
    def _calculate_step_seller_reward(self) -> float:
        """Calculate step reward from seller's perspective for current round
        
        Calculate reward value based on seller's current offer in this round.
        This is calculated every round, not just at the end.
        
        reward = seller profit (from current offer) + round cost
        - seller profit = seller_price - seller_min_price (profit from current offer)
        - round cost = -current_round (penalty for number of rounds taken)
        
        If seller_price is not specified yet, only round cost is returned.
        
        Returns:
            Step reward value from seller's perspective for current round
        """
        # Round cost: negative value based on number of rounds
        round_cost = -self.current_round
        reward = 0.0
        seller_profit = 0.0
        
        # Get seller price from the selected buyer
        seller_price = None
        if self.current_selected_buyer == 1:
            seller_price = self.state_buyer1.seller_price
        elif self.current_selected_buyer == 2:
            seller_price = self.state_buyer2.seller_price
        
        # Calculate seller profit from current offer: seller_price - seller_min_price
        if seller_price is not None and self.seller_min_price is not None:
            seller_profit = seller_price - self.seller_min_price
            reward += seller_profit * self.reward_weights["seller_profit"]
        
        # Add round cost (negative penalty)
        reward += round_cost * self.reward_weights["time_cost"]
        
        return reward
    
    def _get_selected_buyer_max_price(self) -> Optional[float]:
        """Get the final selected buyer's max_price
        
        Returns:
            Final selected buyer's max_price, or None if no buyer is selected
        """
        if self.final_selected_buyer == 1:
            return self.buyer1_max_price
        elif self.final_selected_buyer == 2:
            return self.buyer2_max_price
        return None

    def _get_market_best_ceiling(self) -> Optional[float]:
        """Highest max price across buyers — dual of ``market_best_floor`` in multi-seller envs."""
        caps = [p for p in (self.buyer1_max_price, self.buyer2_max_price) if p is not None]
        if not caps:
            return None
        return max(caps)

    def _get_contract_score_terms(self) -> Optional[Dict[str, float]]:
        """Return normalized utility terms using the market-best buyer utility space."""
        if not self.use_contract_mode or self.final_selected_buyer not in (1, 2):
            return None

        market_best_buyer = self._get_market_best_contract_buyer()
        if market_best_buyer is None:
            return None
        z_market = self.z_max_by_buyer.get(market_best_buyer)
        if not z_market or z_market <= 0:
            return None

        selected_state = self._get_buyer_state(self.final_selected_buyer)
        final_contract = selected_state.metadata.get("agreed_contract")
        if not final_contract:
            final_contract = self._resolve_agreed_contract(self.final_selected_buyer)
            if final_contract:
                selected_state.metadata["agreed_contract"] = final_contract

        if not final_contract or not self._validate_contract(final_contract, self.final_selected_buyer):
            return None

        u_b, u_s = self._calculate_contract_utilities(final_contract, self.final_selected_buyer)
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
            "market_best_buyer": float(market_best_buyer),
            "selected_buyer": float(self.final_selected_buyer),
            "selected_z_max": self.z_max_by_buyer.get(self.final_selected_buyer) or 0.0,
        }
    
    def _calculate_global_score(self, print_details: bool = True) -> float:
        """Calculate GlobalScore based on the optimized formula
        
        Uses **market_best_ceiling** = max(buyer1_max, buyer2_max) for the surplus span Z_market
        (aligned with multi-seller's use of min seller floors for ``Z_market``). The final selected
        buyer's max_price still bounds whether the deal is valid; buyer surplus u_b uses that max.
        
        If no buyer is selected or required prices are missing, calculates failure penalty.
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
                    print(f"  market_best_buyer = Buyer {int(score_terms['market_best_buyer'])}")
                    print(f"  z_market = {score_terms['z_market']:.4f}, selected_buyer_z_max = {score_terms['selected_z_max']:.4f}")
                    print(f"  final_selected_buyer = Buyer {int(score_terms['selected_buyer'])}")
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

        selected_buyer_max_price = self._get_selected_buyer_max_price()
        market_best_ceiling = self._get_market_best_ceiling()
        
        if (
            selected_buyer_max_price is None
            or self.seller_min_price is None
            or market_best_ceiling is None
        ):
            round_index = max(0, self.current_round)
            discount = self.gamma ** round_index
            failure_penalty = -self.failure_penalty_weight * (1.0 - discount)
            
            if print_details:
                print(f"\n[GlobalScore Calculation]")
                print(f"  selected_buyer_max_price, market_best_ceiling, or seller_min_price is None")
                print(f"  round_index = {round_index}, gamma = {self.gamma}, discount = γ^{round_index} = {discount:.6f}")
                print(f"  FailurePenalty = -F({self.failure_penalty_weight:.1f}) * (1 - discount({discount:.6f})) = {failure_penalty:.3f}")
                print(f"  GlobalScore = {failure_penalty:.3f}")
            return failure_penalty
        
        Z_market = market_best_ceiling - self.seller_min_price
        
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
                    f"  market_best_ceiling = max(buyer1_max({self.buyer1_max_price}), "
                    f"buyer2_max({self.buyer2_max_price})) = {market_best_ceiling:.2f}"
                )
                print(f"  Z_market = {market_best_ceiling:.2f} - seller_min({self.seller_min_price:.2f}) = {Z_market:.2f}")
                print(f"  No final price available")
                print(f"  feasible_deal = {feasible_deal}")
                print(f"  round_index = {round_index}, gamma = {self.gamma}, discount = γ^{round_index} = {discount:.6f}")
                print(f"  FailurePenalty = -F({self.failure_penalty_weight:.1f}) * (1 - discount({discount:.6f})) = {failure_penalty:.3f}")
                print(f"  GlobalScore = {failure_penalty:.3f}")
            return failure_penalty
        
        valid_range = (Z_market > 0) and (self.seller_min_price <= final_price <= selected_buyer_max_price)
        
        if feasible_deal and valid_range:
            u_b = (selected_buyer_max_price - final_price) / Z_market
            u_s = (final_price - self.seller_min_price) / Z_market
            Q = 4.0 * u_b * u_s
            
            deal_score = self.deal_score_weight * discount
            quality_score = self.quality_score_weight * Q * discount
            efficiency_score = self.efficiency_score_weight * discount
            global_score = deal_score + quality_score + efficiency_score
            
            if print_details:
                print(f"\n[GlobalScore Calculation]")
                print(
                    f"  market_best_ceiling = max(buyer1_max({self.buyer1_max_price}), "
                    f"buyer2_max({self.buyer2_max_price})) = {market_best_ceiling:.2f}"
                )
                print(f"  Z_market = {market_best_ceiling:.2f} - seller_min({self.seller_min_price:.2f}) = {Z_market:.2f}")
                print(f"  final_price = {final_price:.2f}")
                print(f"  feasible_deal = {feasible_deal} (negotiation status: {self.negotiation_info.status.value})")
                sr = (
                    f"(Z_market > 0) and (seller_min <= final_price({final_price:.2f}) "
                    f"<= selected_buyer_max({selected_buyer_max_price:.2f}))"
                )
                print(f"  valid_range = {sr} = {valid_range}")
                print(f"  round_index = {round_index}, gamma = {self.gamma}, discount = γ^{round_index} = {discount:.6f}")
                print(
                    f"  u_b = (selected_buyer_max({selected_buyer_max_price:.2f}) - final_price) / Z_market({Z_market:.2f}) = {u_b:.4f}"
                )
                print(
                    f"  u_s = (final_price - seller_min({self.seller_min_price:.2f})) / Z_market({Z_market:.2f}) = {u_s:.4f}"
                )
                print(f"  Q = 4 * u_b({u_b:.4f}) * u_s({u_s:.4f}) = {Q:.4f}")
                print(f"  DealScore = D({self.deal_score_weight:.1f}) * discount({discount:.6f}) = {deal_score:.3f}")
                print(f"  QualityScore = W({self.quality_score_weight:.1f}) * Q({Q:.4f}) * discount({discount:.6f}) = {quality_score:.3f}")
                print(f"  EfficiencyScore = E({self.efficiency_score_weight:.1f}) * discount({discount:.6f}) = {efficiency_score:.3f}")
                print(f"  GlobalScore = DealScore({deal_score:.3f}) + QualityScore({quality_score:.3f}) + EfficiencyScore({efficiency_score:.3f}) = {global_score:.3f}")
            
            return global_score
        else:
            failure_penalty = -self.failure_penalty_weight * (1.0 - discount)
            
            if print_details:
                print(f"\n[GlobalScore Calculation]")
                print(
                    f"  market_best_ceiling = max(buyer1_max({self.buyer1_max_price}), "
                    f"buyer2_max({self.buyer2_max_price})) = {market_best_ceiling:.2f}"
                )
                print(f"  Z_market = {Z_market:.2f}, final_price = {final_price:.2f}")
                print(f"  feasible_deal = {feasible_deal} (negotiation status: {self.negotiation_info.status.value})")
                print(f"  valid_range = {valid_range}")
                print(f"  round_index = {round_index}, gamma = {self.gamma}, discount = γ^{round_index} = {discount:.6f}")
                print(f"  FailurePenalty = -F({self.failure_penalty_weight:.1f}) * (1 - discount({discount:.6f})) = {failure_penalty:.3f}")
                print(f"  GlobalScore = {failure_penalty:.3f}")
            
            return failure_penalty
    
    def _calculate_buyer_score(self, print_details: bool = True) -> float:
        """BuyerScore: u_b uses selected buyer's max; denominator uses Z_market (market_best_ceiling - seller_min)."""
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
                    print(f"  market_best_buyer = Buyer {int(score_terms['market_best_buyer'])}")
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

        selected_buyer_max_price = self._get_selected_buyer_max_price()
        market_best_ceiling = self._get_market_best_ceiling()
        
        if (
            selected_buyer_max_price is None
            or self.seller_min_price is None
            or market_best_ceiling is None
        ):
            round_index = max(0, self.current_round)
            discount = self.gamma ** round_index
            buyer_score = -self.buyer_failure_penalty_weight * (1.0 - discount)
            
            if print_details:
                print(f"\n[BuyerScore Calculation]")
                print(f"  selected_buyer_max_price, market_best_ceiling, or seller_min_price is None")
                print(f"  round_index = {round_index}, gamma = {self.gamma}, discount = γ^{round_index} = {discount:.6f}")
                print(f"  BuyerScore = -Fb({self.buyer_failure_penalty_weight:.1f}) * (1 - discount({discount:.6f})) = {buyer_score:.3f}")
            return buyer_score
        
        Z_market = market_best_ceiling - self.seller_min_price
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
                    f"  market_best_ceiling = {market_best_ceiling:.2f}, "
                    f"Z_market = {Z_market:.2f}"
                )
                print(f"  No final price available")
                print(f"  round_index = {round_index}, gamma = {self.gamma}, discount = γ^{round_index} = {discount:.6f}")
                print(f"  BuyerScore = -Fb({self.buyer_failure_penalty_weight:.1f}) * (1 - discount({discount:.6f})) = {buyer_score:.3f}")
            return buyer_score
        
        valid_range = (Z_market > 0) and (self.seller_min_price <= final_price <= selected_buyer_max_price)
        
        if feasible_deal and valid_range:
            u_b = (selected_buyer_max_price - final_price) / Z_market
            buyer_score = discount * (self.buyer_deal_weight + self.buyer_utility_weight * u_b + self.buyer_efficiency_weight)
            
            if print_details:
                print(f"\n[BuyerScore Calculation]")
                print(
                    f"  market_best_ceiling = max(buyer1_max, buyer2_max) = {market_best_ceiling:.2f}, "
                    f"Z_market = {Z_market:.2f}"
                )
                print(f"  final_price = {final_price:.2f}")
                print(f"  feasible_deal = {feasible_deal} (negotiation status: {self.negotiation_info.status.value})")
                print(f"  valid_range = (Z_market > 0) and (seller_min <= p <= selected_buyer_max) = {valid_range}")
                print(f"  round_index = {round_index}, gamma = {self.gamma}, discount = γ^{round_index} = {discount:.6f}")
                print(
                    f"  u_b = (selected_buyer_max({selected_buyer_max_price:.2f}) - final_price) / Z_market({Z_market:.2f}) = {u_b:.4f}"
                )
                print(f"  BuyerScore = {discount:.6f} * ({self.buyer_deal_weight:.1f} + {self.buyer_utility_weight * u_b:.4f} + {self.buyer_efficiency_weight:.1f}) = {buyer_score:.3f}")
            
            return buyer_score
        else:
            buyer_score = -self.buyer_failure_penalty_weight * (1.0 - discount)
            
            if print_details:
                print(f"\n[BuyerScore Calculation]")
                print(f"  Z_market = {Z_market:.2f}, final_price = {final_price:.2f}")
                print(f"  feasible_deal = {feasible_deal} (negotiation status: {self.negotiation_info.status.value})")
                print(f"  valid_range = {valid_range}")
                print(f"  round_index = {round_index}, gamma = {self.gamma}, discount = γ^{round_index} = {discount:.6f}")
                print(f"  BuyerScore = -Fb({self.buyer_failure_penalty_weight:.1f}) * (1 - discount({discount:.6f})) = {buyer_score:.3f}")
            
            return buyer_score
    
    def _calculate_seller_score(self, print_details: bool = True) -> float:
        """SellerScore: u_s = (p - seller_min) / Z_market with Z_market = market_best_ceiling - seller_min."""
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
                    print(f"  market_best_buyer = Buyer {int(score_terms['market_best_buyer'])}")
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

        selected_buyer_max_price = self._get_selected_buyer_max_price()
        market_best_ceiling = self._get_market_best_ceiling()
        
        if (
            selected_buyer_max_price is None
            or self.seller_min_price is None
            or market_best_ceiling is None
        ):
            round_index = max(0, self.current_round)
            discount = self.gamma ** round_index
            seller_score = -self.seller_failure_penalty_weight * (1.0 - discount)
            
            if print_details:
                print(f"\n[SellerScore Calculation]")
                print(f"  selected_buyer_max_price, market_best_ceiling, or seller_min_price is None")
                print(f"  round_index = {round_index}, gamma = {self.gamma}, discount = γ^{round_index} = {discount:.6f}")
                print(f"  SellerScore = -Fs({self.seller_failure_penalty_weight:.1f}) * (1 - discount({discount:.6f})) = {seller_score:.3f}")
            return seller_score
        
        Z_market = market_best_ceiling - self.seller_min_price
        round_index = max(0, self.current_round)
        discount = self.gamma ** round_index
        feasible_deal = (self.negotiation_info.status == NegotiationStatus.AGREED) or (self.final_deal_price is not None)
        
        if self.final_deal_price is not None:
            final_price = self.final_deal_price
        else:
            seller_score = -self.seller_failure_penalty_weight * (1.0 - discount)
            
            if print_details:
                print(f"\n[SellerScore Calculation]")
                print(f"  Z_market = {Z_market:.2f}, no final price")
                print(f"  round_index = {round_index}, gamma = {self.gamma}, discount = γ^{round_index} = {discount:.6f}")
                print(f"  SellerScore = -Fs({self.seller_failure_penalty_weight:.1f}) * (1 - discount({discount:.6f})) = {seller_score:.3f}")
            return seller_score
        
        valid_range = (Z_market > 0) and (self.seller_min_price <= final_price <= selected_buyer_max_price)
        
        if feasible_deal and valid_range:
            u_s = (final_price - self.seller_min_price) / Z_market
            seller_score = discount * (self.seller_deal_weight + self.seller_utility_weight * u_s + self.seller_efficiency_weight)
            
            if print_details:
                print(f"\n[SellerScore Calculation]")
                print(
                    f"  market_best_ceiling = {market_best_ceiling:.2f}, "
                    f"Z_market = {Z_market:.2f}"
                )
                print(f"  final_price = {final_price:.2f}")
                print(f"  feasible_deal = {feasible_deal} (negotiation status: {self.negotiation_info.status.value})")
                print(f"  valid_range = (Z_market > 0) and (seller_min <= p <= selected_buyer_max) = {valid_range}")
                print(f"  round_index = {round_index}, gamma = {self.gamma}, discount = γ^{round_index} = {discount:.6f}")
                print(
                    f"  u_s = (final_price - seller_min({self.seller_min_price:.2f})) / Z_market({Z_market:.2f}) = {u_s:.4f}"
                )
                print(f"  SellerScore = {discount:.6f} * ({self.seller_deal_weight:.1f} + {self.seller_utility_weight * u_s:.4f} + {self.seller_efficiency_weight:.1f}) = {seller_score:.3f}")
            
            return seller_score
        else:
            seller_score = -self.seller_failure_penalty_weight * (1.0 - discount)
            
            if print_details:
                print(f"\n[SellerScore Calculation]")
                print(f"  Z_market = {Z_market:.2f}, final_price = {final_price:.2f}")
                print(f"  feasible_deal = {feasible_deal} (negotiation status: {self.negotiation_info.status.value})")
                print(f"  valid_range = {valid_range}")
                print(f"  round_index = {round_index}, gamma = {self.gamma}, discount = γ^{round_index} = {discount:.6f}")
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

