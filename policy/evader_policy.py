from typing import Literal

import numpy as np
from numpy.typing import NDArray
from typing_extensions import override

from .base import Action, Policy


class EvaderPolicy(Policy):
    def __init__(
        self,
        policy_mode: Literal["scripted", "learned"],
        distance_weight: float = 2.0,
        velocity_weight: float = 1.0,
        hunter_priority: float = 10.0,
        boundary_priority: float = 1.0,
        hunter_threat_radius: float = 1.0,
        boundary_threshold: float = 1.0,
    ) -> None:
        self.policy_mode = policy_mode
        self.distance_weight = distance_weight
        self.velocity_weight = velocity_weight
        self.hunter_priority = hunter_priority
        self.boundary_priority = boundary_priority
        self.hunter_threat_radius = hunter_threat_radius
        self.boundary_threshold = boundary_threshold
        self.prev_pursuer_rel_positions: NDArray[np.float32] | None = None

    @override
    def sample(self, obs: NDArray[np.float32]) -> Action:
        """Choose an action using either the scripted or learned policy."""
        if self.policy_mode == "scripted":
            return self.scripted_policy(obs)
        if self.policy_mode == "learned":
            return self.learned_policy(obs)

        raise ValueError(
            "policy_mode must be either 'scripted' or 'learned'. "
            f"Received {self.policy_mode!r}."
        )

    def scripted_policy(self, obs: NDArray[np.float32]) -> Action:
        """Move away from pursuers while avoiding outward boundary actions."""
        # With 2 landmarks and 3 pursuers, the evader observation has 14 values.
        _, self_position, _, pursuer_rel_flat = np.split(obs, [2, 4, 8])
        pursuer_rel_positions = pursuer_rel_flat.reshape(3, 2)

        epsilon = 1e-6

        # Only hunters inside the threat radius participate in the calculation.
        all_pursuer_distances = np.linalg.norm(
            pursuer_rel_positions,
            axis=1,
            keepdims=True,
        )
        nearby_mask = all_pursuer_distances[:, 0] <= self.hunter_threat_radius

        nearby_rel_positions = pursuer_rel_positions[nearby_mask]
        nearby_distances = all_pursuer_distances[nearby_mask]

        distance_threat_direction = np.zeros(2, dtype=np.float32)
        velocity_threat_direction = np.zeros(2, dtype=np.float32)

        if len(nearby_rel_positions) > 0:
            directions_to_pursuers = nearby_rel_positions / (
                nearby_distances + epsilon
            )

            inverse_distances = 1.0 / (nearby_distances + epsilon)
            distance_weights = inverse_distances / (
                np.sum(inverse_distances) + epsilon
            )
            distance_threat_direction = np.sum(
                distance_weights * directions_to_pursuers,
                axis=0,
            )

            if self.prev_pursuer_rel_positions is not None:
                relative_motion = (
                    pursuer_rel_positions - self.prev_pursuer_rel_positions
                )
                nearby_relative_motion = relative_motion[nearby_mask]

                radial_motion = np.sum(
                    nearby_relative_motion * directions_to_pursuers,
                    axis=1,
                    keepdims=True,
                )
                closing_motion = np.maximum(0.0, -radial_motion)

                total_closing_motion = float(np.sum(closing_motion))
                if total_closing_motion > epsilon:
                    closing_weights = closing_motion / total_closing_motion
                    velocity_threat_direction = np.sum(
                        closing_weights * directions_to_pursuers,
                        axis=0,
                    )

        hunter_escape_direction = -(
            self.distance_weight * distance_threat_direction
            + self.velocity_weight * velocity_threat_direction
        )

        # Boundary recovery: inside the safe area this is [0, 0].
        # Outside, point toward the closest point on the safe area's boundary.
        nearest_safe_position = np.clip(
            self_position,
            -self.boundary_threshold,
            self.boundary_threshold,
        )
        boundary_direction = nearest_safe_position - self_position
        boundary_distance = np.linalg.norm(boundary_direction)
        if boundary_distance > epsilon:
            boundary_direction = boundary_direction / boundary_distance

        escape_direction = (
            self.hunter_priority * hunter_escape_direction
            + self.boundary_priority * boundary_direction
        )

        self.prev_pursuer_rel_positions = pursuer_rel_positions.copy()

        if np.linalg.norm(escape_direction) < epsilon:
            return Action.NO_OP

        action_scores = {
            Action.LEFT: -escape_direction[0],
            Action.RIGHT: escape_direction[0],
            Action.DOWN: -escape_direction[1],
            Action.UP: escape_direction[1],
        }

        return max(action_scores, key=action_scores.__getitem__)

    def learned_policy(self, obs: NDArray[np.float32]) -> Action:
        """Choose an action with a learned policy."""
        raise NotImplementedError
