"""Feature engineering for Wingspan bird card regression analysis.

Transforms raw bird card data into regression-ready features, including:
- Weighted food cost (adjusted for dice rarity under Oceania/Asia expansions)
- Nest and wingspan indicators
- Bonus card eligibility
- Power color indicators

Dice layout (Oceania/Asia expansion, 5 six-sided dice):
    Face 1: Invertebrate
    Face 2: Invertebrate + Seed (shared)
    Face 3: Seed + Nectar (shared)
    Face 4: Nectar + Fruit (shared)
    Face 5: Fish
    Face 6: Rodent

Probability of obtaining each food type on a single die:
    Invertebrate: 2/6 (faces 1, 2)
    Seed:         2/6 (faces 2, 3)
    Fish:         1/6 (face 5)
    Fruit:        1/6 (face 4)  -- shared with nectar
    Rodent:       1/6 (face 6)
    Nectar:       2/6 (faces 3, 4) -- but shared placements

Rarity weights are the inverse of these probabilities, normalized so that
the most common foods (invertebrate, seed) have weight 1.0. This means
rarer foods cost more "effective" food units.
"""

import re

import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Raw probability of each food type appearing as an option on a single die roll.
# Shared faces offer a choice between two foods — both are available as options,
# so each food on a shared face gets full credit for that face.
#   Invertebrate: face 1 (solo) + face 2 (shared w/ seed)       = 2/6
#   Seed:         face 2 (shared w/ invert) + face 3 (shared w/ nectar) = 2/6
#   Nectar:       face 3 (shared w/ seed) + face 4 (shared w/ fruit)    = 2/6
#   Fish:         face 5 (solo)                                  = 1/6
#   Fruit:        face 4 (shared w/ nectar)                      = 1/6
#   Rodent:       face 6 (solo)                                  = 1/6
FOOD_AVAILABILITY = {
    "Invertebrate": 2 / 6,
    "Seed": 2 / 6,
    "Fish": 1 / 6,
    "Fruit": 1 / 6,
    "Rodent": 1 / 6,
    "Nectar": 2 / 6,
}

# Rarity weight = inverse availability, normalized so max availability = 1.0
_max_avail = max(FOOD_AVAILABILITY.values())
FOOD_RARITY_WEIGHT = {
    food: _max_avail / avail for food, avail in FOOD_AVAILABILITY.items()
}

# "Wild (food)" can be paid with anything, so its effective rarity equals
# the most available food type (invertebrate) -- weight 1.0.
FOOD_RARITY_WEIGHT["Wild (food)"] = 1.0

# Bonus card columns (Anatomist through Wildlife Gardener)
BONUS_CARD_COLS = [
    "Anatomist", "Cartographer", "Historian", "Photographer",
    "Backyard Birder", "Bird Bander", "Bird Counter", "Bird Feeder",
    "Diet Specialist", "Enclosure Builder", "Endangered Species Protector",
    "Falconer", "Fishery Manager", "Food Web Expert", "Forester",
    "Large Bird Specialist", "Nest Box Builder", "Omnivore Expert",
    "Passerine Specialist", "Platform Builder", "Prairie Manager",
    "Rodentologist", "Small Clutch Specialist", "Viticulturalist",
    "Wetland Scientist", "Wildlife Gardener",
]

# Nest types in the dataset (wild = star nest)
NEST_TYPES = ["bowl", "cavity", "ground", "platform", "wild", "none"]

# ---------------------------------------------------------------------------
# Power mechanic regex patterns (applied case-insensitively to power_text)
# ---------------------------------------------------------------------------

_FOOD_TOKENS = r"die|invertebrate|seed|fish|fruit|rodent|wild|nectar"

POWER_MECHANIC_PATTERNS_DICT = {
    "mech_gain_food": re.compile(
        rf"gain \d+ \[(?:{_FOOD_TOKENS})\]",
        re.IGNORECASE,
    ),
    "mech_draw_card": re.compile(
        r"draw \d+ \[card\]",
        re.IGNORECASE,
    ),
    "mech_lay_egg": re.compile(
        r"lay \d+ \[egg\]",
        re.IGNORECASE,
    ),
    "mech_tuck": re.compile(
        r"tuck \d+ \[card\]",
        re.IGNORECASE,
    ),
    "mech_cache": re.compile(
        r"cache \d+ \[",
        re.IGNORECASE,
    ),
    "mech_from_supply": re.compile(
        r"from the supply|from your supply",
        re.IGNORECASE,
    ),
    "mech_from_feeder": re.compile(
        r"from the birdfeeder",
        re.IGNORECASE,
    ),
    "mech_reset_feeder": re.compile(
        r"reset the birdfeeder",
        re.IGNORECASE,
    ),
    "mech_bonus_card": re.compile(
        r"bonus card",
        re.IGNORECASE,
    ),
    "mech_play_bird": re.compile(
        r"play (?:an additional|another) bird",
        re.IGNORECASE,
    ),
    "mech_repeat_copy": re.compile(
        r"repeat.*power|copy.*power|you may copy",
        re.IGNORECASE,
    ),
    "mech_other_player": re.compile(
        r"another player|other player|each player|all players"
        r"|player.*left|player.*right",
        re.IGNORECASE,
    ),
    "mech_discard_to_gain": re.compile(
        r"discard.*to (?:tuck|gain|draw|lay|cache)",
        re.IGNORECASE,
    ),
}

# Numeric extraction patterns for magnitude features
_TUCK_COUNT_RE = re.compile(
    r"tuck (\d+) \[card\]|up to (\d+) \[card\].*tuck",
    re.IGNORECASE,
)
_EGG_COUNT_RE = re.compile(
    r"lay (\d+) \[egg\]",
    re.IGNORECASE,
)
_DRAW_COUNT_RE = re.compile(
    r"draw (\d+) \[card\]",
    re.IGNORECASE,
)
_FOOD_COUNT_RE = re.compile(
    rf"gain (\d+) \[(?:{_FOOD_TOKENS})\]",
    re.IGNORECASE,
)

POWER_MECHANIC_FEATURES_LIST = [
    # Binary regex features
    "mech_gain_food",
    "mech_draw_card",
    "mech_lay_egg",
    "mech_tuck",
    "mech_cache",
    "mech_from_supply",
    "mech_from_feeder",
    "mech_reset_feeder",
    "mech_bonus_card",
    "mech_play_bird",
    "mech_repeat_copy",
    "mech_other_player",
    "mech_discard_to_gain",
    # Pre-tagged from raw data
    "mech_flocking",
    # Predator sub-types (replace old mech_predator + mech_predator_tag)
    "pred_look_wingspan",
    "pred_roll_dice",
    "pred_play_on_top",
    "pred_pay_card",
    "pred_other",
    "pred_success_prob",
    # Numeric magnitude
    "mech_max_tuck_count",
    "mech_max_egg_count",
    "mech_max_draw_count",
    "mech_max_food_count",
]


# ---------------------------------------------------------------------------
# Predator sub-type classification and success probability
# ---------------------------------------------------------------------------

# Regex patterns for predator sub-type detection
_PRED_LOOK_WINGSPAN_RE = re.compile(
    r"look at a \[card\].*(?:less than|over) \d+\s*cm",
    re.IGNORECASE | re.DOTALL,
)
_PRED_ROLL_DICE_RE = re.compile(
    r"roll all dice not in birdfeeder"
    r"|roll any \d+ \[die\]"
    r"|choose any \d+ \[die\].*roll"
    r"|roll all \[die\] that are in the birdfeeder"
    r"|roll all 5 \[die\]",
    re.IGNORECASE,
)
_PRED_PLAY_ON_TOP_RE = re.compile(
    r"play this bird on top of another bird",
    re.IGNORECASE,
)
_PRED_PAY_CARD_RE = re.compile(
    r"you may pay 1 \[card\] from your hand instead",
    re.IGNORECASE,
)

# Wingspan threshold extraction for look-type predators
_WINGSPAN_UNDER_RE = re.compile(r"less than (\d+)\s*cm", re.IGNORECASE)
_WINGSPAN_OVER_RE = re.compile(r"over (\d+)\s*cm", re.IGNORECASE)

# Roll-dice food target extraction
_ROLL_FOOD_TARGET_RE = re.compile(
    r"if (?:any are|you roll (?:at least 1|a)) \[(\w+)\]",
    re.IGNORECASE,
)


def _compute_look_wingspan_success_prob(
    text: str,
    wingspan_values_array: np.ndarray,
) -> float:
    """Compute fraction of the card pool that satisfies a wingspan predator's condition.

    Args:
        text: Power text of the predator bird.
        wingspan_values_array: Array of all wingspan values in cm (NaN for star wingspan).

    Returns:
        Probability (0-1) that a random card from the pool satisfies the condition.
    """
    # "less than N cm" — prey must have wingspan < N
    m = _WINGSPAN_UNDER_RE.search(text)
    if m:
        threshold = int(m.group(1))
        valid = np.nansum(wingspan_values_array < threshold)
        return valid / len(wingspan_values_array)

    # "over N cm" — prey must have wingspan > N (e.g. Wedge-Tailed Eagle)
    m = _WINGSPAN_OVER_RE.search(text)
    if m:
        threshold = int(m.group(1))
        valid = np.nansum(wingspan_values_array > threshold)
        return valid / len(wingspan_values_array)

    return 0.0


def _compute_roll_dice_success_prob(text: str) -> float:
    """Compute expected success probability for roll-dice predators.

    Assumes the number of dice outside the birdfeeder is uniformly distributed
    over 0-4 (the feeder holds 5 dice, and should never be completely empty).

    For "roll all dice not in birdfeeder": P(success) = E[1 - (1-p)^k] for k ~ Uniform(0,4)
    where p is the single-die probability of showing the target food.

    For "roll any N [die]" or "choose any N [die]": uses fixed N dice with
    the same per-die probability.

    Returns:
        Expected probability of a successful hunt (0-1).
    """
    t = text.lower()

    # Determine target food and its per-die probability
    food_match = _ROLL_FOOD_TARGET_RE.search(text)
    if food_match:
        target_food = food_match.group(1).capitalize()
        # Map token names to FOOD_AVAILABILITY keys
        token_to_food_dict = {
            "Invertebrate": "Invertebrate",
            "Seed": "Seed",
            "Fish": "Fish",
            "Fruit": "Fruit",
            "Rodent": "Rodent",
            "Nectar": "Nectar",
        }
        p_food = FOOD_AVAILABILITY.get(
            token_to_food_dict.get(target_food, target_food),
            1 / 6,
        )
    else:
        # Default assumption for unrecognized targets
        p_food = 1 / 6

    # "roll all dice not in birdfeeder" — variable number of dice (0-4)
    if "roll all dice not in birdfeeder" in t:
        total_prob = 0.0
        for k in range(5):  # k = 0, 1, 2, 3, 4
            total_prob += 1 - (1 - p_food) ** k
        return total_prob / 5

    # "roll any N [die]" or "choose any N [die]" — fixed N dice
    n_match = re.search(r"(?:roll any|choose any) (\d+) \[die\]", t)
    if n_match:
        n_dice = int(n_match.group(1))
        return 1 - (1 - p_food) ** n_dice

    # "roll all 5 [die]" — all 5 dice
    if "roll all 5 [die]" in t:
        return 1 - (1 - p_food) ** 5

    # "roll all [die] that are in the birdfeeder" — variable (1-5)
    if "roll all [die] that are in the birdfeeder" in t:
        total_prob = 0.0
        for k in range(1, 6):  # k = 1, 2, 3, 4, 5
            total_prob += 1 - (1 - p_food) ** k
        return total_prob / 5

    return 0.0


def _parse_predator_features(
    feat_df: pd.DataFrame,
    raw_df: pd.DataFrame,
) -> pd.DataFrame:
    """Classify predator birds into sub-types and compute success probabilities.

    Sub-types:
        pred_look_wingspan  : "Look at card, tuck if wingspan meets threshold"
        pred_roll_dice      : "Roll dice, cache if target food appears"
        pred_play_on_top    : "Play on top of another bird" (white predators)
        pred_pay_card       : "Pay cards instead of rodent cost" (white predators)
        pred_other          : All other predator-tagged birds
        pred_success_prob   : Estimated success probability (0-1) for look/roll types

    Args:
        feat_df: Feature matrix (needs power_text column).
        raw_df: Raw bird data (needs Predator and Wingspan columns).

    Returns:
        DataFrame with 6 predator feature columns.
    """
    text_series = feat_df["power_text"].fillna("")
    is_predator = raw_df["Predator"].notna()

    pred_df = pd.DataFrame(0, index=feat_df.index, columns=[
        "pred_look_wingspan",
        "pred_roll_dice",
        "pred_play_on_top",
        "pred_pay_card",
        "pred_other",
    ])
    pred_df["pred_success_prob"] = 0.0

    # Pre-compute wingspan values for the entire card pool
    wingspan_values_array = pd.to_numeric(
        raw_df["Wingspan"],
        errors="coerce",
    ).values

    for idx in feat_df.index:
        if not is_predator.loc[idx]:
            continue

        t = text_series.loc[idx]

        if _PRED_LOOK_WINGSPAN_RE.search(t):
            pred_df.loc[idx, "pred_look_wingspan"] = 1
            pred_df.loc[idx, "pred_success_prob"] = (
                _compute_look_wingspan_success_prob(t, wingspan_values_array)
            )
        elif _PRED_ROLL_DICE_RE.search(t):
            pred_df.loc[idx, "pred_roll_dice"] = 1
            pred_df.loc[idx, "pred_success_prob"] = (
                _compute_roll_dice_success_prob(t)
            )
        elif _PRED_PLAY_ON_TOP_RE.search(t):
            pred_df.loc[idx, "pred_play_on_top"] = 1
        elif _PRED_PAY_CARD_RE.search(t):
            pred_df.loc[idx, "pred_pay_card"] = 1
        else:
            pred_df.loc[idx, "pred_other"] = 1

    return pred_df


def _extract_max_match(pattern: re.Pattern, text: str) -> int:
    """Return the max numeric group captured by pattern, or 0 if no match."""
    matches = pattern.findall(text)
    if not matches:
        return 0
    # findall returns tuples when pattern has multiple groups
    if isinstance(matches[0], tuple):
        values_list = [int(v) for m in matches for v in m if v]
    else:
        values_list = [int(m) for m in matches]
    return max(values_list) if values_list else 0


def parse_power_mechanics(feat_df: pd.DataFrame, raw_df: pd.DataFrame) -> pd.DataFrame:
    """Parse power text into binary and numeric mechanic features.

    Args:
        feat_df: Feature matrix (needs power_text column).
        raw_df: Raw bird data (needs Flocking and Predator columns).

    Returns:
        DataFrame with 20 mechanic feature columns, same index as feat_df.
    """
    text_series = feat_df["power_text"].fillna("")
    mech_df = pd.DataFrame(index=feat_df.index)

    # Binary regex features
    for name, pattern in POWER_MECHANIC_PATTERNS_DICT.items():
        mech_df[name] = text_series.apply(
            lambda t, p=pattern: int(bool(p.search(t)))
        )

    # Pre-tagged features from raw data
    mech_df["mech_flocking"] = raw_df["Flocking"].notna().astype(int)

    # Predator sub-types (replaces old mech_predator + mech_predator_tag)
    pred_features_df = _parse_predator_features(feat_df, raw_df)
    for col in pred_features_df.columns:
        mech_df[col] = pred_features_df[col]

    # Numeric magnitude features
    mech_df["mech_max_tuck_count"] = text_series.apply(
        lambda t: _extract_max_match(_TUCK_COUNT_RE, t)
    )
    mech_df["mech_max_egg_count"] = text_series.apply(
        lambda t: _extract_max_match(_EGG_COUNT_RE, t)
    )
    mech_df["mech_max_draw_count"] = text_series.apply(
        lambda t: _extract_max_match(_DRAW_COUNT_RE, t)
    )
    mech_df["mech_max_food_count"] = text_series.apply(
        lambda t: _extract_max_match(_FOOD_COUNT_RE, t)
    )

    return mech_df


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def compute_weighted_food_cost(df: pd.DataFrame) -> pd.Series:
    """Compute rarity-weighted total food cost for each bird.

    Each unit of a food type is multiplied by that food's rarity weight.
    For birds with "/" (or-cost), the listed foods are alternatives, so
    we use the *minimum* rarity weight among the options (the player would
    choose the cheapest-to-obtain food).

    For birds with "*" (predator cost), the food cost is handled normally
    since the * mechanic is part of the bird's power, not a cost discount.
    We flag these separately.

    Nectar required as a cost is weighted by its rarity, but note that
    nectar has additional strategic value (wild input, habitat majority
    scoring) and risk (expires end of round) not captured here.

    Returns:
        Series of weighted food cost values, indexed like the input df.
    """
    specific_foods = ["Invertebrate", "Seed", "Fish", "Fruit", "Rodent", "Nectar"]
    weighted_cost = pd.Series(0.0, index=df.index)

    has_or_cost = df["/ (food cost)"].notna()

    for _, row in df.iterrows():
        idx = row.name
        if has_or_cost.loc[idx]:
            # Or-cost: player picks one food from the listed options.
            # Cost = 1 unit * min rarity weight among listed foods.
            options = [f for f in specific_foods if pd.notna(row[f])]
            if options:
                min_weight = min(FOOD_RARITY_WEIGHT[f] for f in options)
                weighted_cost.loc[idx] = min_weight
            # Any wild food component is additive (not part of the or-choice)
            if pd.notna(row["Wild (food)"]):
                weighted_cost.loc[idx] += row["Wild (food)"] * FOOD_RARITY_WEIGHT["Wild (food)"]
        else:
            # Standard cost: sum each food * its rarity weight
            for food in specific_foods:
                if pd.notna(row[food]):
                    weighted_cost.loc[idx] += row[food] * FOOD_RARITY_WEIGHT[food]
            if pd.notna(row["Wild (food)"]):
                weighted_cost.loc[idx] += row["Wild (food)"] * FOOD_RARITY_WEIGHT["Wild (food)"]

    return weighted_cost


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Build a regression-ready feature matrix from raw bird card data.

    Features created:
        weighted_food_cost  : Rarity-adjusted total food cost
        total_food_cost     : Raw total food cost (for comparison)
        has_or_cost         : 1 if bird has "/" food cost (pick-one flexibility)
        has_predator_cost   : 1 if bird has "*" food cost (predator mechanic)
        nectar_required     : Units of nectar in food cost (special handling)
        egg_limit           : Maximum eggs the bird can hold
        star_nest           : 1 if nest type is "wild" (star nest, fits any category)
        nest_none           : 1 if nest type is "none" (no nest)
        nest_bowl           : 1 if nest type is bowl  (reference: ground)
        nest_cavity         : 1 if nest type is cavity
        nest_platform       : 1 if nest type is platform
        star_wingspan       : 1 if wingspan is "*" (flightless / star)
        wingspan_cm         : Numeric wingspan in cm (0 for star wingspan birds)
        bonus_count         : Number of bonus cards this bird is eligible for
        habitat_count       : Number of habitats the bird can be placed in
        color_brown         : 1 if power color is brown  (reference: no power)
        color_white         : 1 if power color is white
        color_teal          : 1 if power color is teal
        color_pink          : 1 if power color is pink
        color_yellow        : 1 if power color is yellow
        has_power           : 1 if bird has any power

    Power mechanic features (parsed from power text):
        mech_gain_food      : 1 if power gains food tokens
        mech_draw_card      : 1 if power draws bird cards
        mech_lay_egg        : 1 if power lays eggs
        mech_tuck           : 1 if power tucks cards
        mech_cache          : 1 if power caches food on bird
        mech_from_supply    : 1 if food comes from general supply
        mech_from_feeder    : 1 if food comes from birdfeeder
        mech_reset_feeder   : 1 if power resets the birdfeeder
        mech_bonus_card     : 1 if power involves bonus cards
        mech_play_bird      : 1 if power plays an additional bird
        mech_repeat_copy    : 1 if power repeats/copies another power
        mech_other_player   : 1 if power involves other players
        mech_discard_to_gain: 1 if power requires discarding to gain benefit
        mech_flocking       : 1 if bird has flocking flag (from raw data)
        pred_look_wingspan  : 1 if predator hunts by wingspan threshold
        pred_roll_dice      : 1 if predator hunts by rolling dice for food
        pred_play_on_top    : 1 if predator plays on top of another bird
        pred_pay_card       : 1 if predator pays cards instead of rodent
        pred_other          : 1 if predator with other mechanic
        pred_success_prob   : Estimated hunt success probability (0-1)
        mech_max_tuck_count : Max cards tucked per activation
        mech_max_egg_count  : Max eggs laid per activation
        mech_max_draw_count : Max cards drawn per activation
        mech_max_food_count : Max food gained per activation

    Also preserves:
        common_name         : Bird name for identification
        victory_points      : Dependent variable
        color               : Raw power color (for grouping)
        power_text          : Raw power description (for parsing)
    """
    feat = pd.DataFrame(index=df.index)

    # Identity / DV
    feat["common_name"] = df["Common name"]
    feat["victory_points"] = df["Victory points"]
    feat["color"] = df["Color"]
    feat["power_text"] = df["Power text"]

    # --- Food cost features ---
    feat["weighted_food_cost"] = compute_weighted_food_cost(df)
    feat["total_food_cost"] = df["Total food cost"]
    feat["has_or_cost"] = (df["/ (food cost)"].notna()).astype(int)
    feat["has_predator_cost"] = (df["* (food cost)"].notna()).astype(int)
    feat["nectar_required"] = df["Nectar"].fillna(0)

    # --- Nest features ---
    feat["egg_limit"] = df["Egg limit"]
    feat["star_nest"] = (df["Nest type"] == "wild").astype(int)
    feat["nest_none"] = (df["Nest type"] == "none").astype(int)
    # Dummy-code nest types (ground is reference category)
    feat["nest_bowl"] = (df["Nest type"] == "bowl").astype(int)
    feat["nest_cavity"] = (df["Nest type"] == "cavity").astype(int)
    feat["nest_platform"] = (df["Nest type"] == "platform").astype(int)

    # --- Wingspan features ---
    feat["star_wingspan"] = (df["Wingspan"] == "*").astype(int)
    feat["wingspan_cm"] = pd.to_numeric(df["Wingspan"], errors="coerce").fillna(0)

    # --- Bonus card eligibility ---
    bonus_df = df[BONUS_CARD_COLS].notna().astype(int)
    feat["bonus_count"] = bonus_df.sum(axis=1)

    # --- Habitat flexibility ---
    habitat_cols = ["Forest", "Grassland", "Wetland"]
    feat["habitat_count"] = df[habitat_cols].notna().astype(int).sum(axis=1)

    # --- Power color indicators ---
    feat["has_power"] = df["Color"].notna().astype(int)
    feat["color_brown"] = (df["Color"] == "brown").astype(int)
    feat["color_white"] = (df["Color"] == "white").astype(int)
    feat["color_teal"] = (df["Color"] == "teal").astype(int)
    feat["color_pink"] = (df["Color"] == "pink").astype(int)
    feat["color_yellow"] = (df["Color"] == "yellow").astype(int)

    # --- Power mechanic features (parsed from power text + raw flags) ---
    mech_df = parse_power_mechanics(feat, df)
    for col in mech_df.columns:
        feat[col] = mech_df[col]

    return feat


def get_food_rarity_summary() -> pd.DataFrame:
    """Return a summary table of food rarity weights for reference."""
    rows = []
    for food, avail in FOOD_AVAILABILITY.items():
        rows.append({
            "food_type": food,
            "dice_availability": avail,
            "rarity_weight": FOOD_RARITY_WEIGHT[food],
        })
    rows.append({
        "food_type": "Wild (food)",
        "dice_availability": _max_avail,  # effectively the best availability
        "rarity_weight": FOOD_RARITY_WEIGHT["Wild (food)"],
    })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    from src.load_data import load_bird_data

    df = load_bird_data()
    feat = build_feature_matrix(df)

    print("Feature matrix shape:", feat.shape)
    print("\nFeature columns:")
    for col in feat.columns:
        print(f"  {col:<25} {feat[col].dtype}")

    print("\nFood rarity weights:")
    print(get_food_rarity_summary().to_string(index=False))

    print("\nFeature summary stats:")
    numeric_cols = feat.select_dtypes(include=[np.number]).columns
    print(feat[numeric_cols].describe().round(2).to_string())
