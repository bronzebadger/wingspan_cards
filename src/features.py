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

    Also preserves:
        common_name         : Bird name for identification
        victory_points      : Dependent variable
        color               : Raw power color (for grouping)
        power_text          : Raw power description (for later parsing)
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
