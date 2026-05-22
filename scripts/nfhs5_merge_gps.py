#############################################################
############ STEP 1 — Imports and File Paths ############
import geopandas as gpd
import pandas as pd
import os
from scipy import stats

print(f"geopandas: {gpd.__version__}")
print(f"pandas:    {pd.__version__}")

# ── UPDATE THESE PATHS ──────────────────────────────────────────────────────

NFHS_GPS_PATH = "/Users/sonalideliwala/Library/Mobile Documents/com~apple~CloudDocs/Desktop/India Data/NFHS GIS/Geographic data nfhs5/IAGE7AFL.shp"

SHRUG_POLY_PATH = "/Users/sonalideliwala/Library/Mobile Documents/com~apple~CloudDocs/Desktop/[RA work] Womens Reservation & Girls Aspirations/shrug-pc11-village-poly-shp/village_modified.shp"

SHRUG_LGD_PATH = "/Users/sonalideliwala/Library/Mobile Documents/com~apple~CloudDocs/Desktop/India Data/Shrid To Local Governance Body Matching/data/shrug_LGD_matched.csv"

NFHS_WOMEN_PATH = "/Users/sonalideliwala/Desktop/nfhs5_women_truncated.dta"

OUTPUT_PATH = "/Users/sonalideliwala/Desktop/nfhs5_women_gp_full.dta"

# ── CONSTANTS ────────────────────────────────────────────────────────────────

TARGET_STATES      = ["RAJASTHAN", "UTTAR PRADESH"]  # ADM1NAME in GPS file
TARGET_STATE_CODES = ["08", "09"]                    # pc11_s_id in SHRUG
PROJECTED_CRS      = "EPSG:32644"                    # UTM Zone 44N (metres)

print("\nAll imports and paths set.")
#############################################################


#############################################################
###### STEP 2 — Load and Prepare NFHS-5 GPS Clusters ########
# ── Load shapefile ────────────────────────────────────────────────────────────
print("Loading NFHS-5 GPS shapefile...")
gps5 = gpd.read_file(NFHS_GPS_PATH)
print(f"Total clusters all India: {gps5.shape[0]}")

# Keep relevant columns
gps5 = gps5[["DHSCLUST", "LATNUM", "LONGNUM", "URBAN_RURA",
              "ADM1NAME", "DHSREGNA"]].copy()

# Strip whitespace from state name (known encoding issue in DHS files)
gps5["ADM1NAME"] = gps5["ADM1NAME"].str.strip()

# Drop clusters with missing or zero coordinates
gps5 = gps5[(gps5["LATNUM"] != 0) & (gps5["LONGNUM"] != 0)].copy()

# Filter to Rajasthan and UP
gps5_rj_up = gps5[gps5["ADM1NAME"].isin(TARGET_STATES)].copy()
print(f"\nClusters in RJ + UP: {gps5_rj_up.shape[0]}")
print(gps5_rj_up["ADM1NAME"].value_counts())

# Convert to GeoDataFrame with WGS84 CRS, then reproject to metres
gps5_rj_up = gpd.GeoDataFrame(
    gps5_rj_up,
    geometry=gpd.points_from_xy(gps5_rj_up.LONGNUM, gps5_rj_up.LATNUM),
    crs="EPSG:4326"
)
gps5_proj = gps5_rj_up.to_crs(PROJECTED_CRS)
print("\nGPS clusters reprojected to UTM Zone 44N.")
#############################################################


#############################################################
###### STEP 3 — Load and Prepare SHRUG Village Polygons #####
# ── Load shapefile ────────────────────────────────────────────────────────────
print("Loading SHRUG village polygons (all India)...")
shrug = gpd.read_file(SHRUG_POLY_PATH)
print(f"Total villages all India: {shrug.shape[0]}")

# Filter to RJ (08) and UP (09)
shrug_rj_up = shrug[shrug["pc11_s_id"].isin(TARGET_STATE_CODES)].copy()
print(f"\nVillage polygons in RJ + UP: {shrug_rj_up.shape[0]}")
print(shrug_rj_up["pc11_s_id"].value_counts())

# Set CRS and reproject to metres
shrug_rj_up = shrug_rj_up.set_crs("EPSG:4326")
shrug_proj  = shrug_rj_up.to_crs(PROJECTED_CRS)
print("\nSHRUG polygons reprojected to UTM Zone 44N.")
#############################################################


#####################################################################
## STEP 4 — Spatial Join: NFHS Clusters → Nearest SHRUG Village #####

# NOTE: I use sjoin_nearest rather than point-in-polygon because DHS GPS
# coordinates are intentionally displaced up to 5km for rural clusters to
# protect respondent confidentiality. Nearest-neighbour ensures all clusters
# receive a village assignment despite displacement. Aggregation to the GP
# level (Step 5) substantially mitigates the resulting village-level noise
# since a typical GP spans 10-25 km² and displacement rarely crosses GP lines.

print("Running nearest-neighbour spatial join (cluster GPS → village polygon)...")

crosswalk = gpd.sjoin_nearest(
    gps5_proj,
    shrug_proj[["pc11_s_id", "pc11_d_id", "pc11_sd_id",
                "pc11_tv_id", "tv_name", "geometry"]],
    how="left"
)

# Keep only needed columns
crosswalk = crosswalk[[
    "DHSCLUST", "LATNUM", "LONGNUM", "URBAN_RURA",
    "ADM1NAME", "DHSREGNA", "tv_name",
    "pc11_s_id", "pc11_d_id", "pc11_sd_id", "pc11_tv_id"
]].copy()

print(f"Crosswalk rows:              {crosswalk.shape[0]}")
print(f"Clusters matched to village: {crosswalk['pc11_tv_id'].notna().sum()}")
crosswalk.head()
#####################################################################

#####################################################################
###### STEP 5 — Load SHRUG-LGD Crosswalk and Assign GP Codes ########
# ── Load crosswalk ────────────────────────────────────────────────────────────
print("Loading SHRUG-LGD crosswalk...")
shrug_lgd = pd.read_csv(SHRUG_LGD_PATH, encoding="latin-1", low_memory=False)
print(f"SHRUG-LGD rows: {shrug_lgd.shape[0]}")

# ── Build GP code column ──────────────────────────────────────────────────────
# For GP → ULB conversions, use old_gp_lgd_code
# For regular GPs, use LGD_code

def get_gp_code(row):
    if row["gp_to_urban_conversion"] == "Yes" and pd.notna(row["old_gp_lgd_code"]):
        return row["old_gp_lgd_code"]
    elif pd.notna(row["LGD_code"]) and row["local_body_type"] == "Gram Panchayat":
        return row["LGD_code"]
    return None

def get_gp_name(row):
    if row["gp_to_urban_conversion"] == "Yes" and pd.notna(row["old_gp_name"]):
        return row["old_gp_name"]
    elif pd.notna(row["local_body_name"]) and row["local_body_type"] == "Gram Panchayat":
        return row["local_body_name"]
    return None

shrug_lgd["gp_lgd_code"] = shrug_lgd.apply(get_gp_code, axis=1)
shrug_lgd["gp_name"]     = shrug_lgd.apply(get_gp_name, axis=1)

# Keep only rows with a valid GP code
shrug_lgd_gps = shrug_lgd[
    shrug_lgd["gp_lgd_code"].notna()
][["shrid2", "gp_lgd_code", "gp_name",
   "gp_to_urban_conversion", "shrid_part_of_multi_village_GP"]].copy()

shrug_lgd_gps = shrug_lgd_gps.rename(columns={"shrid2": "shrid2_key"})
print(f"Rows with valid GP code: {len(shrug_lgd_gps)}")

# ── Build shrid2-format key from SHRUG polygon fields ────────────────────────
# shrid2 format: "11-{state:02}-{district:03}-{subdistrict:05}-{village:06}"

crosswalk["shrid2_key"] = (
    "11-" +
    crosswalk["pc11_s_id"].astype(str).str.zfill(2)  + "-" +
    crosswalk["pc11_d_id"].astype(str).str.zfill(3)  + "-" +
    crosswalk["pc11_sd_id"].astype(str).str.zfill(5) + "-" +
    crosswalk["pc11_tv_id"].astype(str).str.zfill(6)
)

print(f"\nSample shrid2 keys: {crosswalk['shrid2_key'].head(3).tolist()}")

# ── Merge GP codes onto crosswalk ─────────────────────────────────────────────
crosswalk_with_gp = crosswalk.merge(shrug_lgd_gps, on="shrid2_key", how="left")

print(f"\n── GP match rates ──")
print(crosswalk_with_gp.groupby(["ADM1NAME", "URBAN_RURA"])["gp_lgd_code"].apply(
    lambda x: f"{x.notna().mean():.1%} ({x.notna().sum()} of {len(x)})"
))
#####################################################################

#####################################################################
###### STEP 6 — Load Truncated NFHS-5 Women's Survey ################
# Prepared in Stata: only ~170 autonomy-relevant variables retained
# convert_categoricals=False avoids error from non-unique value labels (e.g. sdist)

print("Loading truncated NFHS-5 women's survey...")
women5 = pd.read_stata(NFHS_WOMEN_PATH, convert_categoricals=False)
print(f"Observations: {women5.shape[0]}, Variables: {women5.shape[1]}")

# Rename cluster variable to match crosswalk key
women5 = women5.rename(columns={"v001": "DHSCLUST"})
print(f"Unique clusters: {women5['DHSCLUST'].nunique()}")
#####################################################################


#####################################################################
###### STEP 7 — Merge Women's Survey with GP Crosswalk ##############
print("Merging women's survey with GP crosswalk...")

# Inner merge retains only RJ + UP women
# (crosswalk only contains clusters from those two states)
merged_final = women5.merge(
    crosswalk_with_gp[[
        "DHSCLUST", "LATNUM", "LONGNUM", "URBAN_RURA",
        "ADM1NAME", "DHSREGNA", "tv_name",
        "pc11_s_id", "pc11_d_id", "pc11_sd_id", "pc11_tv_id",
        "shrid2_key", "gp_lgd_code", "gp_name",
        "gp_to_urban_conversion", "shrid_part_of_multi_village_GP"
    ]],
    on="DHSCLUST",
    how="inner"
)

print(f"\n── Final dataset ──")
print(f"Total observations:      {merged_final.shape[0]}")
print(f"Total variables:         {merged_final.shape[1]}")
print(f"Unique GPs matched:      {merged_final['gp_lgd_code'].nunique()}")
print(f"\nBy state:")
print(merged_final.groupby("ADM1NAME").agg(
    women      = ("DHSCLUST", "count"),
    clusters   = ("DHSCLUST", "nunique"),
    unique_gps = ("gp_lgd_code", "nunique")
))
#####################################################################

#####################################################################
###### STEP 8 — Save Final Dataset ##################################
print(f"Saving to: {OUTPUT_PATH}")

merged_final.to_stata(
    OUTPUT_PATH,
    write_index=False,
    version=118        # Stata 14+ format
)

print(f"\nSaved successfully.")
print(f"Observations: {merged_final.shape[0]}")
print(f"Variables:    {merged_final.shape[1]}")
print("\nNext step: merge gp_lgd_code with GP election reservation status data.")
#####################################################################

#####################################################################
###### STEP 9 — Match Rate Diagnostics () ###########################
# ── Unmatched cluster breakdown ───────────────────────────────────────────────
unmatched      = crosswalk_with_gp[crosswalk_with_gp["gp_lgd_code"].isna()]
unmatched_keys = unmatched["shrid2_key"].tolist()
found_in_shrug = shrug_lgd[shrug_lgd["shrid2"].isin(unmatched_keys)]

print(f"Total unmatched clusters: {len(unmatched)}")
print(f"Found in shrug_lgd (any type): {len(found_in_shrug)}")
print(found_in_shrug["local_body_type"].value_counts())

rural_unmatched = crosswalk_with_gp[
    (crosswalk_with_gp["gp_lgd_code"].isna()) &
    (crosswalk_with_gp["URBAN_RURA"] == "R")
]
og_flag = rural_unmatched["pc11_tv_id"].astype(str).str.startswith("8")
print(f"\nUnmatched rural with OG village codes (start with 8): {og_flag.sum()}")
print(f"Unmatched rural - genuine LGD gap:                    {(~og_flag).sum()}")

# ── Balance check: matched vs unmatched UP rural women ───────────────────────
up_rural = merged_final[
    (merged_final["ADM1NAME"] == "UTTAR PRADESH") &
    (merged_final["URBAN_RURA"] == "R")
].copy()

up_rural["matched"] = up_rural["gp_lgd_code"].notna().astype(int)

check_vars = {
    "v190":  "Wealth index",
    "v149":  "Educational attainment",
    "v012":  "Age",
    "v744a": "Wife-beating justified",
    "v743a": "Decides own healthcare"
}

print(f"\n── Balance: matched vs unmatched UP rural women ──")
print(f"{'Variable':<35} {'Unmatched':>10} {'Matched':>10} {'p-value':>10}")
print("-" * 70)

for var, label in check_vars.items():
    if var not in up_rural.columns:
        continue
    u = up_rural.loc[up_rural["matched"]==0, var].dropna()
    m = up_rural.loc[up_rural["matched"]==1, var].dropna()
    _, pval = stats.ttest_ind(u, m)
    print(f"{label:<35} {u.mean():>10.3f} {m.mean():>10.3f} {pval:>10.3f}")

print(f"\nUnmatched N: {(up_rural['matched']==0).sum()}")
print(f"Matched N:   {(up_rural['matched']==1).sum()}")
#####################################################################
