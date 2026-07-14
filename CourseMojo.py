import kagglehub
from kagglehub import KaggleDatasetAdapter
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency


# =====================================================================
# 1. LOAD DATA
# =====================================================================

file_path = "students_adaptability_level_online_education.csv"

df = kagglehub.dataset_load(
  KaggleDatasetAdapter.PANDAS,
  "mdmahmudulhasansuzan/students-adaptability-level-in-online-education",
  file_path,

  # documenation for more information:
  # https://github.com/Kaggle/kagglehub/blob/main/README.md#kaggledatasetadapterpandas
)


# =====================================================================
# 2. CONFIGURATION
# Constants and groupings used throughout the rest of the script
# =====================================================================

# Adaptivity Level's natural order isn't alphabetical, so enforce it explicitly
# for every crosstab/chart below
adaptivity_order = ['Low', 'Moderate', 'High']

# Every factor we're testing, organized by theme so we can compare
# themes against each other later, not just individual columns
factor_groups = {
    'Access / Infrastructure': ['Internet Type', 'Network Type', 'Device', 'Load-shedding'],
    'Household Context': ['Financial Condition', 'Location'],
    'Institutional Support': ['Institution Type', 'Self Lms', 'Class Duration'],
    'Demographics': ['Gender', 'Age', 'Education Level', 'IT Student']
}

# Flattened list of every factor, derived from factor_groups so there's a
# single source of truth (no risk of the two lists drifting out of sync)
all_factors = [factor for factors in factor_groups.values() for factor in factors]

# Statistical significance cutoff: standard convention across most fields.
# A p-value below this means the relationship is unlikely to be random noise.
SIGNIFICANCE_THRESHOLD = 0.05


# =====================================================================
# 3. RAW CROSSTABS
# Print the % breakdown (Low/Moderate/High) for every factor, for
# manual inspection before any statistical testing happens
# =====================================================================

for factor in all_factors:
    print(f"--- {factor} ---")
    crosstab_pct = pd.crosstab(df[factor], df['Adaptivity Level'], normalize='index')[adaptivity_order] * 100
    print(crosstab_pct.round(1))
    print()


# =====================================================================
# 4. EFFECT SIZE FUNCTION
# =====================================================================

def cramers_v(count_table):
    """Effect size (0-1) for the strength of association between two
    categorical variables. Pair with a chi-square p-value to check
    significance first — Cramér's V alone doesn't tell you if a
    relationship is statistically real, only how strong it is if real."""
    chi2_stat = chi2_contingency(count_table)[0]
    total_students = count_table.sum().sum()
    num_rows, num_cols = count_table.shape
    return np.sqrt(chi2_stat / (total_students * (min(num_rows, num_cols) - 1)))


# =====================================================================
# 5. DASHBOARD: SIGNIFICANCE + EFFECT SIZE PER THEME
# One 2x2 grid of bar charts, one per theme, showing Cramér's V for
# every factor that passed the significance test
# =====================================================================

# One combined figure: 3 rows total — top 2 rows hold the 2x2 group grid,
# bottom row is a single wide subplot spanning both columns for the lift chart
fig = plt.figure(figsize=(14, 12))
grid = gridspec.GridSpec(3, 2, height_ratios=[1, 1, 1.1])

group_subplots = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1]),
                  fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1])]
lift_subplot = fig.add_subplot(grid[2, :])  # spans both columns

# Collected here so we can compare each group's overall influence after the loop too
group_avg_by_name = {}

# Pairs each subplot with one theme from factor_groups, so each iteration
# draws into its own quadrant of the shared figure
for subplot, (group_name, factors_in_group) in zip(group_subplots, factor_groups.items()):
    factor_scores = []
    for factor in factors_in_group:
        count_table = pd.crosstab(df[factor], df['Adaptivity Level'])
        chi2_stat, p_value, degrees_of_freedom, expected_counts = chi2_contingency(count_table)
        effect_size = cramers_v(count_table)
        factor_scores.append({'factor': factor, 'cramers_v': effect_size, 'p_value': p_value})

    scores_df = pd.DataFrame(factor_scores)

    # Only keep factors whose relationship to Adaptivity Level is statistically
    # significant — drop anything that could just be random noise
    significant_scores_df = scores_df[scores_df['p_value'] < SIGNIFICANCE_THRESHOLD]
    scores_df_sorted = significant_scores_df.sort_values('cramers_v', ascending=True)

    # Average Cramér's V across this theme's SIGNIFICANT factors only
    group_avg = scores_df_sorted['cramers_v'].mean()
    group_avg_by_name[group_name] = group_avg

    subplot.barh(scores_df_sorted['factor'], scores_df_sorted['cramers_v'], color='#3d405b')
    subplot.set_xlabel("Cramér's V")
    subplot.set_title(f"{group_name} (avg: {group_avg:.3f})")

    # Force the full 0-1 scale (Cramér's V's true range) even though our
    # values are all small — keeps all four charts on a consistent, honest
    # scale rather than each auto-zooming to its own tiny range
    subplot.set_xlim(0, 1)

# Print all four group averages ranked, for a quick side-by-side comparison
print("Overall influence by group (average Cramér's V, significant factors only):")
for name, avg in sorted(group_avg_by_name.items(), key=lambda x: x[1], reverse=True):
    print(f"  {name}: {avg:.3f}")


# =====================================================================
# 6. LIFT ANALYSIS
# Instead of testing whole columns, find which specific category
# VALUES are most over-represented among High-adaptivity students
# =====================================================================

# Isolate just the students with High adaptivity
high_adaptivity_df = df[df['Adaptivity Level'] == 'High']

# Total High-adaptivity students — used as the denominator for every
# sample-size label on the lift chart, so readers see counts in context
# rather than as isolated, hard-to-interpret numbers
total_high_adaptivity = len(high_adaptivity_df)

lift_scores = []
for factor in all_factors:
    baseline_dist = df[factor].value_counts(normalize=True)              # this category's mix, overall
    high_adaptivity_dist = high_adaptivity_df[factor].value_counts(normalize=True)  # this category's mix, High group only

    # lift = how over/under-represented each value is among High students
    # (1.0 = no difference from baseline; higher = more common among High-adaptivity students)
    lift = (high_adaptivity_dist / baseline_dist).dropna()

    # keep only this column's single most over-represented value
    top_category = lift.idxmax()
    top_lift = lift.max()
    lift_scores.append({'factor': f"{factor} = {top_category}", 'lift': top_lift})

lift_df = pd.DataFrame(lift_scores).sort_values('lift', ascending=True)

# Sample size behind each trait's lift score — small samples can produce
# misleadingly large lift values, so this count is shown directly on the
# chart rather than left hidden behind the ratio
sample_sizes = []
for factor_value in lift_df['factor']:
    factor_name, value = factor_value.split(' = ')
    count_in_high_group = (high_adaptivity_df[factor_name] == value).sum()
    sample_sizes.append(count_in_high_group)
lift_df['sample_size'] = sample_sizes


# =====================================================================
# 7. DRAW LIFT CHART + SHOW FULL DASHBOARD
# =====================================================================

lift_subplot.barh(lift_df['factor'], lift_df['lift'], color='#3d405b')
lift_subplot.axvline(1, color='gray', linestyle='--', linewidth=1)  # reference line: 1 = no effect
lift_subplot.set_xlabel("Lift (1.0 = no difference from baseline)")
lift_subplot.set_title(
    f"Which single trait is most associated with High Adaptivity? "
    f"(n={total_high_adaptivity} High-adaptivity students total)"
)

# Label each bar as "n=X/total" so readers see both how many students back
# this trait AND how many High-adaptivity students there are overall —
# a big lift number next to a small count is easy to spot and treat cautiously
for i, (lift_val, n) in enumerate(zip(lift_df['lift'], lift_df['sample_size'])):
    lift_subplot.text(lift_val + 0.05, i, f"n={n}/{total_high_adaptivity}", va='center', fontsize=9, color='gray')

plt.tight_layout()
plt.show()


# =====================================================================
# 8. FINAL OUTPUT + SANITY CHECK
# =====================================================================

print(lift_df.sort_values('lift', ascending=False))

# Sanity check: how many actual students back the single strongest trait?
# (important since only ~8% of students are High adaptivity, so some
# category combinations may have very few students behind a big-looking lift)
top_trait = lift_df.iloc[-1]['factor']
top_factor_name, top_factor_value = top_trait.split(' = ')
print(f"\nStudents with {top_factor_name} = {top_factor_value} in High group:",
      (high_adaptivity_df[top_factor_name] == top_factor_value).sum())
print(f"Students with {top_factor_name} = {top_factor_value} overall:",
      (df[top_factor_name] == top_factor_value).sum())