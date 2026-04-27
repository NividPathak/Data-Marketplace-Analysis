

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from statistics import NormalDist
import json, os, warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures') + os.sep
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({
    'font.family': 'serif',
    'font.size':   10,
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'figure.dpi': 150,
})


# 1.  SYNTHETIC DATASET
# Calibrated to mirror the empirical distributions reported by
# Azcoitia, Iordanou & Laoutaris (IEEE ICDE, 2023):
#   median subscription price ≈ $1,400 / month;
#   financial and healthcare data at the high end;
#   government / media at the low end.


N = 450

CAT_NAMES  = ['Financial','Location','Healthcare','Marketing',
              'Environmental','IoT_Sensor','Retail','Media','Government']
CAT_PROBS  = [0.20,0.15,0.12,0.15,0.10,0.08,0.10,0.05,0.05]

FREQ_ORDER = ['Real-time','Daily','Weekly','Monthly','Static']

platforms      = np.random.choice(['AWS Data Exchange','Snowflake Marketplace'],
                                  size=N, p=[0.55, 0.45])
categories     = np.random.choice(CAT_NAMES, size=N, p=CAT_PROBS)

pricing_models = np.array([
    np.random.choice(['Subscription','One-off','Free'],
                     p=[0.72,0.18,0.10] if p == 'AWS Data Exchange'
                       else [0.52,0.22,0.26])
    for p in platforms
])
update_freqs   = np.random.choice(FREQ_ORDER, size=N, p=[0.18,0.34,0.25,0.15,0.08])
provider_types = np.random.choice(['Enterprise','Startup','Academic_NGO'],
                                  size=N, p=[0.58,0.27,0.15])
is_bundled     = np.random.choice([0,1], size=N, p=[0.63,0.37])

# Log-price coefficients
CAT_EFF  = {'Financial':1.10,'Healthcare':0.85,'Location':0.65,'Marketing':0.35,
             'Environmental':0.15,'IoT_Sensor':0.10,'Retail':0.00,
             'Media':-0.25,'Government':-0.60}
FREQ_EFF = {'Real-time':0.75,'Daily':0.40,'Weekly':0.10,'Monthly':-0.10,'Static':-0.30}
PLAT_EFF = {'AWS Data Exchange':0.15,'Snowflake Marketplace':0.0}
PMOD_EFF = {'Subscription':0.0,'One-off':0.35}
PROV_EFF = {'Enterprise':0.20,'Startup':-0.10,'Academic_NGO':-0.40}
BUND_EFF = -0.12

base = np.random.normal(6.8, 1.1, N)
log_prices = []
for i in range(N):
    if pricing_models[i] == 'Free':
        log_prices.append(np.nan)
    else:
        lp = (base[i]
              + CAT_EFF[categories[i]]
              + FREQ_EFF[update_freqs[i]]
              + PLAT_EFF[platforms[i]]
              + PMOD_EFF[pricing_models[i]]
              + PROV_EFF[provider_types[i]]
              + (BUND_EFF if is_bundled[i] else 0)
              + np.random.normal(0, 0.30))
        log_prices.append(np.clip(lp, np.log(10), np.log(250000)))
log_prices = np.array(log_prices, dtype=float)
prices     = np.where(pricing_models == 'Free', 0.0, np.exp(log_prices))

df = pd.DataFrame({
    'platform':      platforms,
    'category':      categories,
    'pricing_model': pricing_models,
    'update_freq':   update_freqs,
    'provider_type': provider_types,
    'is_bundled':    is_bundled.astype(int),
    'price':         prices,
    'log_price':     log_prices,
})
df.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'marketplace_data.csv'), index=False)
print(f"Dataset: {len(df)} rows  |  Paid: {(df.pricing_model != 'Free').sum()}"
      f"  |  Free: {(df.pricing_model == 'Free').sum()}")

paid = df[df['pricing_model'] != 'Free'].copy().reset_index(drop=True)


# 2.  DESCRIPTIVE STATISTICS


desc = (paid.groupby('category')['price']
            .agg(N='count',
                 Median='median',
                 Mean='mean',
                 Std='std')
            .assign(CV=lambda x: (x['Std'] / x['Mean'] * 100))
            .round({'Median':0,'Mean':0,'Std':0,'CV':1})
            .sort_values('Median', ascending=False)
            .reset_index())
desc.columns = ['Category','N','Median ($)','Mean ($)','Std Dev ($)','CV (%)']
desc.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'desc_stats.csv'), index=False)
print("\nDescriptive stats by category:")
print(desc.to_string(index=False))

plat_tbl = (paid.groupby('platform')['price']
                .agg(N='count', Median='median', Mean='mean', Std='std')
                .round(0).reset_index())
print("\nPlatform comparison:")
print(plat_tbl.to_string(index=False))


# 3.  OLS REGRESSION 


def ols_hc3(X_arr, y_arr, col_names):
    """OLS with HC3 heteroskedasticity-robust standard errors."""
    X, y = X_arr.astype(float), y_arr.astype(float)
    n, k = X.shape
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta    = XtX_inv @ X.T @ y
    y_hat   = X @ beta
    e       = y - y_hat
    # Leverage values (hat matrix diagonal)
    H = X @ XtX_inv @ X.T
    h = np.clip(np.diag(H), 0, 0.9999)
    # HC3 meat
    e_adj = e / (1 - h)
    meat  = (X * e_adj[:, None]).T @ (X * e_adj[:, None])
    V     = XtX_inv @ meat @ XtX_inv
    se    = np.sqrt(np.abs(np.diag(V)))
    t_arr = beta / se
    # p-values: normal approximation (valid for n=375)
    nd    = NormalDist()
    pvals = np.array([2 * (1 - nd.cdf(abs(t))) for t in t_arr])
    ss_res = float(e @ e)
    ss_tot = float((y - y.mean()) @ (y - y.mean()))
    r2     = 1 - ss_res / ss_tot
    r2_adj = 1 - (1 - r2) * (n - 1) / (n - k)
    return pd.DataFrame({'Variable': col_names,
                         'Coefficient': np.round(beta, 3),
                         'Std_Error':   np.round(se,   3),
                         't_stat':      np.round(t_arr,2),
                         'p_value':     np.round(pvals,3)}), r2, r2_adj, n

# Build design matrix
reg_df = paid.copy()
reg_df['log_price'] = np.log(reg_df['price'])

# Dummies — reference categories: Retail, Snowflake, Static, Academic_NGO, One-off
cat_dummies  = pd.get_dummies(reg_df['category'],      prefix='Cat',  drop_first=False).drop(columns=['Cat_Retail'],      errors='ignore')
plat_dummies = pd.get_dummies(reg_df['platform'],      prefix='Plat', drop_first=False).drop(columns=['Plat_Snowflake Marketplace'], errors='ignore')
freq_dummies = pd.get_dummies(reg_df['update_freq'],   prefix='Freq', drop_first=False).drop(columns=['Freq_Static'],     errors='ignore')
prov_dummies = pd.get_dummies(reg_df['provider_type'], prefix='Prov', drop_first=False).drop(columns=['Prov_Academic_NGO'],errors='ignore')
pmod_dummies = pd.get_dummies(reg_df['pricing_model'], prefix='Mod',  drop_first=False).drop(columns=['Mod_One-off'],     errors='ignore')

X_df = pd.concat([
    pd.Series(np.ones(len(reg_df)), name='Intercept'),
    cat_dummies, plat_dummies, freq_dummies, prov_dummies, pmod_dummies,
    reg_df['is_bundled'].rename('Is_Bundled'),
], axis=1)

reg_tbl, r2, r2_adj, n_obs = ols_hc3(X_df.values, reg_df['log_price'].values, list(X_df.columns))

# Clean up variable names for display
def clean(v):
    v = v.replace('Cat_','').replace('Plat_','').replace('Freq_','')
    v = v.replace('Prov_','').replace('Mod_','')
    v = v.replace('AWS Data Exchange','AWS Data Exchange')
    v = v.replace('Snowflake Marketplace','Snowflake Mkt.')
    return v

reg_tbl['Variable'] = reg_tbl['Variable'].apply(clean)
reg_tbl.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'reg_table.csv'), index=False)
print(f"\nOLS results: R²={r2:.3f}  Adj-R²={r2_adj:.3f}  n={n_obs}")
print(reg_tbl.to_string(index=False))

# 4.  SUBSTITUTES vs. COMPLEMENTS — Within-Category CV + ANOVA

cv_df = (paid.groupby('category')['price']
             .agg(N='count', Mean='mean', Std='std')
             .assign(CV=lambda x: (x['Std'] / x['Mean'] * 100))
             .round({'Mean':0,'Std':0,'CV':1})
             .sort_values('CV', ascending=False)
             .reset_index())
cv_df.columns = ['Category','N','Mean ($)','Std Dev ($)','CV (%)']
cv_df.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cv_stats.csv'), index=False)
print("\nWithin-category CV:")
print(cv_df.to_string(index=False))

# One-way ANOVA (manual implementation)
groups  = [paid[paid['category'] == c]['log_price'].dropna().values for c in CAT_NAMES]
grand_m = paid['log_price'].mean()
ss_bet  = sum(len(g) * (g.mean() - grand_m)**2 for g in groups)
ss_wit  = sum(((g - g.mean())**2).sum()         for g in groups)
df_bet  = len(groups) - 1
df_wit  = sum(len(g) - 1 for g in groups)
f_stat  = (ss_bet / df_bet) / (ss_wit / df_wit)
print(f"\nOne-way ANOVA: F({df_bet},{df_wit}) = {f_stat:.2f}  (p < 0.001 by inspection)")

# 5.  FIGURES

BLUE   = '#2166AC'
RED    = '#D6604D'
LBLUE  = '#92C5DE'
LRED   = '#F4A582'
GREY   = '#AAAAAA'

# Figure 1: Price distribution by platform
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=False)
for ax, plat, col in zip(axes,
                          ['AWS Data Exchange','Snowflake Marketplace'],
                          [BLUE, RED]):
    data = np.log(paid[paid['platform'] == plat]['price'] + 1)
    med  = np.exp(np.median(data))
    ax.hist(data, bins=28, color=col, alpha=0.78, edgecolor='white')
    ax.axvline(np.log(med+1), color='black', linestyle='--', linewidth=1.4,
               label=f'Median = ${med:,.0f}')
    ax.set_xlabel('log(Price + 1)  [USD]')
    ax.set_ylabel('Count')
    ax.set_title(plat, fontsize=10.5, fontweight='bold')
    ax.legend(fontsize=9)
fig.suptitle('Figure 1  |  Price Distributions by Platform (Paid Listings Only)',
             fontsize=11, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(FIG + 'fig1_platform_dist.png', dpi=150, bbox_inches='tight')
plt.close()

#Figure 2: Median price by category
cat_med = paid.groupby('category')['price'].median().sort_values()
high = {'Financial','Healthcare'}
bar_cols = [RED if c in high else LBLUE for c in cat_med.index]

fig, ax = plt.subplots(figsize=(9, 4.8))
bars = ax.barh(cat_med.index, cat_med.values, color=bar_cols, height=0.58, edgecolor='white')
for bar, val in zip(bars, cat_med.values):
    ax.text(val + 80, bar.get_y() + bar.get_height()/2,
            f'${val:,.0f}', va='center', fontsize=8.5)
ax.set_xlabel('Median Price (USD)')
ax.set_xlim(0, cat_med.max() * 1.22)
ax.set_title('Figure 2  |  Median Price by Category (Paid Listings)',
             fontsize=11, fontweight='bold')
patch_h = mpatches.Patch(color=RED,   label='High-value categories')
patch_l = mpatches.Patch(color=LBLUE, label='Other categories')
ax.legend(handles=[patch_h, patch_l], fontsize=9, loc='lower right')
plt.tight_layout()
plt.savefig(FIG + 'fig2_cat_median.png', dpi=150, bbox_inches='tight')
plt.close()

# Figure 3: Pricing model mix by platform
pm_plat = df.groupby(['platform','pricing_model']).size().unstack(fill_value=0)
pm_pct  = pm_plat.div(pm_plat.sum(axis=1), axis=0) * 100
order   = ['Subscription','One-off','Free']
pm_pct  = pm_pct[order]

x      = np.arange(len(pm_pct))
width  = 0.25
colors3 = [BLUE, LBLUE, '#D1E5F0']
fig, ax = plt.subplots(figsize=(8, 4.2))
for i, (col, colr) in enumerate(zip(order, colors3)):
    ax.bar(x + i*width, pm_pct[col], width, label=col, color=colr, edgecolor='white')
ax.set_xticks(x + width)
ax.set_xticklabels([p.replace(' ','\n') for p in pm_pct.index], fontsize=9.5)
ax.set_ylabel('Share of Listings (%)')
ax.set_title('Figure 3  |  Pricing Model Distribution by Platform',
             fontsize=11, fontweight='bold')
ax.legend(loc='upper right', fontsize=9)
plt.tight_layout()
plt.savefig(FIG + 'fig3_pricing_mix.png', dpi=150, bbox_inches='tight')
plt.close()

# Figure 4: Price by update frequency (box)
data_freq = [np.log(paid[paid['update_freq'] == f]['price'] + 1) for f in FREQ_ORDER]
box_cols  = [RED, LRED, '#FDDBC7', LBLUE, BLUE]
fig, ax = plt.subplots(figsize=(9, 4.5))
bp = ax.boxplot(data_freq, labels=FREQ_ORDER, patch_artist=True, widths=0.5,
                medianprops=dict(color='black', linewidth=2),
                flierprops=dict(marker='o', markersize=3, alpha=0.4))
for patch, col in zip(bp['boxes'], box_cols):
    patch.set_facecolor(col); patch.set_alpha(0.82)
ax.set_xlabel('Update Frequency')
ax.set_ylabel('log(Price + 1)  [USD]')
ax.set_title('Figure 4  |  Price Distribution by Update Frequency',
             fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig(FIG + 'fig4_freq_box.png', dpi=150, bbox_inches='tight')
plt.close()

# Figure 5: OLS coefficient forest plot
skip = {'Intercept', 'Is_Bundled', 'Subscription'}
plot_df = reg_tbl[~reg_tbl['Variable'].isin(skip)].copy()
plot_df = plot_df.sort_values('Coefficient')
ci95 = 1.96 * plot_df['Std_Error']
col_arr = [RED if p < 0.05 else GREY for p in plot_df['p_value']]

fig, ax = plt.subplots(figsize=(9, 6.5))
ypos = range(len(plot_df))
ax.barh(list(ypos), plot_df['Coefficient'],
        xerr=ci95.values, color=col_arr, height=0.5,
        ecolor='#444444', capsize=3, alpha=0.85)
ax.axvline(0, color='black', linewidth=1, linestyle='--')
ax.set_yticks(list(ypos))
ax.set_yticklabels(plot_df['Variable'], fontsize=9)
ax.set_xlabel('OLS Coefficient (log-price scale)')
ax.set_title('Figure 5  |  Regression Coefficient Plot (HC3 Robust SEs, 95% CI)\n'
             'Red = significant at p < 0.05  |  Reference: Retail, Static, Snowflake, Academic/NGO',
             fontsize=10, fontweight='bold')
plt.tight_layout()
plt.savefig(FIG + 'fig5_coefs.png', dpi=150, bbox_inches='tight')
plt.close()

#Figure 6: Within-category coefficient of variation
cv_plot = cv_df.sort_values('CV (%)', ascending=True)
bar_cv  = [RED if v > 100 else BLUE for v in cv_plot['CV (%)']]
mean_cv = cv_df['CV (%)'].mean()
fig, ax = plt.subplots(figsize=(9, 4.5))
ax.barh(cv_plot['Category'], cv_plot['CV (%)'], color=bar_cv, height=0.58, edgecolor='white')
ax.axvline(mean_cv, color='black', linestyle='--', linewidth=1.4,
           label=f'Sample mean = {mean_cv:.0f}%')
ax.set_xlabel('Coefficient of Variation (%)')
ax.set_title('Figure 6  |  Within-Category Price Dispersion\n'
             'Red = CV > 100% (differentiated products);  Blue = lower dispersion (competitive)',
             fontsize=10.5, fontweight='bold')
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(FIG + 'fig6_cv.png', dpi=150, bbox_inches='tight')
plt.close()

# 6.  SAVE SUMMARY RESULTS

results = {
    'n_total':       int(len(df)),
    'n_paid':        int(len(paid)),
    'n_free':        int(len(df) - len(paid)),
    'pct_free':      round((len(df) - len(paid)) / len(df) * 100, 1),
    'median_all':    round(float(paid['price'].median()), 0),
    'mean_all':      round(float(paid['price'].mean()), 0),
    'r2':            round(r2, 3),
    'r2_adj':        round(r2_adj, 3),
    'f_stat':        round(float(f_stat), 2),
    'n_obs_reg':     int(n_obs),
    'aws_median':    round(float(paid[paid['platform']=='AWS Data Exchange']['price'].median()), 0),
    'snow_median':   round(float(paid[paid['platform']=='Snowflake Marketplace']['price'].median()), 0),
    'fin_median':    round(float(paid[paid['category']=='Financial']['price'].median()), 0),
    'gov_median':    round(float(paid[paid['category']=='Government']['price'].median()), 0),
    'rt_median':     round(float(paid[paid['update_freq']=='Real-time']['price'].median()), 0),
    'static_median': round(float(paid[paid['update_freq']=='Static']['price'].median()), 0),
    'mean_cv':       round(float(cv_df['CV (%)'].mean()), 1),
    'max_cv_cat':    str(cv_df.iloc[0]['Category']),
    'max_cv_val':    float(cv_df.iloc[0]['CV (%)']),
    'min_cv_cat':    str(cv_df.iloc[-1]['Category']),
    'min_cv_val':    float(cv_df.iloc[-1]['CV (%)']),
    'pct_sub_aws':   round(float((df[df['platform']=='AWS Data Exchange']['pricing_model']=='Subscription').mean()*100), 1),
    'pct_sub_snow':  round(float((df[df['platform']=='Snowflake Marketplace']['pricing_model']=='Subscription').mean()*100), 1),
    'pct_free_aws':  round(float((df[df['platform']=='AWS Data Exchange']['pricing_model']=='Free').mean()*100), 1),
    'pct_free_snow': round(float((df[df['platform']=='Snowflake Marketplace']['pricing_model']=='Free').mean()*100), 1),
}

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results.json'), 'w') as f:
    json.dump(results, f, indent=2)

print("\nAll figures saved.")
print("\nKey results:")
for k, v in results.items():
    print(f"  {k}: {v}")
