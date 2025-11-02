from __future__ import annotations

import os
from pathlib import Path
import math

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns

from scipy import stats
from scipy.stats import chi2_contingency, f_oneway, kruskal, pointbiserialr
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder


def safe_mkdir(p: Path) -> None:
	p.mkdir(parents=True, exist_ok=True)


def analyze_csv(csv_path: Path, out_dir: Path, sample_rows: int = 5000, chunksize: int = 100_000):
	csv_path = csv_path.resolve()
	out_dir = out_dir.resolve()
	safe_mkdir(out_dir)

	summary_lines = []
	summary_lines.append(f"File: {csv_path}")

	try:
		sample = pd.read_csv(csv_path, nrows=sample_rows)
	except Exception as e:
		summary_lines.append(f"Failed to read sample rows: {e}")
		try:
			sample = pd.read_csv(csv_path, nrows=sample_rows, low_memory=False)
		except Exception as e2:
			summary_lines.append(f"Second attempt failed: {e2}")
			sample = None

	if sample is None:
		summary_lines.append("Unable to read any rows from CSV. Exiting.")
		(out_dir / "features_analysis.txt").write_text("\n".join(summary_lines))
		print("Wrote minimal summary to", out_dir / "features_analysis.txt")
		return

	summary_lines.append(f"Sample rows: {len(sample)}")
	summary_lines.append("Columns and dtypes (from sample):")
	summary_lines.extend([f"  {c}: {str(t)}" for c, t in sample.dtypes.items()])

	miss = sample.isnull().sum()
	miss_pct = miss / len(sample) * 100
	summary_lines.append("\nMissing values (sample):")
	summary_lines.extend([f"  {c}: {miss[c]} ({miss_pct[c]:.2f}%)" for c in sample.columns if miss[c] > 0])

	num_cols = sample.select_dtypes(include=[np.number]).columns.tolist()
	summary_lines.append(f"\nNumeric columns (sample): {num_cols}")

	try:
		file_size = os.path.getsize(csv_path)
	except Exception:
		file_size = None

	# large = 50 MB
	# chunks if large
	LARGE_BYTES = 50 * 1024 * 1024

	if file_size is not None and file_size > LARGE_BYTES:
		summary_lines.append(f"Detected large file ({file_size / (1024**2):.1f} MiB) - using chunked aggregation for numeric columns")
		agg_count = {}
		agg_sum = {}
		agg_sum_sq = {}
		agg_min = {}
		agg_max = {}

		for chunk in pd.read_csv(csv_path, chunksize=chunksize):
			nums = chunk[num_cols]
			for col in num_cols:
				col_ser = nums[col].dropna()
				c = int(col_ser.count())
				s = float(col_ser.sum()) if c else 0.0
				ssq = float((col_ser ** 2).sum()) if c else 0.0
				mn = float(col_ser.min()) if c else math.nan
				mx = float(col_ser.max()) if c else math.nan

				agg_count[col] = agg_count.get(col, 0) + c
				agg_sum[col] = agg_sum.get(col, 0.0) + s
				agg_sum_sq[col] = agg_sum_sq.get(col, 0.0) + ssq
				agg_min[col] = col_ser.min() if (col in agg_min and not math.isnan(agg_min[col])) and not math.isnan(mn) else mn if col not in agg_min else min(agg_min[col], mn)
				agg_max[col] = col_ser.max() if (col in agg_max and not math.isnan(agg_max[col])) and not math.isnan(mx) else mx if col not in agg_max else max(agg_max[col], mx)

		summary_lines.append("\nAggregated numeric summary:")
		for col in num_cols:
			n = agg_count.get(col, 0)
			if n == 0:
				summary_lines.append(f"  {col}: no non-null values")
				continue
			s = agg_sum[col]
			ssq = agg_sum_sq[col]
			mean = s / n

			var = (ssq - (s * s) / n) / max(n - 1, 1)
			std = math.sqrt(var) if var >= 0 else float('nan')
			mn = agg_min.get(col, float('nan'))
			mx = agg_max.get(col, float('nan'))
			summary_lines.append(f"  {col}: n={n}, mean={mean:.4g}, std={std:.4g}, min={mn:.4g}, max={mx:.4g}")

		sample_for_plots = pd.read_csv(csv_path, nrows=sample_rows)
	else:
		summary_lines.append("Reading full CSV into memory for analysis (file not too large or unknown size)")
		df = pd.read_csv(csv_path)
		sample_for_plots = df.sample(n=min(len(df), sample_rows), random_state=42)

		summary_lines.append("\nFull numeric describe:")
		desc = df.describe().round(4).T
		summary_lines.extend([f"  {idx}: {row.to_dict()}" for idx, row in desc.iterrows()])

	if len(num_cols) >= 2:
		corr = sample_for_plots[num_cols].corr()
		summary_lines.append("\nCorrelation matrix (sample):")

		pairs = []
		for i, a in enumerate(num_cols):
			for b in num_cols[i+1:]:
				val = corr.loc[a, b]
				pairs.append((abs(val), a, b, val))
		pairs.sort(reverse=True)
		top_pairs = pairs[:10]
		for absval, a, b, val in top_pairs:
			summary_lines.append(f"  {a} <-> {b}: r={val:.4f}")

		plt.figure(figsize=(12, 10))  # Larger heatmap
		sns.heatmap(corr, annot=False, cmap='coolwarm', center=0)
		plt.title('Correlation (sample)')
		plt.tight_layout()
		heatmap_file = out_dir / 'features_correlation_heatmap.png'
		plt.savefig(heatmap_file, dpi=150)
		plt.close()
		summary_lines.append(f"Saved correlation heatmap to {heatmap_file}")

	miss_pct_full = sample_for_plots.isnull().mean() * 100
	miss_pct_full = miss_pct_full.sort_values(ascending=False)
	if len(miss_pct_full) > 0 and miss_pct_full.max() > 0:
		plt.figure(figsize=(12, max(6, len(miss_pct_full) * 0.3)))
		miss_pct_full.plot.bar()
		plt.ylabel('Percent missing')
		plt.title('Missing values (sample)')
		plt.tight_layout()
		miss_file = out_dir / 'features_missing.png'
		plt.savefig(miss_file, dpi=150)
		plt.close()
		summary_lines.append(f"Saved missing-values bar chart to {miss_file}")

	if len(num_cols) > 0:
		cols_to_plot = num_cols[:6]
		plt.figure(figsize=(12, 3.5 * len(cols_to_plot)))
		for i, col in enumerate(cols_to_plot, start=1):
			plt.subplot(len(cols_to_plot), 1, i)
			sample_for_plots[col].dropna().hist(bins=40)
			plt.title(col)
		plt.tight_layout()
		hist_file = out_dir / 'features_histograms.png'
		plt.savefig(hist_file, dpi=150)
		plt.close()
		summary_lines.append(f"Saved histograms to {hist_file}")

	unique_counts = sample_for_plots.nunique()
	unique_counts = unique_counts.sort_values(ascending=False)

	plt.figure(figsize=(12, 8))  # Larger chart for unique value counts
	unique_counts.plot(kind='bar', stacked=True, color='skyblue')
	plt.ylabel('Unique Value Counts')
	plt.title('Distribution of Unique Value Counts per Feature')
	plt.tight_layout()

	unique_counts_file = out_dir / 'unique_value_counts_stacked_chart.png'
	plt.savefig(unique_counts_file, dpi=150)
	plt.close()
	summary_lines.append(f"Saved unique value counts stacked chart to {unique_counts_file}")
	summary_lines.append("STATISTICAL ANALYSIS: FEATURES INFLUENCING 'voice'")
 
	voice_counts = sample_for_plots['voice'].value_counts()
	voice_pcts = sample_for_plots['voice'].value_counts(normalize=True) * 100
	summary_lines.append(f"\nVoice distribution (sample):")
	for cat in voice_counts.index:
		summary_lines.append(f"  {cat}: {voice_counts[cat]} ({voice_pcts[cat]:.2f}%)")

	categorical_features = sample_for_plots.select_dtypes(include=['object', 'bool']).columns.tolist()
	numeric_features = sample_for_plots.select_dtypes(include=[np.number]).columns.tolist()
	
	exclude_cols = ['voice', 'index', 'verb_form', 'verb_lemma', 'verb_embedding', 
					'topic_vector', 'collocates', 'subject_info', 'obj_info',
					'deprel_counts', 'conj_children_ids', 'conj_sibling_ids',
					'srl_predicate', 'subcat_frame']
	categorical_features = [c for c in categorical_features if c not in exclude_cols]
	numeric_features = [c for c in numeric_features if c not in exclude_cols]

	summary_lines.append(f"\nAnalyzing {len(categorical_features)} categorical and {len(numeric_features)} numeric features")

	if len(categorical_features) > 0:
		summary_lines.append("\n--- Chi-Square Tests (Categorical Features vs Voice) ---")
		chi_square_results = []
		
		for feature in categorical_features:
			contingency = pd.crosstab(sample_for_plots[feature].fillna('_MISSING_'), 
										sample_for_plots['voice'])
			
			if contingency.shape[0] > 1 and contingency.shape[1] > 1:
				chi2, p_value, dof, expected = chi2_contingency(contingency)
				
				n = contingency.sum().sum()
				min_dim = min(contingency.shape[0] - 1, contingency.shape[1] - 1)
				cramers_v = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else 0
				
				chi_square_results.append({
					'feature': feature,
					'chi2': chi2,
					'p_value': p_value,
					'cramers_v': cramers_v,
					'n_categories': contingency.shape[0]
				})
		
		chi_square_results.sort(key=lambda x: x['cramers_v'], reverse=True)
		
		summary_lines.append("\nTop 20 Categorical Features by Effect Size (Cramér's V):")
		for i, result in enumerate(chi_square_results[:20], 1):
			sig = "***" if result['p_value'] < 0.001 else "**" if result['p_value'] < 0.01 else "*" if result['p_value'] < 0.05 else ""
			summary_lines.append(f"  {i}. {result['feature']}: V={result['cramers_v']:.4f}, χ²={result['chi2']:.2f}, p={result['p_value']:.4e} {sig}")

		if chi_square_results:
			chi_df = pd.DataFrame(chi_square_results)
			chi_csv = out_dir / 'voice_categorical_chisquare.csv'
			chi_df.to_csv(chi_csv, index=False)
			summary_lines.append(f"\nSaved full chi-square results to {chi_csv}")

	if len(numeric_features) > 0:
		summary_lines.append("\n--- Statistical Tests (Numeric Features vs Voice) ---")
		numeric_results = []
		
		voice_groups = sample_for_plots['voice'].dropna().unique()
		
		for feature in numeric_features:
			groups = []
			for voice_cat in voice_groups:
				group_data = sample_for_plots[sample_for_plots['voice'] == voice_cat][feature].dropna()
				if len(group_data) > 0:
					groups.append(group_data)
			
			if len(groups) >= 2:
				h_stat, p_value_kw = kruskal(*groups)
				
				n_total = sum(len(g) for g in groups)
				eta_squared = h_stat / (n_total - 1) if n_total > 1 else 0
				
				means_by_voice = {}
				for voice_cat in voice_groups:
					mean_val = sample_for_plots[sample_for_plots['voice'] == voice_cat][feature].mean()
					means_by_voice[voice_cat] = mean_val
				
				numeric_results.append({
					'feature': feature,
					'h_statistic': h_stat,
					'p_value': p_value_kw,
					'eta_squared': eta_squared,
					**{f'mean_{k}': v for k, v in means_by_voice.items()}
				})

		
		numeric_results.sort(key=lambda x: x['eta_squared'], reverse=True)
		
		summary_lines.append("\nTop 20 Numeric Features by Effect Size (Eta-Squared):")
		for i, result in enumerate(numeric_results[:20], 1):
			sig = "***" if result['p_value'] < 0.001 else "**" if result['p_value'] < 0.01 else "*" if result['p_value'] < 0.05 else ""
			means_str = ", ".join([f"{k.replace('mean_', '')}={v:.3f}" for k, v in result.items() if k.startswith('mean_')])
			summary_lines.append(f"  {i}. {result['feature']}: η²={result['eta_squared']:.4f}, H={result['h_statistic']:.2f}, p={result['p_value']:.4e} {sig}")
			summary_lines.append(f"      Means: {means_str}")

		if numeric_results:
			num_df = pd.DataFrame(numeric_results)
			num_csv = out_dir / 'voice_numeric_tests.csv'
			num_df.to_csv(num_csv, index=False)
			summary_lines.append(f"\nSaved full numeric test results to {num_csv}")

	summary_lines.append("\n--- Mutual Information (All Features) ---")

	mi_data = sample_for_plots.copy()
	
	le_target = LabelEncoder()
	y = le_target.fit_transform(mi_data['voice'].fillna('_MISSING_'))
	
	X_list = []
	feature_names = []
	
	for feat in numeric_features:
		feat_data = mi_data[feat].fillna(mi_data[feat].median())
		if feat_data.isna().any():
			feat_data = feat_data.fillna(0)
		X_list.append(feat_data.values)
		feature_names.append(feat)
	
	for feat in categorical_features[:50]:
		try:
			le_feat = LabelEncoder()
			encoded = le_feat.fit_transform(mi_data[feat].fillna('_MISSING_').astype(str))
			X_list.append(encoded)
			feature_names.append(feat)
		except:
			pass
	
	if len(X_list) > 0:
		X = np.column_stack(X_list)
		
		if np.isnan(X).any():
			summary_lines.append("Warning: NaN values found in feature matrix, replacing with 0")
			X = np.nan_to_num(X, nan=0.0)
		
		mi_scores = mutual_info_classif(X, y, random_state=42, n_neighbors=3)
		
		mi_results = list(zip(feature_names, mi_scores))
		mi_results.sort(key=lambda x: x[1], reverse=True)
		
		summary_lines.append("\nTop 30 Features by Mutual Information:")
		for i, (feat, score) in enumerate(mi_results[:30], 1):
			summary_lines.append(f"  {i}. {feat}: MI={score:.4f}")
		
		mi_df = pd.DataFrame(mi_results, columns=['feature', 'mutual_information'])
		mi_csv = out_dir / 'voice_mutual_information.csv'
		mi_df.to_csv(mi_csv, index=False)
		summary_lines.append(f"\nSaved mutual information results to {mi_csv}")
		
		plt.figure(figsize=(12, 10))
		top_mi = mi_df.head(30)
		plt.barh(range(len(top_mi)), top_mi['mutual_information'].values)
		plt.yticks(range(len(top_mi)), top_mi['feature'].values)
		plt.xlabel('Mutual Information Score')
		plt.title('Top 30 Features by Mutual Information with Voice')
		plt.gca().invert_yaxis()
		plt.tight_layout()
		mi_plot = out_dir / 'voice_mutual_information_plot.png'
		plt.savefig(mi_plot, dpi=150)
		plt.close()
		summary_lines.append(f"Saved MI plot to {mi_plot}")
			

	summary_lines.append("\n--- Feature Statistics by Voice Category ---")
	
	for voice_cat in voice_counts.index:
		summary_lines.append(f"\n=== Voice: {voice_cat} ===")
		voice_subset = sample_for_plots[sample_for_plots['voice'] == voice_cat]
		summary_lines.append(f"N = {len(voice_subset)}")
		
		summary_lines.append("\nTop numeric features (by mean):")
		for feat in numeric_features[:10]:
			mean_val = voice_subset[feat].mean()
			std_val = voice_subset[feat].std()
			summary_lines.append(f"  {feat}: mean={mean_val:.4f}, std={std_val:.4f}")
		
		summary_lines.append("\nTop categorical features:")
		for feat in categorical_features[:10]:
			top_vals = voice_subset[feat].value_counts().head(3)
			vals_str = ", ".join([f"{k}={v}" for k, v in top_vals.items()])
			summary_lines.append(f"  {feat}: {vals_str}")

	importance_data = []
	
	if chi_square_results:
		for result in chi_square_results[:15]:
			importance_data.append({
				'feature': result['feature'],
				'importance': result['cramers_v'],
				'type': 'Categorical (Cramér\'s V)'
			})
	
	if numeric_results:
		for result in numeric_results[:15]:
			importance_data.append({
				'feature': result['feature'],
				'importance': result['eta_squared'],
				'type': 'Numeric (η²)'
			})
	
	if importance_data:
		imp_df = pd.DataFrame(importance_data)
		imp_df = imp_df.sort_values('importance', ascending=False).head(30)
		
		plt.figure(figsize=(14, 10))
		colors = {'Categorical (Cramér\'s V)': 'steelblue', 'Numeric (η²)': 'coral'}
		for ftype in imp_df['type'].unique():
			subset = imp_df[imp_df['type'] == ftype]
			plt.barh(subset['feature'], subset['importance'], 
					label=ftype, color=colors.get(ftype, 'gray'), alpha=0.7)
		
		plt.xlabel('Effect Size')
		plt.title('Top Features Influencing Voice (by Effect Size)')
		plt.legend()
		plt.gca().invert_yaxis()
		plt.tight_layout()
		combined_plot = out_dir / 'voice_combined_importance.png'
		plt.savefig(combined_plot, dpi=150)
		plt.close()
		summary_lines.append(f"\nSaved combined importance plot to {combined_plot}")

	summary_lines.append("\n--- Features Most Predictive of Each Voice Category ---")
	
	voice_categories = voice_counts.index.tolist()
	
	for target_voice in voice_categories:
		summary_lines.append(f"\n=== Features Most Predictive of '{target_voice}' Voice ===")
		
		voice_specific_features = []
		
		if numeric_results:
			for result in numeric_results:
				feat = result['feature']
				mean_col = f'mean_{target_voice}'
				if mean_col in result:
					target_mean = result[mean_col]
					other_means = [result[f'mean_{v}'] for v in voice_categories if v != target_voice and f'mean_{v}' in result]
					
					if len(other_means) > 0:
						avg_other = np.mean(other_means)
						if avg_other != 0:
							diff_ratio = (target_mean - avg_other) / (abs(avg_other) + 1e-10)
						else:
							diff_ratio = target_mean
						
						voice_specific_features.append({
							'feature': feat,
							'type': 'numeric',
							'target_mean': target_mean,
							'other_mean': avg_other,
							'difference': target_mean - avg_other,
							'ratio': diff_ratio,
							'effect_size': result['eta_squared']
						})
		
		for feature in categorical_features[:30]:
			try:
				contingency = pd.crosstab(sample_for_plots[feature].fillna('_MISSING_'), 
											sample_for_plots['voice'], normalize='columns')
				
				if target_voice in contingency.columns:
					for cat_val in contingency.index:
						target_prop = contingency.loc[cat_val, target_voice]
						other_props = [contingency.loc[cat_val, v] for v in voice_categories 
										if v != target_voice and v in contingency.columns]
						
						if len(other_props) > 0:
							avg_other_prop = np.mean(other_props)
							diff = target_prop - avg_other_prop
							
							if diff > 0.05:
								effect_size = 0
								for chi_result in chi_square_results:
									if chi_result['feature'] == feature:
										effect_size = chi_result['cramers_v']
										break
								
								voice_specific_features.append({
									'feature': f"{feature}={cat_val}",
									'type': 'categorical',
									'target_prop': target_prop,
									'other_prop': avg_other_prop,
									'difference': diff,
									'ratio': diff,
									'effect_size': effect_size
								})
			except:
				pass
		
		voice_specific_features.sort(key=lambda x: abs(x['difference']) * (1 + x['effect_size']), reverse=True)
		
		summary_lines.append(f"\nTop features predicting '{target_voice}' voice:")
		for i, feat_info in enumerate(voice_specific_features[:20], 1):
			if feat_info['type'] == 'numeric':
				summary_lines.append(
					f"  {i}. {feat_info['feature']}: "
					f"mean in {target_voice}={feat_info['target_mean']:.3f}, "
					f"mean in others={feat_info['other_mean']:.3f}, "
					f"diff={feat_info['difference']:.3f}"
				)
			else:
				summary_lines.append(
					f"  {i}. {feat_info['feature']}: "
					f"{target_voice}={feat_info['target_prop']:.1%}, "
					f"others={feat_info['other_prop']:.1%}, "
					f"diff={feat_info['difference']:.1%}"
				)
		
		if len(voice_specific_features) > 0:
			top_features = voice_specific_features[:20]
			
			fig, ax = plt.subplots(figsize=(14, 10))
			
			y_pos = np.arange(len(top_features))
			differences = [f['difference'] for f in top_features]
			feature_labels = [f['feature'] if len(f['feature']) < 40 else f['feature'][:37] + '...' 
								for f in top_features]
			colors_list = ['steelblue' if f['type'] == 'numeric' else 'coral' 
							for f in top_features]
			
			bars = ax.barh(y_pos, differences, color=colors_list, alpha=0.7)
			ax.set_yticks(y_pos)
			ax.set_yticklabels(feature_labels)
			ax.set_xlabel('Difference from Other Voice Categories')
			ax.set_title(f'Top 20 Features Most Predictive of "{target_voice.upper()}" Voice')
			ax.invert_yaxis()
			
			from matplotlib.patches import Patch
			legend_elements = [
				Patch(facecolor='steelblue', alpha=0.7, label='Numeric'),
				Patch(facecolor='coral', alpha=0.7, label='Categorical')
			]
			ax.legend(handles=legend_elements, loc='lower right')
			
			plt.tight_layout()
			voice_specific_plot = out_dir / f'voice_predictors_{target_voice}.png'
			plt.savefig(voice_specific_plot, dpi=150)
			plt.close()
			summary_lines.append(f"\nSaved {target_voice}-specific predictor plot to {voice_specific_plot}")
			
			voice_df = pd.DataFrame(voice_specific_features[:30])
			voice_csv = out_dir / f'voice_predictors_{target_voice}.csv'
			voice_df.to_csv(voice_csv, index=False)
			summary_lines.append(f"Saved {target_voice}-specific data to {voice_csv}")

	summary_lines.append("\n--- Creating comparative heatmap of feature means by voice ---")
	
	top_numeric = [r['feature'] for r in numeric_results[:25]]
	
	heatmap_data = []
	for feat in top_numeric:
		row = []
		for voice_cat in voice_categories:
			mean_val = sample_for_plots[sample_for_plots['voice'] == voice_cat][feat].mean()
			row.append(mean_val)
		heatmap_data.append(row)
	
	heatmap_df = pd.DataFrame(heatmap_data, columns=voice_categories, index=top_numeric)
	
	heatmap_normalized = heatmap_df.apply(lambda x: (x - x.mean()) / (x.std() + 1e-10), axis=1)
	
	plt.figure(figsize=(10, 14))
	sns.heatmap(heatmap_normalized, annot=False, cmap='RdBu_r', center=0, 
				cbar_kws={'label': 'Z-score'}, linewidths=0.5)
	
	plt.title('Feature Means by Voice Category (Z-scores)\nBlue=Below Average, Red=Above Average')
	plt.xlabel('Voice Category')
	plt.ylabel('Feature')
	plt.tight_layout()
	
	heatmap_file = out_dir / 'voice_feature_means_heatmap.png'
	plt.savefig(heatmap_file, dpi=150)
	plt.close()
	summary_lines.append(f"Saved feature means heatmap to {heatmap_file}")
	

	summary_lines.append("\n" + "="*80)

	out_file = out_dir / 'features_analysis.txt'
	out_file.write_text("\n".join(summary_lines), encoding='utf-8')
	print(f"Wrote analysis summary to: {out_file}")


def main():
	csv_path = Path('./predicate_features.csv')
	out_dir = Path('./Report/figures')

	analyze_csv(csv_path, out_dir, sample_rows=5000)


if __name__ == '__main__':
	main()

