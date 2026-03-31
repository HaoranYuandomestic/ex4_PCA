import numpy as np
import matplotlib.pyplot as plt


class StandardScaler:
	def __init__(self):
		self.mean_ = None
		self.std_ = None

	def fit(self, x):
		self.mean_ = np.mean(x, axis=0)
		self.std_ = np.std(x, axis=0)
		self.std_[self.std_ == 0] = 1.0
		return self

	def transform(self, x):
		return (x - self.mean_) / self.std_

	def inverse_transform(self, x):
		return x * self.std_ + self.mean_

	def fit_transform(self, x):
		return self.fit(x).transform(x)


class PCAFromScratch:
	def __init__(self, n_components=2):
		self.n_components = n_components
		self.mean_ = None
		self.components_ = None
		self.eigenvalues_ = None
		self.explained_variance_ratio_ = None

	def fit(self, x):
		self.mean_ = np.mean(x, axis=0)
		x_centered = x - self.mean_

		cov = np.dot(x_centered.T, x_centered) / (x_centered.shape[0] - 1)
		eigenvalues, eigenvectors = np.linalg.eigh(cov)

		sorted_indices = np.argsort(eigenvalues)[::-1]
		eigenvalues = eigenvalues[sorted_indices]
		eigenvectors = eigenvectors[:, sorted_indices]

		self.eigenvalues_ = eigenvalues
		self.components_ = eigenvectors[:, : self.n_components]
		self.explained_variance_ratio_ = eigenvalues / np.sum(eigenvalues)
		return self

	def transform(self, x):
		x_centered = x - self.mean_
		return np.dot(x_centered, self.components_)

	def inverse_transform(self, z):
		return np.dot(z, self.components_.T) + self.mean_

	def fit_transform(self, x):
		self.fit(x)
		return self.transform(x)


def generate_noisy_dataset(n_samples=200, random_state=7):
	rng = np.random.default_rng(random_state)

	z1 = rng.normal(0, 1.0, size=n_samples)
	z2 = rng.normal(0, 0.85, size=n_samples)

	x_clean = np.column_stack([
		2.2 * z1 + 0.3 * z2,
		-1.4 * z1 + 1.7 * z2,
		0.8 * z1 - 1.1 * z2,
		1.6 * z1 + 0.5 * z2,
	])

	noise = rng.normal(0, 0.65, size=x_clean.shape)
	x_noisy = x_clean + noise

	return x_clean, x_noisy


def mse(a, b):
	return float(np.mean((a - b) ** 2))


def feature_correlation(clean, target):
	cor_list = []
	for j in range(clean.shape[1]):
		corr = np.corrcoef(clean[:, j], target[:, j])[0, 1]
		cor_list.append(corr)
	return np.array(cor_list)


def plot_feature_scatter(clean, noisy, denoised):
	fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))

	axes[0].scatter(clean[:, 0], clean[:, 1], s=22, alpha=0.75, c="tab:green", edgecolors="none")
	axes[0].set_title("Clean Data (F1 vs F2)")
	axes[0].set_xlabel("Feature 1")
	axes[0].set_ylabel("Feature 2")
	axes[0].grid(alpha=0.25)

	axes[1].scatter(noisy[:, 0], noisy[:, 1], s=22, alpha=0.75, c="tab:gray", edgecolors="none")
	axes[1].set_title("Noisy Data (F1 vs F2)")
	axes[1].set_xlabel("Feature 1")
	axes[1].set_ylabel("Feature 2")
	axes[1].grid(alpha=0.25)

	axes[2].scatter(denoised[:, 0], denoised[:, 1], s=22, alpha=0.75, c="tab:blue", edgecolors="none")
	axes[2].set_title("Denoised by PCA (F1 vs F2)")
	axes[2].set_xlabel("Feature 1")
	axes[2].set_ylabel("Feature 2")
	axes[2].grid(alpha=0.25)

	plt.tight_layout()
	plt.savefig("data_compare.png", dpi=160)
	plt.show()


def plot_variance_curve(explained_variance_ratio):
	idx = np.arange(1, len(explained_variance_ratio) + 1)
	cumulative = np.cumsum(explained_variance_ratio)

	plt.figure(figsize=(7.3, 4.8))
	plt.bar(idx, explained_variance_ratio, alpha=0.7, color="tab:orange", label="Single ratio")
	plt.plot(idx, cumulative, marker="o", color="tab:red", linewidth=2, label="Cumulative ratio")
	plt.xticks(idx)
	plt.ylim(0, 1.05)
	plt.xlabel("Principal Component")
	plt.ylabel("Explained Variance Ratio")
	plt.title("Explained Variance by PCA Components")
	plt.grid(alpha=0.25)
	plt.legend()
	plt.tight_layout()
	plt.savefig("variance_curve.png", dpi=160)
	plt.show()


def plot_correlation_compare(corr_noisy, corr_denoised):
	features = np.arange(1, len(corr_noisy) + 1)
	width = 0.34

	plt.figure(figsize=(7.6, 4.8))
	plt.bar(features - width / 2, corr_noisy, width=width, color="tab:gray", label="Clean vs Noisy")
	plt.bar(features + width / 2, corr_denoised, width=width, color="tab:blue", label="Clean vs Denoised")
	plt.xticks(features, [f"F{i}" for i in features])
	plt.ylim(0, 1.05)
	plt.xlabel("Feature")
	plt.ylabel("Correlation Coefficient")
	plt.title("Feature-wise Correlation Comparison")
	plt.grid(axis="y", alpha=0.25)
	plt.legend()
	plt.tight_layout()
	plt.savefig("corr_compare.png", dpi=160)
	plt.show()


def main():
	# 1) Generate 4-D clean data and noisy data.
	x_clean, x_noisy = generate_noisy_dataset(n_samples=200, random_state=7)

	# 2) Standardize noisy data, then run from-scratch PCA.
	scaler = StandardScaler()
	x_noisy_std = scaler.fit_transform(x_noisy)

	pca = PCAFromScratch(n_components=2)
	z = pca.fit_transform(x_noisy_std)
	x_recon_std = pca.inverse_transform(z)
	x_denoised = scaler.inverse_transform(x_recon_std)

	# 3) Evaluate denoising effect.
	mse_noisy = mse(x_clean, x_noisy)
	mse_denoised = mse(x_clean, x_denoised)

	corr_noisy = feature_correlation(x_clean, x_noisy)
	corr_denoised = feature_correlation(x_clean, x_denoised)

	print("==== From-scratch PCA Denoising (No sklearn PCA) ====")
	print(f"Samples: {x_clean.shape[0]}, Features: {x_clean.shape[1]}")
	print(f"MSE(clean, noisy): {mse_noisy:.6f}")
	print(f"MSE(clean, denoised): {mse_denoised:.6f}")
	print(f"MSE improvement: {mse_noisy - mse_denoised:.6f}")
	print("Explained variance ratio:")
	print(pca.explained_variance_ratio_)
	print("Feature-wise correlation (clean vs noisy):")
	print(corr_noisy)
	print("Feature-wise correlation (clean vs denoised):")
	print(corr_denoised)

	# 4) Visualization.
	plot_feature_scatter(x_clean, x_noisy, x_denoised)
	plot_variance_curve(pca.explained_variance_ratio_)
	plot_correlation_compare(corr_noisy, corr_denoised)


if __name__ == "__main__":
	main()