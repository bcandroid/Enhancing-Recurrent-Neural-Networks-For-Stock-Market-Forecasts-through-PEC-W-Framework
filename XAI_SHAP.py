!pip install shap
import shap
import matplotlib.pyplot as plt
shap.initjs()

explainer = shap.GradientExplainer(model, X_train)
shap_values = explainer.shap_values(X_test)

print(type(shap_values))


# Flatten SHAP values if needed for summary and beeswarm plots
shap_values_flat = shap_values.reshape(shap_values.shape[0], shap_values.shape[1])  # Shape: (124, 16)
X_test_flat = X_test.reshape(X_test.shape[0], X_test.shape[1])  # Shape: (124, 16)

# Summary plot
shap.summary_plot(shap_values_flat, X_test_flat, feature_names=['Feature_' + str(i) for i in range(X_test_flat.shape[1])])

import shap
import matplotlib.pyplot as plt

# Convert SHAP values to shap.Explanation objects
explanation = shap.Explanation(values=shap_values_flat, data=X_test_flat, feature_names=['Feature_' + str(i) for i in range(X_test_flat.shape[1])])
# Summary plot
shap.summary_plot(explanation)

# Bar plot
shap.plots.bar(explanation)

# Beeswarm plot
shap.plots.beeswarm(explanation)

# Violin plot
shap.summary_plot(explanation, plot_type='violin')

# Convert SHAP values to shap.Explanation object for a single instance
def create_explanation(shap_values_instance, data_instance, feature_names):
    return shap.Explanation(
        values=shap_values_instance,
        base_values=0,  # Adjust this if needed
        data=data_instance,
        feature_names=feature_names
    )

# Prepare Explanation for the first instance
explanation_instance = create_explanation(shap_values_flat[0], X_test_flat[0], ['Feature_' + str(i) for i in range(X_test_flat.shape[1])])
# Summary plot
shap.summary_plot(explanation)

# Bar plot
shap.plots.bar(explanation)

# Beeswarm plot
shap.plots.beeswarm(explanation)

# Violin plot
shap.summary_plot(explanation, plot_type='violin')

# Waterfall plot (for the first instance)
shap.plots.waterfall(explanation_instance)

# Force plot (for the first instance)
shap.plots.force(explanation_instance)
