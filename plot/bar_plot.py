import matplotlib.pyplot as plt

# ----------------------------
# 1. Customize these variables
# ----------------------------
models = ["Model A", "Model B", "Model C"]  # Model names
gaps = [2.5, 3.8, 1.2]                     # Gap (%) for each model
chart_title = "Gap (%) by Model"           # Chart title
output_file = "gap_bar_chart.png"          # Optional: save chart as image

# ----------------------------
# 2. Create the bar chart
# ----------------------------
plt.figure(figsize=(8, 5))  # Width, height in inches
bars = plt.bar(models, gaps, color='skyblue')

# Add value labels on top of bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height + 0.05, f'{height:.2f}%', 
             ha='center', va='bottom', fontsize=10)

plt.title(chart_title, fontsize=14)
plt.ylabel("Gap (%)")
plt.ylim(0, max(gaps)*1.2)  # Add some space above the tallest bar
plt.grid(axis='y', linestyle='--', alpha=0.7)

# ----------------------------
# 3. Show and/or save the chart
# ----------------------------
plt.tight_layout()
plt.show()                  # Display chart
# plt.savefig(output_file)  # Uncomment to save as PNG
