#%%
import matplotlib.pyplot as plt

sample_indicator = "number_of_newborns_weighed_at_birth"
hmis_data.filter(pl.col("subdistrict_code") == 3460).select(["date", sample_indicator]).sort("date").to_pandas().plot(x="date", y=sample_indicator)
plt.show()

# %%


fig = px.line(hmis_data.filter(pl.col("subdistrict_code") == 3460).sort("date"), 
                x="date", y="number_of_newborns_weighed_at_birth")
fig.show()

# %%
import matplotlib.pyplot as plt
import polars as pl
import os

subdistricts = [660, 361, 3462]

for indicator in indicator_columns:
    fig, ax = plt.subplots(figsize=(10, 6))  # Create a single Matplotlib figure
    
    for subdistrict in subdistricts:
        # Filter data for the current subdistrict
        subdistrict_data = hmis_data.filter(pl.col("subdistrict_code") == subdistrict).sort("date")
        
        # Plot the data
        ax.plot(
            subdistrict_data["date"],
            subdistrict_data[indicator],
            label=f"Subdistrict {subdistrict}"
        )
    
    # Customize the plot
    ax.set_title(f"{indicator} Trends Across Subdistricts", fontsize=16)
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel(indicator, fontsize=12)
    ax.legend(title="Subdistricts", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.6)
    
    # Save the plot as a PNG image
    save_path = f"{REPORTS_PATH}/{indicator}_combined.png"
    plt.tight_layout()
    plt.savefig(save_path, format="png", dpi=300)
    plt.close(fig)  # Close the figure to free memory

print("All images saved successfully.")