import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from scipy.stats import gaussian_kde, pearsonr
from shapely import wkt
from shapely.geometry import Polygon

# --- Utility functions ---


def create_combined_gdf(fseastatecolocs):
    """
    Reads a NetCDF file and creates a GeoDataFrame containing polygon geometries
    and all relevant data (SAR SWH, SWOT SWH, etc.).
    """

    ordered = [0, 1, 3, 2]
    records = []

    # Define the mapping from NetCDF variable names to GeoDataFrame column names
    variable_mapping = {
        "hs_most_likely": "hs_sar",
        "swh_karin_mean": "swh_swot",
        "hs_conf": "hs_conf",
        "swh_karin_med": "swh_karin_med",
        "swh_karin_std": "swh_karin_std",  # selection after filters
        "swh_karin_mean2": "swh_karin_sel",
        "t0m1_most_likely": "t0m1_most_likely",
        "phs0_most_likely": "phs0_most_likely",
    }
    print("nb files", len(fseastatecolocs))
    footprints_subswath_sar = []
    for fseastatecoloc in fseastatecolocs:
        print("fseastatecoloc", fseastatecoloc)
        ds_l2c = xr.open_dataset(fseastatecoloc)
        footprints_subswath_sar.append(ds_l2c.attrs["footprint"])
        corner_lon, corner_lat = ds_l2c["corner_longitude"], ds_l2c["corner_latitude"]
        for i in range(ds_l2c.dims["tile_line"]):
            for j in range(ds_l2c.dims["tile_sample"]):
                lon_corners = corner_lon.isel(
                    tile_line=i, tile_sample=j
                ).values.ravel()[ordered]
                lat_corners = corner_lat.isel(
                    tile_line=i, tile_sample=j
                ).values.ravel()[ordered]
                poly = Polygon(list(zip(lon_corners, lat_corners)))

                record = {"geometry": poly}
                for nc_var, gdf_col in variable_mapping.items():
                    if nc_var == "swh_karin_mean2":
                        nc_var = "swh_karin_mean"
                    # print('nc_var',nc_var,'->',gdf_col)
                    if nc_var in ds_l2c:
                        record[gdf_col] = (
                            ds_l2c[nc_var].isel(tile_line=i, tile_sample=j).item()
                        )
                    else:
                        print(
                            f"Warning: Variable '{nc_var}' not found in the NetCDF file. Filling with NaN."
                        )
                        record[gdf_col] = np.nan
                records.append(record)

    combined_gdf = gpd.GeoDataFrame(records, crs="EPSG:4326")
    # The mask is only for the scatter plot, to ensure both variables are valid
    mask = (
        np.isfinite(combined_gdf["swh_swot"])
        & np.isfinite(combined_gdf["hs_sar"])
        & (combined_gdf["swh_karin_std"] > 0)
        & (combined_gdf["swh_swot"] > 0)
        & (combined_gdf["swh_karin_std"] < 1)
    )

    # Return the full dataframe for maps, and the filtered one for the scatter plot
    return (
        combined_gdf,
        combined_gdf[mask].reset_index(drop=True),
        footprints_subswath_sar,
    )


def calculate_statistics(gdf, x_col="swh_swot", y_col="hs_sar"):
    """Calculates basic statistics."""
    if gdf.empty:
        return {
            "N": 0,
            "Bias": np.nan,
            "RMSE": np.nan,
            "Correlation (R)": np.nan,
            "SI (%)": np.nan,
        }
    x, y = gdf[x_col], gdf[y_col]
    difference = y - x
    stats = {
        "N": len(x),
        "Bias": np.mean(difference),
        "RMSE": np.sqrt(np.mean(difference**2)),
        "Correlation (R)": pearsonr(x, y)[0] if len(x) > 1 else np.nan,
        "SI (%)": (
            (np.sqrt(np.mean(difference**2)) / np.mean(x)) * 100
            if np.mean(x) != 0
            else np.nan
        ),
    }
    return stats


# --- Static plotting functions ---


def plot_map(ax, gdf, column, cmap, norm, title, subswath_polygons):
    """
    Helper function to plot a map of colored polygons on a Cartopy axis.
    """
    ax.set_title(title)
    ax.coastlines("50m", linewidth=0.8)
    ax.add_feature(cfeature.LAND, facecolor="lightgray")
    gl = ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=True,
        linewidth=1,
        color="gray",
        alpha=0.5,
        linestyle="--",
    )
    gl.top_labels = False
    gl.right_labels = False

    # Use GeoPandas' built-in plot function for better performance
    if not gdf.empty:
        gdf.plot(
            column=column,
            cmap=cmap,
            norm=norm,
            ax=ax,
            transform=ccrs.PlateCarree(),
            edgecolor="black",
            linewidth=0.2,
        )

    # Add the SWOT swath outline
    for subswath_polygon in subswath_polygons:
        ax.add_geometries(
            [wkt.loads(subswath_polygon)],
            crs=ccrs.PlateCarree(),
            facecolor="none",
            edgecolor="red",
            linewidth=1.5,
            linestyle="--",
        )

    # Set the map extent with a margin
    if not gdf.empty:
        bounds = gdf.total_bounds
        ax.set_extent(
            [bounds[0] - 0.5, bounds[2] + 0.5, bounds[1] - 0.5, bounds[3] + 0.5],
            crs=ccrs.PlateCarree(),
        )


def create_static_dashboard(data_files, output_filename="dashboard_statique.png"):
    """
    Creates and saves a static 3x3 figure with maps and a scatter plot.
    """
    print("--- STEP 1: Reading and processing data file... ---")
    gdf_all, gdf_filtered, footprints_subswath_sar = create_combined_gdf(data_files)
    print(
        f"--- Data loaded successfully. {len(gdf_filtered)} valid collocation points found. ---"
    )

    # --- Figure setup for a 3x3 grid ---
    fig, axes = plt.subplots(
        nrows=3,
        ncols=3,
        figsize=(22, 18),  # Increased size for more plots
        subplot_kw={"projection": ccrs.PlateCarree()},
    )
    # The last axis is for the scatter plot, so we replace it
    axes[2, 2].remove()
    axes[2, 2] = fig.add_subplot(3, 3, 9)

    # Flatten axes for easy iteration
    ax_list = axes.flatten()

    # --- Color setup for all variables ---
    cmap_swh = plt.get_cmap("viridis")
    norm_swh = Normalize(vmin=0, vmax=8)
    cmap_conf = plt.get_cmap("bwr")
    norm_conf = Normalize(vmin=-1, vmax=1)
    cmap_std = plt.get_cmap("magma")
    norm_std = Normalize(vmin=0, vmax=4)
    cmap_period = plt.get_cmap("plasma")
    norm_period = Normalize(vmin=0, vmax=20)
    # Corrected normalization for Windsea SWH (phs0)
    norm_windsea_swh = Normalize(vmin=0, vmax=8)

    # sar_subswath_polygon = wkt.loads(ds_l2c.attrs['footprint'])

    # --- Plotting all maps ---
    print("--- STEP 2: Creating maps... ---")

    # Define plot configurations for easy looping
    plot_configs = [
        {
            "ax": ax_list[0],
            "col": "hs_sar",
            "cmap": cmap_swh,
            "norm": norm_swh,
            "label": "SWH (m)",
            "title_base": "SAR S-1 IW VV (SWH)",
        },
        {
            "ax": ax_list[1],
            "col": "swh_swot",
            "cmap": cmap_swh,
            "norm": norm_swh,
            "label": "SWH (m)",
            "title_base": "SWOT KarIn (SWH Mean)",
        },
        {
            "ax": ax_list[2],
            "col": "swh_karin_med",
            "cmap": cmap_swh,
            "norm": norm_swh,
            "label": "SWH (m)",
            "title_base": "SWOT KarIn (SWH Median)",
        },
        {
            "ax": ax_list[3],
            "col": "hs_conf",
            "cmap": cmap_conf,
            "norm": norm_conf,
            "label": "Confidence",
            "title_base": "SAR SWH Confidence",
        },
        {
            "ax": ax_list[4],
            "col": "swh_karin_std",
            "cmap": cmap_std,
            "norm": norm_std,
            "label": "Std Dev (m)",
            "title_base": "SWOT KarIn (SWH Std)",
        },
        {
            "ax": ax_list[5],
            "col": "swh_karin_sel",
            "cmap": cmap_swh,
            "norm": None,
            "label": "Hs",
            "title_base": "SWOT KarIn (selection of points after filter.)",
        },
        {
            "ax": ax_list[6],
            "col": "t0m1_most_likely",
            "cmap": cmap_period,
            "norm": norm_period,
            "label": "Period (s)",
            "title_base": "SAR Period (T0m1)",
        },
        # Corrected configuration for phs0_most_likely
        {
            "ax": ax_list[7],
            "col": "phs0_most_likely",
            "cmap": cmap_swh,
            "norm": norm_windsea_swh,
            "label": "SWH (m)",
            "title_base": "SAR Windsea SWH (Phs0)",
        },
    ]

    for config in plot_configs:
        # if config['col']=='swh_karin_std':
        #     print(gdf_all[config['col']])
        if config["col"] == "swh_karin_sel":
            col_data = gdf_filtered["swh_swot"]
            gdf4plot = gdf_filtered
        else:
            col_data = gdf_all[config["col"]]
            gdf4plot = gdf_all

        # Handle dynamic normalization for 'swh_karin_nb'
        norm = config["norm"]
        if config["col"] == "swh_karin_nb" and not col_data.dropna().empty:
            norm = Normalize(vmin=col_data.min(), vmax=col_data.max())

        # Generate title with min/max
        if not col_data.dropna().empty:
            data_min, data_max = col_data.min(), col_data.max()
            title = f"{config['title_base']}\nmin: {data_min:.2f}, max: {data_max:.2f}"
        else:
            title = config["title_base"]

        # Plot the map
        plot_map(
            config["ax"],
            gdf4plot,
            config["col"],
            config["cmap"],
            norm,
            title,
            footprints_subswath_sar,
        )

        # Add colorbar
        if norm is not None:
            fig.colorbar(
                ScalarMappable(norm=norm, cmap=config["cmap"]),
                ax=config["ax"],
                orientation="vertical",
                shrink=0.6,
                label=config["label"],
            )

    # --- Plotting the scatter plot with density ---
    print("--- STEP 3: Creating scatter plot... ---")
    scatter_ax = ax_list[8]
    x, y = gdf_filtered["swh_swot"], gdf_filtered["hs_sar"]

    if not x.empty and not y.empty:
        # Calculate point density
        xy = np.vstack([x, y])
        z = gaussian_kde(xy)(xy)
        # Sort points by density
        idx = z.argsort()
        x, y, z = x.iloc[idx], y.iloc[idx], z[idx]
        scatter = scatter_ax.scatter(x, y, c=z, s=15, cmap="inferno", alpha=0.7)
        fig.colorbar(
            scatter, ax=scatter_ax, orientation="vertical", label="Point Density"
        )
    else:
        # Plot an empty scatter if no data
        scatter_ax.scatter(x, y, alpha=0.6, s=15)

    # Add 1:1 line
    scatter_ax.plot(
        [0, 8], [0, 8], "r--", label="1:1 Line"
    )  # Assuming a fixed range for the 1:1 line

    # Configure scatter plot axes and title
    scatter_ax.set_xlabel("SWOT mean SWH [m]")
    scatter_ax.set_ylabel("S-1 IW SWH most likely [m]")
    scatter_ax.set_title("SWOT vs SAR SWH")
    scatter_ax.set_xlim(0, 8)
    scatter_ax.set_ylim(0, 8)
    scatter_ax.grid(True, linestyle="--", alpha=0.6)
    scatter_ax.set_aspect("equal", adjustable="box")

    # Calculate and display statistics
    stats = calculate_statistics(gdf_filtered)
    stats_text = (
        f"N points: {stats['N']}\n"
        f"Bias: {stats['Bias']:.3f} m\n"
        f"RMSE: {stats['RMSE']:.3f} m\n"
        f"Corr (R): {stats['Correlation (R)']:.3f}\n"
        f"SI: {stats['SI (%)']:.2f} %"
    )
    scatter_ax.text(
        0.05,
        0.95,
        stats_text,
        transform=scatter_ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.5", fc="aliceblue", alpha=0.8),
    )
    scatter_ax.legend(loc="lower right")

    # --- Finalization and saving ---
    fig.suptitle("SWOT / Sentinel-1 Collocation Analysis", fontsize=20, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for main title

    if output_filename is not None:
        print(f"--- STEP 4: Saving figure to '{output_filename}'... ---")
        plt.savefig(output_filename, dpi=150, bbox_inches="tight")
        plt.close(fig)
    print("--- Done. ---")
    return fig
