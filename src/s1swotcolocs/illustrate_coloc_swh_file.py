import os

import geoviews as gv
import holoviews as hv
import hvplot.pandas  # noqa
import numpy as np
import panel as pn
from scipy.stats import (
    gaussian_kde,  # <-- Add this import at the top of your file
    pearsonr,
)
from shapely import wkt

from s1swotcolocs.illustrate_coloc_swh_file_pyplot import create_combined_gdf

# --- Configuration ---
hv.extension("bokeh")
pn.extension()


# # --- 1. Data Loading and Helper Functions ---
# def create_combined_gdf(fseastatecoloc):
#     """
#     Reads a NetCDF file and creates a single GeoDataFrame containing polygon geometries
#     and all relevant data (SAR SWH, SWOT SWH) for linking.
#     """
#     # This function is correct and remains unchanged.
#     ds_l2c = xr.open_dataset(fseastatecoloc)
#     corner_lon, corner_lat = ds_l2c["corner_longitude"], ds_l2c["corner_latitude"]
#     ordered = [0, 1, 3, 2]
#     records = []
#     for i in range(ds_l2c.dims["tile_line"]):
#         for j in range(ds_l2c.dims["tile_sample"]):
#             lon_corners = corner_lon.isel(tile_line=i, tile_sample=j).values.ravel()[
#                 ordered
#             ]
#             lat_corners = corner_lat.isel(tile_line=i, tile_sample=j).values.ravel()[
#                 ordered
#             ]
#             poly = Polygon(list(zip(lon_corners, lat_corners)))
#             records.append(
#                 {
#                     "geometry": poly,
#                     "hs_sar": ds_l2c["hs_most_likely"]
#                     .isel(tile_line=i, tile_sample=j)
#                     .item(),
#                     "swh_swot": ds_l2c["swh_karin_mean"]
#                     .isel(tile_line=i, tile_sample=j)
#                     .item(),
#                     "hs_conf": ds_l2c["hs_conf"]
#                     .isel(tile_line=i, tile_sample=j)
#                     .item(),
#                 }
#             )
#     combined_gdf = gpd.GeoDataFrame(records, crs="EPSG:4326")
#     mask = np.isfinite(combined_gdf["swh_swot"]) & np.isfinite(combined_gdf["hs_sar"])
#     return combined_gdf[mask].reset_index(drop=True), ds_l2c


def calculate_statistics(gdf, x_col="swh_swot", y_col="hs_sar"):
    """Calculates statistics. Unchanged."""
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


# --- Main Application Function ---
def create_linked_dashboard(data_file):
    print("--- STEP 1: Reading and processing data file... ---")
    gdf, ds_l2c = create_combined_gdf(data_file)
    print(f"--- Data loaded successfully. Found {len(gdf)} valid data points. ---")

    # --- NEW: Calculate point density ---
    print("--- STEP 1.5: Calculating point density... ---")
    if not gdf.empty:
        try:
            # Extract x and y data for the density calculation
            x = gdf["swh_swot"].values
            y = gdf["hs_sar"].values
            # mask = np.isfinite(x) & np.isfinite(y)
            xy = np.vstack([x, y])

            # Calculate density using Gaussian Kernel Density Estimation
            z = gaussian_kde(xy)(xy)
            # Add the density values as a new column to the DataFrame
            gdf["density"] = z
            # gdf['density'] = np.ones(len(x))
            print("--- Density calculation successful. ---")
        except Exception as e:
            print(
                f"--- WARNING: Density calculation failed: {e}. Points will not be colored by density. ---"
            )
            gdf["density"] = 1  # Assign a default value if calculation fails
    else:
        gdf["density"] = []

    # --- Plotting Configuration ---
    vmin, vmax = (0, 8)
    cmap = "viridis"
    width, height = (550, 500)

    print("--- STEP 2: Creating plot components... ---")
    # Manually create the stream we will use for the statistics pane.
    selection_stream = hv.streams.Selection1D()

    sar_subswath_polygon = wkt.loads(ds_l2c.attrs["footprint"])
    tools = ["box_select", "lasso_select", "hover"]
    # Create the map plots (unchanged)
    polygons_sar = gdf.hvplot.polygons(
        geo=True,
        color="hs_sar",
        cmap=cmap,
        clim=(vmin, vmax),
        hover_cols=tools,
        line_color="black",
        line_width=0.5,
        title="SAR S-1 IW VV",
        colorbar=True,
        width=width,
        height=height,
    ).opts(
        hv.opts.Polygons(tools=tools, nonselection_alpha=0.2, selection_color="orange")
    )
    polygons_swot = gdf.hvplot.polygons(
        geo=True,
        color="swh_swot",
        cmap=cmap,
        clim=(vmin, vmax),
        hover_cols=tools,
        line_color="black",
        line_width=0.5,
        title="SWOT KarIn",
        colorbar=True,
        width=width,
        height=height,
    ).opts(
        hv.opts.Polygons(tools=tools, nonselection_alpha=0.2, selection_color="orange")
    )
    polygons_sar_hs_conf = gdf.hvplot.polygons(
        geo=True,
        color="hs_conf",
        cmap="bwr",
        clim=(-1, 1),
        hover_cols=tools,
        line_color="black",
        line_width=0.5,
        title="SAR Hs conf",
        colorbar=True,
        width=width,
        height=height,
    ).opts(
        hv.opts.Polygons(tools=tools, nonselection_alpha=0.2, selection_color="orange")
    )

    subswath_contour_polygon = gv.Path(sar_subswath_polygon).opts(alpha=0.4, color="r")

    # --- MODIFIED: Create a standard, interactive scatter plot colored by density ---
    # scatter_plot = gdf.hvplot.scatter(
    #     x='swh_swot', y='hs_sar',
    #     # c='density',  # Use the new 'density' column for color
    #     color=hv.dim('density'),
    #     cmap='inferno',  # 'inferno' or 'magma' are great for density plots
    #     hover_cols=['hs_sar', 'swh_swot'] # , 'density'
    # ).opts(
    #     hv.opts.Scatter(
    #         size=4,
    #         selection_color='orange',
    #         nonselection_alpha=0.2,
    #         tools=tools,
    #         colorbar=True,  # Add a colorbar for the density scale
    #         clabel='Point Density',
    #         line_color=None,  # <--- ADD THIS LINE: Set line_color to None
    #         hatch_color=None    # <-- THE FIX: Explicitly set hatch_color to None
    #     )
    # )
    # --- THE CORRECTED CODE ---
    def create_scatter_plot(gdf, c_col, cmap, clabel, title):
        """Creates a configured and styled scatter plot."""
        return gdf.hvplot.scatter(
            x="swh_swot", y="hs_sar", c=c_col, hover_cols=["hs_sar", "swh_swot", c_col]
        ).opts(
            hv.opts.Scatter(
                cmap=cmap,
                size=6,
                alpha=0.8,
                selection_color="orange",
                nonselection_alpha=0.2,
                tools=["box_select", "lasso_select", "hover"],
                colorbar=True,
                clabel=clabel,
                line_color=None,
            )
            # Note: The overall width, height, and title are best set on the final Overlay
        )

    # In create_linked_dashboard:
    scatter_plot = create_scatter_plot(
        gdf[["swh_swot", "hs_sar", "density"]],
        "density",
        "inferno",
        "Point Density",
        "SAR vs SWOT with Point Density",
    )
    scatter_plot_conf = create_scatter_plot(
        gdf[["swh_swot", "hs_sar", "hs_conf"]],
        "hs_conf",
        "bwr",
        "Hs SAR confidence",
        "SAR vs SWOT with Hs SAR confidence colors",
    )

    print("--- STEP 3: Linking plots and attaching streams... ---")
    # Manually attach our stream to the scatter plot to get the selection indices.
    selection_stream.source = scatter_plot

    # Use the linker ONLY for visual highlighting between all plots.
    linker = hv.link_selections.instance()
    data_to_link = (
        polygons_sar
        + polygons_swot
        + polygons_sar_hs_conf
        + scatter_plot
        + scatter_plot_conf
    )
    linked_layout = linker(data_to_link)

    # Extract the visually linked plots from the layout
    # linked_sar = linked_layout[0, 0]
    # linked_swot = linked_layout[0, 1]
    # linked_sar_conf = linked_layout[0, 2]
    # linked_scatter_plot = linked_layout[0, 3]
    # linked_scatter_conf_plot = linked_layout[0,4]
    linked_plots = list(linked_layout)
    linked_sar = linked_plots[0]
    linked_swot = linked_plots[1]
    linked_sar_conf = linked_plots[2]
    linked_scatter_plot = linked_plots[3]
    linked_scatter_conf_plot = linked_plots[4]

    print("--- STEP 4: Assembling final dashboard layout... ---")
    # Build the final scatter view by overlaying the identity line.
    identity_line = hv.Slope(1, 0).opts(color="red", line_width=1.5)

    scatter_color_density = (linked_scatter_plot * identity_line).opts(
        hv.opts.Overlay(
            width=width,
            height=height,
            xlabel="SWOT mean SWH [m]",
            ylabel="S-1 IW SWH most likely [m]",
            title="SWOT vs SAR SWH",
            show_grid=True,
            xlim=(0, 8),
            ylim=(0, 8),
        )
    )

    final_scatter_conf = (linked_scatter_conf_plot * identity_line).opts(
        hv.opts.Overlay(
            width=width,
            height=height,
            xlabel="SWOT mean SWH [m]",
            ylabel="S-1 IW SWH most likely [m]",
            title="SWOT vs SAR SWH",
            show_grid=True,
            xlim=(0, 8),
            ylim=(0, 8),
        )
    )

    # Create the Dynamic Statistics Pane (unchanged)
    def create_stats_pane(index):
        subset_gdf = gdf if not index else gdf.iloc[index]
        stats = calculate_statistics(subset_gdf)
        title = "### Global Statistics" if not index else "### Selection Statistics"
        stats_text = f"""
        {title}
        | Metric          | Value      |
        |-----------------|------------|
        | **N points**    | {stats['N']}         |
        | **Bias (m)**    | {stats['Bias']:.3f}    |
        | **RMSE (m)**    | {stats['RMSE']:.3f}    |
        | **SI (%)**      | {stats['SI (%)']:.2f}    |
        | **Correlation** | {stats['Correlation (R)']:.3f}    |
        """
        return pn.pane.Markdown(stats_text, width=250, align="center")

    dynamic_stats_pane = pn.bind(create_stats_pane, index=selection_stream.param.index)

    # Assemble the Final Dashboard Layout
    background_tiles = hv.element.tiles.EsriImagery().opts(width=width, height=height)
    final_map_sar = background_tiles * linked_sar * subswath_contour_polygon
    final_map_swot = background_tiles * linked_swot * subswath_contour_polygon
    final_map_sar_confhs = background_tiles * linked_sar_conf * subswath_contour_polygon
    # scatter_with_stats = pn.Column(final_scatter, )
    # scatter_color_density = final_scatter
    # scatter_color_density_conf = final_scatter_conf
    final_app = pn.Row(
        pn.Tabs(
            ("SAR", final_map_sar),
            ("SWOT", final_map_swot),
            ("SAR_Hs_confidence", final_map_sar_confhs),
        ),
        pn.Tabs(
            ("density", scatter_color_density),
            ("Hs confidence SAR", final_scatter_conf),
        ),
        dynamic_stats_pane,
    )

    print("--- Dashboard object created successfully. Ready to be served. ---")
    return final_app


# --- This is the main execution block for a script ---
if __name__ == "__main__":
    # Define the path to your data file
    # Replace this with the actual path to your NetCDF file
    fseastatecoloc = "dummy_coloc.nc"

    if not os.path.exists(fseastatecoloc):
        print(f"ERROR: Data file not found at '{fseastatecoloc}'")
        # Here you could add a function call to create a dummy file if needed
    else:
        # Create the dashboard object
        app = create_linked_dashboard(fseastatecoloc)

        # This line marks the 'app' object as the one to be displayed
        # when you use the `panel serve` command.
        app.servable(title="Linked SWH Analysis Dashboard")
