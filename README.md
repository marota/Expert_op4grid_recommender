# ExpertOp4Grid Recommender

[![License: MPL 2.0](https://img.shields.io/badge/License-MPL_2.0-brightgreen.svg)](https://opensource.org/licenses/MPL-2.0)
[![Python Version](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

Expert system recommender for power grid contingency analysis based on ExpertOp4Grid principles. This tool analyzes N-1 contingencies in Grid2Op/pypowsybl environments, builds overflow graphs, applies expert rules to filter potential actions, and identifies relevant corrective measures to alleviate line overloads.

---

## Features

* **Contingency Simulation**: Simulates N-1 contingencies in a Grid2Op environment.
* **Overflow Graph Generation**: Builds and visualizes overflow graphs using `alphaDeesp` and `networkx`.
* **Expert Rule Engine**: Filters potential grid actions (line switching, topology changes) based on predefined rules derived from operator expertise.
* **Action Prioritization**: Identifies and scores relevant corrective actions (line reconnections, disconnections, node splitting/merging).
* **Modular Structure**: Organized code for better maintainability and testing.

---

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/marota/Expert_op4grid_recommender.git
    cd Expert_op4grid_recommender
    ```

2.  **Recommended: Create a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install the package and dependencies:**
    * **Core dependencies:** Make sure you have the necessary libraries installed. If `alphaDeesp` or specific `grid2op` versions are not on PyPI, you might need to install them manually first according to their own instructions.
    * **Install this package:** For development (recommended), use editable mode:
        ```bash
        pip install -e .
        ```
        Or for a standard installation:
        ```bash
        pip install .
        ```
    * **Install test dependencies (optional):**
        ```bash
        pip install -e .[test]
        ```

---

## Usage Example

Configure the desired scenario in `expert_op4grid_recommender/config.py` (Date, Timestep, Contingency Lines, etc.).

Then, run the main analysis script from the **project root directory**:

```bash
python expert_op4grid_recommender/main.py --date 2024-08-28 --timestep 36 --lines-defaut FRON5L31LOUHA P.SAOL31RONCI
```

The script will:

1.  Set up the Grid2Op environment.
2.  Simulate the specified contingency.
3.  Build and save an overflow graph visualization in the `Overflow_Graph/` directory.
4.  Apply expert rules to filter actions loaded from the action space file.
5.  Identify and print a list of prioritized corrective actions.

-----

An option that can be activated for specific use is to rebuild an action space from one segmentation of a grid to another or the full grid:

```bash
python expert_op4grid_recommender/main.py --rebuild-actions --repas-file allLogics.json --grid-snapshot-file data/snapshot/pf_20240828T0100Z_20240828T0100Z.xiidm
```

From all known logics on the full grid, and targeted action ids in the ACTION_FILE, it rebuilds the actions to be applied on the grid snapshot (in detailed topology format with switches) at the date of interest.

## Configuration

Key parameters can be adjusted in `expert_op4grid_recommender/config.py`:

  * `DATE`, `TIMESTEP`, `LINES_DEFAUT`: Define the specific case to analyze.
  * `ENV_FOLDER`, `ENV_NAME`: Specify the Grid2Op environment location.
  * `ACTION_FILE_PATH`: Path to the JSON file containing the action space.
  * `USE_DC_LOAD_FLOW`: Set to `True` to use DC power flow if AC flow fails.
  * `PARAM_OPTIONS_EXPERT_OP`: Thresholds and parameters for the overflow graph analysis.

-----

## Action Discovery and Scoring

After building the overflow graph and filtering candidate actions with expert rules, the `ActionDiscoverer` evaluates and scores each candidate action by type. The resulting scores are returned in an `action_scores` dictionary with four keys:

```python
action_scores = {
    "line_reconnection":  {action_id: score, ...},
    "line_disconnection": {action_id: score, ...},
    "open_coupling":      {action_id: score, ...},
    "close_coupling":     {},  # placeholder
}
```

### Line Reconnection Score (delta-theta)

Reconnection candidates are lines that appear on dispatch paths of the overflow graph and are currently disconnected but reconnectable. The score is the **voltage angle difference** (delta-theta) across the line's endpoints:

```
score = |theta_or - theta_ex|
```

A lower delta-theta indicates that the line can be reconnected with less stress on the grid. Actions are sorted by ascending delta-theta (lower is better).

### Line Disconnection Score (asymmetric bell curve)

Disconnection candidates are lines on the constrained path (blue path) of the overflow graph. The score evaluates whether the redispatch flow from disconnecting the line falls within a useful range:

**Flow bounds:**
- `max_overload_flow`: maximum absolute redispatch flow on the overflow graph (MW)
- `min_redispatch = (rho_max_overloaded - 1.0) * max_overload_flow` -- the minimum flow needed to bring the worst overloaded line below 100%
- `max_redispatch`: the binding constraint across all lines with increased loading, computed as:

```
For each line with delta_rho > 0:
    ratio = capacity_line * (1 - rho_before) / (rho_after - rho_before)
    max_redispatch = min(max_redispatch, ratio)
```

**Scoring function:** An asymmetric bell curve based on a Beta(3.0, 1.5) kernel, normalized so the peak equals 1 and occurs at 80% of the [min, max] range (i.e., closer to max_redispatch):

```
x = (observed_flow - min_redispatch) / (max_redispatch - min_redispatch)

If 0 <= x <= 1:  score = Beta_kernel(x; alpha=3.0, beta=1.5) / peak_value
If x < 0:        score = -2.0 * x^2        (quadratic penalty)
If x > 1:        score = -2.0 * (x - 1)^2  (quadratic penalty)
```

The score is positive when the disconnection relieves the right amount of flow, with higher scores for actions closer to the maximum useful redispatch. It becomes negative when the redispatch is too small (ineffective) or too large (would create new overloads).

### Node Splitting Score (open coupling -- weighted repulsion)

Node splitting candidates are substations that are either hubs of the overflow graph or lie on the constrained path. The scoring uses `AlphaDeesp` to evaluate how well splitting a substation into two buses separates opposing flows.

The score is based on the **weighted repulsion** of flows on the bus of interest:

```
TotalFlow = NegativeInflow + NegativeOutflow + PositiveInflow + PositiveOutflow

For upstream (amont) nodes:
    Repulsion = NegativeOutflow - PositiveOutflow
    WeightFactor = (NegativeOutflow - OtherFlows) / TotalFlow

For downstream (aval) nodes:
    Repulsion = NegativeInflow - PositiveInflow
    WeightFactor = (NegativeInflow - OtherFlows) / TotalFlow

Score = WeightFactor * Repulsion
```

A higher score indicates a better separation of the overload-relieving (negative/red) flows from the overload-aggravating (positive/green) flows.

### Node Merging Score (close coupling)

Node merging scoring is not yet implemented (placeholder).

-----

## Dependencies

This project relies on several external libraries, including:

  * `numpy`
  * `pandas`
  * `networkx`
  * `pypowsybl`
  * `grid2op` (Ensure you have a compatible version installed)
  * `alphaDeesp` (Ensure this library is installed in your environment)
  * `expertop4grid>=0.2.8`

See `pyproject.toml` for the full list.

-----

## Testing

To run the unit and integration tests, navigate to the project root and use `pytest`:

```bash
pytest
```

*Note: Some integration tests (`@pytest.mark.slow`) require the Grid2Op environment data to be present and may take longer to run.*

-----

## License

This project is licensed under the Mozilla Public License 2.0 (MPL 2.0). See the [LICENSE](LICENSE) file for details.
