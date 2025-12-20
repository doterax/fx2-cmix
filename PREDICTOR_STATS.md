# Predictor Statistics Tracking

This feature tracks the accuracy of each predictor model during compression and generates visualization graphs.

## How It Works

1. **Statistics Collection**: During compression, each predictor's output is tracked and compared against the actual bit value. Every 100,000 bits, statistics are written to `predictor_stats.csv`.

2. **Metrics**:
   - **Hit**: When a predictor's probability > 0.5 and the actual bit is 1, OR probability < 0.5 and the actual bit is 0
   - **Miss**: All other cases
   - **Accuracy**: Percentage of hits out of total predictions in each interval

3. **Output**: CSV file with columns:
   - `bit_position`: Current bit position in the compression
   - `file_percentage`: Estimated percentage of file processed (0-100%)
   - `<predictor_name>`: Accuracy percentage for each predictor

## Usage

### 1. Run Compression with Statistics

```powershell
./cmix.exe -p generic no-preprocess ./prof_input/input .\res.bin
```

This will create `predictor_stats.csv` in the current directory.

### 2. Visualize Results

```powershell
python plot_predictor_stats.py predictor_stats.csv predictor_accuracy.png
```

Or using the virtual environment:

```powershell
& .venv\Scripts\python.exe plot_predictor_stats.py predictor_stats.csv predictor_accuracy.png
```

### 3. View the Plot

The script will:
- Generate a PNG image with multiple colored lines showing each predictor's accuracy over time
- Print a summary table with mean, standard deviation, and range for each predictor
- Display the plot (if running interactively)

## Configuration

You can adjust the statistics collection interval by modifying `stats_interval_` in `generic_full_predictor.h`:

```cpp
unsigned long long stats_interval_ = 100000; // Write stats every N bits
```

## Tracked Predictors

The following predictors are tracked:
- **bracket_model**: Bracket context model
- **fxcm_model**: FXCM model
- **direct_0**: Direct hash model
- **match_0-9**: Match models
- **indirect_ns_0-14**: Indirect non-stationary models
- **indirect_r_0**: Indirect run-map model
- **byte_model**: PPMD byte model
- **byte_mixer**: LSTM byte mixer (typically highest accuracy)

## Example Output

```
Predictor Accuracy Summary:
============================================================
bracket_model       : Mean= 46.5% Std=  1.6% Range=[ 45.0%- 48.6%]
fxcm_model          : Mean= 46.5% Std=  1.6% Range=[ 45.0%- 48.6%]
...
byte_mixer          : Mean= 94.7% Std=  2.3% Range=[ 91.5%- 96.6%]
```

The byte_mixer typically shows the highest accuracy as it operates at the byte level rather than bit level.

## Notes

- Statistics are reset after each interval write, so each data point represents accuracy for that specific 100K-bit window
- The file percentage is estimated based on bit progress; accuracy may vary depending on file size
- Lower accuracy predictors are still valuable as they contribute diverse information to the ensemble
